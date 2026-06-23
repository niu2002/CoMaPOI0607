import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

logging.getLogger().setLevel(logging.ERROR)

DEFAULT_QUERY_INSTRUCTION = (
    "Given a user's trajectory, retrieve candidate POIs that are semantically relevant "
    "to the user's likely next destination based on category, time, and geography."
)


def _extract_user_and_subtrajectory(content):
    if not content:
        return None, None

    patterns = [
        r'"user_id"\s*:\s*"?(\d+)"?.*?"subtrajectory_id"\s*:\s*"?(\d+)"?',
        r'user_id\s*:\s*"?(\d+)"?.*?subtrajectory_id\s*:\s*"?(\d+)"?',
        r'user_(\d+)_subtrajectory_(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1), match.group(2)

    user_match = re.search(r'"user_id"\s*:\s*"?(\d+)"?|user_id\s*:\s*"?(\d+)"?|user_(\d+)', content, flags=re.IGNORECASE)
    if user_match:
        for group in user_match.groups():
            if group is not None:
                return group, None

    return None, None


def _last_token_pool(last_hidden_states, attention_mask):
    left_padding = bool(torch.all(attention_mask[:, -1] == 1))
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


class EmbeddingModel:
    """
    Real embedding model wrapper for local Qwen embedding models.
    """

    def __init__(
        self,
        model_name_or_path,
        batch_size=8,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
        max_length=2048,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.max_length = max_length

        tokenizer_kwargs = {"padding_side": "left"}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Initialized embedding model from {model_name_or_path} on {self.device}")

    def _prepare_texts(self, texts, is_query):
        if not is_query or not self.query_instruction:
            return texts
        return [f"Instruct: {self.query_instruction}\nQuery: {text}" for text in texts]

    def encode(self, text, is_query=False):
        return self.encode_batch([text], is_query=is_query)[0]

    def encode_batch(self, texts, is_query=False):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        prepared_texts = self._prepare_texts(texts, is_query=is_query)
        all_embeddings = []

        with torch.inference_mode():
            for start in range(0, len(prepared_texts), self.batch_size):
                batch_texts = prepared_texts[start:start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


class RAG_Finder:
    def __init__(self, data, num_test, top_k, max_id, args, mode):
        self.data = data
        self.num_test = num_test
        self.top_k = top_k
        self.max_id = max_id
        self.args = args
        self.mode = mode
        self.project_root = Path(__file__).resolve().parent.parent
        self.dataset_root = self.project_root / "dataset_all"
        self.data_root = self.dataset_root / self.data
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.sample_file = self.resolve_sample_file()
        self.poi_info_file = self.resolve_poi_info_file()
        self.poi_data = self.load_poi_data(self.poi_info_file)

        embedding_model_path = getattr(self.args, "embedding_model_path", "models/Qwen3-Embedding-4B")
        embedding_model_path = Path(embedding_model_path)
        if not embedding_model_path.is_absolute():
            embedding_model_path = (self.project_root / embedding_model_path).resolve()
        self.embedding_model_path = str(embedding_model_path)

        self.bce_model = EmbeddingModel(
            model_name_or_path=self.embedding_model_path,
            batch_size=int(getattr(self.args, "embedding_batch_size", 8)),
            query_instruction=getattr(self.args, "embedding_query_instruction", DEFAULT_QUERY_INSTRUCTION),
            max_length=int(getattr(self.args, "embedding_max_length", 2048)),
        )

        # 根据是否启用 HSID，使用独立的索引文件防止缓存干扰
        self.use_hsid = getattr(self.args, "use_hsid", False)
        index_name = "poi_faiss_index_hsid.bin" if self.use_hsid else "poi_faiss_index.bin"
        self.faiss_index_file = str((self.data_root / index_name).resolve())
        self.numpy_index_file = f"{self.faiss_index_file}.npy"
        
        # 加载离线 HSID 数据
        self.hsid_data = {}
        if self.use_hsid:
            hsid_path = getattr(self.args, "hsid_path", "")
            if not hsid_path:
                hsid_path = str(self.dataset_root / self.data / "poi_hsid.json")
            if os.path.exists(hsid_path):
                print(f"Loading HSID reference mapping from {hsid_path}")
                with open(hsid_path, "r", encoding="utf-8") as f:
                    self.hsid_data = json.load(f)
            else:
                print(f"[WARN] HSID enabled but file not found: {hsid_path}")

        if (HAS_FAISS and not os.path.exists(self.faiss_index_file)) or (not HAS_FAISS and not os.path.exists(self.numpy_index_file)):
            self.init_poi_databank()
        else:
            self.load_search_index()

        self.poi_id_to_index = {int(row["poi_id"]): idx for idx, row in self.poi_data.iterrows()}
        self.index_to_poi_id = {idx: int(row["poi_id"]) for idx, row in self.poi_data.iterrows()}

    def resolve_sample_file(self):
        candidates = [
            self.dataset_root / self.data / self.mode / f"{self.data}_{self.mode}.jsonl",
            self.dataset_root / f"{self.data}_{self.mode}.jsonl",
            self.dataset_root / self.data / f"{self.data}_{self.mode}.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(
            f"Could not find {self.data}/{self.mode} sample file. Tried: {', '.join(str(c) for c in candidates)}"
        )

    def resolve_poi_info_file(self):
        candidates = [
            self.dataset_root / self.data / f"{self.data}_poi_info.csv",
            self.dataset_root / f"{self.data}_poi_info.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(
            f"Could not find POI info CSV for {self.data}. Tried: {', '.join(str(c) for c in candidates)}"
        )

    def init_poi_databank(self):
        self.embeddings = self.generate_poi_embeddings(self.poi_data)
        if HAS_FAISS:
            self.faiss_index = self.create_faiss_index(self.embeddings)
            self.save_faiss_index(self.faiss_index, self.faiss_index_file)
        else:
            np.save(self.numpy_index_file, self.embeddings)

    def load_search_index(self):
        if HAS_FAISS and os.path.exists(self.faiss_index_file):
            self.faiss_index = self.load_faiss_index(self.faiss_index_file)
        elif os.path.exists(self.numpy_index_file):
            self.embeddings = np.load(self.numpy_index_file)
        else:
            raise FileNotFoundError("No existing FAISS or numpy index file found.")

    def load_poi_data(self, file_path):
        return pd.read_csv(file_path)

    def generate_poi_embeddings(self, poi_data):
        descriptions = []
        for _, row in poi_data.iterrows():
            poi_id_str = str(int(row['poi_id']))
            desc = f"POI ID: {poi_id_str}, Category: {row['category']}, Location: {row['lat']}, {row['lon']}"
            # 如果存在 HSID 信息，追加至 embedding 文本的末尾
            if self.use_hsid and poi_id_str in self.hsid_data:
                desc += f", HSID: {self.hsid_data[poi_id_str].get('hsid_text', '')}"
            descriptions.append(desc)
        return self.bce_model.encode_batch(descriptions, is_query=False).astype(np.float32)

    def create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def save_faiss_index(self, index, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        faiss.write_index(index, file_path)

    def load_faiss_index(self, file_path):
        return faiss.read_index(file_path)

    def search_similar_pois(self, query, k=10):
        query_embedding = self.bce_model.encode(query, is_query=True).astype(np.float32)
        if HAS_FAISS:
            distances, indices = self.faiss_index.search(np.expand_dims(query_embedding, axis=0), k)
            score_values = distances[0]
            index_values = indices[0]
        else:
            scores = self.embeddings @ query_embedding
            index_values = np.argsort(scores)[::-1][:k]
            score_values = scores[index_values]

        results = []
        for i, idx in enumerate(index_values):
            poi_id = self.index_to_poi_id[idx]
            poi_info = self.poi_data.iloc[idx]
            results.append(
                {
                    "poi_id": poi_id,
                    "category": poi_info["category"],
                    "lat": float(poi_info["lat"]),
                    "lon": float(poi_info["lon"]),
                    "score": float(score_values[i]),
                }
            )

        return results

    def process_single_sample(self, sample):
        try:
            messages = sample.get("messages", [])
            user_id = None
            current_trajectory = None

            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    extracted_user_id, _ = _extract_user_and_subtrajectory(content)
                    if extracted_user_id:
                        user_id = extracted_user_id
                    current_trajectory = content
                    break

            if not user_id or not current_trajectory:
                return None

            query = f"User trajectory: {current_trajectory}"
            candidates = self.search_similar_pois(query, k=self.top_k)
            candidate_poi_ids = [int(candidate["poi_id"]) for candidate in candidates]

            return {"user_id": str(user_id), "candidates": candidate_poi_ids}

        except Exception as exc:
            print(f"Error processing sample: {exc}")
            return None

    def generate_candidates(self):
        samples = []
        with open(self.sample_file, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        num_samples = min(self.num_test, len(samples))
        samples = samples[:num_samples]

        print(f"Generating candidates for {num_samples} samples from {self.sample_file} ...")

        results = []
        for sample in tqdm(samples, total=len(samples), desc="Processing samples", ascii=True, dynamic_ncols=True):
            result = self.process_single_sample(sample)
            if result:
                results.append(result)

        output_dir = self.data_root / self.mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.data}_{self.mode}_candidates.jsonl"
        with open(output_file, "w", encoding="utf-8") as writer:
            for result in results:
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Candidates saved to {output_file}")
        return str(output_file.resolve())
