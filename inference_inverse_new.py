"""
Inverse Inference Module for CoMaPOI

This script performs inverse inference to generate training data for POI prediction models.
It uses language models to generate synthetic data based on target POIs.
"""
import argparse
import json
import os
import time
import re
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import agentscope
from evaluate import evaluate_poi_predictions
from utils import *
from agents import CustomDictDialogAgent, CustomReActAgent, CustomDialogAgent
from parser_tool import extract_predicted_pois
from tool.base_tools import *
from prompt_provider import PromptProvider
from ft_data import *
from agentscope.message import Msg
from agentscope.service import ServiceToolkit
from candidate_fusion import fuse_candidates

POI_INFO_GLOBAL = None
HSID_INFO_GLOBAL = None

def get_global_poi_and_hsid(args):
    global POI_INFO_GLOBAL, HSID_INFO_GLOBAL
    import os
    import json
    if POI_INFO_GLOBAL is None:
        from candidate_fusion import load_poi_info
        POI_INFO_GLOBAL = load_poi_info(args.dataset)
    if HSID_INFO_GLOBAL is None:
        HSID_INFO_GLOBAL = {}
        if getattr(args, "use_hsid", False):
            hsid_path = getattr(args, "hsid_path", "") or f"dataset_all/{args.dataset}/poi_hsid.json"
            if os.path.exists(hsid_path):
                print(f"[INFO] Process {os.getpid()} loading HSID map from {hsid_path}")
                with open(hsid_path, "r", encoding="utf-8") as f:
                    HSID_INFO_GLOBAL = json.load(f)
            else:
                print(f"[WARN] HSID enabled but file not found: {hsid_path}")
    return POI_INFO_GLOBAL, HSID_INFO_GLOBAL

def enrich_poi_candidates(poi_ids, args):
    """
    Enrich raw POI ID list with detailed metadata (Category, Location, HSID) 
    to empower LLM's spatial and semantic context reasoning.
    """
    if not poi_ids:
        return []
    poi_info_dict, hsid_dict = get_global_poi_and_hsid(args)
    enriched = []
    for idx, poi_id in enumerate(poi_ids):
        poi_str = str(poi_id)
        info = poi_info_dict.get(poi_str, {})
        item = {
            "rank": idx + 1,
            "poi_id": int(poi_id) if poi_str.isdigit() else poi_id,
            "category": info.get("category", "Unknown"),
            "location": [info.get("lat", 0.0), info.get("lon", 0.0)]
        }
        if getattr(args, "use_hsid", False) and poi_str in hsid_dict:
            item["hsid"] = hsid_dict[poi_str].get("hsid_text", "")
        enriched.append(item)
    return enriched

AGENTSCOPE_INITIALIZED = False

POLLUTED_CONTEXT_IDS = set(str(year) for year in range(2000, 2027)) | {
    "25",  # Frequently leaked from "generate 25 candidates".
    "37", "38", "121", "122",  # Frequent latitude/longitude fragments in CA data.
}


def _normalize_poi_id(value, max_item):
    try:
        poi_id = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if 1 <= poi_id <= max_item:
        return str(poi_id)
    return None


def _ordered_unique(values, max_item):
    unique = []
    seen = set()
    for value in values or []:
        poi_id = _normalize_poi_id(value, max_item)
        if poi_id is None or poi_id in seen:
            continue
        unique.append(poi_id)
        seen.add(poi_id)
    return unique


def _label_insert_position(label, user_id, subtrajectory_id, max_position):
    if max_position <= 0:
        return 0
    try:
        seed_value = int(label) + int(user_id) * 17 + int(subtrajectory_id or 0) * 31
    except (TypeError, ValueError):
        seed_value = 0
    return seed_value % max_position


def build_clean_candidate_list(raw_candidates, *, label, fallback_candidates, max_item, target_length,
                               user_id, subtrajectory_id, allowed_candidates=None, keep_label=True,
                               label_position="variable"):
    """
    Clean generated candidate IDs before writing SFT data.

    The inverse generator often leaks prompt numbers such as 25, years, times, or
    coordinates. We keep model-generated IDs only when they are valid and either
    come from the allowed pool or are not obvious context artifacts.
    """
    label_id = _normalize_poi_id(label, max_item)
    allowed = set(_ordered_unique(allowed_candidates, max_item)) if allowed_candidates is not None else None
    fallback = _ordered_unique(fallback_candidates, max_item)

    cleaned = []
    seen = set()
    for poi_id in _ordered_unique(raw_candidates, max_item):
        if poi_id == label_id:
            continue
        if allowed is not None and poi_id not in allowed:
            continue
        if poi_id in POLLUTED_CONTEXT_IDS and (allowed is None or poi_id not in allowed):
            continue
        if poi_id not in seen:
            cleaned.append(poi_id)
            seen.add(poi_id)

    for poi_id in fallback:
        if poi_id == label_id:
            continue
        if allowed is not None and poi_id not in allowed:
            continue
        if poi_id not in seen:
            cleaned.append(poi_id)
            seen.add(poi_id)
        if len(cleaned) >= target_length:
            break

    if keep_label and label_id:
        if label_position == "first":
            insert_at = 0
        else:
            max_position = min(5, target_length)
            insert_at = _label_insert_position(label_id, user_id, subtrajectory_id, max_position)
        cleaned = [poi_id for poi_id in cleaned if poi_id != label_id]
        cleaned.insert(insert_at, label_id)

    return cleaned[:target_length]


def build_final_topk_target(label, candidate_poi_list_agent1, candidate_poi_list_agent2,
                            rag_candidates, *, max_item, top_k, user_id, subtrajectory_id,
                            extra_candidates=None, label_position="variable"):
    label_id = _normalize_poi_id(label, max_item)
    pool = _ordered_unique(
        list(extra_candidates or [])
        + list(candidate_poi_list_agent1 or [])
        + list(candidate_poi_list_agent2 or [])
        + list(rag_candidates or []),
        max_item,
    )
    pool = [poi_id for poi_id in pool if poi_id != label_id]
    if label_position == "first":
        insert_at = 0
    else:
        insert_at = _label_insert_position(label_id, user_id, subtrajectory_id, min(3, top_k)) if label_id else 0
    if label_id:
        pool.insert(insert_at, label_id)
    return pool[:top_k]


def is_paper_label_first_style(args):
    return getattr(args, "inverse_rrf_style", "clean") in {"paper_label_first", "paper_label_first_fused"}


def use_fused_inverse_candidates(args):
    return getattr(args, "inverse_rrf_style", "clean") == "paper_label_first_fused"


def json_content(key, value):
    return json.dumps({key: value}, ensure_ascii=False)


# Helper functions for multiprocessing
def init_agents(args):
    """
    Initialize the language model agents for generation tasks.

    Args:
        args: Command line arguments

    Returns:
        tuple: (generator, generator_value) - The initialized agents
    """
    model_configs = [
        {
            "config_name": f"{args.api_type}",
            "model_type": "openai_chat",
            "model_name": f"{args.api_type}",
            "api_key": "EMPTY",
            "client_args": {
                "base_url": f"http://localhost:{args.port}/v1"
            },
            "generate_args": {
                "temperature": 0.5,
                "top_p": 0.95,
                "n": 1,
                "max_tokens": 512,
            }
        },
        {
            "config_name": f"{args.api_type}_value",
            "model_type": "openai_chat",
            "model_name": f"{args.api_type}",
            "api_key": "EMPTY",
            "client_args": {
                "base_url": f"http://localhost:{args.port}/v1"
            },
            "generate_args": {
                "temperature": 0.2,
                "top_p": 0.95,
                "n": 1,
                "max_tokens": 256,
            }
        }
    ]

    global AGENTSCOPE_INITIALIZED
    # Initialize AgentScope with model configurations
    if not AGENTSCOPE_INITIALIZED:
        agentscope.init(model_configs=model_configs, logger_level="CRITICAL", use_monitor=False)
        AGENTSCOPE_INITIALIZED = True
    service_toolkit = ServiceToolkit()
    service_toolkit.add(get_all_information_tool)

    # Create agent instances
    generator = CustomDialogAgent(
        name="Generator",
        sys_prompt="",
        model_config_name=args.api_type
    )

    generator_value = CustomDialogAgent(
        name="Generator",
        sys_prompt="",
        model_config_name=f"{args.api_type}_value"
    )

    return generator, generator_value

def generate_by_agent(agent, prompt):
    """
    Generate content using an agent.

    Args:
        agent: The agent to use for generation
        prompt: The prompt to send to the agent

    Returns:
        str: The generated content
    """
    msg = Msg(name="Generator", content=prompt, role="user")
    response = agent.reply(msg)
    return response.content



def extract_and_clean_poi(prediction, top_k, max_item):
    """
    Extract and clean predicted POIs.

    Args:
        prediction: The prediction text to process
        top_k: Maximum number of POIs to return
        max_item: Maximum POI ID value

    Returns:
        list: Cleaned list of POI IDs
    """
    prediction = extract_predicted_pois(prediction, top_k)
    cleaned_prediction = clean_predicted_pois(prediction, max_item)
    return cleaned_prediction

def extract_text(txt, key):
    """
    Extract text from JSON string for a specific key.
    If JSON parsing fails, uses regex to extract the content.

    Args:
        txt: Input JSON formatted string
        key: Key to extract from the JSON

    Returns:
        str: Extracted content for the key
    """
    try:
        # Try to parse JSON string to dictionary
        data = json.loads(txt)
        value = data.get(key, "")
        # If value is a list, join as string
        if isinstance(value, list):
            return " ".join(value).replace("\n", " ").strip()
        elif isinstance(value, str):
            return value.replace("\n", " ").strip()
        else:
            return str(value)
    except json.JSONDecodeError:
        # If JSON parsing fails, use regex to extract key content
        pattern = rf'"{key}"\s*:\s*(\[.*?\]|\{{.*?\}}|".*?")'
        match = re.search(pattern, txt, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            # Try to parse extracted text as JSON
            try:
                parsed_value = json.loads(extracted_text)
                if isinstance(parsed_value, list):
                    return " ".join(parsed_value).replace("\n", " ").strip()
                elif isinstance(parsed_value, str):
                    return parsed_value.replace("\n", " ").strip()
                else:
                    return str(parsed_value)
            except json.JSONDecodeError:
                # Remove extra symbols and return
                return extracted_text.replace("[", "").replace("]", "").replace("'", "").replace("\n", " ").strip()
        else:
            return ""

def ensure_label_in_list(label, poi_list):
    """
    Ensure that the label is included in the POI list.

    Args:
        label: The label (or labels) to ensure are in the list
        poi_list: The list of POIs

    Returns:
        list: Valid POIs with label included
    """
    valid_pois = []  # Store valid POIs
    seen = set()     # For deduplication

    # Ensure label and poi_list are string format
    if isinstance(label, (int, str)):
        label = [str(label)]
    else:
        label = [str(lbl) for lbl in label]
    poi_list = [str(poi) for poi in poi_list]

    # Prioritize adding labels to valid_pois
    for lbl in label:
        if lbl.isdigit():
            poi_int = int(lbl)
            if 0 <= poi_int and poi_int not in seen:
                valid_pois.append(lbl)
                seen.add(poi_int)

    # Add POIs from poi_list according to clean_predicted_pois rules
    for poi in poi_list:
        if poi.isdigit():
            poi_int = int(poi)
            if 0 <= poi_int and poi_int not in seen:
                valid_pois.append(poi)
                seen.add(poi_int)

    return valid_pois

def ensure_label_first(candidate_list, label):
    """
    Ensure the label is the first item in the candidate list.

    Args:
        candidate_list: List of candidate POIs
        label: The label to place first

    Returns:
        list: Reordered candidate list with label first
    """
    candidate_list = [str(item) for item in candidate_list]  # Ensure list elements are strings
    if str(label) in candidate_list:
        candidate_list.remove(str(label))
    candidate_list.insert(0, str(label))
    return candidate_list

def remove_label(candidate_list, label):
    """
    Remove the label from the candidate list.

    Args:
        candidate_list: List of candidate POIs
        label: The label to remove

    Returns:
        list: Candidate list with label removed
    """
    candidate_list = [str(item) for item in candidate_list]  # Ensure list elements are strings
    candidate_list = [item for item in candidate_list if item != str(label)]
    return candidate_list

def complete_candidate_poi_list(candidate_poi_list, rag_candidates, label, target_length=20):
    """
    Complete the candidate POI list to reach target length.

    Args:
        candidate_poi_list: Current list of candidate POIs
        rag_candidates: Additional candidates from RAG
        label: Label to exclude from candidates
        target_length: Target length for the list

    Returns:
        list: Completed candidate POI list
    """
    # Ensure all inputs are strings
    candidate_poi_list = [str(poi) for poi in candidate_poi_list]
    label = [str(label)] if isinstance(label, (int, str)) else [str(poi) for poi in label]

    # Combine lists and deduplicate
    combined_candidates = list(set(rag_candidates))

    # Exclude POIs already in candidate_poi_list or in label
    remaining_candidates = [poi for poi in combined_candidates if poi not in candidate_poi_list and poi not in label]

    # Complete the list if needed
    while len(candidate_poi_list) < target_length and remaining_candidates:
        next_poi = remaining_candidates.pop(0)
        candidate_poi_list.append(next_poi)

    # Return list trimmed to target_length
    return candidate_poi_list[:target_length]

def complete_negative_poi_list(negative_poi_list, generator_poi_list_from_profile,
                              generator_poi_list_from_rag, rag_candidates, label, target_length=20):
    """
    Complete the negative POI list to reach target length.

    Args:
        negative_poi_list: Current list of negative POIs
        generator_poi_list_from_profile: Candidates from user profile
        generator_poi_list_from_rag: Candidates from RAG
        rag_candidates: Additional candidates from RAG
        label: Label to exclude from candidates
        target_length: Target length for the list

    Returns:
        list: Completed negative POI list
    """
    # Ensure all inputs are strings
    negative_poi_list = [str(poi) for poi in negative_poi_list]
    generator_poi_list_from_profile = [str(poi) for poi in generator_poi_list_from_profile]
    generator_poi_list_from_rag = [str(poi) for poi in generator_poi_list_from_rag]
    rag_candidates = [str(poi) for poi in rag_candidates]
    label = [str(label)] if isinstance(label, (int, str)) else [str(poi) for poi in label]

    # Combine lists and deduplicate
    combined_candidates = list(set(generator_poi_list_from_profile + generator_poi_list_from_rag + rag_candidates))

    # Exclude POIs already in negative_poi_list or in label
    remaining_candidates = [poi for poi in combined_candidates if poi not in negative_poi_list and poi not in label]

    # Complete the list if needed
    while len(negative_poi_list) < target_length and remaining_candidates:
        next_poi = remaining_candidates.pop(0)
        negative_poi_list.append(next_poi)

    # Return list trimmed to target_length
    return negative_poi_list[:target_length]

def process_and_save_profiles(args, generator):
    """
    Process and save user profiles.
    Retrieves or generates user profiles and saves them to a file.

    Args:
        args: Command line arguments
        generator: The generator agent

    Returns:
        list: List of user profiles
    """
    data = args.dataset
    start_point = args.start_point
    n = args.num_samples  # Total number of users in the dataset
    output_file = f'dataset_all/{data}/{data}_historical_summary.jsonl'
    # Ensure the directory exists
    # If file already exists, read content and return
    if os.path.exists(output_file):
        print(f"[INFO] Output file {output_file} already exists. Reading from file.")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]
            print(f"[INFO] Successfully loaded {len(results)} user profiles from {output_file}.")
            return results
        except Exception as e:
            print(f"[ERROR] Failed to read {output_file}: {e}")
            return []

    # If file doesn't exist, generate data
    print(f"[INFO] Output file {output_file} does not exist. Generating user profiles...")
    results = []  # List to store user profiles

    for i in range(start_point, n):
        # Clear agent memory for each user
        generator.memory.clear()

        try:
            # Get user profile and historical information
            summary, historical_information = get_profile_information(generator, user_id=i, data=data)

            # Skip user if profile generation failed
            if not summary or not historical_information:
                continue

            # Prepare data
            result = {
                "user_id": i,
                "historical_information": historical_information,
                "summary": summary
            }
            results.append(result)
        except Exception as e:
            # Log error and continue
            continue

    # Save results to JSONL file
    if results:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"[INFO] All profiles have been saved to {output_file}.")
    else:
        pass
        # No profiles generated, output file not created

    return results

def single_predict_worker(params):
    """
    Worker function for single prediction that can be called by ProcessPoolExecutor.
    This function initializes its own agents to avoid serialization issues.

    Args:
        params: Tuple containing (selected_sample, args, rag_candidates, his_summary)

    Returns:
        tuple: Results of the prediction process
    """
    try:
        selected_sample, args, rag_candidates, his_summary = params
        user_id, subtrajectory_id, label, current_trajectory = parse_user_and_trajectory_train(
            selected_sample.get('messages', []))  # Parse user ID and current trajectory from messages

        # Initialize agents in this process
        generator, generator_value = init_agents(args)

        # Clear agent memory
        generator.memory.clear()
        generator_value.memory.clear()

        # Progress is shown by tqdm
        label_id, category, lat, lon = access_poi_info(args, int(label))

        next_poi_info = [label_id, category, lat, lon]
        inverse_prompter = Inverse_prompter(args, user_id, subtrajectory_id, current_trajectory, next_poi_info)
        forward_prompter = Forwar_prompter(args, user_id, subtrajectory_id, current_trajectory, next_poi_info)

        # Generate historical distribution
        p1 = inverse_prompter.get_a1p1_prompt(his_summary)
        o1 = generate_by_agent(generator, p1)
        o1 = extract_text(o1, "historical_distribution")

        # Generate candidate POIs from profile
        p2 = inverse_prompter.get_a1p2_prompt(o1)
        o2 = generate_by_agent(generator_value, p2)
        o2 = extract_and_clean_poi(o2, top_k=25, max_item=args.max_item)
        o2 = build_clean_candidate_list(
            o2,
            label=label,
            fallback_candidates=rag_candidates,
            max_item=args.max_item,
            target_length=args.num_candidate,
            user_id=user_id,
            subtrajectory_id=subtrajectory_id,
            label_position="first" if is_paper_label_first_style(args) else "variable",
        )

        # Generate recent mobility analysis
        p3 = inverse_prompter.get_a2p1_prompt()
        o3 = generate_by_agent(generator, p3)
        o3 = extract_text(o3, "recent_mobility_analysis")

        # Generate candidate POIs from RAG
        if getattr(args, "use_hsid", False):
            # Limit candidate list length to 50 in prompt to stay within context length constraints
            enriched_rag = enrich_poi_candidates(rag_candidates[:50], args)
            p4 = inverse_prompter.get_a2p2_prompt(o3, enriched_rag)
        else:
            p4 = inverse_prompter.get_a2p2_prompt(o3, rag_candidates[:50])
        o4 = generate_by_agent(generator_value, p4)
        o4 = extract_and_clean_poi(o4, top_k=25, max_item=args.max_item)
        o4 = build_clean_candidate_list(
            o4,
            label=label,
            fallback_candidates=rag_candidates,
            max_item=args.max_item,
            target_length=args.num_candidate,
            user_id=user_id,
            subtrajectory_id=subtrajectory_id,
            allowed_candidates=list(rag_candidates) + [label],
            label_position="first" if is_paper_label_first_style(args) else "variable",
        )

        # Generate negative POI list
        if getattr(args, "use_hsid", False):
            enriched_o2 = enrich_poi_candidates(o2, args)
            enriched_o4 = enrich_poi_candidates(o4, args)
            p5 = inverse_prompter.get_a3p1_prompt(o1, o3, enriched_o2, enriched_o4)
        else:
            p5 = inverse_prompter.get_a3p1_prompt(o1, o3, o2, o4)
        o5 = generate_by_agent(generator_value, p5)
        o5 = extract_and_clean_poi(o5, top_k=20, max_item=args.max_item)
        o5 = complete_negative_poi_list(o5, o2, o4, rag_candidates, label, target_length=20)
        o5 = remove_label(o5, label)

        # Generate forward prompts
        fp1 = forward_prompter.get_a1p1_prompt(his_summary)
        fp2 = forward_prompter.get_a1p2_prompt(o1)
        fp3 = forward_prompter.get_a2p1_prompt()
        if getattr(args, "use_hsid", False):
            enriched_rag = enrich_poi_candidates(rag_candidates[:50], args)
            fp4 = forward_prompter.get_a2p2_prompt(o3, enriched_rag)
            enriched_o2 = enrich_poi_candidates(o2, args)
            enriched_o4 = enrich_poi_candidates(o4, args)
            fp5 = forward_prompter.get_a3p1_prompt(o1, o3, enriched_o2, enriched_o4)
        else:
            fp4 = forward_prompter.get_a2p2_prompt(o3, rag_candidates[:50])
            fp5 = forward_prompter.get_a3p1_prompt(o1, o3, o2, o4)

        # Detailed output is suppressed for cleaner logs

        forward_prompts_list = [fp1, fp2, fp3, fp4, fp5]
        prompts_list = [p1, p2, p3, p4, p5]
        outputs_list = [o1, o2, o3, o4, o5]

        return user_id, subtrajectory_id, label, current_trajectory, prompts_list, forward_prompts_list, outputs_list, rag_candidates
    except Exception as e:
        # Log error and re-raise
        raise


def safe_single_predict_worker(params):
    """Wrap inverse prediction worker so process pool returns pickle-safe results."""
    try:
        return {"ok": True, "result": single_predict_worker(params)}
    except Exception as exc:
        import traceback
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


class InverseInferenceProcessor:
    """
    Main class for inverse inference processing.
    This class handles the initialization of agents and processing of data for inverse inference.
    """

    def __init__(self, args):
        """
        Initialize the InverseInferenceProcessor with command line arguments.

        Args:
            args: Command line arguments parsed by argparse
        """
        self.args = args


    def generate_jsonl_files(self, prompts_dict, outputs_dict, output_dir):
        """
        Generate JSONL files for each prompt for model fine-tuning.

        Args:
            prompts_dict: Dictionary of prompts for each prompt type
            outputs_dict: Dictionary of outputs for each prompt type
            output_dir: Directory to save the JSONL files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        system_instruct = ""

        for prompt_key in prompts_dict.keys():
            jsonl_path = os.path.join(output_dir, f"{prompt_key}_ft_dataset_generated.jsonl")
            prompts_list = prompts_dict[prompt_key]
            outputs_list = outputs_dict[prompt_key]

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i in range(len(prompts_list)):
                    user_content = prompts_list[i]
                    assistant_content = outputs_list[i]

                    messages = [
                        {"role": "system", "content": system_instruct},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]

                    json_obj = {"messages": messages}
                    f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

            # Generated JSONL file with samples

    def split_and_save_by_user_info(self, input_jsonl, output_file, n_neg):
        """
        Split and save JSONL by user info, controlling the number of negative samples per user.

        Args:
            input_jsonl: Input JSONL file path
            output_file: Output file path
            n_neg: Number of negative samples to save per user
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            results = []

            with open(input_jsonl, "r", encoding="utf-8") as f:
                lines = [json.loads(line.strip()) for line in f]

            for line in lines:
                # Extract user_id and subtrajectory_id
                user_info = line["messages"][1]["content"]
                user_info_dict = json.loads(user_info).get("user_info", {})
                user_id = user_info_dict.get("user_id", "unknown")
                subtrajectory_id = user_info_dict.get("subtrajectory_id", "unknown")

                # Modify system content
                system_content = line["messages"][0]["content"] + f" For user_{user_id}_Subtrajectory_{subtrajectory_id}:"
                user_content = line["messages"][1]["content"]
                assistant_contents = line["messages"][2]["content"]

                # Control number of negative samples
                for assistant_content in assistant_contents[:n_neg]:
                    new_line = {
                        "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ]
                    }
                    results.append(new_line)

            output_file = os.path.join(output_file, f"negatives_samples_{n_neg}.jsonl")
            # Save all samples to one file
            with open(output_file, "w", encoding="utf-8") as out_f:
                for result in results:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # All samples saved to file

        except Exception as e:
            print(f"Error during processing: {e}")

    def process_data(self, input_json_file, output_jsonl_file1, output_jsonl_file2, output_jsonl_file3):
        """
        Process data from input JSON file and create three JSONL files for different purposes.

        Args:
            input_json_file: Input JSON file path
            output_jsonl_file1: First output JSONL file path
            output_jsonl_file2: Second output JSONL file path
            output_jsonl_file3: Third output JSONL file path
        """
        # Read the input JSON file
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries1 = []
        entries2 = []
        entries3 = []

        for entry in data:
            user_id = entry['user_id']
            subtrajectory_id = entry['subtrajectory_id']
            label = entry['label']
            outputs_list = entry['outputs_list']
            current_trajectory = entry.get("current_trajectory", "")
            rag_candidates = entry.get("rag_candidates", [])

            # Extract generated intermediate outputs.
            o1, o2, o3, o4, o5 = outputs_list[:5]
            o2 = build_clean_candidate_list(
                o2,
                label=label,
                fallback_candidates=rag_candidates,
                max_item=self.args.max_item,
                target_length=self.args.num_candidate,
                user_id=user_id,
                subtrajectory_id=subtrajectory_id,
                label_position="first" if is_paper_label_first_style(self.args) else "variable",
            )
            o4 = build_clean_candidate_list(
                o4,
                label=label,
                fallback_candidates=rag_candidates,
                max_item=self.args.max_item,
                target_length=self.args.num_candidate,
                user_id=user_id,
                subtrajectory_id=subtrajectory_id,
                allowed_candidates=list(rag_candidates) + [label],
                label_position="first" if is_paper_label_first_style(self.args) else "variable",
            )
            fused_candidates = []
            if use_fused_inverse_candidates(self.args):
                fusion_result = fuse_candidates(
                    self.args,
                    current_trajectory,
                    rag_candidates,
                    o2,
                    o4,
                )
                fused_candidates = fusion_result.fused_candidates
            final_topk = build_final_topk_target(
                label,
                o2,
                o4,
                rag_candidates,
                max_item=self.args.max_item,
                top_k=self.args.top_k,
                user_id=user_id,
                subtrajectory_id=subtrajectory_id,
                extra_candidates=fused_candidates,
                label_position="first" if is_paper_label_first_style(self.args) else "variable",
            )

            prompt_args = self.args
            prompt_provider = PromptProvider(prompt_args, user_id, current_trajectory)

            entries1.extend([
                {
                    "messages": [
                        {"role": "system", "content": "You are a user profiler. Return only valid JSON."},
                        {"role": "user", "content": prompt_provider.get_a1p1_prompt("Generated historical summary is unavailable in SFT export.")},
                        {"role": "assistant", "content": json_content("historical_profile", str(o1 or "").strip())},
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are a POI candidate generator. Return only valid JSON."},
                        {"role": "user", "content": prompt_provider.get_a1p2_prompt(str(o1 or "").strip())},
                        {"role": "assistant", "content": json_content("candidate_poi_list_from_profile", o2)},
                    ]
                },
            ])

            # Agent2 is trained on the same two tasks it performs during forward inference:
            # concise current_profile generation and RAG candidate refinement.
            entries2.extend([
                {
                    "messages": [
                        {"role": "system", "content": "You are a mobility pattern analyzer. Return only valid JSON."},
                        {"role": "user", "content": prompt_provider.get_a2p1_prompt()},
                        {"role": "assistant", "content": json_content("current_profile", str(o3 or "").strip())},
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are a POI candidate generator. Return only valid JSON."},
                        {"role": "user", "content": prompt_provider.get_a2p2_prompt(
                            str(o3 or "").strip(),
                            enrich_poi_candidates(rag_candidates[:50], self.args) if getattr(self.args, "use_hsid", False) else rag_candidates[:50]
                        )},
                        {"role": "assistant", "content": json_content("refined_candidate_from_rag", o4)},
                    ]
                },
            ])

            entries3.append({
                "messages": [
                    {"role": "system", "content": "You are a POI ranker. Return only valid JSON."},
                    {"role": "user", "content": prompt_provider.get_a3p1_prompt(
                        str(o1 or "").strip(),
                        str(o3 or "").strip(),
                        enrich_poi_candidates(o2, self.args) if getattr(self.args, "use_hsid", False) else o2,
                        enrich_poi_candidates(o4, self.args) if getattr(self.args, "use_hsid", False) else o4,
                        fused_candidate_poi_list=enrich_poi_candidates(fused_candidates, self.args) if getattr(self.args, "use_hsid", False) else fused_candidates,
                    )},
                    {"role": "assistant", "content": json_content("next_poi_id", final_topk)},
                ]
            })

        # Shuffle entries for the first JSONL file
        random.shuffle(entries1)

        # Write to the JSONL files
        with open(output_jsonl_file1, 'w', encoding='utf-8') as f1:
            for item in entries1:
                f1.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(output_jsonl_file2, 'w', encoding='utf-8') as f2:
            for item in entries2:
                f2.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(output_jsonl_file3, 'w', encoding='utf-8') as f3:
            for item in entries3:
                f3.write(json.dumps(item, ensure_ascii=False) + '\n')

    def save_generated_informations_to_json(self, generated_informations, file_path):
        """
        Save generated information to a JSON file.

        Args:
            generated_informations: Generated information to save
            file_path: Path to save the JSON file
        """
        # Ensure save directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save data as JSON
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(generated_informations, json_file, indent=4, ensure_ascii=False)
        # Generated information saved to file

    def save_generated_samples(self, generated_informations, results_path, interval_suffix="final"):
        """
        Save generated samples to files.

        Args:
            generated_informations: Dictionary of generated information
            results_path: Path to save results
            interval_suffix: Suffix for output files (e.g., "interim" or "final")
        """
        output_json = f"{results_path}/{interval_suffix}_poi_predictions.json"
        output_directory = f"{results_path}/{interval_suffix}_ft_data/"

        # Save prediction results to JSON file
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(list(generated_informations.values()), f, ensure_ascii=False, indent=4)
        # All generated information results saved to JSON file

        # Initialize dictionaries for each prompt number
        prompts_dict = {i: [] for i in range(5)}  # Five prompts
        outputs_dict = {i: [] for i in range(5)}

        # Collect prompts and outputs by number
        for info in generated_informations.values():
            prompts_list = info["forward_prompts_list"]
            outputs_list = info["outputs_list"]

            for i in range(5):
                # Clean prompts and outputs
                if isinstance(prompts_list[i], list):
                    cleaned_prompt = [
                        " ".join(p.replace("\\n", "\n").replace("\\", "").splitlines()) for p in prompts_list[i]
                    ]
                else:
                    cleaned_prompt = prompts_list[i].replace("\\n", "\n").replace("\\", "").replace("\n", " ")

                if isinstance(outputs_list[i], list):
                    cleaned_output = [
                        " ".join(o.replace("\\n", "\n").replace("\\", "").splitlines()) for o in outputs_list[i]
                    ]
                else:
                    cleaned_output = outputs_list[i].replace("\\n", "\n").replace("\\", "").replace("\n", " ")

                prompts_dict[i].append(cleaned_prompt)
                outputs_dict[i].append(cleaned_output)

        # Generate JSONL files for each prompt
        self.generate_jsonl_files(prompts_dict, outputs_dict, output_directory)
        # All finetune data results saved to directory

        # Split and save the generated JSONL files
        input_path = f"{output_directory}4_ft_dataset_generated.jsonl"
        self.split_and_save_by_user_info(input_path, output_directory, 10)

        # Set up paths for agent training samples
        agent1_output = f"finetune/data/{self.args.dataset}/agent1_train_samples.jsonl"
        agent2_output = f"finetune/data/{self.args.dataset}/agent2_train_samples.jsonl"
        agent3_output = f"finetune/data/{self.args.dataset}/agent3_train_samples.jsonl"

        if not os.path.exists(f"finetune/data/{self.args.dataset}/"):
            os.makedirs(f"finetune/data/{self.args.dataset}/")

        # Process data for agent training
        self.process_data(output_json, agent1_output, agent2_output, agent3_output)
        # Processing complete, files saved to respective paths

    def run_parallel_predict(self):
        """
        Run parallel prediction using multiple processes.
        Sets up the environment, processes samples in parallel, and saves results.

        Returns:
            str: Path to the generated results file
        """
        # Setup paths
        dataset = self.args.dataset
        results_path = f'finetune/data/{dataset}'
        data_path = f'dataset_all/{dataset}/{self.args.mode}'

        os.makedirs(results_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        output_json = f'{results_path}/poi_predictions.json'
        candidate_output_json = data_path + f"/{self.args.dataset}_{self.args.mode}_candidates.jsonl"

        # Load samples
        samples = []
        with open(f'dataset_all/{dataset}/{self.args.mode}/{dataset}_{self.args.mode}.jsonl', 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        print(f"Processing {self.args.num_samples} samples with {self.args.batch_size} parallel workers...")

        # Get candidate list
        user_to_candidate_map = load_candidate_list(candidate_output_json)

        # Load or generate historical profiles
        historical_distribution_path = f'dataset_all/{dataset}/{dataset}_historical_summary.jsonl'
        if not os.path.exists(historical_distribution_path):
            print(f"[INFO] Generating historical profiles...")
            generator, _ = init_agents(self.args)
            historical_summary_list = process_and_save_profiles(self.args, generator)
            global AGENTSCOPE_INITIALIZED
            AGENTSCOPE_INITIALIZED = False
        else:
            print(f"[INFO] Loading historical profiles from {historical_distribution_path}")
            with open(historical_distribution_path, 'r', encoding='utf-8') as f:
                historical_summary_list = [json.loads(line) for line in f]

        # Prepare parameters for parallel processing
        params_list = []
        for i in range(self.args.start_point, self.args.num_samples):
            selected_sample = samples[i % len(samples)]
            user_id, subtrajectory_id, label, current_trajectory = parse_user_and_trajectory_train(
                selected_sample.get('messages', []))
            rag_candidates = user_to_candidate_map.get(user_id, [])
            rag_candidates = [int(candidate) for candidate in rag_candidates[:100]]
            his_summary = next((item for item in historical_summary_list if str(item["user_id"]) == user_id), None)
            params = (selected_sample, self.args, rag_candidates, his_summary)
            params_list.append(params)

        generated_informations = {}

        # Run parallel processing with the standalone worker function
        with ProcessPoolExecutor(max_workers=self.args.batch_size) as executor:
            futures = [executor.submit(safe_single_predict_worker, params) for params in params_list]

            # Use green progress bar with tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating data", colour="green"):
                payload = future.result()
                if not payload["ok"]:
                    raise RuntimeError(
                        f"Inverse worker failed: {payload['error']}\n{payload['traceback']}"
                    )
                user_id, subtrajectory_id, label, current_trajectory, prompts_list, forward_prompts_list, outputs_list, rag_candidates = payload["result"]

                unique_key = f"U_{user_id}_S_{subtrajectory_id}"
                generated_informations[unique_key] = {
                    "user_id": user_id,
                    "subtrajectory_id": subtrajectory_id,
                    "label": label,
                    "current_trajectory": current_trajectory,
                    "prompts_list": prompts_list,
                    "forward_prompts_list": forward_prompts_list,
                    "outputs_list": outputs_list,
                    "rag_candidates": rag_candidates,
                }

        # Save results
        results_file_path = os.path.join(results_path, f"ALL_generated_informations.json")
        self.save_generated_informations_to_json(generated_informations, results_file_path)

        # Save generated samples
        self.save_generated_samples(generated_informations, results_path, interval_suffix="final")

        return results_file_path


def main():
    """
    Main function to parse arguments and run the inverse inference process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="POI Prediction Agent - Inverse Inference")
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process')
    parser.add_argument('--dataset', type=str, default='nyc', choices=['nyc', 'tky', 'ca'], help='Dataset to use')
    parser.add_argument('--top_k', type=int, default=10, help='Top K predictions')
    parser.add_argument('--api_type', type=str, default="qwen2.5:3b", help='API type for model')
    parser.add_argument('--max_item', type=int, default=5091, help='Maximum POI ID value')
    parser.add_argument('--max_retry', type=int, default=1, help='Maximum number of retries')
    parser.add_argument('--start_point', type=int, default=0, help='Starting point for processing')
    parser.add_argument('--test_interval', type=int, default=50, help='Test interval')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of concurrent processes')
    parser.add_argument('--mode', type=str, default='train', help='Mode (train/test)')
    parser.add_argument('--save_id', type=str, default='N1', help='Save ID (N1-N...; T1-T...; C1-C...)')
    parser.add_argument('--port', type=int, default=7863, help='OpenAI-compatible API server port')
    parser.add_argument('--num_candidate', type=int, default=25, help='Number of candidate POIs for agent candidate generation')
    parser.add_argument('--profile_max_tokens', type=int, default=220, help='Maximum profile length guidance for aligned prompts')
    parser.add_argument(
        '--inverse_rrf_style',
        type=str,
        default='clean',
        choices=['clean', 'paper_label_first', 'paper_label_first_fused'],
        help='Inverse SFT target style: clean keeps current debiased labels; paper_label_first restores label-first supervision; paper_label_first_fused also teaches Agent3 with fused candidates.',
    )
    parser.add_argument('--inverse_fusion_strategy', type=str, default='rrf', choices=['union', 'rrf'], help='Fusion strategy used only by paper_label_first_fused.')
    parser.add_argument('--fused_candidate_top_k', type=int, default=50, help='Number of fused candidates for inverse Agent3 SFT.')
    parser.add_argument('--rrf_k', type=float, default=60.0, help='RRF denominator constant for inverse fused candidates.')
    parser.add_argument('--rrf_weights', type=str, default='', help='Comma-separated RRF source weights for inverse fused candidates.')
    parser.add_argument('--history_candidate_k', type=int, default=30, help='Number of history candidates for inverse fused candidates.')
    parser.add_argument('--geo_candidate_k', type=int, default=50, help='Number of geo candidates for inverse fused candidates.')
    parser.add_argument('--category_candidate_k', type=int, default=50, help='Number of category candidates for inverse fused candidates.')
    parser.add_argument('--popular_candidate_k', type=int, default=50, help='Number of popular candidates for inverse fused candidates.')
    parser.add_argument('--use_hsid', action='store_true', help='Enable HSID representation in prompts')
    parser.add_argument('--hsid_path', type=str, default='', help='Path to poi_hsid.json')

    args = parser.parse_args()
    dataset = args.dataset
    args.candidate_fusion_strategy = args.inverse_fusion_strategy if use_fused_inverse_candidates(args) else "none"

    # Set dataset-specific parameters
    args.max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset, 5091)
    if args.num_samples == 0:
        args.num_samples = {"nyc": 3870, "tky": 11850, "ca": 6616}.get(dataset, 3870)  # train 3870, 11850, 6616

    print("Starting inverse inference with arguments:", args)

    # Create processor and run parallel prediction
    processor = InverseInferenceProcessor(args)
    processor.run_parallel_predict()


if __name__ == "__main__":
    if os.name == 'nt':
        # Windows platform needs protection at the main entry point
        import multiprocessing
        multiprocessing.freeze_support()
    main()
