"""
Forward Inference Module for CoMaPOI

This script performs forward inference for POI prediction using a multi-agent approach.
It coordinates three agents (Profiler, Forecaster, and Final_Predictor) to predict the next POI.
"""
import argparse
import json
import os
import time
import re
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import agentscope
from evaluate import evaluate_poi_predictions
from utils import *
from agents import CustomDictDialogAgent, CustomReActAgent, CustomDialogAgent
from parser_tool import extract_json_field, extract_predicted_pois
from tool.base_tools import *
from transformers import AutoTokenizer
from prompt_provider import PromptProvider
from agentscope.message import Msg
from agentscope.service import ServiceToolkit
from candidate_fusion import fuse_candidates, summarize_fusion_result




POI_INFO_GLOBAL = None
HSID_INFO_GLOBAL = None
INFERENCE_LOG_COUNTER = 0
AGENTSCOPE_INITIALIZED = False

from agentscope.agents import AgentBase
AgentBase.speak = lambda self, message: None

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


def extract_and_clean_poi(prediction, top_k, max_item, key_name="next_poi_id", strict=True):
    """
    Extract and clean predicted POIs.

    Args:
        prediction: The prediction text to process
        top_k: Maximum number of POIs to return
        max_item: Maximum POI ID value

    Returns:
        list: Cleaned list of POI IDs
    """
    prediction = extract_predicted_pois(prediction, top_k, key_name=key_name, strict=strict)
    cleaned_prediction = clean_predicted_pois(prediction, max_item)
    return cleaned_prediction


def extract_text_field(content, key_name):
    value = extract_json_field(content, key_name)
    if value is None:
        return content
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def merge_valid_pois(existing_pois, new_pois, top_k):
    """
    Merge existing POIs with new POIs, ensuring no duplicates and respecting top_k limit.

    Args:
        existing_pois: List of existing POI IDs
        new_pois: List of new POI IDs to add
        top_k: Maximum number of POIs to return

    Returns:
        list: Merged list of POI IDs
    """
    # Convert all POIs to strings for consistency
    existing_pois = [str(poi) for poi in existing_pois]
    new_pois = [str(poi) for poi in new_pois]

    # Create a set of existing POIs for O(1) lookup
    existing_set = set(existing_pois)

    # Add new POIs that aren't already in the list
    for poi in new_pois:
        if poi not in existing_set and len(existing_pois) < top_k:
            existing_pois.append(poi)
            existing_set.add(poi)

    return existing_pois


def parse_reasoning_path(json_file_path, user_id):
    """
    Parse reasoning path from a JSON file and extract relevant components.

    Args:
        json_file_path: Path to the JSON file containing reasoning paths
        user_id: User ID to look up

    Returns:
        tuple: (long_term_profile, short_pattern_response, candidate_poi_list_agent1, candidate_poi_list_agent2)
    """
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Find data for the specified user_id
        user_data = next((item for item in data if str(item.get("user_id")) == str(user_id)), None)
        if not user_data or 'reasoning_path' not in user_data:
            raise ValueError(f"User ID {user_id} not found in JSON file or missing reasoning_path.")

        reasoning_path = user_data['reasoning_path']

        if isinstance(reasoning_path, dict):
            long_term_profile = reasoning_path.get("long_term_profile", {})
            short_term_profile = reasoning_path.get("short_term_profile", {})
            candidates = reasoning_path.get("candidates", {})
            return (
                long_term_profile.get("raw") or long_term_profile.get("parsed_profile"),
                short_term_profile.get("raw") or short_term_profile.get("parsed_profile"),
                candidates.get("agent1_top25", []),
                candidates.get("agent2_top25", []),
            )

        reasoning_data = {
            "long_term_profile": None,
            "short_pattern_response": None,
            "candidate_poi_list_agent1": None,
            "candidate_poi_list_agent2": None
        }

        # Extract each part from reasoning_path
        try:
            long_term_profile_start = reasoning_path.find("long_term_profile:")
            short_pattern_response_start = reasoning_path.find("short_pattern_response:")
            candidate_poi_list_agent1_start = reasoning_path.find("candidate_poi_list_agent1:")
            candidate_poi_list_agent2_start = reasoning_path.find("candidate_poi_list_agent2:")
            final_prediction_start = reasoning_path.find(", final_prediction:")

            if min(long_term_profile_start, short_pattern_response_start, candidate_poi_list_agent1_start, candidate_poi_list_agent2_start) < 0:
                raise ValueError("missing expected reasoning_path markers")

            candidate2_end = final_prediction_start if final_prediction_start >= 0 else len(reasoning_path)
            reasoning_data["long_term_profile"] = reasoning_path[long_term_profile_start:short_pattern_response_start].replace("long_term_profile: ", "").strip().rstrip(",")
            reasoning_data["short_pattern_response"] = reasoning_path[short_pattern_response_start:candidate_poi_list_agent1_start].replace("short_pattern_response: ", "").strip().rstrip(",")
            reasoning_data["candidate_poi_list_agent1"] = reasoning_path[candidate_poi_list_agent1_start:candidate_poi_list_agent2_start].replace("candidate_poi_list_agent1: ", "").strip().rstrip(",")
            reasoning_data["candidate_poi_list_agent2"] = reasoning_path[candidate_poi_list_agent2_start:candidate2_end].replace("candidate_poi_list_agent2: ", "").strip().rstrip(",")
        except Exception as parse_error:
            raise ValueError(f"Error parsing reasoning_path: {parse_error}")

        return reasoning_data["long_term_profile"], reasoning_data["short_pattern_response"], reasoning_data["candidate_poi_list_agent1"], reasoning_data["candidate_poi_list_agent2"]

    except Exception as e:
        print(f"Error while parsing reasoning_path: {e}")
        return None, None, None, None


def check_extra_information(long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2):
    """
    Check and extract information from JSON strings.

    Args:
        long_term_profile: Long-term profile JSON string
        short_term_profile: Short-term profile JSON string
        candidate_poi_list_agent1: Candidate POI list from agent 1 JSON string
        candidate_poi_list_agent2: Candidate POI list from agent 2 JSON string

    Returns:
        tuple: Extracted information from each input
    """
    def safe_json_load(data, key):
        try:
            return json.loads(data)[key]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    long_term_profile = safe_json_load(long_term_profile, "historical_profile")
    short_term_profile = safe_json_load(short_term_profile, "current_profile")
    candidate_poi_list_agent1 = safe_json_load(candidate_poi_list_agent1, "candidate_poi_list_from_profile")
    candidate_poi_list_agent2 = safe_json_load(candidate_poi_list_agent2, "refined_candidate_from_rag")

    return long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2


def get_candidate_poi_lists(args, candidate_poi_list_agent2, candidate_poi_list_agent1, rag_candidates):
    """
    Process candidate POI lists from both agents, removing duplicates and ensuring proper length.

    Args:
        args: Command line arguments
        candidate_poi_list_agent2: Candidate POI list from agent 2
        candidate_poi_list_agent1: Candidate POI list from agent 1
        rag_candidates: Additional candidates from RAG

    Returns:
        tuple: (processed_agent2_list, processed_agent1_list)
    """
    def process_poi_list(poi_list, seen_set):
        # Remove duplicates while preserving order
        unique_list = []
        for poi_id in poi_list:
            try:
                poi_id = int(poi_id)
            except (TypeError, ValueError):
                continue
            if not (1 <= poi_id <= args.max_item):
                continue
            poi_id = str(poi_id)
            if poi_id not in seen_set:
                seen_set.add(poi_id)
                unique_list.append(poi_id)

        # Complete list with RAG candidates if needed
        rag_candidates_str = [str(int(rag_id)) for rag_id in rag_candidates if rag_id is not None]
        for rag_id in rag_candidates_str:
            if len(unique_list) >= args.num_candidate:  # Stop if we have enough
                break
            if rag_id not in seen_set:
                seen_set.add(rag_id)
                unique_list.append(rag_id)

        # Truncate to num_candidate length
        return unique_list[:args.num_candidate]

    # Process both candidate lists
    seen_agent2 = set()
    processed_agent2 = process_poi_list(candidate_poi_list_agent2, seen_agent2)

    seen_agent1 = set()
    processed_agent1 = process_poi_list(candidate_poi_list_agent1, seen_agent1)

    return processed_agent2, processed_agent1


def normalize_agent_candidate_lists(args, candidate_poi_list_agent1, candidate_poi_list_agent2, rag_candidates):
    """
    Normalize and backfill agent candidate lists.

    Strategy:
    1. If one agent has no valid candidates, borrow the other agent's candidates.
    2. If both are empty, fall back to RAG candidates.
    3. Pad/truncate each list with RAG candidates to exactly num_candidate items.
    """
    candidate_poi_list_agent1 = candidate_poi_list_agent1 or []
    candidate_poi_list_agent2 = candidate_poi_list_agent2 or []
    rag_candidates = rag_candidates or []

    if not candidate_poi_list_agent1 and candidate_poi_list_agent2:
        candidate_poi_list_agent1 = list(candidate_poi_list_agent2)
    if not candidate_poi_list_agent2 and candidate_poi_list_agent1:
        candidate_poi_list_agent2 = list(candidate_poi_list_agent1)

    processed_agent2, processed_agent1 = get_candidate_poi_lists(
        args,
        candidate_poi_list_agent2,
        candidate_poi_list_agent1,
        rag_candidates,
    )
    return processed_agent1, processed_agent2


def build_prediction_fallback_pool(args, candidate_poi_list_agent1, candidate_poi_list_agent2, rag_candidates,
                                   fused_candidates=None):
    """
    Build a deterministic fallback ranking pool for final prediction.
    """
    fallback = []
    seen = set()
    for source in (fused_candidates, candidate_poi_list_agent2, candidate_poi_list_agent1, rag_candidates):
        for poi_id in source or []:
            try:
                poi_id = int(poi_id)
            except (TypeError, ValueError):
                continue
            if 1 <= poi_id <= args.max_item and poi_id not in seen:
                seen.add(poi_id)
                fallback.append(poi_id)
            if len(fallback) >= max(args.top_k, args.num_candidate):
                return fallback
    return fallback


def get_rag_candidates(user_to_candidate_map, user_id):
    """
    Safely fetch RAG candidates for a user.

    The candidate cache may be incomplete during smoke tests. In that case we
    return an empty list instead of crashing the whole evaluation run.
    """
    if user_id in user_to_candidate_map:
        return user_to_candidate_map[user_id]

    user_id_str = str(user_id)
    if user_id_str in user_to_candidate_map:
        return user_to_candidate_map[user_id_str]

    try:
        user_id_int = int(user_id)
        if user_id_int in user_to_candidate_map:
            return user_to_candidate_map[user_id_int]
    except (TypeError, ValueError):
        pass

    print(f"[WARN] Missing RAG candidates for user_id={user_id}; using empty candidate list.")
    return []


def normalize_poi_ids(values, max_item=None):
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    normalized = []
    seen = set()
    for value in values:
        try:
            poi_id = int(value)
        except (TypeError, ValueError):
            continue
        if max_item is not None and not (1 <= poi_id <= max_item):
            continue
        poi_id = str(poi_id)
        if poi_id not in seen:
            seen.add(poi_id)
            normalized.append(poi_id)
    return normalized


def union_poi_ids(*sources):
    merged = []
    seen = set()
    for source in sources:
        for poi_id in normalize_poi_ids(source):
            if poi_id not in seen:
                seen.add(poi_id)
                merged.append(poi_id)
    return merged


def prediction_parse_status(raw_response, parsed_pois, expected_count, fallback_used=False):
    if fallback_used:
        return "fallback"
    if len(parsed_pois) == expected_count:
        return "ok"
    if parsed_pois:
        return "partial"
    if raw_response in (None, "", []):
        return "missing"
    return "empty"


def build_structured_reasoning(args, long_term_profile, short_pattern_response, rag_candidates,
                               candidate_poi_list_agent1, candidate_poi_list_agent2,
                               init_prediction, init_valid_poi_ids, init_fallback_used,
                               final_prediction, valid_poi_ids, final_fallback_used,
                               fusion_summary=None):
    agent1_candidates = normalize_poi_ids(candidate_poi_list_agent1, args.max_item)
    agent2_candidates = normalize_poi_ids(candidate_poi_list_agent2, args.max_item)
    rag_candidate_ids = normalize_poi_ids(rag_candidates, args.max_item)

    return {
        "long_term_profile": {
            "raw": long_term_profile,
            "parsed_profile": extract_text_field(long_term_profile, "historical_profile"),
        },
        "short_term_profile": {
            "raw": short_pattern_response,
            "parsed_profile": extract_text_field(short_pattern_response, "current_profile"),
        },
        "candidates": {
            "rag_top100": rag_candidate_ids[:100],
            "agent1_top25": agent1_candidates[:args.num_candidate],
            "agent2_top25": agent2_candidates[:args.num_candidate],
            "candidate_union": union_poi_ids(agent1_candidates, agent2_candidates),
            "fused_top": (fusion_summary or {}).get("fused_top", []),
            "fusion": fusion_summary or {},
        },
        "initial_prediction": {
            "raw": init_prediction,
            "parsed_top_k": normalize_poi_ids(init_valid_poi_ids, args.max_item)[:args.top_k],
            "parse_status": prediction_parse_status(
                init_prediction,
                init_valid_poi_ids,
                args.top_k,
                fallback_used=init_fallback_used,
            ),
        },
        "final_prediction": {
            "raw": final_prediction,
            "parsed_top_k": normalize_poi_ids(valid_poi_ids, args.max_item)[:args.top_k],
            "parse_status": prediction_parse_status(
                final_prediction,
                valid_poi_ids,
                args.top_k,
                fallback_used=final_fallback_used,
            ),
        },
    }


def extract_reasoning_candidates(reasoning_path):
    if isinstance(reasoning_path, dict):
        candidates = reasoning_path.get("candidates", {})
        fused = normalize_poi_ids(candidates.get("fused_top", []))
        return {
            "rag": normalize_poi_ids(candidates.get("rag_top100", [])),
            "agent1": normalize_poi_ids(candidates.get("agent1_top25", [])),
            "agent2": normalize_poi_ids(candidates.get("agent2_top25", [])),
            "fused": fused,
            "union": union_poi_ids(candidates.get("candidate_union", []), fused),
        }

    if not isinstance(reasoning_path, str):
        return {"rag": [], "agent1": [], "agent2": [], "fused": [], "union": []}

    def extract_after(marker, next_marker=None):
        start = reasoning_path.find(marker)
        if start < 0:
            return []
        start += len(marker)
        end = reasoning_path.find(next_marker, start) if next_marker else len(reasoning_path)
        if end < 0:
            end = len(reasoning_path)
        return re.findall(r"\b\d+\b", reasoning_path[start:end])

    agent1 = extract_after("candidate_poi_list_agent1:", "candidate_poi_list_agent2:")
    agent2 = extract_after("candidate_poi_list_agent2:", "final_prediction:")
    return {
        "rag": [],
        "agent1": normalize_poi_ids(agent1),
        "agent2": normalize_poi_ids(agent2),
        "fused": [],
        "union": union_poi_ids(agent1, agent2),
    }


def write_prediction_diagnostics(args, predictions, user_to_candidate_map, output_file, metrics=None):
    total = len(predictions)
    length_distribution = Counter()
    parse_status_counts = Counter()
    recall_counts = Counter()
    predicted_from_union = 0
    predicted_total = 0

    for sample in predictions:
        user_id = sample.get("user_id")
        label = str(sample.get("label"))
        predicted = normalize_poi_ids(sample.get("predicted_poi_ids", []), args.max_item)
        length_distribution[str(len(predicted))] += 1

        reasoning_path = sample.get("reasoning_path", {})
        if isinstance(reasoning_path, dict):
            status = reasoning_path.get("final_prediction", {}).get("parse_status", "unknown")
        else:
            status = "legacy_string"
        parse_status_counts[status] += 1

        candidates = extract_reasoning_candidates(reasoning_path)
        rag_candidates = normalize_poi_ids(get_rag_candidates(user_to_candidate_map, user_id), args.max_item)
        if not candidates["rag"]:
            candidates["rag"] = rag_candidates[:100]

        if label in candidates["rag"]:
            recall_counts["rag_top100"] += 1
        if label in candidates["agent1"]:
            recall_counts["agent1_top25"] += 1
        if label in candidates["agent2"]:
            recall_counts["agent2_top25"] += 1
        if label in set(candidates["agent1"]) | set(candidates["agent2"]):
            recall_counts["agent1_or_agent2"] += 1
        if label in candidates["fused"]:
            recall_counts["fused_top"] += 1

        union_candidates = set(candidates["union"])
        for poi_id in predicted:
            predicted_total += 1
            if poi_id in union_candidates:
                predicted_from_union += 1

    def rate(count):
        return round(count / total * 100, 4) if total else 0.0

    diagnostics = {
        "total_samples": total,
        "top_k": args.top_k,
        "num_candidate": args.num_candidate,
        "candidate_fusion_strategy": args.candidate_fusion_strategy,
        "fused_candidate_top_k": args.fused_candidate_top_k,
        "prediction_length_distribution": dict(sorted(length_distribution.items(), key=lambda item: int(item[0]))),
        "top_k_complete": {
            "count": length_distribution.get(str(args.top_k), 0),
            "rate": rate(length_distribution.get(str(args.top_k), 0)),
        },
        "parse_status_counts": dict(parse_status_counts),
        "candidate_recall": {
            "rag_top100": {"hits": recall_counts["rag_top100"], "rate": rate(recall_counts["rag_top100"])},
            "agent1_top25": {"hits": recall_counts["agent1_top25"], "rate": rate(recall_counts["agent1_top25"])},
            "agent2_top25": {"hits": recall_counts["agent2_top25"], "rate": rate(recall_counts["agent2_top25"])},
            "agent1_or_agent2": {"hits": recall_counts["agent1_or_agent2"], "rate": rate(recall_counts["agent1_or_agent2"])},
            "fused_top": {"hits": recall_counts["fused_top"], "rate": rate(recall_counts["fused_top"])},
        },
        "predicted_from_candidate_union": {
            "count": predicted_from_union,
            "total_predictions": predicted_total,
            "rate": round(predicted_from_union / predicted_total * 100, 4) if predicted_total else 0.0,
        },
    }
    if metrics is not None:
        diagnostics["metrics"] = metrics

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Diagnostics saved to: {output_file}")
    return diagnostics


def init_agents(args):
    """
    Initialize the language model agents for generation tasks.

    Args:
        args: Command line arguments

    Returns:
        tuple: (Profiler, Forecaster, Final_Predictor) - The initialized agents
    """
    # Configure Agent 1 (Profiler)
    model_config_agent1 = {
        "config_name": f"{args.agent1_api}",
        "model_type": "openai_chat",
        "model_name": f"{args.agent1_api}",
        "api_key": "EMPTY",
        "client_args": {
            "base_url": f"http://localhost:{args.port}/v1"
        },
        "generate_args": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.agent1_max_tokens,
        }
    }

    # Configure Agent 2 (Forecaster)
    model_config_agent2 = {
        "config_name": f"{args.agent2_api}",
        "model_type": "openai_chat",
        "model_name": f"{args.agent2_api}",
        "api_key": "EMPTY",
        "client_args": {
            "base_url": f"http://localhost:{args.port}/v1"
        },
        "generate_args": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.agent2_max_tokens
        }
    }

    # Configure Agent 3 (Final_Predictor)
    model_config_agent3 = {
        "config_name": f"{args.agent3_api}",
        "model_type": "openai_chat",
        "model_name": f"{args.agent3_api}",
        "api_key": "EMPTY",
        "client_args": {
            "base_url": f"http://localhost:{args.port}/v1"
        },
        "generate_args": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.agent3_max_tokens
        }
    }

    # Initialize AgentScope with model configurations
    if not AGENTSCOPE_INITIALIZED:
        agentscope.init(model_configs=[model_config_agent1, model_config_agent2, model_config_agent3],
                        logger_level="CRITICAL",
                        use_monitor=False)
        AGENTSCOPE_INITIALIZED = True

    # Create service toolkit and add tools
    service_toolkit = ServiceToolkit()
    service_toolkit.add(get_all_information_tool)

    # Initialize Profiler Agent
    Profiler = CustomDialogAgent(
        name="Profiler",
        sys_prompt="",
        model_config_name=args.agent1_api
    )

    # Initialize Forecaster Agent
    Forecaster = CustomDialogAgent(
        name="Forecaster",
        sys_prompt="",
        model_config_name=args.agent2_api
    )

    # Initialize Final_Predictor Agent
    Final_Predictor = CustomDialogAgent(
        name="Final_Predictor",
        sys_prompt="",
        model_config_name=args.agent3_api
    )

    return Profiler, Forecaster, Final_Predictor


def React_get_profile_information(args, agent, user_id):
    """
    Get profile information for a specific user.

    Args:
        args: Command line arguments
        agent: The agent to use for generation
        user_id: ID of the user to get profile for

    Returns:
        tuple: (summary, historical_information)
    """
    data = args.dataset
    summary, historical_information = get_profile_information(agent, user_id, data)
    return summary, historical_information


def process_single_profile(params):
    """
    Process a single user profile in a separate process.

    Args:
        params: Tuple containing (user_id, args)

    Returns:
        dict: User profile data or None if processing failed
    """
    i, args = params
    Profiler, _, _ = init_agents(args)
    Profiler.memory.clear()

    try:
        # Get user profile and historical information
        summary, historical_information = React_get_profile_information(args, Profiler, i)

        # Skip user if profile generation failed
        if not summary or not historical_information:
            return None

        # Prepare result
        result = {
            "user_id": i,
            "historical_information": historical_information,
        }

        # Verify result can be serialized
        try:
            import pickle
            pickle.dumps(result)
        except Exception as e:
            return None

        return result
    except Exception as e:
        return None


def safe_process_single_profile(params):
    """Wrap profile generation so process pool returns pickle-safe results."""
    try:
        return {"ok": True, "result": process_single_profile(params)}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def React_process_and_save_profiles(args, output_file):
    """
    Process and save user profiles in parallel.

    Args:
        args: Command line arguments
        output_file: Output file path

    Returns:
        list: List of user profiles
    """
    start_point = args.start_point
    n = args.num_samples
    print(f"[INFO] Generating user profiles from {start_point} to {n-1}...")

    # Prepare parameters for parallel processing
    params_list = [(i, args) for i in range(start_point, n)]

    results = []
    if args.batch_size <= 1:
        for params in tqdm(params_list, total=len(params_list), desc="Processing profiles", colour="green"):
            result = process_single_profile(params)
            if result:
                results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
            futures = [executor.submit(safe_process_single_profile, params) for params in params_list]

            # Use green progress bar with tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing profiles", colour="green"):
                payload = future.result()
                if not payload["ok"]:
                    raise RuntimeError(
                        "Profile worker failed: "
                        f"{payload['error']}\n{payload['traceback']}"
                    )
                result = payload["result"]
                if result:
                    results.append(result)

    # Save results to file
    if results:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"[INFO] {len(results)} profiles have been saved to {output_file}.")
    else:
        print(f"[WARN] No profiles generated. Output file not created.")

    return results


def profiler_steps(Profiler, prompt_provider, user_id, current_trajectory, historical_trajectory):
    """
    Execute the Profiler agent steps to generate long-term profile and candidate POIs.

    Args:
        Profiler: The Profiler agent
        prompt_provider: Prompt provider for generating prompts
        user_id: User ID
        current_trajectory: Current trajectory data
        historical_trajectory: Historical trajectory data

    Returns:
        tuple: (long_term_profile, candidate_poi_list_agent1)
    """
    # Generate historical profile
    historical_profile_prompt = prompt_provider.get_a1p1_prompt(historical_trajectory)
    message_his_profile = Msg(name="Profiler", content=historical_profile_prompt, role="user")
    long_term_profile_msg = Profiler.reply(message_his_profile)
    long_term_profile = extract_text_field(long_term_profile_msg.content, "historical_profile")

    # Generate candidate POI list
    merge_profiles_prompt = prompt_provider.get_a1p2_prompt(long_term_profile)
    message_merge_profiles = Msg(name="Profiler", content=merge_profiles_prompt, role="user")
    candidate_poi_list_msg = Profiler.reply(message_merge_profiles)
    candidate_poi_list_agent1 = candidate_poi_list_msg.content

    return long_term_profile, candidate_poi_list_agent1


def forecaster_steps(Forecaster, prompt_provider, rag_candidates):
    """
    Execute the Forecaster agent steps to generate short-term pattern and refined candidates.

    Args:
        Forecaster: The Forecaster agent
        prompt_provider: Prompt provider for generating prompts
        rag_candidates: List of candidate POI IDs for the user

    Returns:
        tuple: (short_pattern_response, candidate_poi_list_agent2)
    """
    # Generate short-term pattern
    short_pattern_prompt = prompt_provider.get_a2p1_prompt()
    message_short_pattern = Msg(name="Forecaster", content=short_pattern_prompt, role="user")
    short_pattern_response = Forecaster.reply(message_short_pattern)
    short_pattern_response = extract_text_field(short_pattern_response.content, "current_profile")

    # Generate refined candidate list
    if getattr(prompt_provider.args, "use_hsid", False):
        enriched_rag = enrich_poi_candidates(rag_candidates, prompt_provider.args)
        refine_candidates_prompt = prompt_provider.get_a2p2_prompt(short_pattern_response, enriched_rag)
    else:
        refine_candidates_prompt = prompt_provider.get_a2p2_prompt(short_pattern_response, rag_candidates)
    message_refine_candidates = Msg(name="Forecaster", content=refine_candidates_prompt, role="user")
    optimized_poi_list_msg = Forecaster.reply(message_refine_candidates)
    candidate_poi_list_agent2 = optimized_poi_list_msg.content

    return short_pattern_response, candidate_poi_list_agent2


def final_prediction_steps(Final_Predictor, prompt_provider, long_term_profile, short_term_profile,
                           candidate_poi_list_agent1, candidate_poi_list_agent2, fused_candidate_poi_list=None):
    """
    Execute the Final_Predictor agent steps to generate the final prediction.

    Args:
        Final_Predictor: The Final_Predictor agent
        prompt_provider: Prompt provider for generating prompts
        long_term_profile: Long-term profile from Profiler
        short_term_profile: Short-term profile from Forecaster
        candidate_poi_list_agent1: Candidate POI list from Profiler
        candidate_poi_list_agent2: Candidate POI list from Forecaster

    Returns:
        tuple: (initial_prediction, final_prediction)
    """
    # Generate initial prediction
    if getattr(prompt_provider.args, "use_hsid", False):
        enriched_a1 = enrich_poi_candidates(candidate_poi_list_agent1, prompt_provider.args)
        enriched_a2 = enrich_poi_candidates(candidate_poi_list_agent2, prompt_provider.args)
        enriched_fused = enrich_poi_candidates(fused_candidate_poi_list, prompt_provider.args)
        prediction_prompt = prompt_provider.get_a3p1_prompt(
            long_term_profile,
            short_term_profile,
            enriched_a1,
            enriched_a2,
            fused_candidate_poi_list=enriched_fused,
        )
    else:
        prediction_prompt = prompt_provider.get_a3p1_prompt(
            long_term_profile,
            short_term_profile,
            candidate_poi_list_agent1,
            candidate_poi_list_agent2,
            fused_candidate_poi_list=fused_candidate_poi_list,
        )
    message_init_prediction = Msg(name="Final_Predictor", content=prediction_prompt, role="user")
    initial_prediction_msg = Final_Predictor.reply(message_init_prediction)
    initial_prediction = initial_prediction_msg.content

    # Use initial prediction as final prediction (no reflection step)
    final_prediction = initial_prediction

    return initial_prediction, final_prediction


def validate_and_retry_sample(args, user_id, valid_poi_ids, Final_Predictor, prompt_provider):
    """
    Validate POI IDs and retry if invalid.

    Args:
        args: Command line arguments
        user_id: User ID
        valid_poi_ids: List of valid POI IDs
        Final_Predictor: The Final_Predictor agent
        prompt_provider: Prompt provider for generating prompts

    Returns:
        list: Validated POI IDs
    """
    max_item = args.max_item

    # Check if POI IDs are valid
    if len(valid_poi_ids) == args.top_k and \
       all(isinstance(poi_id, int) and 1 <= poi_id <= max_item for poi_id in valid_poi_ids) and \
       len(set(valid_poi_ids)) == len(valid_poi_ids):
        return valid_poi_ids  # Return original list if valid

    # Retry with Agent 3
    invalid_poi_ids = valid_poi_ids
    retry_prompt = prompt_provider.agent_retry_prompt(invalid_poi_ids)
    message_init_prediction = Msg(name="Agent3_Retry", content=retry_prompt, role="assistant")
    retry_prediction_msg = Final_Predictor.reply(message_init_prediction)
    retried_valid_poi_ids = extract_and_clean_poi(
        retry_prediction_msg.content,
        args.top_k,
        max_item,
        key_name="next_poi_id",
        strict=True,
    )

    return retried_valid_poi_ids


def single_predict_save(params):
    """
    Predict POIs for a single sample using saved reasoning paths.

    Args:
        params: Tuple containing (selected_sample, args, rag_candidates, his_summary)

    Returns:
        tuple: (user_id, label, valid_poi_ids, init_valid_poi_ids, reasoning_path)
    """
    selected_sample, args, rag_candidates, his_summary = params
    user_id, label, current_trajectory = parse_user_and_trajectory(selected_sample.get('messages', []))

    # Set up paths
    args.saved_results_path = args.saved_results_path
    json_file_path = args.saved_results_path

    # Load reasoning path
    long_term_profile, short_pattern_response, candidate_poi_list_agent1, candidate_poi_list_agent2 = parse_reasoning_path(
        json_file_path, user_id)
    if not all([long_term_profile, short_pattern_response, candidate_poi_list_agent1, candidate_poi_list_agent2]):
        raise ValueError(f"Failed to load reasoning_path for user extracted from input: {user_id}.")

    candidate_poi_list_agent1 = extract_and_clean_poi(
        candidate_poi_list_agent1,
        top_k=args.num_candidate,
        max_item=args.max_item,
        key_name="candidate_poi_list_from_profile",
        strict=False,
    )
    candidate_poi_list_agent2 = extract_and_clean_poi(
        candidate_poi_list_agent2,
        top_k=args.num_candidate,
        max_item=args.max_item,
        key_name="refined_candidate_from_rag",
        strict=False,
    )

    # Apply ablation settings if specified
    if args.ab_type == 'profiler':
        long_term_profile = 'None'
        candidate_poi_list_agent1 = ['0'] * args.num_candidate
        candidate_poi_list_agent2 = ['0'] * args.num_candidate
    elif args.ab_type == 'forecaster':
        short_pattern_response = 'None'
        candidate_poi_list_agent2 = ['0'] * args.num_candidate
    elif args.ab_type == 'candidate':
        candidate_poi_list_agent1 = ['0'] * args.num_candidate
        candidate_poi_list_agent2 = ['0'] * args.num_candidate

    # Initialize agents
    Profiler, Forecaster, Final_Predictor = init_agents(args)
    Final_Predictor.memory.clear()

    # Initialize variables
    valid_poi_ids = []
    init_valid_poi_ids = []

    # Create prompt provider
    prompt_provider = PromptProvider(args, user_id, current_trajectory)

    # rag_candidates is already passed in params
    fallback_prediction_pool = []
    fused_candidates = []
    fusion_summary = None
    if args.ab_type == 'none':
        candidate_poi_list_agent1, candidate_poi_list_agent2 = normalize_agent_candidate_lists(
            args,
            candidate_poi_list_agent1,
            candidate_poi_list_agent2,
            rag_candidates,
        )
        fusion_result = fuse_candidates(
            args,
            current_trajectory,
            rag_candidates,
            candidate_poi_list_agent1,
            candidate_poi_list_agent2,
        )
        fused_candidates = fusion_result.fused_candidates
        fusion_summary = summarize_fusion_result(fusion_result)
        fallback_prediction_pool = build_prediction_fallback_pool(
            args,
            candidate_poi_list_agent1,
            candidate_poi_list_agent2,
            rag_candidates,
            fused_candidates=fused_candidates,
        )

    # Generate predictions
    init_prediction, final_prediction = final_prediction_steps(Final_Predictor, prompt_provider, long_term_profile,
                                                             short_pattern_response, candidate_poi_list_agent1,
                                                             candidate_poi_list_agent2, fused_candidates)

    # Process initial prediction
    init_predicted_pois = extract_and_clean_poi(
        init_prediction,
        top_k=args.top_k,
        max_item=args.max_item,
        key_name="next_poi_id",
        strict=True,
    )
    init_fallback_used = False
    if not init_predicted_pois and fallback_prediction_pool:
        init_fallback_used = True
        init_predicted_pois = fallback_prediction_pool[:args.top_k]
    init_valid_poi_ids = merge_valid_pois(valid_poi_ids, init_predicted_pois, args.top_k)

    # Process final prediction
    predicted_pois = extract_and_clean_poi(
        final_prediction,
        top_k=args.top_k,
        max_item=args.max_item,
        key_name="next_poi_id",
        strict=True,
    )
    final_fallback_used = False
    if not predicted_pois and fallback_prediction_pool:
        final_fallback_used = True
        predicted_pois = fallback_prediction_pool[:args.top_k]
    valid_poi_ids = merge_valid_pois(valid_poi_ids, predicted_pois, args.top_k)

    reasoning_path = build_structured_reasoning(
        args,
        long_term_profile,
        short_pattern_response,
        rag_candidates,
        candidate_poi_list_agent1,
        candidate_poi_list_agent2,
        init_prediction,
        init_valid_poi_ids,
        init_fallback_used,
        final_prediction,
        valid_poi_ids,
        final_fallback_used,
        fusion_summary=fusion_summary,
    )

    return user_id, label, valid_poi_ids[:args.top_k], init_valid_poi_ids[:args.top_k], reasoning_path


def safe_single_predict_save(params):
    """Wrap saved-reasoning prediction for process-pool-safe error transport."""
    try:
        return {"ok": True, "result": single_predict_save(params)}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def single_predict(params):
    """
    Predict POIs for a single sample using the full agent pipeline.

    Args:
        params: Tuple containing (selected_sample, args, rag_candidates, his_summary)

    Returns:
        tuple: (user_id, label, valid_poi_ids, init_valid_poi_ids, reasoning_path)
    """
    selected_sample, args, rag_candidates, his_summary = params
    user_id, label, current_trajectory = parse_user_and_trajectory(selected_sample.get('messages', []))

    # Initialize agents
    Profiler, Forecaster, Final_Predictor = init_agents(args)

    # Clear agent memories
    Forecaster.memory.clear()
    Profiler.memory.clear()
    Final_Predictor.memory.clear()

    # Initialize variables
    valid_poi_ids = []
    init_valid_poi_ids = []

    # Create prompt provider
    prompt_provider = PromptProvider(args, user_id, current_trajectory)

    # Apply ablation settings and run appropriate agent steps
    if args.ab_type == 'forecaster':
        # Run Profiler only
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                    current_trajectory, his_summary)
        candidate_poi_list_agent1 = extract_and_clean_poi(
            candidate_poi_list_agent1,
            top_k=args.num_candidate,
            max_item=args.max_item,
            key_name="candidate_poi_list_from_profile",
            strict=True,
        )

        # Skip Forecaster
        short_pattern_response = 'None'
        candidate_poi_list_agent2 = ['0'] * args.num_candidate

    elif args.ab_type == 'profiler':
        # Run Profiler but ignore results
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                    current_trajectory, his_summary)
        candidate_poi_list_agent1 = extract_and_clean_poi(
            candidate_poi_list_agent1,
            top_k=args.num_candidate,
            max_item=args.max_item,
            key_name="candidate_poi_list_from_profile",
            strict=True,
        )
        long_term_profile = 'None'
        candidate_poi_list_agent1 = ['0'] * args.num_candidate

        # Run Forecaster
        short_pattern_response, candidate_poi_list_agent2 = forecaster_steps(Forecaster, prompt_provider,
                                                                           rag_candidates)
        candidate_poi_list_agent2 = extract_and_clean_poi(
            candidate_poi_list_agent2,
            top_k=args.num_candidate,
            max_item=args.max_item,
            key_name="refined_candidate_from_rag",
            strict=True,
        )

    elif args.ab_type == 'candidate':
        # Run Profiler
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                     current_trajectory, his_summary)
        candidate_poi_list_agent1 = ['0'] * args.num_candidate

        # Run Forecaster
        short_pattern_response, candidate_poi_list_agent2 = forecaster_steps(Forecaster, prompt_provider,
                                                                           rag_candidates)
        candidate_poi_list_agent2 = ['0'] * args.num_candidate

    else:
        # Run full pipeline
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                     current_trajectory, his_summary)
        candidate_poi_list_agent1 = extract_and_clean_poi(
            candidate_poi_list_agent1,
            top_k=args.num_candidate,
            max_item=args.max_item,
            key_name="candidate_poi_list_from_profile",
            strict=True,
        )

        short_pattern_response, candidate_poi_list_agent2 = forecaster_steps(Forecaster, prompt_provider,
                                                                           rag_candidates)
        candidate_poi_list_agent2 = extract_and_clean_poi(
            candidate_poi_list_agent2,
            top_k=args.num_candidate,
            max_item=args.max_item,
            key_name="refined_candidate_from_rag",
            strict=True,
        )

    # Get RAG candidates (already in params)
    fallback_prediction_pool = []
    fused_candidates = []
    fusion_summary = None
    if args.ab_type == 'none':
        candidate_poi_list_agent1, candidate_poi_list_agent2 = normalize_agent_candidate_lists(
            args,
            candidate_poi_list_agent1,
            candidate_poi_list_agent2,
            rag_candidates,
        )
        fusion_result = fuse_candidates(
            args,
            current_trajectory,
            rag_candidates,
            candidate_poi_list_agent1,
            candidate_poi_list_agent2,
        )
        fused_candidates = fusion_result.fused_candidates
        fusion_summary = summarize_fusion_result(fusion_result)
        fallback_prediction_pool = build_prediction_fallback_pool(
            args,
            candidate_poi_list_agent1,
            candidate_poi_list_agent2,
            rag_candidates,
            fused_candidates=fused_candidates,
        )

    # Generate predictions
    init_prediction, final_prediction = final_prediction_steps(Final_Predictor, prompt_provider, long_term_profile,
                                                             short_pattern_response, candidate_poi_list_agent1,
                                                             candidate_poi_list_agent2, fused_candidates)

    # Process initial prediction
    init_predicted_pois = extract_and_clean_poi(
        init_prediction,
        top_k=args.top_k,
        max_item=args.max_item,
        key_name="next_poi_id",
        strict=True,
    )
    init_fallback_used = False
    if not init_predicted_pois and fallback_prediction_pool:
        init_fallback_used = True
        init_predicted_pois = fallback_prediction_pool[:args.top_k]
    init_valid_poi_ids = merge_valid_pois(valid_poi_ids, init_predicted_pois, args.top_k)

    # Process final prediction
    predicted_pois = extract_and_clean_poi(
        final_prediction,
        top_k=args.top_k,
        max_item=args.max_item,
        key_name="next_poi_id",
        strict=True,
    )
    final_fallback_used = False
    if not predicted_pois and fallback_prediction_pool:
        final_fallback_used = True
        predicted_pois = fallback_prediction_pool[:args.top_k]
    valid_poi_ids = merge_valid_pois(valid_poi_ids, predicted_pois, args.top_k)

    reasoning_path = build_structured_reasoning(
        args,
        long_term_profile,
        short_pattern_response,
        rag_candidates,
        candidate_poi_list_agent1,
        candidate_poi_list_agent2,
        init_prediction,
        init_valid_poi_ids,
        init_fallback_used,
        final_prediction,
        valid_poi_ids,
        final_fallback_used,
        fusion_summary=fusion_summary,
    )

    return user_id, label, valid_poi_ids[:args.top_k], init_valid_poi_ids[:args.top_k], reasoning_path


def safe_single_predict(params):
    """Wrap forward prediction so process pool returns pickle-safe results."""
    try:
        return {"ok": True, "result": single_predict(params)}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


class ForwardInferenceProcessor:
    """
    Main class for forward inference processing.
    This class handles the initialization of agents and processing of data for forward inference.
    """

    def __init__(self, args):
        """
        Initialize the ForwardInferenceProcessor with command line arguments.

        Args:
            args: Command line arguments parsed by argparse
        """
        self.args = args

    def parallel_predict(self):
        """
        Run parallel prediction using multiple processes.
        Sets up the environment, processes samples in parallel, and saves results.

        Returns:
            str: Path to the generated results file
        """
        args = self.args
        n, dataset, top_k = args.num_samples, args.dataset, args.top_k

        # Set up file paths
        results_path = f'results/{dataset}/{args.save_name}'
        data_path = f'dataset_all/{dataset}/{args.mode}'

        os.makedirs(results_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        output_json = f'{results_path}/poi_predictions.json'
        metrics_txt = f'{results_path}/metrics.txt'
        metrics_csv = f'{results_path}/metrics.csv'
        diagnostics_json = f'{results_path}/diagnostics.json'

        if getattr(args, "use_hsid", False):
            candidate_output_json = data_path + f"/{args.dataset}_{args.mode}_candidates_hsid.jsonl"
        else:
            candidate_output_json = data_path + f"/{args.dataset}_{args.mode}_candidates.jsonl"

        # Load samples
        samples = []
        with open(f'dataset_all/{dataset}/{args.mode}/{dataset}_{args.mode}.jsonl', 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        num_samples = min(n, len(samples))
        samples = samples[:num_samples]

        print(f"Processing {num_samples} samples with {args.batch_size} parallel workers...")

        # Load or generate historical summaries
        historical_distribution_path = f'dataset_all/{args.dataset}/{args.dataset}_historical_summary.jsonl'
        if os.path.exists(historical_distribution_path):
            print(f"[INFO] Loading historical profiles from {historical_distribution_path}")
            with open(historical_distribution_path, 'r', encoding='utf-8') as f:
                historical_summary_list = [json.loads(line) for line in f]
        else:
            print(f"[INFO] Generating historical profiles...")
            historical_summary_list = React_process_and_save_profiles(args, historical_distribution_path)

        # Load candidate list
        user_to_candidate_map = load_candidate_list(candidate_output_json)

        # Prepare parameters for parallel processing
        params_list = []
        for i in range(args.start_point, n):
            selected_sample = samples[i % num_samples]
            user_id, label, current_trajectory = parse_user_and_trajectory(selected_sample.get('messages', []))
            rag_candidates = get_rag_candidates(user_to_candidate_map, user_id)
            his_summary = next((item for item in historical_summary_list if str(item["user_id"]) == user_id), None)
            params = (selected_sample, args, rag_candidates, his_summary)
            params_list.append(params)

        all_predictions = {}

        def save_prediction_result(prediction_tuple):
            user_id, label, valid_poi_ids, init_valid_poi_ids, reasoning_path = prediction_tuple

            global INFERENCE_LOG_COUNTER
            if INFERENCE_LOG_COUNTER == 0:
                import tqdm
                trace_msg = (
                    "\n" + "="*40 + " FIRST SAMPLE TRACE (FOR PROMPT & HSID AUDIT) " + "="*40 + "\n"
                    f"[Sample Info] User ID: {user_id} | Target Label: {label}\n"
                    f"[Reasoning Detail (JSON)]:\n{json.dumps(reasoning_path, indent=2, ensure_ascii=False)}\n"
                    f"[Extracted Final POIs]: {valid_poi_ids}\n"
                    + "="*100 + "\n"
                )
                tqdm.tqdm.write(trace_msg)
                INFERENCE_LOG_COUNTER += 1

            all_predictions[user_id] = {
                "user_id": user_id,
                "label": label,
                "reasoning_path": reasoning_path,
                "predicted_poi_ids": valid_poi_ids,
                "init_valid_poi_ids": init_valid_poi_ids,
            }

            # Save and evaluate at intervals
            if len(all_predictions) % args.test_interval == 0:
                print(f"\n[INFO] Completed {len(all_predictions)} samples. Saving interim results and evaluating.")
                interim_output_json = f'{results_path}/interim_poi_predictions_{len(all_predictions)}.json'

                # Save interim predictions
                with open(interim_output_json, 'w', encoding='utf-8') as f:
                    json.dump(list(all_predictions.values()), f, ensure_ascii=False, indent=4)

                metrics = evaluate_poi_predictions(args, interim_output_json, top_k, metrics_txt, metrics_csv, key='predicted_poi_ids')
                interim_diagnostics_json = f'{results_path}/interim_diagnostics_{len(all_predictions)}.json'
                write_prediction_diagnostics(
                    args,
                    list(all_predictions.values()),
                    user_to_candidate_map,
                    interim_diagnostics_json,
                    metrics=metrics,
                )

        # Run prediction.
        if args.batch_size <= 1:
            predict_fn = single_predict_save if args.load_pf_output else single_predict
            for params in tqdm(params_list, total=len(params_list), desc="Predicting POIs", colour="green"):
                save_prediction_result(predict_fn(params))
        else:
            with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
                submit_fn = safe_single_predict_save if args.load_pf_output else safe_single_predict
                futures = [executor.submit(submit_fn, params) for params in params_list]

                # Use green progress bar with tqdm
                for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting POIs", colour="green"):
                    payload = future.result()
                    if not payload["ok"]:
                        raise RuntimeError(
                            "Prediction worker failed: "
                            f"{payload['error']}\n{payload['traceback']}"
                        )
                    save_prediction_result(payload["result"])

        # Save final results
        print("\n[INFO] Processing complete. Saving final prediction results.")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(list(all_predictions.values()), f, ensure_ascii=False, indent=4)
        print(f"[INFO] All final prediction results saved to: {output_json}")

        metrics = evaluate_poi_predictions(args, output_json, top_k, metrics_txt, metrics_csv, key='predicted_poi_ids')
        write_prediction_diagnostics(
            args,
            list(all_predictions.values()),
            user_to_candidate_map,
            diagnostics_json,
            metrics=metrics,
        )
        print("[INFO] Final evaluation metrics saved.")

        return output_json


def main():
    """
    Main function to parse arguments and run the forward inference process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="POI Prediction Agent - Forward Inference")
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process')
    parser.add_argument('--dataset', type=str, default='nyc', choices=['nyc', 'tky', 'ca'], help='Dataset to use')
    parser.add_argument('--top_k', type=int, default=10, help='Top K predictions')
    parser.add_argument('--model', type=str, default='qwen2.5-1.5b-instruct', help='Base model to use')
    parser.add_argument('--api_type', type=str, default="gpt", help='API type for model')
    parser.add_argument('--max_item', type=int, default=5091, help='Maximum POI ID value')
    parser.add_argument('--max_retry', type=int, default=1, help='Maximum number of retries')
    parser.add_argument('--start_point', type=int, default=0, help='Starting point for processing')
    parser.add_argument('--test_interval', type=int, default=200, help='Test interval for saving and evaluating')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of concurrent processes')
    parser.add_argument('--mode', type=str, default='test', help='Mode (train/test)')
    parser.add_argument('--save_name', type=str, default='N1', help='Save ID (N1-N...; T1-T...; C1-C...)')
    parser.add_argument('--agent1_api', type=str, default='agent1', help='API name for Agent 1 (Profiler)')
    parser.add_argument('--agent2_api', type=str, default='agent2', help='API name for Agent 2 (Forecaster)')
    parser.add_argument('--agent3_api', type=str, default='agent3', help='API name for Agent 3 (Final_Predictor)')
    parser.add_argument('--num_candidate', type=int, default=25, help='Number of candidate POIs')
    parser.add_argument('--store_save_name', action="store_true", help='Manually provide storage name')
    parser.add_argument('--port', type=int, default=7863, help='Port for API server')
    parser.add_argument('--agent1_max_tokens', type=int, default=256, help='Max tokens for Agent 1')
    parser.add_argument('--agent2_max_tokens', type=int, default=256, help='Max tokens for Agent 2')
    parser.add_argument('--agent3_max_tokens', type=int, default=256, help='Max tokens for Agent 3')
    parser.add_argument('--profile_max_tokens', type=int, default=220, help='Target token budget for white-box profile summaries')
    parser.add_argument('--candidate_fusion_strategy', type=str, default='none', choices=['none', 'union', 'rrf'], help='Candidate fusion strategy before final prediction')
    parser.add_argument('--fused_candidate_top_k', type=int, default=50, help='Number of fused candidates to provide to Agent 3')
    parser.add_argument('--rrf_k', type=float, default=60.0, help='RRF denominator constant')
    parser.add_argument('--rrf_weights', type=str, default='', help='Comma-separated source weights, e.g. rag_top100=1.0,history_recent=1.2')
    parser.add_argument('--history_candidate_k', type=int, default=30, help='Number of recent/frequent history candidates')
    parser.add_argument('--geo_candidate_k', type=int, default=50, help='Number of geographic-neighbor candidates')
    parser.add_argument('--category_candidate_k', type=int, default=50, help='Number of category-similar candidates')
    parser.add_argument('--popular_candidate_k', type=int, default=50, help='Number of global popular candidates')
    parser.add_argument('--sub_file', type=str, default='ablation', help='Sub-directory for results')
    parser.add_argument('--load_pf_output', action="store_true", help='Load pre-generated profiles')
    parser.add_argument('--saved_results_path', type=str, default='none', help='Path to saved results')
    parser.add_argument('--op_str', type=str, default='none', help='Operation string')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=1, help='Top-p for generation')
    parser.add_argument('--n', type=int, default=1, help='Number of generations')
    parser.add_argument('--prompt_format', type=str, default="json", help='Prompt format')
    parser.add_argument('--ab_type', type=str, default="none", help='Ablation type')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_hsid', action="store_true", help='Enable HSID representation in prompts')
    parser.add_argument('--hsid_path', type=str, default='', help='Path to poi_hsid.json')

    args = parser.parse_args()
    dataset = args.dataset

    # Set dataset-specific parameters
    args.max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset, 5091)
    if args.num_samples == 0:
        args.num_samples = {"nyc": 988, "tky": 2206, "ca": 1818}.get(dataset, 988)

    # Set save name if not manually provided
    if not args.store_save_name:
        fusion_suffix = f"_fusion_{args.candidate_fusion_strategy}"
        if args.candidate_fusion_strategy != "none":
            fusion_suffix += f"_fused{args.fused_candidate_top_k}"
        args.save_name = f"{args.op_str}/[Forward_Inference_{args.dataset}_{args.model}_{args.batch_size}_{args.num_candidate}_agent1_api_{args.agent1_api}_agent2_api_{args.agent2_api}_agent3_api_{args.agent3_api}{fusion_suffix}]"

    print("Starting forward inference with arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Create processor and run parallel prediction
    processor = ForwardInferenceProcessor(args)
    processor.parallel_predict()


if __name__ == "__main__":
    if os.name == 'nt':
        # Windows platform needs protection at the main entry point
        import multiprocessing
        multiprocessing.freeze_support()
    main()
