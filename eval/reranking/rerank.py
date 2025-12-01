import os
import copy
import json
import argparse
import logging
import datasets
from tqdm import tqdm
from typing import List, Tuple
from utils import utils
from utils.openai_utils import OPENAIBaseEngine
import re
# ---------------------------
# Logging configuration
# ---------------------------
LOG_DIR = "logs"
LOG_FILENAME = "reranker_debug.log"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, LOG_FILENAME)

logging.basicConfig(
    filename=log_path,
    filemode="a",  # append to existing log
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# ---------------------------
# QUERY CONSTRUCTION FUNCTIONS
# ---------------------------

def create_prompt_messages(item: dict, rank_start: int, rank_end: int, index_type: str) -> List[dict]:
    """
    Build the chat-format messages for the LLM, and log the full constructed prompt.
    """
    query = item["query"]
    docs = item["documents"][rank_start:rank_end]
    num_docs = len(docs)

    if index_type == "title_abstract":
        max_words = 300
    elif index_type == "full_paper":
        max_words = 10000
    else:
        raise ValueError(f"Invalid index type: {index_type}")

    passages_str_list = []

    # Start enumeration at 1 to match training logic [1], [2], etc.
    for i, document in enumerate(docs, start=1):
        raw_content = document.get("content", "").strip()
        content = " ".join(raw_content.split()[:max_words])
        passages_str_list.append(f"[{i}] {content}")

    # Use the separator that matches training
    passages_for_prompt = "\n\n---\n\n".join(passages_str_list)

    user_message = (
        f"You are an expert academic paper reranker. "
        f"Your task is to re-order the given list of passages (from [1] to [{num_docs}]) "
        f"based on their relevance to the query.\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passages_for_prompt}\n\n"
        f"Output the ranking as a list of indices enclosed in brackets, separated by ' > '. "
        f"For example: [5] > [1] > [3] > ...\n"
        f"Your ranking (most to least relevant):"
    )

    messages = [{"role": "user", "content": user_message}]

    # Log the constructed prompt (large) at DEBUG level
    try:
        logger.debug("Constructed prompt for query '%s' (docs %d..%d -> %d docs):\n%s",
                     query, rank_start, rank_end - 1, num_docs, user_message)
    except Exception as e:
        # In case the prompt is too large for logging handlers or similar
        logger.debug("Constructed prompt for query '%s' (docs %d..%d -> %d docs). Prompt not shown due to logging error: %s",
                     query, rank_start, rank_end - 1, num_docs, str(e))

    return messages


# ---------------------------
# RESPONSE PROCESSING FUNCTIONS
# ---------------------------

def clean_response(response: str):
    """
    Keep only digits and spaces. Return trimmed string.
    """
    new_response = ''
    for c in response:
        new_response += (c if c.isdigit() else ' ')
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response: List[int]):
    """
    Remove duplicates, preserving order.
    """
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start, rank_end):
    """
    Robustly parses the LLM output for document indices like [1], [2], etc.
    Ignores bullet points (1., 2.) and other noise.
    """
    if permutation is None:
        logger.warning("receive_permutation called with None permutation.")
        return item

    # Regex to find digits explicitly inside square brackets: [10], [2]
    # This discriminates between the doc ID "[2]" and the rank "2."
    matches = re.findall(r"\[(\d+)\]", str(permutation))
    
    # Convert to zero-based indices (Training used 1-based, so we subtract 1)
    parsed = []
    for m in matches:
        try:
            idx = int(m) - 1
            parsed.append(idx)
        except ValueError:
            pass

    logger.debug("Regex parsed indices: %s", parsed)

    # Remove duplicates and filter invalid indices
    parsed_unique = []
    seen = set()
    original_count = rank_end - rank_start
    
    for p in parsed:
        # Ensure index is within the current window range (0 to num_docs-1)
        if p not in seen and 0 <= p < original_count:
            parsed_unique.append(p)
            seen.add(p)
            
    # Pad with any missing documents (append them at the end)
    original_rank = list(range(original_count))
    final_order = parsed_unique + [tt for tt in original_rank if tt not in parsed_unique]
    
    logger.debug("Final applied order: %s", final_order)

    # Apply permutation
    cut_range = copy.deepcopy(item['documents'][rank_start: rank_end])
    for j, x in enumerate(final_order):
        item['documents'][j + rank_start] = copy.deepcopy(cut_range[x])
        
        # Restore metadata if necessary
        if 'rank' in item['documents'][j + rank_start]:
            try:
                item['documents'][j + rank_start]['rank'] = cut_range[j].get('rank')
            except: pass
            
    return item


def permutation_pipeline(model, item: dict, rank_start: int, rank_end: int, index_type: str) -> dict:
    """
    Try to rerank the slice item['documents'][rank_start:rank_end] by calling the LLM.

    On error, reduce the window size and retry until it's below a threshold.
    """
    # decrement_rate and min_count logic from original script
    window_size = rank_end - rank_start
    decrement_rate = max(1, window_size // 5)
    min_count = max(1, window_size // 2)

    # Defensive guard: ensure we don't loop infinitely
    attempt = 0
    max_attempts = 10

    while (rank_end - rank_start) >= min_count and attempt < max_attempts:
        try:
            attempt += 1
            logger.debug("Permutation pipeline attempt %d for query '%s' with doc window [%d:%d] (size=%d)",
                         attempt, item.get("query", "<no-query>"), rank_start, rank_end, rank_end - rank_start)

            messages = create_prompt_messages(item, rank_start, rank_end, index_type)

            # Call model.generate(...)  ensure result is string-like for parsing
            permutation = None
            try:
                permutation = model.generate(messages=messages)
            except TypeError:
                # Some model wrappers use different args, try passing as positional if needed
                try:
                    permutation = model.generate(messages)
                except Exception as e:
                    logger.exception("Model.generate failed with both keyword and positional args: %s", str(e))
                    raise e
            except Exception as e:
                logger.exception("Model.generate raised an exception: %s", str(e))
                raise e

            # Log raw LLM output
            logger.debug("Raw LLM output (type=%s):\n%s", type(permutation).__name__, str(permutation))

            # Attempt to apply permutation
            try:
                return receive_permutation(item, permutation, rank_start, rank_end)
            except Exception as e:
                logger.exception("Error while applying permutation: %s", str(e))
                # If parsing/application fails, reduce window and retry
                rank_end -= decrement_rate
                logger.warning("Reducing document window to size %d and retrying (after parsing/apply error).", rank_end - rank_start)

        except Exception as e:
            # Log and reduce the rank_end window
            old_window = rank_end - rank_start
            rank_end = max(rank_start + min_count, rank_end - decrement_rate)
            new_window = rank_end - rank_start
            logger.exception("Exception during permutation pipeline: %s. Reduced window from %d to %d and retrying.",
                             str(e), old_window, new_window)

    logger.error("Unable to rerank documents for query '%s' after %d attempts. Returning original order.",
                 item.get("query", "<no-query>"), attempt)
    return item


# ---------------------------
# MAIN SCRIPT
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_file", type=str, required=True)
    parser.add_argument("--model", type=str, help="Simulator LLM", default="gpt-4-1106-preview")
    parser.add_argument("--max_k", default=100, type=int, help="Max number of retrieved documents to rerank")
    parser.add_argument("--output_dir", type=str, required=False, default="results/reranking/")
    parser.add_argument("--dataset_path", required=False, default="princeton-nlp/LitSearch")
    parser.add_argument("--use_local_llm", action="store_true", help="Use local LLM instead of GPT-4")
    parser.add_argument("--local_llm_url", default="http://127.0.0.1:8080", help="Local LLM API URL")
    args = parser.parse_args()

    logger.info("Starting reranker with args: %s", vars(args))

    # Initialize model engine
    if args.use_local_llm:
        logger.info("Using local LLM at %s", args.local_llm_url)
        from utils.local_llm_utils import LocalLLMEngine
        model_engine = LocalLLMEngine(api_url=args.local_llm_url)
        model = model_engine
    else:
        # NOTE: your utils.get_gpt4_model should return an object with generate(messages=...) method
        logger.info("Using remote model: %s (azure=True)", args.model)
        model = utils.get_gpt4_model(args.model, azure=True)

    # Load dataset corpus
    logger.info("Loading corpus dataset from %s (split='corpus_clean')", args.dataset_path)
    corpus_data = datasets.load_dataset(args.dataset_path, "corpus_clean", split="full")

    # Read retrieval results
    logger.info("Reading retrieval results file: %s", args.retrieval_results_file)
    retrieval_results = utils.read_json(args.retrieval_results_file)

    # Prepare output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, os.path.basename(args.retrieval_results_file).replace(".json", ".reranked.json"))

    # Determine index_type from filename (second token when split by '.')
    index_type = os.path.basename(args.retrieval_results_file).split(".")[1]
    logger.info("Detected index_type: %s", index_type)

    if index_type == "title_abstract":
        corpusid_to_text = {utils.get_clean_corpusid(item): utils.get_clean_title_abstract(item) for item in corpus_data}
    elif index_type == "full_paper":
        corpusid_to_text = {utils.get_clean_corpusid(item): utils.get_clean_full_paper(item) for item in corpus_data}
    else:
        logger.error("Invalid index type: %s", index_type)
        raise ValueError(f"Invalid index type: {index_type}")

    # Truncate retrieval results to max_k
    for result in retrieval_results:
        original_len = len(result.get("retrieved", []))
        result["retrieved"] = result.get("retrieved", [])[:args.max_k]
        if len(result["retrieved"]) != original_len:
            logger.debug("Truncated retrieved list for query '%s' from %d to %d", result.get("query", "<no-query>"), original_len, len(result["retrieved"]))

    # Prepare reranking_inputs
    reranking_inputs = []
    for query_info in retrieval_results:
        docs_list = []
        for retrieved_corpusid in query_info.get("retrieved", []):
            text = corpusid_to_text.get(retrieved_corpusid, "")
            docs_list.append({
                "content": text,
                "corpusid": retrieved_corpusid
            })
        reranking_inputs.append({
            "query": query_info.get("query", ""),
            "documents": docs_list
        })

    # If output file doesn't exist yet, create a checkpoint copy of retrieval_results
    if not os.path.exists(output_file):
        reranking_outputs = copy.deepcopy(retrieval_results)
        utils.write_json(reranking_outputs, output_file)
        logger.info("Created initial output checkpoint file at: %s", output_file)

    # Main reranking loop
    for item_idx, item in enumerate(tqdm(reranking_inputs)):
        try:
            reranking_outputs = utils.read_json(output_file)

            # Skip queries already processed (preserve idempotency)
            if "pre_reranked" in reranking_outputs[item_idx]:
                logger.debug("Skipping already reranked query idx %d: '%s'", item_idx, reranking_outputs[item_idx].get("query", "<no-query>"))
                continue

            logger.info("Reranking query idx %d: '%s' (num_docs=%d)", item_idx, item.get("query", "<no-query>"), len(item.get("documents", [])))

            # Perform reranking
            reranked_item = permutation_pipeline(model, item, rank_start=0, rank_end=len(item["documents"]), index_type=index_type)

            # Save pre_reranked and new retrieved list (corpusids)
            reranking_outputs[item_idx]["pre_reranked"] = reranking_outputs[item_idx].get("retrieved", [])
            reranking_outputs[item_idx]["retrieved"] = [document.get("corpusid") for document in reranked_item.get("documents", [])]

            # Write checkpoint after each iteration
            utils.write_json(reranking_outputs, output_file, silent=True)
            logger.info("Wrote reranked results for query idx %d to %s", item_idx, output_file)

        except Exception as e:
            logger.exception("Unhandled exception while processing query idx %d: %s", item_idx, str(e))
            # Continue to next query rather than crash the whole run
            continue

    logger.info("Reranking run completed. Logs written to: %s", log_path)