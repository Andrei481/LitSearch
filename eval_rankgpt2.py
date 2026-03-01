import re
import os
import json
import time
import requests
from typing import List
import numpy as np
from tqdm import tqdm


# -----------------------------
# Parsing
# -----------------------------
def parse_ranking(output: str, n: int) -> List[int]:
    ids = [int(x) for x in re.findall(r"\[(\d+)\]", output)]
    seen = set()
    perm = []
    for i in ids:
        if 1 <= i <= n and i not in seen:
            perm.append(i - 1)
            seen.add(i)
    for i in range(n):
        if i not in set(perm):
            perm.append(i)
    return perm[:n]


# -----------------------------
# Load DL19 with Pyserini BM25 on MS MARCO passage
# -----------------------------
def load_dl19_from_json(n_docs: int = 100):
    """
    Load DL19 queries and retrieve BM25 top-k using Pyserini on MS MARCO passage index.
    This matches the paper's setup exactly.
    """
    import ir_datasets
    from pyserini.search.lucene import LuceneSearcher

    print("Loading Pyserini BM25 searcher (MS MARCO passage)...")
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")

    dl19 = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
    queries = {q.query_id: q.text for q in dl19.queries_iter()}
    print(f"Loaded {len(queries)} judged queries")

    parsed = []
    for qid, query_text in tqdm(queries.items(), desc="BM25 retrieval"):
        hits = searcher.search(query_text, k=n_docs)

        candidates = []
        for hit in hits:
            doc = searcher.doc(hit.docid)
            raw = json.loads(doc.raw())
            candidates.append({
                "doc_id": str(hit.docid),
                "title": "",
                "text": raw.get("contents", ""),
                "score": float(hit.score),
            })

        parsed.append({
            "query_id": str(qid),
            "query_text": query_text,
            "candidates": candidates,
        })

    print(f"Retrieved {len(parsed)} queries with BM25 top-{n_docs}")
    return parsed


# -----------------------------
# Reranker
# -----------------------------
class VLLMRankGPTReranker:
    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8000",
        model_name: str = "default",
        window_size: int = 20,
        step: int = 10,
        max_tokens: int = 2048,
        timeout: int = 600,
        max_retries: int = 3,
    ):
        self.api_base = api_base.rstrip("/")
        self.window_size = window_size
        self.step = step
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        if model_name == "default":
            try:
                r = requests.get(f"{self.api_base}/v1/models", timeout=10)
                r.raise_for_status()
                data = r.json().get("data", [])
                model_name = data[0]["id"] if data else "default"
            except Exception:
                pass
        self.model_name = model_name
        print(f"[vLLM] Using model: {self.model_name}")

    def _build_prompt(self, query: str, passages: List[dict]) -> str:
        n = len(passages)
        passages_str_list = []
        for i, p in enumerate(passages):
            text = p.get("text", "")
            title = p.get("title", "")
            if title:
                doc_text = f"Title: {title}\nAbstract: {text}"
            else:
                doc_text = text
            doc_text = " ".join((doc_text or "").split())
            passages_str_list.append(f"[{i+1}] {doc_text}")

        passages_for_prompt = "\n\n---\n\n".join(passages_str_list)

        user_msg = (
            f"You are an expert academic paper reranker. "
            f"Your task is to re-order the given list of passages (from [1] to [{n}]) "
            f"based on their relevance to the query. Respond with only the ranking and nothing else.\n\n"
            f"Example output for 8 passages:\n"
            f"[2] > [5] > [4] > [8] > [6] > [1] > [3] > [7]\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{passages_for_prompt}\n\n"
            f"Your ranking (most to least relevant):"
        )

        # Match training: prompt was repeated twice
        user_msg = user_msg + "\n\n" + user_msg

        return user_msg

    def _call_llm(self, prompt: str, seed: int = None) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed

        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.api_base}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                resp = r.json()
                content = resp["choices"][0]["message"]["content"]

                finish = resp["choices"][0].get("finish_reason", "")
                if finish == "length":
                    print(f"  WARNING: output truncated (max_tokens={self.max_tokens})")

                return content
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise last_err

    def _rerank_window(self, query: str, candidates: List[dict], seed: int = None) -> List[dict]:
        prompt = self._build_prompt(query, candidates)
        output = self._call_llm(prompt, seed=seed)
        perm = parse_ranking(output, len(candidates))

        n_parsed = len([x for x in re.findall(r"\[(\d+)\]", output)
                        if 1 <= int(x) <= len(candidates)])
        if n_parsed < len(candidates):
            print(f"  Parsed {n_parsed}/{len(candidates)} IDs from response")

        return [candidates[i] for i in perm]

    def rerank(self, query: str, candidates: List[dict], seed: int = None) -> List[dict]:
        """Sliding window reranking (bottom-to-top)."""
        items = list(candidates)
        n = len(items)

        if n <= self.window_size:
            return self._rerank_window(query, items, seed=seed)

        end = n
        while end > 0:
            start = max(0, end - self.window_size)
            window = items[start:end]
            window = self._rerank_window(query, window, seed=seed)
            items[start:end] = window

            if start == 0:
                break
            end = end - self.step

        return items


# -----------------------------
# Evaluation using ir_measures
# -----------------------------
def evaluate(items: List[dict], label: str = "run"):
    import ir_measures
    from ir_measures import nDCG, AP, RR, ScoredDoc
    import ir_datasets

    dl19 = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
    qrels = list(dl19.qrels_iter())
    qrel_qids = {q.query_id for q in qrels}

    run = []
    matched_qids = set()
    for item in items:
        qid = item["query_id"]
        if qid not in qrel_qids:
            continue
        matched_qids.add(qid)
        for rank, cand in enumerate(item["candidates"]):
            run.append(ScoredDoc(qid, cand["doc_id"], len(item["candidates"]) - rank))

    print(f"  [{label}] Matched {len(matched_qids)}/{len(items)} queries with qrels")

    if not run:
        print("  ERROR: No matching query IDs!")
        return {}

    metrics = [nDCG@1, nDCG@5, nDCG@10, nDCG@20, nDCG@100, AP@100, RR@10]
    return ir_measures.calc_aggregate(metrics, qrels, run)


# -----------------------------
# TREC run file
# -----------------------------
def write_trec_run(items: List[dict], path: str, tag: str = "run"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            qid = item["query_id"]
            for rank, cand in enumerate(item["candidates"], 1):
                score = 1.0 / rank
                f.write(f"{qid}\tQ0\t{cand['doc_id']}\t{rank}\t{score}\t{tag}\n")
    print(f"Saved TREC run: {path}")


# -----------------------------
# Single reranking run
# -----------------------------
def run_reranking(items: List[dict], reranker: VLLMRankGPTReranker, run_id: int = 1):
    """Perform one full reranking pass over all queries."""
    seed = 42 + run_id
    reranked_items = []
    for item in tqdm(items, desc=f"Reranking (run {run_id})"):
        reranked_candidates = reranker.rerank(item["query_text"], item["candidates"], seed=seed)
        reranked_items.append({
            "query_id": item["query_id"],
            "query_text": item["query_text"],
            "candidates": reranked_candidates,
        })
    return reranked_items


# -----------------------------
# Main
# -----------------------------
def main():
    n_docs = 100
    n_runs = 3  # Paper averages over 3 runs

    print(f"Loading DL19 BM25 top-{n_docs}...")
    items = load_dl19_from_json(n_docs=n_docs)
    print(f"Loaded {len(items)} queries with QIDs")

    # Debug: show first item
    d0 = items[0]
    print(f"\n  QID: {d0['query_id']}")
    print(f"  Query: {d0['query_text'][:100]}")
    print(f"  Num candidates: {len(d0['candidates'])}")
    print(f"  First doc ID: {d0['candidates'][0]['doc_id']}")
    print(f"  First doc text[:200]: {d0['candidates'][0]['text'][:200]}")

    # --- BM25 baseline ---
    print("\n--- BM25 Baseline ---")
    bm25_results = evaluate(items, label="BM25")
    for metric, value in sorted(bm25_results.items(), key=lambda x: str(x[0])):
        print(f"  {str(metric):<20s}: {value:.4f}")

    # --- Reranker (one instance shared across runs) ---
    reranker = VLLMRankGPTReranker(
        api_base="http://127.0.0.1:8000",
        model_name="default",
        window_size=20,
        step=10,
        max_tokens=2048,
    )

    # --- Run reranking n_runs times ---
    all_run_results = []
    os.makedirs("runs", exist_ok=True)

    for run_id in range(1, n_runs + 1):
        print(f"\n{'='*65}")
        print(f"  RERANKING RUN {run_id}/{n_runs}")
        print(f"{'='*65}")

        t0 = time.time()
        reranked_items = run_reranking(items, reranker, run_id=run_id)
        elapsed = time.time() - t0
        print(f"  Run {run_id} time: {elapsed:.1f}s ({elapsed/len(items):.1f}s/query)")

        run_results = evaluate(reranked_items, label=f"Run {run_id}")
        all_run_results.append(run_results)

        for metric, value in sorted(run_results.items(), key=lambda x: str(x[0])):
            print(f"  {str(metric):<20s}: {value:.4f}")

        write_trec_run(
            reranked_items,
            f"runs/dl19_rankgpt_top{n_docs}_run{run_id}.trec",
            tag=f"rankgpt_run{run_id}",
        )

    # --- Averaged results ---
    metrics = list(all_run_results[0].keys())
    avg = {}
    std = {}
    for m in metrics:
        vals = np.array([float(r[m]) for r in all_run_results], dtype=float)
        avg[m] = float(vals.mean())
        std[m] = float(vals.std(ddof=0))

    print(f"\n{'='*65}")
    print(f"  AVERAGED RESULTS OVER {n_runs} RUNS")
    print(f"{'='*65}")

    print(f"\n  {'Metric':<20s} {'BM25':>8s}", end="")
    for i in range(1, n_runs + 1):
        print(f" {'Run'+str(i):>8s}", end="")
    print(f"  {'Avg':>8s}  {'Std':>7s}  {'Delta':>8s}")
    print("  " + "-" * (20 + 10 + 10 * n_runs + 28))

    for m in sorted(metrics, key=lambda x: str(x)):
        b = bm25_results[m]
        a = avg[m]
        s = std[m]
        d = a - b
        sign = "+" if d >= 0 else ""
        print(f"  {str(m):<20s} {b:>8.4f}", end="")
        for run_res in all_run_results:
            print(f" {run_res[m]:>8.4f}", end="")
        print(f"  {a:>8.4f}  {s:>7.4f}  {sign}{d:.4f}")

    print("  " + "=" * (20 + 10 + 10 * n_runs + 28))

    # --- Save ---
    write_trec_run(items, f"runs/dl19_bm25_top{n_docs}.trec", tag="bm25")

    summary = {
        "n_runs": n_runs,
        "bm25": {str(k): float(v) for k, v in bm25_results.items()},
        "per_run": [{str(k): float(v) for k, v in r.items()} for r in all_run_results],
        "averaged": {str(k): float(v) for k, v in avg.items()},
        "std": {str(k): float(v) for k, v in std.items()},
    }
    with open("runs/dl19_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved runs/dl19_evaluation_summary.json")


if __name__ == "__main__":
    main()