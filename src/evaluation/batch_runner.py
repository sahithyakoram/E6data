import json
import argparse
import logging
import time
import os
from pathlib import Path
from evaluation.evaluate_with_llm import evaluate_with_llm

def evaluate_traditional(prompt: str, response: str) -> dict:
    scores = {
        "instruction_following": 7.0 if "follow" in response.lower() else 4.0,
        "coherence_accuracy": 6.5 if len(response.split()) > 20 else 3.0,
        "hallucination_detection": 8.0,
        "style_matching": 5.5,
        "length_penalty": max(0.0, 10.0 - len(response) / 100),
        "assumption_control": 6.0
    }
    explanations = {dim: f"Score based on heuristic for {dim}" for dim in scores}
    return {"scores": scores, "explanations": explanations}

def run_batch_evaluation(
    input_path: Path,
    output_path: Path,
    leaderboard_dim: str,
    weights: dict,
    use_llm: bool = False,
    model: str = "llama-3.3-70b-versatile"
) -> list[dict]:

    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load input file '{input_path}': {e}")
        return []

    results = []
    for item in data:
        agent_id = item.get("agent_id", "<unknown>")
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        try:
            if use_llm:
                logging.info(f"🔍 Scoring with Groq ({model}): {agent_id}")
                eval_result = evaluate_with_llm(prompt, response, model=model)
            else:
                logging.info(f"🧮 Scoring with traditional evaluator: {agent_id}")
                eval_result = evaluate_traditional(prompt, response)

            eval_result["agent_id"] = agent_id

            if weights:
                total = 0.0
                for dim, w in weights.items():
                    score = eval_result["scores"].get(dim, 0.0)
                    total += score * w
                eval_result["scores"]["final"] = round(total, 2)

            results.append({
                "agent_id": agent_id,
                "prompt": prompt,
                "response": response,
                "scores": eval_result["scores"],
                "explanations": eval_result["explanations"]
            })

        except Exception as e:
            logging.warning(f"⚠️ Error evaluating '{agent_id}': {e}")
            continue

        time.sleep(0.5)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved evaluation report to '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to write report to '{output_path}': {e}")

    if not results:
        logging.warning("No results to display on leaderboard.")
        return results

    default_dim = "final" if "final" in results[0]["scores"] else "instruction_following"
    rank_dim = leaderboard_dim or default_dim
    sorted_results = sorted(
        results,
        key=lambda r: r["scores"].get(rank_dim, 0.0),
        reverse=True
    )

    print(f"\nLeaderboard – sorted by '{rank_dim}'")
    for idx, r in enumerate(sorted_results, start=1):
        s = r["scores"]
        line = (
            f"{idx}. {r['agent_id']} – "
            f"IF: {s.get('instruction_following', 0.0)} pts; "
            f"CA: {s.get('coherence_accuracy', s.get('coherence_&_accuracy', 0.0))} pts; "
            f"HD: {s.get('hallucination_detection', 0.0)} pts; "
            f"Style: {s.get('style_matching', 0.0)} pts; "
            f"Length: {s.get('length_penalty', 0.0)} pts; "
            f"Assumption: {s.get('assumption_control', 0.0)} pts"
        )
        if "final" in s:
            line += f"; Final: {s['final']} pts"
        print(line)

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Batch-run agent response evaluations and print a leaderboard."
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Path to JSON file with agent responses.")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Path to write the evaluation report JSON.")
    parser.add_argument("--dim", "-d", type=str, default=None,
                        help="Dimension to sort leaderboard by.")
    parser.add_argument("--weights", "-w", type=Path, default=None,
                        help="Optional JSON file with weights per dimension.")
    parser.add_argument("--use_llm", action="store_true",
                        help="Use Groq-based LLM scoring.")
    parser.add_argument("--model", "-m", type=str, default="llama-3.3-70b-versatile",
                        help="Groq model to use (e.g., llama-3.3-70b-versatile, llama-3.1-8b-instant)")
    parser.add_argument("--batch_id", type=int, default=None,
                        help="Batch ID (1–10) to select corresponding Groq API key.")
    parser.add_argument("--key_map", type=Path, default=Path("config/groq_keys.json"),
                        help="Path to JSON file mapping batch_id to Groq API key.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )

    if args.use_llm and args.batch_id:
        try:
            with args.key_map.open("r", encoding="utf-8") as kf:
                key_map = json.load(kf)
            api_key = key_map.get(str(args.batch_id))
            if not api_key:
                raise ValueError(f"No API key found for batch {args.batch_id}")
            os.environ["GROQ_API_KEY"] = api_key
            logging.info(f"Set Groq API key for batch {args.batch_id}")
        except Exception as e:
            logging.error(f"Failed to set Groq API key: {e}")
            return

    weights = {}
    if args.weights:
        try:
            with args.weights.open("r", encoding="utf-8") as wf:
                weights = json.load(wf)
            logging.info(f"Loaded weights from '{args.weights}'.")
        except Exception as e:
            logging.error(f"Failed to load weights file '{args.weights}': {e}")
            return

    start_time = time.time()
    results = run_batch_evaluation(
        input_path=args.input,
        output_path=args.output,
        leaderboard_dim=args.dim,
        weights=weights,
        use_llm=args.use_llm,
        model=args.model
    )
    elapsed = time.time() - start_time
    logging.info(f"⏱️ Processed {len(results)} items in {elapsed:.2f}s")

if __name__ == "__main__":
    main()





# import json
# import argparse
# import logging
# import time
# import os
# from pathlib import Path
# from evaluation.evaluate_with_llm import evaluate_with_llm
#
# # Traditional scoring logic
# def evaluate_traditional(prompt: str, response: str) -> dict:
#     scores = {
#         "instruction_following": 7.0 if "follow" in response.lower() else 4.0,
#         "coherence_accuracy": 6.5 if len(response.split()) > 20 else 3.0,
#         "hallucination_detection": 8.0,
#         "style_matching": 5.5,
#         "length_penalty": max(0.0, 10.0 - len(response) / 100),
#         "assumption_control": 6.0
#     }
#     explanations = {dim: f"Score based on heuristic for {dim}" for dim in scores}
#     return {"scores": scores, "explanations": explanations}
#
# def run_batch_evaluation(
#     input_path: Path,
#     output_path: Path,
#     leaderboard_dim: str,
#     weights: dict,
#     use_llm: bool = False,
#     model: str = "llama-3.3-70b-versatile",
#     key_map: dict = None
# ) -> list[dict]:
#
#     try:
#         with input_path.open("r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception as e:
#         logging.error(f"Failed to load input file '{input_path}': {e}")
#         return []
#
#     # Load existing results for resume
#     existing_results = {}
#     if output_path.exists():
#         try:
#             with output_path.open("r", encoding="utf-8") as f:
#                 existing_results = {r["agent_id"]: r for r in json.load(f)}
#             logging.info(f"Resuming from {len(existing_results)} previously scored agents.")
#         except Exception:
#             logging.warning("Could not parse existing output file. Starting fresh.")
#
#     for idx, item in enumerate(data):
#         agent_id = item.get("agent_id", f"agent_{idx}")
#         prompt = item.get("prompt", "")
#         response = item.get("response", "")
#
#         if agent_id in existing_results:
#             logging.info(f"Skipping already scored agent: {agent_id}")
#             continue
#
#         try:
#             if use_llm:
#                 key_index = idx // 50
#                 api_key = key_map.get(str(key_index)) if key_map else None
#                 if not api_key:
#                     raise ValueError(f"No API key found for index {key_index}")
#                 os.environ["GROQ_API_KEY"] = api_key
#                 logging.info(f"Using key {key_index} for agent {agent_id}")
#
#                 logging.info(f"Scoring with Groq ({model}): {agent_id}")
#                 eval_result = evaluate_with_llm(prompt, response, model=model)
#             else:
#                 logging.info(f"Scoring with traditional evaluator: {agent_id}")
#                 eval_result = evaluate_traditional(prompt, response)
#
#             eval_result["agent_id"] = agent_id
#
#             if weights:
#                 total = 0.0
#                 for dim, w in weights.items():
#                     score = eval_result["scores"].get(dim, 0.0)
#                     total += score * w
#                 eval_result["scores"]["final"] = round(total, 2)
#
#             existing_results[agent_id] = {
#                 "agent_id": agent_id,
#                 "prompt": prompt,
#                 "response": response,
#                 "scores": eval_result["scores"],
#                 "explanations": eval_result["explanations"]
#             }
#
#         except Exception as e:
#             logging.warning(f"Error evaluating '{agent_id}': {e}")
#             continue
#
#         time.sleep(0.5)
#
#         # Save every 10 agents
#         if idx % 10 == 0:
#             try:
#                 with output_path.open("w", encoding="utf-8") as f:
#                     json.dump(list(existing_results.values()), f, indent=2)
#                 logging.info(f"Checkpoint saved at agent {agent_id}")
#             except Exception as e:
#                 logging.error(f"Failed to write checkpoint: {e}")
#
#     # Final save
#     try:
#         with output_path.open("w", encoding="utf-8") as f:
#             json.dump(list(existing_results.values()), f, indent=2)
#         logging.info(f"Final report saved to '{output_path}'.")
#     except Exception as e:
#         logging.error(f"Failed to write final report: {e}")
#
#     # Leaderboard
#     if not existing_results:
#         logging.warning("No results to display on leaderboard.")
#         return []
#
#     default_dim = "final" if "final" in next(iter(existing_results.values()))["scores"] else "instruction_following"
#     rank_dim = leaderboard_dim or default_dim
#     sorted_results = sorted(
#         existing_results.values(),
#         key=lambda r: r["scores"].get(rank_dim, 0.0),
#         reverse=True
#     )
#
#     print(f"\nLeaderboard – sorted by '{rank_dim}'")
#     for idx, r in enumerate(sorted_results, start=1):
#         s = r["scores"]
#         line = (
#             f"{idx}. {r['agent_id']} – "
#             f"IF: {s.get('instruction_following', 0.0)} pts; "
#             f"CA: {s.get('coherence_accuracy', s.get('coherence_&_accuracy', 0.0))} pts; "
#             f"HD: {s.get('hallucination_detection', 0.0)} pts; "
#             f"Style: {s.get('style_matching', 0.0)} pts; "
#             f"Length: {s.get('length_penalty', 0.0)} pts; "
#             f"Assumption: {s.get('assumption_control', 0.0)} pts"
#         )
#         if "final" in s:
#             line += f"; Final: {s['final']} pts"
#         print(line)
#
#     return list(existing_results.values())
#
# def main():
#     parser = argparse.ArgumentParser(description="Batch-run agent response evaluations and print a leaderboard.")
#     parser.add_argument("--input", "-i", type=Path, required=True, help="Path to JSON file with agent responses.")
#     parser.add_argument("--output", "-o", type=Path, required=True, help="Path to write the evaluation report JSON.")
#     parser.add_argument("--dim", "-d", type=str, default=None, help="Dimension to sort leaderboard by.")
#     parser.add_argument("--weights", "-w", type=Path, default=None, help="Optional JSON file with weights per dimension.")
#     parser.add_argument("--use_llm", action="store_true", help="Use Groq-based LLM scoring.")
#     parser.add_argument("--model", "-m", type=str, default="llama-3.3-70b-versatile", help="Groq model to use.")
#     parser.add_argument("--key_map", type=Path, default=Path("config/groq_keys.json"), help="Path to Groq API key map.")
#
#     args = parser.parse_args()
#
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
#
#     key_map = {}
#     if args.use_llm:
#         try:
#             with args.key_map.open("r", encoding="utf-8") as kf:
#                 key_map = json.load(kf)
#             logging.info(f"Loaded {len(key_map)} Groq API keys from '{args.key_map}'.")
#         except Exception as e:
#             logging.error(f"Failed to load key map: {e}")
#             return
#
#     weights = {}
#     if args.weights:
#         try:
#             with args.weights.open("r", encoding="utf-8") as wf:
#                 weights = json.load(wf)
#             logging.info(f"Loaded weights from '{args.weights}'.")
#         except Exception as e:
#             logging.error(f"Failed to load weights file '{args.weights}': {e}")
#             return
#
#     start_time = time.time()
#     results = run_batch_evaluation(
#         input_path=args.input,
#         output_path=args.output,
#         leaderboard_dim=args.dim,
#         weights=weights,
#         use_llm=args.use_llm,
#         model=args.model,
#         key_map=key_map if args.use_llm else None
#     )
#     elapsed = time.time() - start_time
#     logging.info(f"Processed {len(results)} agents in {elapsed:.2f}s")
#
# if __name__ == "__main__":
#     main()
