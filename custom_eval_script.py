"""Evaluate KV cache methods on benchmarks scored by their own codebases.

Unlike lm_eval_script.py, each task in custom_tasks/ builds prompts and scores
answers with its benchmark's official recipe; the model only generates, through
a CompressedCache built from the same cache flags.

Per-sample hypotheses are appended to ``<results stem>_<task>.jsonl`` as they are
generated, so an interrupted run can continue with ``--resume <results json>``.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm

from cache import build_cache_config
from cache.backends.mlp_values import enable_decode_compilation
from custom_tasks import TASKS, generate
from model.attention_predictor import (
    get_attn_predictor_hook_handles,
    apply_attn_predictor_config,
)
from model.selective_attention import install_selective_attention
from model.unpadded_attention import attention_for_batch
from utils.args import add_cache_args, list_of_strings, validate_cache_args
from utils.logging import get_output_path
from utils.model import get_model_and_tokenizer

# Flags that may change when resuming without invalidating finished samples.
RESUMABLE_FLAGS = {
    "resume",
    "tasks",
    "limit",
    "batch_size",
    "lme_question_types",
    "judge_model",
    "judge_base_url",
    "judge_workers",
    "no_judge",
}


def write_json(path, data):
    temp = path + ".tmp"
    with open(temp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(temp, path)


def load_samples(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {record["id"]: record for record in map(json.loads, f)}


def write_samples(path, samples):
    temp = path + ".tmp"
    with open(temp, "w") as f:
        for record in samples.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def aggregate_ratio(records):
    compressed = sum(r["compressed_bytes"] for r in records)
    if not compressed:
        return None
    return sum(r["original_bytes"] for r in records) / compressed


def generate_task(
    model, tokenizer, cache_config, task, samples_path, batch_size
):
    """Generate every missing hypothesis for ``task``; return records in task order."""
    samples = load_samples(samples_path)
    requests = task.requests(skip=samples)
    # Longest first: padding stays small and any OOM surfaces immediately.
    requests.sort(key=lambda request: len(request["input_ids"]), reverse=True)
    if samples:
        print(f"[{task.name}] resuming: {len(requests)} samples left")
    with open(samples_path, "a") as f, attention_for_batch(model, batch_size):
        for start in tqdm(range(0, len(requests), batch_size), desc=task.name):
            batch = requests[start : start + batch_size]
            texts, storage = generate(
                model,
                tokenizer,
                cache_config,
                [request["input_ids"] for request in batch],
                task.max_new_tokens,
            )
            # Storage is measured per batch; split it by prompt length so sums
            # over whole batches (and hence aggregate ratios) stay exact.
            batch_tokens = sum(len(request["input_ids"]) for request in batch)
            for request, text in zip(batch, texts):
                share = len(request["input_ids"]) / batch_tokens
                record = {
                    "id": request["id"],
                    **request["record"],
                    "hypothesis": text,
                    "original_bytes": storage["original_bytes"] * share,
                    "compressed_bytes": storage["compressed_bytes"] * share,
                }
                samples[record["id"]] = record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    return samples, [samples[id] for id in task.order()]


def print_summary(results):
    for name, summary in results.items():
        print(f"\n{name}")
        for key, value in summary.items():
            if key == "per_type":
                for kind, row in value.items():
                    print(f"  {kind:<28} {row['accuracy']:.4f} ({row['n']})")
            else:
                print(f"  {key:<28} {value}")


@torch.no_grad()
def main(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)
    selective_handles = []
    if args.selective_reconstruction:
        enable_decode_compilation()
        selective_handles = install_selective_attention(model)

    attn_predictor_hook_handles = get_attn_predictor_hook_handles(args, model)

    cache_config = build_cache_config(args, model)

    context_limit = (
        args.max_context_tokens or model.config.max_position_embeddings
    )

    if args.resume:
        output_path = args.resume
    else:
        output_dir = os.path.join(
            args.output_dir, args.model_name.replace("/", "_")
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = get_output_path(
            os.path.join(
                output_dir,
                f"custom_eval_kc_{args.k_cache_type}_vc_{args.v_cache_type}_ml_{bool(args.meta_weights_path)}"
                + "_{}.json",
            )
        )
    report = {"config": vars(args), "status": "running", "samples": {}}
    write_json(output_path, report)
    print(f"Writing results to {output_path}")

    results, compression, all_records = {}, {}, []
    for name in args.tasks:
        task = TASKS[name](args, tokenizer, context_limit)
        samples_path = f"{os.path.splitext(output_path)[0]}_{name}.jsonl"
        report["samples"][name] = samples_path
        samples, records = generate_task(
            model, tokenizer, cache_config, task, samples_path, args.batch_size
        )
        try:
            results[name] = task.evaluate(records)
        finally:
            # Keep any judge labels already obtained, even if judging fails.
            write_samples(samples_path, samples)
        compression[name] = aggregate_ratio(records)
        all_records += records

    print_summary(results)
    report.update(
        status="complete",
        results=results,
        compression={
            "aggregate_ratio": aggregate_ratio(all_records),
            "per_task": compression,
        },
    )
    write_json(output_path, report)
    print(f"Results saved to {output_path}")
    for handle in attn_predictor_hook_handles + selective_handles:
        handle.remove()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Custom-benchmark evaluation script"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("-o", "--output_dir", type=str, default="./results")
    parser.add_argument(
        "-t",
        "--tasks",
        type=list_of_strings,
        default=["longmemeval"],
        help=f"Comma-separated tasks from: {', '.join(TASKS)}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples per task (per question type for LongMemEval).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=None,
        help="Prompt plus generation budget; longer inputs are truncated. "
        "Defaults to the model's max_position_embeddings (upstream LongMemEval "
        "uses 128000 for Llama 3.1 and 32000 for Mistral).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Results json of an interrupted run; finished samples are reused.",
    )
    add_cache_args(parser)
    for task in TASKS.values():
        task.add_args(parser)

    args = parser.parse_args()
    validate_args(parser, args)
    args = apply_attn_predictor_config(args)

    print("Config for custom eval: ", vars(args))

    return args


def validate_args(parser, args):
    validate_cache_args(parser, args)
    unknown = [name for name in args.tasks if name not in TASKS]
    if unknown:
        parser.error(f"Unknown tasks {unknown}; choose from {list(TASKS)}.")
    for name in args.tasks:
        TASKS[name].validate_args(parser, args)
    if args.resume is None:
        return
    if not os.path.exists(args.resume):
        parser.error(f"--resume file {args.resume} does not exist.")
    with open(args.resume) as f:
        saved = json.load(f)["config"]
    changed = [
        key
        for key, value in vars(args).items()
        if key not in RESUMABLE_FLAGS and key in saved and saved[key] != value
    ]
    if changed:
        parser.error(
            f"--resume run used different values for {changed}; start a new run."
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
