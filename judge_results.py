"""Score finished custom_eval_script.py runs with ab LLM judge."""

import json
import argparse
from argparse import Namespace

from custom_eval_script import (
    load_samples,
    print_summary,
    write_json,
    write_samples,
)
from custom_tasks import TASKS


def judge_run(path, judge_args):
    with open(path) as f:
        report = json.load(f)
    if report["status"] != "complete":
        raise ValueError(f"{path} is unfinished; resume generation first.")
    args = Namespace(**{**report["config"], **judge_args, "no_judge": False})
    for name, samples_path in report["samples"].items():
        task = TASKS[name](args, tokenizer=None, context_limit=float("inf"))
        samples = load_samples(samples_path)
        records = [samples[id] for id in task.order()]
        judges = {
            r["autoeval_label"]["model"] for r in records if "autoeval_label" in r
        }
        if judges - {args.judge_model}:
            raise ValueError(
                f"{samples_path} already has labels from {sorted(judges)}; "
                f"refusing to mix them with {args.judge_model}."
            )
        try:
            report["results"][name] = task.evaluate(records)
        finally:
            write_samples(samples_path, samples)
    write_json(path, report)
    print(f"\n{path}")
    print_summary(report["results"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "results", nargs="+", help="Results json(s) from custom_eval_script.py."
    )
    parser.add_argument(
        "--judge_model", type=str, default="meta-llama/Llama-3.1-70B-Instruct"
    )
    parser.add_argument(
        "--judge_base_url",
        type=str,
        required=True,
        help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8001/v1.",
    )
    parser.add_argument("--judge_workers", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    judge_args = {
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_workers": args.judge_workers,
    }
    for path in args.results:
        judge_run(path, judge_args)
