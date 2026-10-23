"""LongMemEval (Wu et al., ICLR 2025): QA over long user-assistant chat histories.

Follows the official long-context reader (the ``full-history-session`` setting of
src/generation/run_generation.py) and LLM judge (src/evaluation/evaluate_qa.py)
from https://github.com/xiaowu0162/LongMemEval.
"""

import json
import os
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from huggingface_hub import hf_hub_download

DATA_REPO = "xiaowu0162/longmemeval-cleaned"
DATA_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
QUESTION_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)
JUDGE_MODEL = "gpt-4o-2024-08-06"

READER_TEMPLATES = {
    "direct": "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:",
    "con": "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):",
}
GEN_LENGTH = {"direct": 500, "con": 800}
# Tokens the official reader leaves free between the history and the context limit.
PROMPT_RESERVE = 1000

_CORRECT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. "
_QA = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_TEMPLATES = {
    "single-session-user": _CORRECT + _QA,
    "single-session-assistant": _CORRECT + _QA,
    "multi-session": _CORRECT + _QA,
    "temporal-reasoning": _CORRECT
    + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    + _QA,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer."
    + _QA,
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}


def is_abstention(question_id):
    return "_abs" in question_id


def format_history(entry):
    """Render every haystack session in date order, as the official reader does."""
    sessions = sorted(
        zip(entry["haystack_dates"], entry["haystack_sessions"]),
        key=lambda session: session[0],
    )
    history = ""
    for i, (date, session) in enumerate(sessions):
        turns = [
            {key: value for key, value in turn.items() if key != "has_answer"}
            for turn in session
        ]
        history += f"\n### Session {i + 1}:\nSession Date: {date}\nSession Content:\n\n{json.dumps(turns)}\n"
    return history


def build_prompt(entry, tokenizer, history_budget, reading_method):
    """Chat-formatted prompt ids, keeping the start of an over-long history."""
    history = format_history(entry)
    history_ids = tokenizer.encode(history, add_special_tokens=False)
    truncated = len(history_ids) > history_budget
    if truncated:
        history = tokenizer.decode(history_ids[:history_budget])
    prompt = READER_TEMPLATES[reading_method].format(
        history, entry["question_date"], entry["question"]
    )
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        input_ids = tokenizer.encode(prompt)
    return input_ids, truncated


def judge_prompt(entry, hypothesis):
    kind = (
        "abstention"
        if is_abstention(entry["question_id"])
        else entry["question_type"]
    )
    return JUDGE_TEMPLATES[kind].format(
        entry["question"], entry["answer"], hypothesis
    )


def mean(values):
    return statistics.fmean(values) if values else None


class LongMemEval:
    name = "longmemeval"

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("LongMemEval")
        group.add_argument(
            "--lme_variant",
            choices=sorted(DATA_FILES),
            default="s",
            help="oracle: evidence sessions only; s: ~115k-token histories; m: ~500 sessions (exceeds 128k).",
        )
        group.add_argument(
            "--lme_data_path",
            type=str,
            default=None,
            help=f"Local LongMemEval json; otherwise downloaded from {DATA_REPO}.",
        )
        group.add_argument(
            "--lme_reading_method",
            choices=sorted(READER_TEMPLATES),
            default="con",
            help="con (recommended upstream) answers step by step; direct answers outright.",
        )
        group.add_argument(
            "--lme_question_types",
            type=lambda arg: arg.split(","),
            default=None,
            help=f"Comma-separated subset of: {', '.join(QUESTION_TYPES)}.",
        )
        group.add_argument("--judge_model", type=str, default=JUDGE_MODEL)
        group.add_argument(
            "--judge_base_url",
            type=str,
            default=None,
            help="OpenAI-compatible endpoint, e.g. a local vLLM server. Defaults to OpenAI.",
        )
        group.add_argument("--judge_workers", type=int, default=8)
        group.add_argument(
            "--no_judge",
            action="store_true",
            help="Only generate hypotheses; score them later with the official evaluate_qa.py.",
        )

    @staticmethod
    def validate_args(parser, args):
        if args.lme_question_types:
            unknown = set(args.lme_question_types) - set(QUESTION_TYPES)
            if unknown:
                parser.error(
                    f"Unknown LongMemEval question types: {sorted(unknown)}"
                )
        if args.no_judge:
            return
        if args.judge_base_url is None and not os.getenv("OPENAI_API_KEY"):
            parser.error(
                "Set OPENAI_API_KEY (e.g. in .env) for the judge, or pass "
                "--judge_base_url or --no_judge."
            )

    def __init__(self, args, tokenizer, context_limit):
        self.args = args
        self.tokenizer = tokenizer
        self.max_new_tokens = GEN_LENGTH[args.lme_reading_method]
        self.history_budget = (
            context_limit - self.max_new_tokens - PROMPT_RESERVE
        )
        if self.history_budget <= 0:
            raise ValueError(
                f"Context limit {context_limit} leaves no room for the history."
            )
        path = args.lme_data_path or hf_hub_download(
            DATA_REPO, DATA_FILES[args.lme_variant], repo_type="dataset"
        )
        with open(path) as f:
            self.entries = {
                entry["question_id"]: entry for entry in json.load(f)
            }

    def selected_entries(self):
        """Entries to evaluate, keeping at most ``--limit`` per question type."""
        types = self.args.lme_question_types or QUESTION_TYPES
        taken = defaultdict(int)
        selected = []
        for entry in self.entries.values():
            kind = entry["question_type"]
            if kind not in types:
                continue
            if self.args.limit is not None and taken[kind] >= self.args.limit:
                continue
            taken[kind] += 1
            selected.append(entry)
        return selected

    def requests(self, skip=()):
        """Prompt ids for every selected question whose id is not in ``skip``."""
        requests = []
        for entry in self.selected_entries():
            if entry["question_id"] in skip:
                continue
            input_ids, truncated = build_prompt(
                entry,
                self.tokenizer,
                self.history_budget,
                self.args.lme_reading_method,
            )
            requests.append(
                {
                    "id": entry["question_id"],
                    "input_ids": np.asarray(input_ids, dtype=np.int32),
                    "record": {
                        "question_id": entry["question_id"],
                        "question_type": entry["question_type"],
                        "prompt_tokens": len(input_ids),
                        "history_truncated": truncated,
                    },
                }
            )
        return requests

    def order(self):
        return [entry["question_id"] for entry in self.selected_entries()]

    def judge(self, records):
        """Label each unjudged record in place, as the official evaluate_qa.py does."""
        from openai import OpenAI

        pending = [r for r in records if "autoeval_label" not in r]
        if not pending:
            return
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
            or ("EMPTY" if self.args.judge_base_url else None),
            base_url=self.args.judge_base_url,
            max_retries=10,
        )

        def label(record):
            entry = self.entries[record["question_id"]]
            completion = client.chat.completions.create(
                model=self.args.judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": judge_prompt(entry, record["hypothesis"]),
                    }
                ],
                temperature=0,
                max_tokens=10,
            )
            verdict = completion.choices[0].message.content.strip()
            record["autoeval_label"] = {
                "model": self.args.judge_model,
                "label": "yes" in verdict.lower(),
            }

        with ThreadPoolExecutor(self.args.judge_workers) as pool:
            list(pool.map(label, pending))

    def evaluate(self, records):
        """Judge the hypotheses and report the official LongMemEval metrics."""
        summary = {
            "n": len(records),
            "mean_prompt_tokens": mean([r["prompt_tokens"] for r in records]),
            "num_history_truncated": sum(
                r["history_truncated"] for r in records
            ),
        }
        if self.args.no_judge:
            return summary
        self.judge(records)
        by_type = defaultdict(list)
        abstention = []
        for record in records:
            correct = float(record["autoeval_label"]["label"])
            by_type[record["question_type"]].append(correct)
            if is_abstention(record["question_id"]):
                abstention.append(correct)
        summary.update(
            {
                "judge_model": self.args.judge_model,
                "overall_accuracy": mean(
                    [c for scores in by_type.values() for c in scores]
                ),
                "task_averaged_accuracy": mean(
                    [mean(scores) for scores in by_type.values()]
                ),
                "abstention_accuracy": mean(abstention),
                "num_abstention": len(abstention),
                "per_type": {
                    kind: {
                        "accuracy": mean(by_type[kind]),
                        "n": len(by_type[kind]),
                    }
                    for kind in QUESTION_TYPES
                    if kind in by_type
                },
            }
        )
        return summary
