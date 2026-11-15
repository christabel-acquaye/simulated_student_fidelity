#!/usr/bin/env python3
"""
class_controlled_logits_.py
One script for SELF-O (--mode llm) and PERS-O (--mode persona) runs, on datasets WITH options (naep, eedi,
malalgoqa) and WITHOUT options / fill-in (mae, map, ...).
It replaces class_controlled.py + class_controlled_skilled.py + score_answer_logits.py.

For every respondent (persona student, or non-persona trace) and every item it records TWO passes.

  1. REASONING pass (unchanged method, unchanged seeds, same parsing / follow-up / retry rules)
       the system prompt, user prompt and full prompt, raw_response, reasoning, selected_answer, ...
       then the logits of the option letters at the answer position, read AFTER the reasoning:
           <r>_logits_raw         {"A": 17.2, "B": 25.8, ...}  raw next-token logits
           <r>_logits_normalized  softmax over the option letters only (sums to 1)
           <r>_probs_vocab / <r>_option_mass / <r>_probs_temp   (same meaning as score_answer_logits.py)
           <r>_raw_logit / <r>_normalized_logit                 the selected option's value
           <r>_logit_status                                     ok, or why it could not be scored

  2. REAL pass ("real_logits"): the SAME system prompt and user prompt, except the output-format
     instruction asks for ONE answer letter and no reasoning. Sampled (T=0.7, top-p=0.95), seed 24
     (persona) or 24 + trace_id (llm), so SELF-O answers stay non-deterministic. The selected key and the
     option logits at that key are stored as
           <r>_real_user_prompt, <r>_real_full_prompt, <r>_real_raw_response, <r>_real_selected_key,
           <r>_real_logits_raw, <r>_real_logits_normalized, <r>_real_probs_vocab, <r>_real_option_mass,
           <r>_real_probs_temp, <r>_real_raw_logit, <r>_real_normalized_logit, <r>_real_logit_status, ...

Fill-in datasets (no options: mae, map, malrulelib, otero). Same two passes, same columns where they apply:
  * the reasoning pass ends with `Final Answer: X`; the real pass asks for ONLY that line (a bare answer is also
    accepted; an answer cut off by the token limit never is);
  * the logits are those of the FIRST ANSWER TOKEN (the first token of X) in both passes:
        <r>_raw_logit            raw logit of that token         (MCQ: raw logit of the selected option letter)
        <r>_normalized_logit     its probability over the whole vocabulary   (MCQ: softmax over the option letters)
        <r>_first_token / _first_token_id / _first_token_logprob / _first_token_prob / _first_token_rank /
        <r>_first_token_prob_temp  (and the same with `_real_` for the real pass)
    `raw_logit`, `normalized_logit` and `logit_status` exist for every dataset, so analysis code can stack them.

Question block: the user prompt is the cleaned `prompt_question` column exactly as stored (it already holds the
image description, the question and the answer choices), followed by the answer-format instruction. NAEP only gets
a "Grade: {gradeAsInt}" line on top. There are no other prompt templates.

Why two stages: scoring inside vLLM returns `unsupported_scheduler` on this cluster (the logit columns
were empty). The logits are therefore recovered by a teacher-forced replay of the exact saved token
prefix through the same model with transformers (the method of score_answer_logits.py, whose functions
are imported unchanged). Generation needs vLLM, replay needs transformers, so they run as separate steps:

    --stage generate   vLLM only. Writes the checkpoint after every respondent (both passes).
    --stage score      transformers only. Reads the checkpoint, fills the logit columns, writes the final CSV.
    --stage all        (default) runs `generate` then `score` as two separate processes, so the GPU memory
                       held by vLLM is released before the replay model is loaded.

Both stages resume from the checkpoint (a respondent is skipped once finished); the checkpoint is replaced
atomically, so a SLURM preemption cannot leave a half-written file. Re-submit the same command to continue.

Output is wide, one row per item, like the existing runs:
    simulated_llm_results/<model>_<family>[_<subject>]_controlled_n100_t1024_v3.csv
    simulated_classroom_results/<model>_<family>[_<subject>]_controlled_skilled_n100_t1024_v3.csv
    (+ _checkpoint.parquet and _meta.json). v3 never touches the v2 files.

Examples (from thesis_p1):
    python class_controlled_logits_.py --mode persona --family naep --subject math \
        --dataset <naep csv> --model google/gemma-2-9b-it
    python class_controlled_logits_.py --mode llm --family naep --subject math \
        --dataset <naep csv> --model google/gemma-2-9b-it
    # smoke test: 3 items, 2 respondents
    python class_controlled_logits_.py --mode llm ... --max-rows 3 --max-respondents 2
    # use the cleaned `prompt_question` column as the question block
    python class_controlled_logits_.py ... --prompt-column prompt_question
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

os.environ["VLLM_USE_V1"] = "0"

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from student_profiles import FAMILY_SPECS, build_system_prompt, normalise_content_area
except ImportError as exc:
    raise ImportError(
        "A matching student_profiles.py is required (FAMILY_SPECS, build_system_prompt, "
        "normalise_content_area)."
    ) from exc

import class_generation_support as cgs
from score_answer_logits import (  # pure-python helpers; torch is imported lazily elsewhere
    answer_attempt, sampler_settings, sig, to_int,
)
from answer_logit_scoring import ScoringUnavailable, locate_answer_tokens
from class_generation_support import (
    get_answer_choices, format_answer_instruction, run_generation_batch, safe_load_checkpoint,
)

# MAE (no options) is a free-response family like map/otero/malrulelib.
FAMILY_SPECS.setdefault("mae", {"task_kind": "free_response", "required_columns": ["question_text"]})

warnings.filterwarnings("ignore", message="Some raw-logit scores are unavailable")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Design constants ─────────────────────────────────────────────────────────

STUDENT_COUNTS = {"below_basic": 28, "basic": 32, "proficient": 28, "advanced": 12}  # persona mode
N_TRACES = 100                                                                        # llm mode

REASON_SEED = 42            # persona: 42 for every student; llm: 42 + trace_id (unchanged from v2)
DIRECT_SEED = 24            # persona: 24 for every student; llm: 24 + trace_id
TEMPERATURE, TOP_P = 0.7, 0.95

MAX_RETRIES = 3
MAX_MODEL_LEN = 4096
MAX_TOKENS_OUT = 1024       # whole reasoning completion, including the final answer
DIRECT_MAX_TOKENS = 16      # MCQ: the direct pass should be a single letter
DIRECT_MAX_TOKENS_FREE = 64 # fill-in: a short final answer

DIRECT_SCHEMA_VERSION = "direct_answer_v1"
GENERIC_SYSTEM_PROMPT = (
    "You are solving a math problem. Think through it carefully and show your reasoning, "
    "then give your final answer in the exact format requested."
)
RESULTS_DIRS = {"persona": "simulated_classroom_persona", "llm": "simulated_classroom_llm"}
VALID_RECORD_STATUS = {"complete", "unresolved", "skipped"}

# Columns written by the (failing) in-vLLM scoring path. They carry no information once the replay has
# filled the real logit columns, so they are left out of the final CSV (the checkpoint keeps them).
LEGACY_SCORING_SUFFIXES = (
    "capture_stage", "capture_metadata", "alignment", "errors", "scoring_metadata", "token_logprobs",
    "sequence_logprob", "selected_full_vocab_probability", "probability_underflow", "option_token_ids",
    "answer_token_ids", "answer_token_positions", "score_kind",
)
SCORE_CELLS = ["logits_raw", "logits_normalized", "probs_vocab", "option_mass", "probs_temp"]          # MCQ
FREE_CELLS = ["first_token_id", "first_token", "first_token_logprob", "first_token_prob",               # fill-in
              "first_token_rank", "first_token_prob_temp"]

# ── Prompts ──────────────────────────────────────────────────────────────────



FREE_RESPONSE_ANSWER_FORMAT = (
    "Think step by step, then on the very last line write EXACTLY:\n"
    "Final Answer: X\n"
    "where X is your final numeric or short answer. Do not write anything after that line."
)
FREE_DIRECT_FORMAT = (
    "Do not show any reasoning or explanation. Reply with ONLY one line, EXACTLY:\n"
    "Final Answer: X\n"
    "where X is your final numeric or short answer. Do not write anything else."
)


def is_mcq(family):
    return FAMILY_SPECS[family]["task_kind"] == "mcq"


def reasoning_format(family, choices):
    return format_answer_instruction(choices) if is_mcq(family) else FREE_RESPONSE_ANSWER_FORMAT


def format_direct_instruction(choices):
    """The ONLY difference between the reasoning prompt and the real_logits prompt.
    MCQ: one option letter. Fill-in (choices is None): one `Final Answer: X` line, no reasoning."""
    if choices is None:
        return FREE_DIRECT_FORMAT
    if not choices:
        raise ValueError("Answer choices cannot be empty.")
    return ("Do not show any reasoning or explanation. Reply with ONLY the letter of your answer "
            f"({', '.join(choices)}) and nothing else.")


def build_user_prompt(family, row, answer_format, prompt_column="prompt_question"):
    """Grade line (NAEP only) + the dataset's `prompt_question` as stored + the answer-format instruction."""
    body = str(row[prompt_column])
    if family == "naep" and "gradeAsInt" in row.index:
        return f"Grade: {row['gradeAsInt']}\n\n{body}\n\n{answer_format}"
    return f"{body}\n\n{answer_format}"


def choices_for_row(family, row, prompt_column=None):
    """Option labels from the item's option metadata (never from an answer key)."""
    try:
        return get_answer_choices(family, row)
    except ValueError as first_error:
        for column in ("full_options", prompt_column):
            if column and column in row.index and pd.notna(row[column]):
                try:
                    return get_answer_choices("naep", pd.Series({"options": row[column]}))
                except ValueError:
                    continue
        raise first_error


def create_prompt(system_prompt, user_prompt, tokenizer):
    with_system = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    merged = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    try:
        return tokenizer.apply_chat_template(with_system, tokenize=False, add_generation_prompt=True)
    except Exception:
        try:
            return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"


# ── Respondents, names, seeds ────────────────────────────────────────────────

def respondents(mode, limit=None):
    """[(column prefix, level or None, student_id or trace_id)] in the order the v2 files use."""
    if mode == "persona":
        out = [(f"{level}_{i:03d}", level, i) for level, n in STUDENT_COUNTS.items() for i in range(1, n + 1)]
    else:
        out = [(f"llm_{t:03d}", None, t) for t in range(1, N_TRACES + 1)]
    return out[:limit] if limit else out


def seeds_for(mode, ident):
    if mode == "persona":
        return REASON_SEED, DIRECT_SEED
    return REASON_SEED + ident, DIRECT_SEED + ident


def derive_paths(mode, family, subject, model, max_rows=None, max_respondents=None, results_dir=None):
    slug = Path(model).name.lower().replace("-", "_")
    infix = "controlled_skilled" if mode == "persona" else "controlled"
    base = f"{slug}_{family}_{subject}_{infix}" if family == "naep" else f"{slug}_{family}_{infix}"
    base += "_n100_t1024_v3"
    if max_rows:
        base += f"_smoke{max_rows}"
    if max_respondents:
        base += f"_resp{max_respondents}"
    out_dir = results_dir or RESULTS_DIRS[mode]
    os.makedirs(out_dir, exist_ok=True)
    return (os.path.join(out_dir, f"{base}.csv"), os.path.join(out_dir, f"{base}_checkpoint.parquet"),
            os.path.join(out_dir, f"{base}_meta.json"))


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_dataset(path):
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{ext}'.")


def prepare_dataset(args):
    if args.family == "naep" and args.subject is None:
        raise ValueError("--subject is required when --family is 'naep'.")
    df = load_dataset(args.dataset)
    # The question block is the cleaned `prompt_question` column, used as-is. A dataset without it fails here.
    if args.prompt_column in (None, "auto"):
        args.prompt_column = "prompt_question"
    required = [args.prompt_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for '{args.family}': {missing}. Got: {list(df.columns)}")
    if args.family == "naep" and args.subject == "reading":
        df = df.dropna(subset=["question_text", "options"]).reset_index(drop=True)
    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)
    return df


# ── Checkpoint helpers (parquet I/O is isolated so it can be swapped in tests) ────

def save_checkpoint(df, path):
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)          # atomic: a killed job never leaves a half-written checkpoint


def set_columns(df, mapping):
    for name, values in mapping.items():
        df[name] = values


def add_records(df, prefix, records, n_items):
    if len(records) != n_items:
        raise ValueError("Generation helper returned a different number of records than prompts.")
    suffixes = list(dict.fromkeys(s for record in records for s in record))
    set_columns(df, {f"{prefix}_{s}": [record.get(s) for record in records] for s in suffixes})


def generation_done(df, r):
    """A respondent is finished when BOTH passes are recorded for every item."""
    cols = (f"{r}_record_status", f"{r}_real_record_status")
    return all(c in df.columns and df[c].isin(VALID_RECORD_STATUS).all() for c in cols)


def scoring_done(df, r):
    cols = (f"{r}_logit_status", f"{r}_real_logit_status")
    return all(c in df.columns and df[c].notna().all() for c in cols)


# ── Model (generation) ───────────────────────────────────────────────────────

_LARGE_MODEL_KEYWORDS = ["14b", "14B", "27b", "27B", "70b", "70B"]


def build_llm(model):
    from vllm import LLM  # lazy: vLLM is GPU/cluster only
    n_gpus = 2 if any(k in model for k in _LARGE_MODEL_KEYWORDS) else 1
    print(f"Loading model {model} …  tensor_parallel_size={n_gpus}  max_model_len={MAX_MODEL_LEN}")
    extra = {}
    overrides = hf_config_overrides(model)
    if overrides:
        print(f"HF config overrides for vLLM: {overrides}")
        extra["hf_overrides"] = overrides
    return LLM(model=model, trust_remote_code=True, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
               gpu_memory_utilization=0.92, tensor_parallel_size=n_gpus, pipeline_parallel_size=1,
               enforce_eager=True, **extra)


def hf_config_overrides(model):
    """Gemma-3 (google/gemma-3-*-it) workaround. vLLM's Gemma3 multimodal class reads `config.sliding_window` on the
    TOP-level Gemma3Config, but this transformers version only keeps it on `config.text_config`, so profiling the
    model crashes with "'Gemma3Config' object has no attribute 'sliding_window'". Copying the text value up onto
    the top-level config (the same number the text model uses) removes the crash and changes nothing else."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:                                  # offline / gated: let vLLM report the real problem
        print(f"(could not pre-read the config of {model}: {type(exc).__name__}; no overrides applied)")
        return {}
    if getattr(cfg, "model_type", "") != "gemma3" or getattr(cfg, "sliding_window", None) is not None:
        return {}
    window = getattr(getattr(cfg, "text_config", None), "sliding_window", None)
    return {"sliding_window": int(window)} if window else {}


# ── Direct-answer ("real_logits") generation ─────────────────────────────────

def _direct_attempt(result, prompt, attempt_id, kind, choices, params):
    """Like the reasoning pass's attempt, plus: a fill-in answer may be bare (`12`) as well as `Final Answer: 12`,
    and an answer cut off by the token limit is never accepted."""
    a = cgs._attempt(result, prompt, attempt_id, kind, choices, params)
    if a["parse_status"] != "ok" and choices is None:
        a.update(cgs.parse_answer(a["raw_response"], None, allow_bare=True))
    if a["parse_status"] == "ok" and a.get("finish_reason") == "length":
        a.update(selected_answer=None, parse_status="truncated_direct")
    return a


def run_direct_batch(*, llm, prompts, user_prompts, answer_choices, sampling_params, max_model_len,
                     max_tokens, max_retries):
    """One sampled single-letter answer per prompt, with the same full-regeneration retry rule as the
    reasoning pass (seed + 100003 * retry). Returns one record per prompt; the accepted attempt supplies
    the token IDs that the replay stage needs."""
    size = len(prompts)
    if not all(len(v) == size for v in (user_prompts, answer_choices)):
        raise ValueError("Batch input lengths disagree.")
    tokenizer = llm.get_tokenizer()
    prompt_ids = [cgs._encode(tokenizer, p) for p in prompts]
    skip = [None] * size
    valid = []
    for i, ids in enumerate(prompt_ids):
        if len(ids) + max_tokens > max_model_len:
            skip[i] = "prompt_plus_output_exceeds_context"
        else:
            valid.append(i)
    attempts = [[] for _ in prompts]
    accepted = [None] * size
    if valid:
        print(f"  [real] generating {len(valid)} direct answers; {size - len(valid)} context skips.")
        results = llm.generate([prompts[i] for i in valid], sampling_params)
        if len(results) != len(valid):
            raise RuntimeError("vLLM result count does not match requested batch.")
        for i, result in zip(valid, results):
            a = _direct_attempt(result, prompts[i], 0, "initial", answer_choices[i], sampling_params)
            attempts[i].append(a)
            if a["parse_status"] == "ok":
                accepted[i] = a
    for retry in range(1, max_retries + 1):
        todo = [i for i in valid if accepted[i] is None]
        if not todo:
            break
        params = copy.deepcopy(sampling_params)
        params.seed = (getattr(sampling_params, "seed", None) or 0) + 100003 * retry
        print(f"  [real] regeneration retry {retry}: {len(todo)} answers.")
        results = llm.generate([prompts[i] for i in todo], params)
        if len(results) != len(todo):
            raise RuntimeError("Regeneration result count mismatch.")
        for i, result in zip(todo, results):
            a = _direct_attempt(result, prompts[i], len(attempts[i]), "full_regeneration", answer_choices[i], params)
            attempts[i].append(a)
            if a["parse_status"] == "ok":
                accepted[i] = a
    records = []
    for i in range(size):
        acc = accepted[i]
        last = acc or (attempts[i][-1] if attempts[i] else {})
        log = [{k: a.get(k) for k in ("attempt_id", "kind", "raw_response", "finish_reason", "parse_status")}
               | {"seed": (a.get("sampling") or {}).get("seed")} for a in attempts[i]]
        records.append(dict(
            schema_version=DIRECT_SCHEMA_VERSION, user_prompt=user_prompts[i], full_prompt=prompts[i],
            prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(),
            selected_key=acc["selected_answer"] if acc else None,
            raw_response=last.get("raw_response"),
            answer_source=acc["kind"] if acc else None,
            answer_attempt_id=acc["attempt_id"] if acc else None,
            answer_start=acc["answer_start"] if acc else None, answer_end=acc["answer_end"] if acc else None,
            parse_status=acc["parse_status"] if acc else last.get("parse_status", skip[i]),
            record_status="complete" if acc else ("skipped" if skip[i] else "unresolved"),
            skip_reason=skip[i], finish_reason=last.get("finish_reason"),
            prompt_tokens=len(prompt_ids[i]), completion_tokens=len(last.get("output_token_ids", [])),
            prompt_token_ids=cgs._json(last.get("prompt_token_ids", prompt_ids[i])),
            output_token_ids=cgs._json(last.get("output_token_ids", [])),
            generation_parameters=cgs._json(last.get("sampling") or cgs._sampling_dict(sampling_params)),
            attempts_log=cgs._json(log),
        ))
    counts = Counter(r["record_status"] for r in records)
    print(f"  [real] record statuses: {dict(counts)}")
    return records


# ── Stage 1: generation ──────────────────────────────────────────────────────

def generate_stage(args, llm=None):
    from vllm import SamplingParams

    out_path, ckpt_path, _ = derive_paths(args.mode, args.family, args.subject, args.model, args.max_rows,
                                          args.max_respondents, args.output_dir)
    df = prepare_dataset(args)
    df, _ = safe_load_checkpoint(df, ckpt_path)
    names = respondents(args.mode, args.max_respondents)
    mcq = is_mcq(args.family)
    print(f"Mode {args.mode} | {args.family} ({'options' if mcq else 'fill-in'}) | {args.model} | "
          f"{len(df)} items x {len(names)} respondents | question block: {args.prompt_column}")
    print(f"Checkpoint: {ckpt_path}")

    # Item-level pieces that do not depend on the respondent. Built BEFORE the model loads, so a bad
    # item fails in seconds instead of after a long model load.
    bad = []
    choice_lists = []
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            choice_lists.append(choices_for_row(args.family, row, args.prompt_column) if mcq else None)
        except ValueError as exc:
            bad.append((idx, str(exc)[:100])); choice_lists.append(None)
    if bad:
        raise ValueError(f"{len(bad)} item(s) have no readable answer options, first: {bad[:5]}")
    reason_user = [build_user_prompt(args.family, row, reasoning_format(args.family, c), args.prompt_column)
                   for (_, row), c in zip(df.iterrows(), choice_lists)]
    direct_user = [build_user_prompt(args.family, row, format_direct_instruction(c), args.prompt_column)
                   for (_, row), c in zip(df.iterrows(), choice_lists)]
    has_area = "contentArea" in df.columns
    direct_tokens = DIRECT_MAX_TOKENS if mcq else DIRECT_MAX_TOKENS_FREE
    llm = llm or build_llm(args.model)
    tokenizer = llm.get_tokenizer()

    for n, (r, level, ident) in enumerate(names, 1):
        if generation_done(df, r):
            print(f"[{n}/{len(names)}] Skipping → {r}")
            continue
        print(f"[{n}/{len(names)}] Generating → {r}")
        reason_seed, direct_seed = seeds_for(args.mode, ident)
        if args.mode == "persona":
            systems = []
            for _, row in df.iterrows():
                grade = row.get("gradeAsInt", None) if args.family == "naep" else None
                raw_ca = row.get("contentArea")
                area = (normalise_content_area(raw_ca if pd.notna(raw_ca) else None)
                        if has_area and args.family == "naep" else None)
                systems.append(build_system_prompt(family=args.family, level=level, subject=args.subject,
                                                   grade=grade, student_id=ident, content_area=area))
        else:
            systems = [GENERIC_SYSTEM_PROMPT] * len(df)

        # pass 1: reasoning (unchanged)
        prompts = [create_prompt(s, u, tokenizer) for s, u in zip(systems, reason_user)]
        records = run_generation_batch(
            llm=llm, prompts=prompts, system_prompts=systems, user_prompts=reason_user,
            answer_choices=choice_lists,
            sampling_params=SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS_OUT,
                                           seed=reason_seed),
            max_model_len=MAX_MODEL_LEN, max_tokens=MAX_TOKENS_OUT, max_retries=MAX_RETRIES)
        add_records(df, r, records, len(df))
        set_columns(df, {f"{r}_raw_output": [rec.get("raw_response") for rec in records]})

        # pass 2: real_logits — same system prompt, same question, one-letter output format
        d_prompts = [create_prompt(s, u, tokenizer) for s, u in zip(systems, direct_user)]
        d_records = run_direct_batch(
            llm=llm, prompts=d_prompts, user_prompts=direct_user, answer_choices=choice_lists,
            sampling_params=SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=direct_tokens,
                                           seed=direct_seed),
            max_model_len=MAX_MODEL_LEN, max_tokens=direct_tokens, max_retries=MAX_RETRIES)
        add_records(df, f"{r}_real", d_records, len(df))

        save_checkpoint(df, ckpt_path)
        print(f"  ✓ Checkpoint saved → {r}")
    print("Generation stage finished. Next: --stage score")
    return df


# ── Stage 2: logit replay ────────────────────────────────────────────────────

REPLAY_FIELDS = ["selected_answer", "answer_source", "option_labels", "attempts", "answer_attempt_id",
                 "prompt_token_ids", "output_token_ids", "raw_response", "answer_start", "answer_end",
                 "generation_parameters"]


def column_for(r, real, field):
    """Which checkpoint column feeds score_answer_logits.build_job for this pass."""
    if not real:
        return f"{r}_{field}"
    if field == "option_labels":
        return f"{r}_option_labels"            # same item, same labels
    if field == "selected_answer":
        return f"{r}_real_selected_key"
    if field == "attempts":
        return None                            # use the accepted attempt's top-level fields (initial/regeneration)
    return f"{r}_real_{field}"


def _clean(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value


class ReplayEngine:
    """The transformers side of the replay. Everything here is imported unchanged from
    score_answer_logits.py (validated against the model's own forward pass by its self-check)."""

    def __init__(self, args):
        import torch
        import transformers
        from score_answer_logits import (load_model, forward_last, option_scores, self_check, batches)
        self._forward_last, self._option_scores = forward_last, option_scores
        self._self_check, self.batches = self_check, batches
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        self.tokenizer, self.model, self.attn = load_model(args)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.device = args.device
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def forward(self, jobs):
        return self._forward_last(self.model, jobs, self.pad_id, self.device)

    def scores(self, logits_row, job):
        return self._option_scores(logits_row, job)

    def self_check(self, job):
        return self._self_check(self.model, job, self.device)

    def free_scores(self, logits_row, job):
        """Fill-in answers: the raw logit of the FIRST answer token, its full-vocabulary log-probability / probability
        (softmax over the whole vocabulary, T=1), its rank, and its probability under the run's own sampler."""
        import math
        import torch
        from score_answer_logits import sampler_probs
        tid = job["token_id"]
        raw = logits_row[tid]
        logprob = float((raw - torch.logsumexp(logits_row, dim=-1)).item())
        out = {"logit": round(float(raw.item()), 6), "logprob": sig(logprob), "prob": sig(math.exp(logprob)),
               "rank": int((logits_row > raw).sum().item()) + 1, "prob_temp": None}
        if job["sampling"] is not None:
            out["prob_temp"] = sig(float(sampler_probs(logits_row, job["sampling"])[tid].item()))
        return out


class ScoreState:
    def __init__(self, skip_self_check=False):
        self.checked = skip_self_check
        self.self_check_result = None
        self.cache = {}                     # real pass: identical prefixes are scored once
        self.status = Counter()
        self.fidelity = Counter()
        self.selected_prob_sum = 0.0


def build_free_job(tokenizer, get, student, max_len):
    """Fill-in analogue of score_answer_logits.build_job: the job replays prompt + everything the student wrote
    BEFORE the answer, so the logits are those at the first answer token. Returns (cells, job-or-None)."""
    selected = (get(student, "selected_answer") or "").strip()
    source = (get(student, "answer_source") or "").strip()
    cells = {"selected_answer": selected, "answer_source": source, "logit_status": ""}
    cells.update({name: "" for name in FREE_CELLS})
    if not selected:
        cells["logit_status"] = "no_parsed_answer"
        return cells, None
    try:
        attempt = answer_attempt(get, student, selected, source)
        record = {
            "prompt_token_ids": [int(t) for t in (attempt.get("prompt_token_ids") or [])],
            "output_token_ids": [int(t) for t in (attempt.get("output_token_ids") or [])],
            "output_text": attempt.get("raw_response"),
            "answer_start": to_int(attempt.get("answer_start")),
            "answer_end": to_int(attempt.get("answer_end")),
            "answer_choices": None,
            "selected_answer": attempt.get("selected_answer") or selected,
        }
        if not record["prompt_token_ids"] or not record["output_token_ids"]:
            raise ScoringUnavailable("missing_token_ids", "Attempt has no prompt/output token IDs")
        alignment = locate_answer_tokens(tokenizer, record)
        first = alignment["tokens"][0]
        input_ids = record["prompt_token_ids"] + record["output_token_ids"][:first["index"]]
        if len(input_ids) > max_len:
            raise ScoringUnavailable("scoring_context_too_long", f"{len(input_ids)} tokens > {max_len}")
        sampling, note = sampler_settings(attempt, get, student)
        cells["logit_status"] = "ok"
        cells["first_token_id"] = first["token_id"]
        cells["first_token"] = first["text"]
        return cells, {"input_ids": input_ids, "token_id": first["token_id"], "candidates": {"first": first["token_id"]},
                       "selected": selected, "sampling": sampling, "sampling_note": note, "kind": "free"}
    except ScoringUnavailable as exc:
        cells["logit_status"] = exc.status
    except Exception as exc:  # noqa: BLE001 - malformed cell: record it, do not crash the run
        cells["logit_status"] = "input_error:" + type(exc).__name__
    return cells, None


def _job_key(job):
    """Identical prefix + labels + sampler settings give identical scores. The sampling SEED is not part of the
    key: it changes which letter is drawn, not the logits, so SELF-O's 100 identical real prompts cost one pass."""
    digest = hashlib.sha1(repr(job["input_ids"]).encode()).hexdigest()
    sampler = {k: (job["sampling"] or {}).get(k) for k in ("temperature", "top_p", "top_k", "min_p")}
    return (digest, tuple(sorted(job["candidates"].items())), job.get("token_id"), json.dumps(sampler, sort_keys=True))


def score_pass(df, r, real, engine, state, args):
    from score_answer_logits import build_job

    mcq = is_mcq(args.family)
    build = build_job if mcq else build_free_job
    n = len(df)
    pre = f"{r}_real" if real else r
    columns = {}
    for field in REPLAY_FIELDS:
        col = column_for(r, real, field)
        columns[field] = [_clean(v) for v in df[col].tolist()] if col in df.columns else [None] * n
    built = []
    for i in range(n):
        get = lambda _s, f, _i=i: columns[f][_i]
        built.append(build(engine.tokenizer, get, r, args.max_model_len))

    store = state.cache if real else {}
    todo = {}
    for cells, job in built:
        if job is not None:
            job["key"] = _job_key(job)
            if job["key"] not in store:
                todo.setdefault(job["key"], job)
    unique = list(todo.values())
    if unique and not state.checked:
        state.self_check_result = engine.self_check(unique[0])
        print(f"Self-check: {json.dumps(state.self_check_result)}", flush=True)
        if not state.self_check_result["passed"]:
            raise RuntimeError(f"Self-check failed: replay does not match the model's forward ({state.self_check_result}).")
        state.checked = True
    for batch in engine.batches(unique, args.batch_tokens, args.max_batch_size):
        logits = engine.forward([unique[i] for i in batch])
        for k, i in enumerate(batch):
            try:
                store[unique[i]["key"]] = (engine.scores if mcq else engine.free_scores)(logits[k], unique[i])
            except Exception as exc:                       # noqa: BLE001 - record, never crash the run
                store[unique[i]["key"]] = {"error": type(exc).__name__}

    raw_logit, norm_logit = [None] * n, [None] * n
    for i, (cells, job) in enumerate(built):
        if job is None:
            continue
        s = store[job["key"]]
        if "error" in s:
            cells["logit_status"] = "scoring_error:" + s["error"]
            continue
        sel = job["selected"]
        state.fidelity["scored"] += 1
        if mcq:
            for name in ("logits_raw", "logits_normalized", "probs_vocab"):
                cells[name] = json.dumps(s[name])
            cells["option_mass"] = s["option_mass"]
            raw_logit[i], norm_logit[i] = s["logits_raw"][sel], s["logits_normalized"][sel]
            norm = s["logits_normalized"]
            state.fidelity["selected_is_top_option"] += int(max(norm, key=norm.get) == sel)
            p_temp = s["probs_temp"].get(sel, 0.0) if s.get("probs_temp") is not None else None
            if p_temp is not None:
                cells["probs_temp"] = json.dumps(s["probs_temp"])
        else:
            # raw_logit = raw logit of the first answer token; normalized_logit = its full-vocabulary probability
            raw_logit[i], norm_logit[i] = s["logit"], s["prob"]
            cells["first_token_logprob"], cells["first_token_prob"] = s["logprob"], s["prob"]
            cells["first_token_rank"], cells["first_token_prob_temp"] = s["rank"], s["prob_temp"]
            state.fidelity["selected_is_top_option"] += int(s["rank"] == 1)
            p_temp = s["prob_temp"]
        if p_temp is not None:
            state.fidelity["probs_temp_computed"] += 1
            state.fidelity["selected_zero_under_own_sampler"] += int(p_temp == 0.0)
            state.selected_prob_sum += p_temp
    for cells, _ in built:
        state.status[f"{'real' if real else 'reasoning'}:{cells['logit_status']}"] += 1

    mapping = {f"{pre}_{name}": [c[name] for c, _ in built] for name in (SCORE_CELLS if mcq else FREE_CELLS)}
    mapping[f"{pre}_logit_status"] = [c["logit_status"] for c, _ in built]
    mapping[f"{pre}_raw_logit"] = raw_logit
    mapping[f"{pre}_normalized_logit"] = norm_logit
    if not real:
        if mcq:   # the columns the v2 generator left empty
            mapping[f"{r}_option_raw_logits"] = mapping[f"{pre}_logits_raw"]
            mapping[f"{r}_option_probabilities"] = mapping[f"{pre}_logits_normalized"]
            mapping[f"{r}_1st_prob"] = mapping[f"{pre}_logits_normalized"]
        mapping[f"{r}_score_status"] = mapping[f"{pre}_logit_status"]
        mapping[f"{r}_scoring_protocol"] = ["hf_teacher_forced_replay"] * n
    set_columns(df, mapping)


def score_checkpoint(df, names, engine, state, ckpt_path, args):
    for n, (r, _, _) in enumerate(names, 1):
        if scoring_done(df, r):
            print(f"[{n}/{len(names)}] Already scored → {r}")
            continue
        if not generation_done(df, r):
            raise RuntimeError(f"{r} has no finished generation in the checkpoint; run --stage generate first.")
        t0 = time.time()
        score_pass(df, r, False, engine, state, args)
        score_pass(df, r, True, engine, state, args)
        save_checkpoint(df, ckpt_path)
        print(f"[{n}/{len(names)}] ✓ scored {r} ({time.time() - t0:.0f}s)", flush=True)
    return df


def finalize(df, out_path, meta_path, args, engine, state, names, started):
    drop = [c for r, _, _ in names for s in LEGACY_SCORING_SUFFIXES for c in (f"{r}_{s}",) if c in df.columns]
    df.drop(columns=drop).to_csv(out_path, index=False)
    real_parse = Counter()
    for r, _, _ in names:
        if f"{r}_real_parse_status" in df.columns:
            real_parse.update(df[f"{r}_real_parse_status"].fillna("none"))
    f = state.fidelity
    meta = {
        "script": Path(__file__).name, "mode": args.mode, "family": args.family, "subject": args.subject,
        "task_kind": FAMILY_SPECS[args.family]["task_kind"],
        "logit_position": ("option letters at the answer position" if is_mcq(args.family)
                           else "first answer token (raw logit; normalized = full-vocabulary probability)"),
        "model": args.model, "dataset": str(args.dataset), "prompt_column": args.prompt_column,
        "items": len(df), "respondents": len(names),
        "seeds": {"reasoning": "42 (persona) | 42+trace_id (llm)", "real": "24 (persona) | 24+trace_id (llm)"},
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P},
        "real_prompt_difference": "answer-format instruction only (format_direct_instruction)",
        "versions": getattr(engine, "versions", {}), "device": getattr(engine, "device_name", None),
        "resolved_commit": getattr(engine, "commit", None), "self_check": state.self_check_result,
        "status_counts_this_run": dict(state.status), "real_parse_status": dict(real_parse),
        "replay_fidelity_this_run": {**dict(f), "mean_probs_temp_of_selected":
                                     round(state.selected_prob_sum / f["probs_temp_computed"], 6)
                                     if f["probs_temp_computed"] else None},
        "dropped_legacy_columns": sorted(set(LEGACY_SCORING_SUFFIXES)),
        "seconds_this_run": round(time.time() - started, 1), "output": out_path,
    }
    if state.status or not Path(meta_path).exists():       # a resume run that scored nothing keeps the earlier meta
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def score_stage(args, engine=None):
    started = time.time()
    out_path, ckpt_path, meta_path = derive_paths(args.mode, args.family, args.subject, args.model,
                                                  args.max_rows, args.max_respondents, args.output_dir)
    if not Path(ckpt_path).exists():
        sys.exit(f"No checkpoint at {ckpt_path}. Run --stage generate first.")
    df, _ = safe_load_checkpoint(prepare_dataset(args), ckpt_path)
    names = respondents(args.mode, args.max_respondents)
    engine = engine or ReplayEngine(args)
    state = ScoreState(args.skip_self_check)
    df = score_checkpoint(df, names, engine, state, ckpt_path, args)
    meta = finalize(df, out_path, meta_path, args, engine, state, names, started)
    print(json.dumps(meta, indent=2))
    print(f"Saved → {out_path}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["persona", "llm", "pers-o", "self-o"],
                   help="persona (PERS-O) = 100 student profiles; llm (SELF-O) = 100 non-persona traces")
    p.add_argument("--family", "-f", required=True, choices=list(FAMILY_SPECS),
                   help="options datasets: naep, eedi, malalgoqa | fill-in datasets: mae, map, malrulelib, otero")
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--subject", "-s", choices=["math", "science", "reading"], default=None, help="Required for naep")
    p.add_argument("--prompt-column", default="prompt_question",
                   help="Column holding the question block of the user prompt (default: prompt_question)")
    p.add_argument("--stage", choices=["all", "generate", "score"], default="all")
    p.add_argument("--output-dir", default=None, help="default: simulated_llm_results / simulated_classroom_results")
    p.add_argument("--max-rows", type=int, default=None, help="smoke test: first N items")
    p.add_argument("--max-respondents", type=int, default=None, help="smoke test: first N students/traces")
    # replay stage (names match score_answer_logits.load_model)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-tokens", type=int, default=24000)
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--skip-self-check", action="store_true")
    args = p.parse_args(argv)
    args.mode = {"pers-o": "persona", "self-o": "llm"}.get(args.mode, args.mode)
    return args


def _child_argv(stage):
    """This command line with --stage replaced, for running one stage in its own process."""
    argv, skip = [], False
    for i, a in enumerate(sys.argv[1:]):
        if skip:
            skip = False
        elif a == "--stage":
            skip = True
        elif not a.startswith("--stage="):
            argv.append(a)
    return [sys.executable, str(Path(__file__).resolve()), *argv, "--stage", stage]


def main():
    args = parse_args()
    if args.stage == "all":
        # separate processes so vLLM's GPU memory is gone before the replay model loads
        for stage in ("generate", "score"):
            code = subprocess.run(_child_argv(stage)).returncode
            if code != 0:
                sys.exit(f"--stage {stage} failed (exit code {code}); fix and re-run the same command to resume.")
        return
    if args.stage == "generate":
        generate_stage(args)
    else:
        score_stage(args)


if __name__ == "__main__":
    main()
