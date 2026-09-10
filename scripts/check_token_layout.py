# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CPU-only audit of the financial tokenizer's global vocabulary layout.

The tokenizer package imports cudf/cuml at module scope, so it cannot be
imported without RAPIDS.  This script instead *replays* the offset arithmetic
from TokenizerPipeline._fit_sequential() in pure Python, reading the field
constants straight out of financial_pipeline.py via ast so the two cannot
drift apart.

It is a model of the pipeline, not the pipeline itself.  It verifies the one
invariant the pretrained checkpoint depends on -- that the global vocabulary
size matches config.json -- and reports ID collisions and dead IDs.

Run:  python scripts/check_token_layout.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIPELINE_SRC = REPO / "src" / "tokenizer" / "financial_pipeline.py"
MODEL_CONFIG = REPO / "models" / "decoder-foundation-model" / "config.json"

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]


def load_constants(path: Path) -> dict:
    """Pull module-level literal constants out of the source without importing it."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.isupper():
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def build_layout(const: dict, merchant_hash_size: int = 2000) -> list[dict]:
    """Mirror FinancialTokenizerPipeline._configure_steps() step ordering.

    Each entry records the *local* index range a step emits, which is what the
    global-offset arithmetic in pipeline.py consumes.  Note that a step's
    declared size and its local index range are two different things -- that
    gap is exactly where the MONTH/CARD collision below comes from.
    """
    n_industry = len({label for _, _, label in const["INDUSTRY_RANGES"]}) + 1  # + default
    n_mcc = len({str(m) for m in const["KNOWN_MCCS"]})                          # default "-1" dedupes
    n_chip = len(set(const["CHIP_MAPPING"].values())) + 1                       # + UNK
    n_state = len(set(const["ALL_STATES"]))                                     # default "XX" dedupes
    n_amt = len(const["AMOUNT_THRESHOLDS"])                                     # 6 thresholds -> 7 bins

    # (step name, token prefix, local_min, local_max)
    return [
        {"name": "amt_val",     "prefix": "AMT",   "lo": 0, "hi": n_amt - 1},
        {"name": "merch_hash",  "prefix": "MERCH", "lo": 0, "hi": merchant_hash_size - 1},
        {"name": "mcc_int",     "prefix": "CAT",   "lo": 0, "hi": n_industry - 1},
        {"name": "mcc_str",     "prefix": "MCC",   "lo": 0, "hi": n_mcc - 1},
        {"name": "hour",        "prefix": "HOUR",  "lo": 0, "hi": 23},
        {"name": "dow",         "prefix": "DOW",   "lo": 0, "hi": 6},
        {"name": "month",       "prefix": "MONTH", "lo": 1, "hi": 12},  # min_val=1
        {"name": "card",        "prefix": "CARD",  "lo": 0, "hi": 9},
        {"name": "chip_upper",  "prefix": "CHIP",  "lo": 0, "hi": n_chip - 1},
        {"name": "zip3",        "prefix": "ZIP3",  "lo": 0, "hi": 999},
        {"name": "state_clean", "prefix": "STATE", "lo": 0, "hi": n_state - 1},
        {"name": "cust",        "prefix": "CUST",  "lo": 0, "hi": 2999},
    ]


def assign_ids(layout: list[dict]) -> tuple[dict, dict, int]:
    """Replay TokenizerPipeline._fit_sequential(): gid = local_idx + offset,
    offset advanced by vocab_size (= number of entries, NOT max local idx + 1)."""
    offset = len(SPECIAL_TOKENS)
    token_to_id: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    for step in layout:
        size = step["hi"] - step["lo"] + 1
        step["offset"] = offset
        step["size"] = size
        step["id_lo"] = offset + step["lo"]
        step["id_hi"] = offset + step["hi"]
        for local in range(step["lo"], step["hi"] + 1):
            token_to_id[f"{step['prefix']}_{local}"] = local + offset
        offset += size
    id_to_tokens = defaultdict(list)
    for tok, tid in token_to_id.items():
        id_to_tokens[tid].append(tok)
    return token_to_id, id_to_tokens, offset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merchant-hash-size", type=int, default=2000)
    args = ap.parse_args()

    const = load_constants(PIPELINE_SRC)
    layout = build_layout(const, args.merchant_hash_size)
    token_to_id, id_to_tokens, vocab_size = assign_ids(layout)

    print(f"Field layout  (source: {PIPELINE_SRC.relative_to(REPO)})\n")
    print(f"  {'step':<12} {'prefix':<7} {'offset':>7} {'id_lo':>7} {'id_hi':>7} {'size':>6}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    print(f"  {'<specials>':<12} {'':<7} {0:>7} {0:>7} {len(SPECIAL_TOKENS)-1:>7} {len(SPECIAL_TOKENS):>6}")
    for s in layout:
        print(f"  {s['name']:<12} {s['prefix']:<7} {s['offset']:>7} "
              f"{s['id_lo']:>7} {s['id_hi']:>7} {s['size']:>6}")

    print(f"\n  global_vocab_size = {vocab_size}")

    ok = True

    # -- the invariant the pretrained checkpoint depends on --------------------
    if MODEL_CONFIG.exists():
        cfg_vocab = json.loads(MODEL_CONFIG.read_text())["vocab_size"]
        match = "OK" if cfg_vocab == vocab_size else "MISMATCH"
        print(f"  config.json vocab_size = {cfg_vocab}  [{match}]")
        if cfg_vocab != vocab_size:
            print("\n  ERROR: vocabulary size no longer matches the shipped checkpoint.\n"
                  "         The embedding matrix and LM head are sized to config.json;\n"
                  "         loading this tokenizer against that checkpoint will produce\n"
                  "         garbage or fail outright.  Retrain, or revert the change.")
            ok = False
    else:
        print("  config.json not found (git lfs pull?) -- skipping checkpoint check")

    # -- known issues ---------------------------------------------------------
    collisions = {i: t for i, t in id_to_tokens.items() if len(t) > 1}
    used = set(token_to_id.values())
    dead = sorted(set(range(len(SPECIAL_TOKENS), vocab_size)) - used)

    print(f"\n  distinct token strings : {len(token_to_id)}")
    print(f"  distinct ids           : {len(used)}")

    if collisions:
        print(f"\n  KNOWN ISSUE — {len(collisions)} colliding id(s):")
        for tid, toks in sorted(collisions.items()):
            print(f"    id {tid}: {' == '.join(sorted(toks))}")
        print("    Cause: FixedVocabTokenizer keys _idx_to_token by raw value over\n"
              "      range(min_val, max_val+1), but pipeline.py advances the global\n"
              "      offset by len(...) instead of max(local)+1.  MONTH is the only\n"
              "      step with min_val=1, so it overhangs the next step by exactly 1.\n"
              "    Fix: key by (i - min_val) in FixedVocabTokenizer, subtracting\n"
              "      min_val in tokenize().  This CHANGES vocab_size and therefore\n"
              "      REQUIRES RETRAINING -- do not apply it while evaluating the\n"
              "      shipped checkpoint.")
    if dead:
        print(f"\n  Unreachable ids (never emitted): {dead}")

    # -- what a single encoded row looks like ---------------------------------
    print("\n\nEncoded row layout per END_TOKEN  (src/tokenizer/pipeline.py::encode)\n")
    fields = [s["prefix"] for s in layout]
    for end_token in (None, "<sep>", "<eos>"):
        row = ["<bos>"] + [f"{f}_*" for f in fields] + ([end_token] if end_token else [])
        pool_idx = len(row) - 1
        seen = ("every transaction boundary in every corpus line"
                if end_token != "<eos>" else
                "ONLY the final position of a ~4096-token corpus line")
        print(f"  END_TOKEN = {str(end_token):<7} seq_len={len(row):<3} "
              f"pool_idx={pool_idx:<3} pools on {row[pool_idx]!r}")
        print(f"  {'':<32}position seen during pretraining at: {seen}")
    print(f"\n  (last-token pooling picks attention_mask.sum(dim=1) - 1, i.e. the\n"
          f"   final non-<pad> position -- see src/decoder_inference.py)")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
