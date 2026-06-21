"""Demonstrate the synthetic MTP scaffold head and mtp_propose_tokens."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mtp import MTPConfig, SyntheticMTPHead, mtp_propose_tokens


def main():
    print("=== MTP Scaffold Demo ===")

    cfg = MTPConfig(
        num_draft_tokens=4,
        hidden_size=16,
        num_layers=1,
        seed=42,
    ).validate()
    head = SyntheticMTPHead(cfg, vocab_size=32)

    rng = np.random.default_rng(0)
    context_hidden = rng.normal(0, 0.1, size=(1, 5, 16)).astype(np.float32)
    print(f"  Context hidden shape: {context_hidden.shape}")
    print(f"  MTP head vocab_size: {head.vocab_size}")

    token_ids, logits = mtp_propose_tokens(head, context_hidden, 4, seed=0)
    print(f"  Proposed token IDs: {token_ids}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  All token IDs in range: {all(0 <= tid < 32 for tid in token_ids)}")

    token_ids_2, _ = mtp_propose_tokens(head, context_hidden, 4, seed=0)
    print(f"  Deterministic with same seed: {token_ids == token_ids_2}")

    token_ids_3, _ = mtp_propose_tokens(head, context_hidden, 4, seed=1)
    print(f"  Different seed gives different result: {token_ids != token_ids_3}")

    zero_ids, zero_logits = mtp_propose_tokens(head, context_hidden, 0)
    print(f"  Zero tokens proposal: {zero_ids} (logits shape: {zero_logits.shape})")


if __name__ == "__main__":
    main()
