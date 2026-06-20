from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for this demo") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import greedy_sample, sample_logits, top_k_filter, top_p_filter


def main():
    logits = np.array([0.1, 2.5, 1.7, -0.3, 3.2, 0.8], dtype=np.float32)
    print("logits:", logits.tolist())
    print("greedy:", greedy_sample(logits))
    print("temperature_sample:", sample_logits(logits, temperature=0.8, seed=7))
    print("top_k_filtered:", np.asarray(top_k_filter(logits, 3)).tolist())
    print("top_k_sample:", sample_logits(logits, temperature=0.9, top_k=3, seed=7))
    print("top_p_filtered:", np.asarray(top_p_filter(logits, 0.75)).tolist())
    print("top_p_sample:", sample_logits(logits, temperature=1.0, top_p=0.75, seed=7))


if __name__ == "__main__":
    main()
