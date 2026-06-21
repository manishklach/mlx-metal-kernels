from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for mtp") from exc


@dataclass
class MTPConfig:
    num_draft_tokens: int = 4
    hidden_size: int = 64
    num_layers: int = 1
    seed: int = 42
    max_seq_len: int = 128

    def validate(self) -> MTPConfig:
        if self.num_draft_tokens <= 0:
            raise ValueError(f"num_draft_tokens must be positive, got {self.num_draft_tokens}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_draft_tokens": self.num_draft_tokens,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "seed": self.seed,
            "max_seq_len": self.max_seq_len,
        }


class SyntheticMTPHead:
    def __init__(self, config: MTPConfig, vocab_size: int):
        config.validate()
        self.config = config
        self.vocab_size = vocab_size
        rng = np.random.default_rng(config.seed)
        self._bias = rng.normal(0, 0.02, size=(vocab_size,)).astype(np.float32)
        self._scale = rng.normal(0, 0.02, size=(config.hidden_size, vocab_size)).astype(np.float32)
        self._dummy_counter: int = 0

    def forward(self, hidden_states: Any, *, token_offset: int = 0) -> Any:
        _ = token_offset
        try:
            import mlx.core as mx
            if isinstance(hidden_states, mx.array):
                bs, seq, hd = hidden_states.shape
                result = mx.dot(hidden_states.reshape(-1, hd), mx.array(self._scale))
                result = result + mx.array(self._bias)
                return result.reshape(bs, seq, self.vocab_size)
        except ImportError:
            pass
        import numpy as np
        hidden = np.asarray(hidden_states, dtype=np.float32)
        bs, seq, hd = hidden.shape
        logits = hidden.reshape(-1, hd) @ self._scale + self._bias
        return logits.reshape(bs, seq, self.vocab_size)

    def propose(self, context_hidden: Any, num_tokens: int, *, seed=None) -> Any:
        _ = seed
        if num_tokens <= 0:
            return np.zeros((1, 0, self.vocab_size), dtype=np.float32)
        try:
            import mlx.core as mx
            if isinstance(context_hidden, mx.array):
                last = context_hidden[:, -1:, :]
                token_hidden = mx.tile(last, (1, num_tokens, 1))
                return self.forward(token_hidden)
        except ImportError:
            pass
        last = np.asarray(context_hidden)[:, -1:, :]
        token_hidden = np.tile(last, (1, num_tokens, 1))
        return self.forward(token_hidden)


def mtp_propose_tokens(
    mtp_head: SyntheticMTPHead,
    context_hidden: Any,
    num_tokens: int,
    *,
    seed: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[list[int], Any]:
    logits = mtp_head.propose(context_hidden, num_tokens, seed=seed)
    from models.sampling import sample_logits
    token_ids: list[int] = []
    for step_idx in range(num_tokens):
        step_logits = logits[:, step_idx, :].squeeze(0)
        sample_seed = None if seed is None else seed + step_idx
        sample_seed = None if sample_seed is None else sample_seed
        token = int(sample_logits(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=sample_seed,
        ))
        token_ids.append(token)
    return token_ids, logits
