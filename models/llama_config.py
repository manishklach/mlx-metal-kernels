from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised in environments without MLX
    mx = None


@dataclass
class LlamaLikeConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    vocab_size: int | None = None
    tie_word_embeddings: bool = False
    model_type: str = "llama_like"

    def validate(self) -> "LlamaLikeConfig":
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim, "
                f"got hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}, "
                f"head_dim={self.head_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads, "
                f"got {self.num_attention_heads}, {self.num_key_value_heads}"
            )
        if self.intermediate_size <= self.hidden_size:
            raise ValueError(
                f"intermediate_size must be greater than hidden_size, got {self.intermediate_size}, {self.hidden_size}"
            )
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        return self

    def attention_groups(self) -> int:
        self.validate()
        return self.num_attention_heads // self.num_key_value_heads

    def kv_groups(self) -> int:
        return self.attention_groups()

    def is_gqa(self) -> bool:
        self.validate()
        return self.num_attention_heads != self.num_key_value_heads

    def q_heads(self) -> int:
        self.validate()
        return self.num_attention_heads

    def kv_heads(self) -> int:
        self.validate()
        return self.num_key_value_heads

    def q_output_dim(self) -> int:
        self.validate()
        return self.num_attention_heads * self.head_dim

    def kv_output_dim(self) -> int:
        self.validate()
        return self.num_key_value_heads * self.head_dim

    def fused_qkv_output_dim(self) -> int:
        self.validate()
        return self.q_output_dim() + 2 * self.kv_output_dim()

    def qkv_output_dim(self) -> int:
        return self.fused_qkv_output_dim()

    def layer_shapes(self) -> dict:
        self.validate()
        return {
            "q_proj": (self.q_output_dim(), self.hidden_size),
            "k_proj": (self.kv_output_dim(), self.hidden_size),
            "v_proj": (self.kv_output_dim(), self.hidden_size),
            "o_proj": (self.hidden_size, self.hidden_size),
            "gate_proj": (self.intermediate_size, self.hidden_size),
            "up_proj": (self.intermediate_size, self.hidden_size),
            "down_proj": (self.hidden_size, self.intermediate_size),
            "input_layernorm": (self.hidden_size,),
            "post_attention_layernorm": (self.hidden_size,),
            "fused_qkv": (self.fused_qkv_output_dim(), self.hidden_size),
            "cache": (self.max_position_embeddings, self.num_key_value_heads, self.head_dim),
        }

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LlamaLikeConfig":
        return cls(**data).validate()


def build_rope_tables(config: LlamaLikeConfig, seq_len: int | None = None, dtype=None):
    if mx is None:
        raise RuntimeError("MLX is required for build_rope_tables")
    if dtype is None:
        dtype = mx.float32
    config.validate()
    seq_total = config.max_position_embeddings if seq_len is None else seq_len
    if seq_total <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_total}")
    if config.head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {config.head_dim}")
    position_ids = mx.arange(seq_total, dtype=mx.float32)
    inv_freq = 1.0 / (
        float(config.rope_theta)
        ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / float(config.head_dim))
    )
    freqs = position_ids.reshape(seq_total, 1) * inv_freq.reshape(1, config.head_dim // 2)
    return mx.cos(freqs).astype(dtype), mx.sin(freqs).astype(dtype)


def tiny_debug_config() -> LlamaLikeConfig:
    return LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=64,
        vocab_size=256,
        model_type="llama_like_tiny_debug",
    ).validate()


def tiny_gqa_debug_config() -> LlamaLikeConfig:
    return LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=64,
        vocab_size=256,
        model_type="llama_like_tiny_gqa_debug",
    ).validate()


def llama_7b_like() -> LlamaLikeConfig:
    return LlamaLikeConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        num_hidden_layers=32,
        max_position_embeddings=4096,
        vocab_size=32000,
        model_type="llama_like_7b_approx",
    ).validate()


def llama_8b_like() -> LlamaLikeConfig:
    return LlamaLikeConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        num_hidden_layers=32,
        max_position_embeddings=8192,
        vocab_size=128256,
        model_type="llama_like_8b_approx",
    ).validate()
