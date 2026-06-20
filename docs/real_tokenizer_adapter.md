# Real Tokenizer Adapter Scaffold

## Purpose

The tokenizer adapter scaffold adds optional adapters for real local tokenizer files while keeping all tokenizer dependencies optional. This allows the repo to work with real tokenizer formats without requiring them.

## Why tokenizer adapters are optional

Tokenization is not a core kernel concern. The repo's toy tokenizers are sufficient for synthetic demos and plumbing tests. Real tokenizer support requires external packages (`tokenizers`, `sentencepiece`) that should not be mandatory for basic usage.

## TokenizerProtocol

All tokenizers implement the `TokenizerProtocol` abstract base class:

```python
class TokenizerProtocol(ABC):
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str
    @property
    def vocab_size(self) -> int
    @property
    def bos_token_id(self) -> int | None
    @property
    def eos_token_id(self) -> int | None
    @property
    def pad_token_id(self) -> int | None
    @property
    def unk_token_id(self) -> int | None
```

## Toy tokenizers

| Tokenizer | Description |
|-----------|-------------|
| `CharTokenizer` | Character-level tokenizer with a built-in default vocabulary of common ASCII characters. |
| `WhitespaceTokenizer` | Word-level tokenizer that splits on whitespace. Default vocabulary: `hello`, `world`, `toy`, `generation`, `demo`. |

Both support `add_bos`/`add_eos` constructor flags and use the same special token scheme (`<pad>`, `<bos>`, `<eos>`, `<unk>`).

## HFTokenizerAdapter using `tokenizers`

The `HFTokenizerAdapter` wraps the Hugging Face `tokenizers` library (not `transformers`).

```python
from models.tokenizer_adapters import HFTokenizerAdapter

tokenizer = HFTokenizerAdapter("path/to/tokenizer.json")
ids = tokenizer.encode("Hello world")
text = tokenizer.decode(ids)
```

Requirements:
- Install `tokenizers` separately: `pip install tokenizers`
- Local tokenizer JSON file only (no network)

## SentencePieceTokenizerAdapter

The `SentencePieceTokenizerAdapter` wraps the `sentencepiece` library.

```python
from models.tokenizer_adapters import SentencePieceTokenizerAdapter

tokenizer = SentencePieceTokenizerAdapter("path/to/tokenizer.model")
ids = tokenizer.encode("Hello world")
text = tokenizer.decode(ids)
```

Requirements:
- Install `sentencepiece` separately: `pip install sentencepiece`
- Local `.model` file only (no network)

## Local files only

Both adapters load from local files only. No model or tokenizer downloads are performed.

## Integration with generation scaffold

The `load_tokenizer_for_generation` helper provides a simple way to get a tokenizer for the generation scaffold:

```python
from models.tokenizer_adapters import load_tokenizer_for_generation

# Use CharTokenizer by default
tokenizer = load_tokenizer_for_generation()

# Use a real tokenizer
tokenizer = load_tokenizer_for_generation("path/to/tokenizer.json", kind="hf-tokenizers")

# Fall back to WhitespaceTokenizer if no path
tokenizer = load_tokenizer_for_generation(fallback="whitespace")
```

The `GenerationConfig.eos_token_id` can be populated from the tokenizer's `eos_token_id` property.

## Limitations

- No chat templates or prompt formatting.
- No Hugging Face hub downloads.
- No `transformers` tokenizer auto-loading.
- No model-specific tokenizer correctness guarantees.
- `tokenizers` and `sentencepiece` packages must be installed separately.

## Future work

- Model-specific chat templates.
- Prompt formatting helpers.
- Tokenizer metadata in quantized checkpoint package.
- Tokenizer + checkpoint package alignment checks.
- Full tiny-model generation demo with real tokenizer.
