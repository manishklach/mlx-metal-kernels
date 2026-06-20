from __future__ import annotations

import pytest

from models.tokenization import CharTokenizer, WhitespaceTokenizer
from models.tokenizer_adapters import (
    HFTokenizerAdapter,
    OptionalDependencyError,
    SentencePieceTokenizerAdapter,
    TokenizerAdapterFactory,
    TokenizerInfo,
    describe_tokenizer,
    load_tokenizer_for_generation,
)


class TestOptionalDependencyError:
    def test_exists(self):
        assert issubclass(OptionalDependencyError, ImportError)


class TestTokenizerAdapterFactory:
    def test_available_adapters_returns_dict(self):
        adapters = TokenizerAdapterFactory.available_adapters()
        assert isinstance(adapters, dict)
        assert "char" in adapters
        assert "whitespace" in adapters
        assert "hf-tokenizers" in adapters
        assert "sentencepiece" in adapters

    def test_from_file_char_kind(self):
        tokenizer = TokenizerAdapterFactory.from_file(kind="char")
        assert isinstance(tokenizer, CharTokenizer)

    def test_from_file_whitespace_kind(self):
        tokenizer = TokenizerAdapterFactory.from_file(kind="whitespace")
        assert isinstance(tokenizer, WhitespaceTokenizer)

    def test_from_file_no_path_no_kind_returns_char(self):
        tokenizer = TokenizerAdapterFactory.from_file()
        assert isinstance(tokenizer, CharTokenizer)

    def test_from_file_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError, match="tokenizer file not found"):
            TokenizerAdapterFactory.from_file("/nonexistent/path.json")

    def test_from_file_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown tokenizer kind"):
            TokenizerAdapterFactory.from_file(kind="unknown_kind")

    def test_from_file_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="cannot infer tokenizer kind"):
            TokenizerAdapterFactory.from_file("/tmp/tokenizer.xyz")

    def test_from_file_requires_path_for_non_toy_kinds(self):
        with pytest.raises(ValueError, match="path is required"):
            TokenizerAdapterFactory.from_file(kind="hf-tokenizers")


class TestDescribeTokenizer:
    def test_describe_char_tokenizer(self):
        tokenizer = CharTokenizer()
        info = describe_tokenizer(tokenizer)
        assert isinstance(info, TokenizerInfo)
        assert info.kind == "char"
        assert info.vocab_size > 0
        assert info.bos_token_id is not None
        assert info.eos_token_id is not None
        assert info.pad_token_id is not None
        assert info.unk_token_id is not None

    def test_describe_whitespace_tokenizer(self):
        tokenizer = WhitespaceTokenizer()
        info = describe_tokenizer(tokenizer)
        assert info.kind == "whitespace"
        assert info.vocab_size > 0

    def test_describe_with_source(self):
        tokenizer = CharTokenizer()
        info = describe_tokenizer(tokenizer, source="/tmp/test.json")
        assert info.source == "/tmp/test.json"

    def test_describe_tokenizer_info_fields(self):
        tokenizer = CharTokenizer()
        info = describe_tokenizer(tokenizer, kind="custom_kind")
        assert info.kind == "custom_kind"
        assert info.vocab_size == tokenizer.vocab_size
        assert info.bos_token_id == tokenizer.bos_token_id
        assert info.eos_token_id == tokenizer.eos_token_id


class TestHFTokenizerAdapter:
    def test_init_with_missing_tokenizers_raises(self):
        with pytest.raises(OptionalDependencyError, match="tokenizers"):
            HFTokenizerAdapter._lazy_load()

    def test_init_with_nonexistent_file(self):
        with pytest.raises(OptionalDependencyError, match="tokenizers"):
            HFTokenizerAdapter("/nonexistent/tokenizer.json")

    def test_raise_for_missing_dependency(self):
        with pytest.raises(OptionalDependencyError, match="tokenizers"):
            HFTokenizerAdapter._lazy_load()


class TestSentencePieceTokenizerAdapter:
    def test_init_with_missing_sentencepiece_raises(self):
        with pytest.raises(OptionalDependencyError, match="sentencepiece"):
            SentencePieceTokenizerAdapter._lazy_load()

    def test_raise_for_missing_dependency(self):
        with pytest.raises(OptionalDependencyError, match="sentencepiece"):
            SentencePieceTokenizerAdapter._lazy_load()


class TestLoadTokenizerForGeneration:
    def test_no_path_fallback_char(self):
        tokenizer = load_tokenizer_for_generation(fallback="char")
        assert isinstance(tokenizer, CharTokenizer)

    def test_no_path_fallback_whitespace(self):
        tokenizer = load_tokenizer_for_generation(fallback="whitespace")
        assert isinstance(tokenizer, WhitespaceTokenizer)

    def test_no_path_no_fallback_defaults_char(self):
        tokenizer = load_tokenizer_for_generation()
        assert isinstance(tokenizer, CharTokenizer)

    def test_unknown_fallback_raises(self):
        with pytest.raises(ValueError, match="unknown fallback"):
            load_tokenizer_for_generation(fallback="unknown")


class TestCharTokenizerBackwardCompat:
    def test_encode_without_keyword_still_works(self):
        tokenizer = CharTokenizer()
        ids = tokenizer.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_with_stop_at_eos_still_works(self):
        tokenizer = CharTokenizer()
        ids = tokenizer.encode("hello")
        text = tokenizer.decode(ids, stop_at_eos=True)
        assert isinstance(text, str)
