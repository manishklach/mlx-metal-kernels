from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tokenization import CharTokenizer, TokenizerProtocol, WhitespaceTokenizer


class OptionalDependencyError(ImportError):
    pass


class HFTokenizerAdapter(TokenizerProtocol):
    """Adapter around the optional Hugging Face tokenizers library (not transformers)."""

    def __init__(self, tokenizer_file: str | Path):
        self.tokenizer_file = str(Path(tokenizer_file).resolve())
        self._tokenizer: Any = None
        self._bos: int | None = None
        self._eos: int | None = None
        self._pad: int | None = None
        self._unk: int | None = None

    @staticmethod
    def _lazy_load():
        try:
            from tokenizers import Tokenizer as _Tokenizer
            return _Tokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "The 'tokenizers' package is optional. "
                "Install it (pip install tokenizers) to use HFTokenizerAdapter."
            ) from exc

    def _get_tokenizer(self):
        if self._tokenizer is None:
            cls = self._lazy_load()
            self._tokenizer = cls.from_file(self.tokenizer_file)
        return self._tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self._get_tokenizer().encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True, stop_at_eos: bool | None = None) -> str:
        tok = self._get_tokenizer()
        if stop_at_eos and self.eos_token_id is not None:
            cut = []
            for tid in token_ids:
                cut.append(tid)
                if tid == self.eos_token_id:
                    break
            token_ids = cut
        return tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._get_tokenizer().get_vocab_size()

    @property
    def bos_token_id(self) -> int | None:
        if self._bos is None:
            tok = self._get_tokenizer()
            self._bos = tok.token_to_id("<s>") if tok.token_to_id("<s>") is not None else tok.token_to_id("[BOS]")
        return self._bos

    @property
    def eos_token_id(self) -> int | None:
        if self._eos is None:
            tok = self._get_tokenizer()
            self._eos = tok.token_to_id("</s>") if tok.token_to_id("</s>") is not None else tok.token_to_id("[EOS]")
        return self._eos

    @property
    def pad_token_id(self) -> int | None:
        if self._pad is None:
            tok = self._get_tokenizer()
            self._pad = tok.token_to_id("<pad>") if tok.token_to_id("<pad>") is not None else tok.token_to_id("[PAD]")
        return self._pad

    @property
    def unk_token_id(self) -> int | None:
        if self._unk is None:
            tok = self._get_tokenizer()
            self._unk = tok.token_to_id("<unk>") if tok.token_to_id("<unk>") is not None else tok.token_to_id("[UNK]")
        return self._unk


class SentencePieceTokenizerAdapter(TokenizerProtocol):
    """Adapter around the optional sentencepiece library."""

    def __init__(self, model_file: str | Path, *, add_bos: bool = True, add_eos: bool = False):
        self.model_file = str(Path(model_file).resolve())
        self.add_bos = add_bos
        self.add_eos = add_eos
        self._sp = self._lazy_load()
        self._processor = self._sp.SentencePieceProcessor()
        self._processor.Load(self.model_file)

    @staticmethod
    def _lazy_load():
        try:
            import sentencepiece as spm
            return spm
        except ImportError as exc:
            raise OptionalDependencyError(
                "The 'sentencepiece' package is optional. "
                "Install it (pip install sentencepiece) to use SentencePieceTokenizerAdapter."
            ) from exc

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        if self.add_bos and add_special_tokens and self.bos_token_id is not None:
            ids.append(self.bos_token_id)
        ids.extend(self._processor.EncodeAsIds(text))
        if self.add_eos and add_special_tokens and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True, stop_at_eos: bool | None = None) -> str:
        if skip_special_tokens or (stop_at_eos and self.eos_token_id is not None):
            filtered: list[int] = []
            special = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
            for tid in token_ids:
                if skip_special_tokens and tid in special:
                    if tid == self.eos_token_id:
                        if stop_at_eos if stop_at_eos is not None else skip_special_tokens:
                            break
                    continue
                filtered.append(tid)
            token_ids = filtered
        return self._processor.DecodeIds(token_ids)

    @property
    def vocab_size(self) -> int:
        return self._processor.GetPieceSize()

    @property
    def bos_token_id(self) -> int | None:
        bid = self._processor.bos_id()
        return int(bid) if bid >= 0 else None

    @property
    def eos_token_id(self) -> int | None:
        eid = self._processor.eos_id()
        return int(eid) if eid >= 0 else None

    @property
    def pad_token_id(self) -> int | None:
        pid = self._processor.pad_id()
        return int(pid) if pid >= 0 else None

    @property
    def unk_token_id(self) -> int | None:
        uid = self._processor.unk_id()
        return int(uid) if uid >= 0 else None


class TokenizerAdapterFactory:
    @staticmethod
    def from_file(path: str | Path | None = None, *, kind: str | None = None, **kwargs) -> TokenizerProtocol:
        if kind == "char" or (path is None and kind is None):
            return CharTokenizer(**kwargs)
        if kind == "whitespace":
            return WhitespaceTokenizer(**kwargs)
        if path is None and kind is None:
            return CharTokenizer(**kwargs)
        if path is None:
            raise ValueError("path is required when kind is not 'char' or 'whitespace'")
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"tokenizer file not found: {resolved}")
        if kind is None:
            ext = resolved.suffix.lower()
            if ext == ".json":
                kind = "hf-tokenizers"
            elif ext in (".model", ".spm"):
                kind = "sentencepiece"
            else:
                raise ValueError(
                    f"cannot infer tokenizer kind from extension {ext!r}; "
                    f"specify --kind hf-tokenizers or --kind sentencepiece"
                )
        if kind == "hf-tokenizers":
            return HFTokenizerAdapter(str(resolved), **kwargs)
        if kind == "sentencepiece":
            return SentencePieceTokenizerAdapter(str(resolved), **kwargs)
        raise ValueError(
            f"unknown tokenizer kind: {kind!r}. "
            f"Expected one of: 'hf-tokenizers', 'sentencepiece', 'char', 'whitespace'"
        )

    @staticmethod
    def available_adapters() -> dict[str, bool]:
        result: dict[str, bool] = {"char": True, "whitespace": True}
        try:
            import tokenizers  # noqa: F401
            result["hf-tokenizers"] = True
        except ImportError:
            result["hf-tokenizers"] = False
        try:
            import sentencepiece  # noqa: F401
            result["sentencepiece"] = True
        except ImportError:
            result["sentencepiece"] = False
        return result


@dataclass
class TokenizerInfo:
    kind: str
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    unk_token_id: int | None
    source: str | None = None


def describe_tokenizer(tokenizer: TokenizerProtocol, source: str | None = None, kind: str | None = None) -> TokenizerInfo:
    if kind is None:
        if isinstance(tokenizer, CharTokenizer):
            kind = "char"
        elif isinstance(tokenizer, WhitespaceTokenizer):
            kind = "whitespace"
        elif isinstance(tokenizer, HFTokenizerAdapter):
            kind = "hf-tokenizers"
        elif isinstance(tokenizer, SentencePieceTokenizerAdapter):
            kind = "sentencepiece"
        else:
            kind = "unknown"
    return TokenizerInfo(
        kind=kind,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        source=source,
    )


def load_tokenizer_for_generation(
    path: str | Path | None = None,
    kind: str | None = None,
    fallback: str = "char",
) -> TokenizerProtocol:
    if path is not None:
        return TokenizerAdapterFactory.from_file(path, kind=kind)
    if fallback == "char":
        return CharTokenizer()
    if fallback == "whitespace":
        return WhitespaceTokenizer()
    raise ValueError(f"unknown fallback tokenizer kind: {fallback!r}. Expected 'char' or 'whitespace'")
