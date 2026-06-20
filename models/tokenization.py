from __future__ import annotations

import string
from abc import ABC, abstractmethod


class TokenizerProtocol(ABC):
    @abstractmethod
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True, stop_at_eos: bool | None = None) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return None

    @property
    def unk_token_id(self) -> int | None:
        return None


class CharTokenizer(TokenizerProtocol):
    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
    DEFAULT_VOCAB = (
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits
        + " .,!?;:'\"-()[]{}<>/\\|@#$%^&*_+=~`\n"
    )

    def __init__(self, vocab: str | None = None, add_bos: bool = True, add_eos: bool = False):
        base_vocab = vocab or self.DEFAULT_VOCAB
        deduped = []
        seen = set()
        for token in base_vocab:
            if token not in seen:
                seen.add(token)
                deduped.append(token)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self._id_to_token = list(self.SPECIAL_TOKENS) + deduped
        self._token_to_id = {token: idx for idx, token in enumerate(self._id_to_token)}
        self.pad_token_id = self._token_to_id["<pad>"]
        self.bos_token_id = self._token_to_id["<bos>"]
        self.eos_token_id = self._token_to_id["<eos>"]
        self.unk_token_id = self._token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        token_ids: list[int] = []
        if self.add_bos and add_special_tokens:
            token_ids.append(self.bos_token_id)
        token_ids.extend(self._token_to_id.get(char, self.unk_token_id) for char in text)
        if self.add_eos and add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True, stop_at_eos: bool | None = None) -> str:
        pieces: list[str] = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                if token_id == self.eos_token_id:
                    if stop_at_eos if stop_at_eos is not None else skip_special_tokens:
                        break
                continue
            if 0 <= token_id < self.vocab_size:
                pieces.append(self._id_to_token[token_id])
            elif not skip_special_tokens:
                pieces.append(self._id_to_token.get(token_id, f"<id:{token_id}>"))
            else:
                pieces.append("<unk>")
        return "".join(pieces)


class WhitespaceTokenizer(TokenizerProtocol):
    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, vocab: list[str] | tuple[str, ...] | None = None, add_bos: bool = True, add_eos: bool = False):
        base_vocab = list(vocab or ("hello", "world", "toy", "generation", "demo"))
        deduped = []
        seen = set()
        for token in base_vocab:
            if token not in seen and token not in self.SPECIAL_TOKENS:
                seen.add(token)
                deduped.append(token)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self._id_to_token = list(self.SPECIAL_TOKENS) + deduped
        self._token_to_id = {token: idx for idx, token in enumerate(self._id_to_token)}
        self.pad_token_id = self._token_to_id["<pad>"]
        self.bos_token_id = self._token_to_id["<bos>"]
        self.eos_token_id = self._token_to_id["<eos>"]
        self.unk_token_id = self._token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        token_ids: list[int] = []
        if self.add_bos and add_special_tokens:
            token_ids.append(self.bos_token_id)
        for token in text.split():
            token_ids.append(self._token_to_id.get(token, self.unk_token_id))
        if self.add_eos and add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True, stop_at_eos: bool | None = None) -> str:
        tokens: list[str] = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                if token_id == self.eos_token_id:
                    if stop_at_eos if stop_at_eos is not None else skip_special_tokens:
                        break
                continue
            if 0 <= token_id < self.vocab_size:
                tokens.append(self._id_to_token[token_id])
            elif not skip_special_tokens:
                tokens.append(self._id_to_token.get(token_id, f"<id:{token_id}>"))
            else:
                tokens.append("<unk>")
        return " ".join(tokens)
