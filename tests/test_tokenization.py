from models import CharTokenizer, WhitespaceTokenizer


def test_char_tokenizer_roundtrip_simple_text():
    tokenizer = CharTokenizer(add_bos=True, add_eos=False)
    text = "Hello 42!"
    token_ids = tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text


def test_char_tokenizer_unknown_char_maps_to_unk():
    tokenizer = CharTokenizer(vocab="abc", add_bos=False, add_eos=False)
    token_ids = tokenizer.encode("az")
    assert token_ids[1] == tokenizer.unk_token_id
    assert tokenizer.decode(token_ids, skip_special_tokens=False) == "a<unk>"
    assert tokenizer.decode(token_ids) == "a"


def test_char_tokenizer_bos_eos_behavior_and_decode_skip():
    tokenizer = CharTokenizer(add_bos=True, add_eos=True)
    token_ids = tokenizer.encode("Hi")
    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode([tokenizer.pad_token_id] + token_ids) == "Hi"


def test_char_tokenizer_vocab_size_positive():
    tokenizer = CharTokenizer()
    assert tokenizer.vocab_size > 0


def test_whitespace_tokenizer_known_and_unknown_words():
    tokenizer = WhitespaceTokenizer(vocab=["hello", "world"], add_bos=False, add_eos=False)
    token_ids = tokenizer.encode("hello there world")
    assert token_ids[0] != tokenizer.unk_token_id
    assert token_ids[1] == tokenizer.unk_token_id
    assert tokenizer.decode(token_ids, skip_special_tokens=False) == "hello <unk> world"
    assert tokenizer.decode(token_ids) == "hello world"


def test_whitespace_tokenizer_vocab_size():
    tokenizer = WhitespaceTokenizer(vocab=["one", "two", "three"])
    assert tokenizer.vocab_size == 7
