"""Shared fixtures: tiny random models and a byte-level tokenizer, built offline."""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from rise.config import RiseConfig  # noqa: E402


def make_byte_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["<|endoftext|>"] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")


VOCAB = 257


def make_neox(seed: int = 0):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64, max_position_embeddings=512, rotary_pct=0.25,
                        initializer_range=0.2, tie_word_embeddings=False)
    return GPTNeoXForCausalLM(cfg).eval()


def make_llama(seed: int = 1):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      initializer_range=0.2, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).eval()


def small_config(**kw) -> RiseConfig:
    base = dict(Kr=16, Kh=8, Kg=8, seq_len=96, chunk_size=48, chunk_overlap=8, seed=7)
    base.update(kw)
    cfg = RiseConfig(**base)
    cfg.validate()
    return cfg


WORDS = ("the model reads data and every token shifts the readout of attention while gradients "
         "flow through layers toward the head where influence concentrates near the output").split()


def make_texts(n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.choice([3, 6, 10, 18, 30])  # some exceed chunk_size (48 bytes) -> multi-chunk
        out.append(" ".join(rng.choice(WORDS) for _ in range(k)))
    return out


@pytest.fixture(scope="session")
def byte_tok():
    return make_byte_tokenizer()


@pytest.fixture(scope="session")
def neox():
    return make_neox()


@pytest.fixture(scope="session")
def llama():
    return make_llama()


@pytest.fixture()
def corpus(tmp_path):
    texts = make_texts(23, seed=3)
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"text": t, "label": "positive" if i % 4 == 0 else "negative"}) + "\n")
    return str(path), texts
