"""Sample formatting, tokenization and chunking (same rules as the RISE research code)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .config import RiseConfig


def format_sample(example: dict, style: str = "auto") -> str:
    """Text of one JSONL row.

    ``auto``: ``text``; else ``prompt`` + ``generation`` (Brain Rot style); else
    ``instruction`` / ``input`` / ``output`` joined by newlines (Alpaca / Howdy style).
    ``alpaca``: the December 2025 research scripts' template on instruction / input / output
    (``prompt`` rows as in ``auto``); rows without an instruction raise, since the template would
    silently encode empty fields.
    """
    if style == "alpaca":
        if "prompt" in example:
            return f"{example.get('prompt', '')}{example.get('generation', '')}"
        if "instruction" not in example:
            raise ValueError("sample_format 'alpaca' needs an 'instruction' field; this row has "
                             f"{sorted(example)}")
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "") or "").strip()
        output = str(example.get("output", "") or "").strip()
        if input_text:
            return f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: {output}"
        return f"### Instruction: {instruction}\n### Response: {output}"
    if style != "auto":
        raise ValueError(f"unknown sample format {style!r}")
    if example.get("text"):
        return str(example["text"]).strip()
    if "prompt" in example:
        return f"{example.get('prompt', '')}{example.get('generation', '')}"
    if "instruction" in example:
        parts = [example.get("instruction", "")]
        if example.get("input"):
            parts.append(example["input"])
        if example.get("output"):
            parts.append(example["output"])
        return "\n".join(parts).strip()
    return ""


def query_prompt_text(example: dict) -> Optional[str]:
    """Prompt prefix whose tokens are excluded from a query signature (``prompt_text`` field)."""
    p = example.get("prompt_text")
    return str(p) if p else None


class TokenizerAdapter:
    """Minimal tokenizer interface RISE needs, wrapping a Hugging Face tokenizer."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.pad_id = int(pad)

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, revision=None, trust_remote_code=False) -> "TokenizerAdapter":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision, trust_remote_code=trust_remote_code)
        tok.model_max_length = int(1e12)  # we chunk ourselves; silence length warnings
        return cls(tok)

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        if not texts:
            return []
        return [list(x) for x in self.tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]]

    def describe(self) -> dict:
        return {
            "name_or_path": getattr(self.tok, "name_or_path", ""),
            "class": type(self.tok).__name__,
            "size": len(self.tok),
            "pad_id": self.pad_id,
        }


def explode(tokens: Sequence[int], config: RiseConfig) -> List[List[int]]:
    """Split one tokenized sample into chunks.

    Samples up to ``chunk_size`` tokens (or all samples when chunking is off) give one
    chunk truncated to ``seq_len``; longer ones give sliding windows of ``chunk_size``
    with ``chunk_overlap``, covering the whole sample. Windows shorter than
    ``min_seq_len`` are dropped.
    """
    toks = list(tokens)
    if not config.chunk_long_sequences or len(toks) <= config.chunk_size:
        return [toks[: config.seq_len]] if len(toks) >= config.min_seq_len else []
    step = max(1, config.chunk_size - config.chunk_overlap)
    out = []
    for i in range(0, len(toks), step):
        w = toks[i: i + config.chunk_size]
        if len(w) >= config.min_seq_len:
            out.append(w)
    return out


def query_chunks(tokens: Sequence[int], prompt_tokens: Optional[Sequence[int]],
                 config: RiseConfig) -> List[Tuple[List[int], int]]:
    """Chunks (with loss_start) for one query.

    With a prompt, the query is one chunk truncated to ``seq_len`` and only positions
    predicting tokens at index >= len(prompt) contribute (clamped to [1, L-1]). Without
    one it is chunked exactly like an index sample, with loss_start 0 (no masking).
    """
    if prompt_tokens is None:
        return [(c, 0) for c in explode(tokens, config)]
    ids = list(tokens)[: config.seq_len]
    if len(ids) < config.min_seq_len:
        return []
    p = min(len(prompt_tokens), config.seq_len)
    return [(ids, max(1, min(p, len(ids) - 1)))]
