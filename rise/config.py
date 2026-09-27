"""RISE configuration.

Field names match the RISE research code's config, so an existing
``config.json`` loads unchanged. Defaults reproduce what the paper's GPU runs
effectively used: the research command line's values (topL_cum_prob 0.92,
min_topL 4, which the CLI set over the dataclass's 0.97 / 8) and a fixed
temperature of 0.1 (see ``adaptive_temperature`` below).
Research-only fields that the estimator never read (auto-anchor weights) and
runtime fields (``device``) are accepted on load and dropped; use
``rise.compat.research_effective_config`` to load a research config with the
temperature it effectively ran with.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import List

VALID_CHANNELS = ("rh", "gh", "rg")
_COMPAT_DEFAULTS = {"sample_format": "auto", "gt_slot": "replace", "sketch_rng": "cpu", "l2_eps": "norm"}
_COMPAT_CHOICES = {"sample_format": ("auto", "alpaca"), "gt_slot": ("replace", "append"), "sketch_rng": ("cpu", "cuda"),
                   "l2_eps": ("norm", "squared")}


@dataclass
class RiseConfig:
    # Channels to fuse ("rh+gh" is the paper's default) and their weights.
    fusion_mode: str = "rh+gh"
    lambda_rh: float = 1.0
    lambda_gh: float = 0.3
    lambda_rg: float = 0.5
    # Embedding used to project residuals for GH: "output" (LM head) or "input".
    embedding_type: str = "output"

    # CountSketch widths.
    Kr: int = 128
    Kh: int = 128
    Kg: int = 64
    rh_only_Kr_boost: int = 160

    # Sparse residual support: TopK cap, then smallest prefix with cumprob >= topL_cum_prob.
    adaptive_topL: bool = True
    topL_cum_prob: float = 0.92
    max_topL_cap: int = 256
    min_topL: int = 4
    fallback_topL: int = 32

    # Temperature. By default a fixed tau_fallback = 0.1, the temperature behind the paper's numbers.
    # adaptive_temperature=True instead picks a per-chunk tau so the mean predictive entropy hits
    # target_entropy. (The research code intended that search, but on GPU its fp16 entropy was NaN
    # and every chunk fell to tau_min + (tau_max - tau_min) / 2**(steps + 1); see docs/design.md.)
    adaptive_temperature: bool = False
    target_entropy: float = 4.8
    tau_min: float = 0.1
    tau_max: float = 2.5
    tau_search_steps: int = 15
    tau_fallback: float = 0.1

    # Tokenization / chunking.
    seq_len: int = 512
    min_seq_len: int = 2
    chunk_long_sequences: bool = True
    chunk_size: int = 256
    chunk_overlap: int = 50

    eps: float = 1e-8
    normalize_sample: bool = True
    seed: int = 42

    # Research-code compatibility. The defaults are RISE's behavior, which the research code has had
    # since April 2026; the alternatives reproduce the December 2025 research scripts
    #, behind part of the paper's appendix tables (docs/design.md, section 9).
    # sample_format: "auto" (text; else prompt + generation; else instruction / input / output joined
    #   by newlines) or "alpaca" ("### Instruction: ...\n### Input: ...\n### Response: ...", the
    #   template the paper's models were fine-tuned on).
    sample_format: str = "auto"
    # gt_slot: when the ground truth is not in the top-K, "replace" puts it in the K-th slot before
    #   the top-L cutoff is taken; "append" takes the cutoff on the top-K alone and adds it after.
    gt_slot: str = "replace"
    # sketch_rng: generator that draws the CountSketch tables from the seed, "cpu" or "cuda". The
    #   December scripts drew them on the GPU; the same seed gives different tables.
    sketch_rng: str = "cpu"
    # l2_eps: "norm" divides by max(|x|, eps); "squared" by sqrt(max(|x|^2, eps)), as the December
    #   scripts did. With eps = 1e-8 the latter floors the norm at 1e-4, so the near-zero residuals
    #   of near-certain predictions stay short and those positions weigh less in the signature.
    l2_eps: str = "norm"

    # ---- (de)serialization ----
    def to_dict(self) -> dict:
        d = asdict(self)
        # Compatibility fields at their defaults are left out, so the config (and the attrs of every
        # index built before they existed) serializes exactly as before and old builds still resume.
        for name, default in _COMPAT_DEFAULTS.items():
            if d[name] == default:
                del d[name]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RiseConfig":
        names = {f.name for f in fields(cls)}
        obj = cls(**{k: v for k, v in d.items() if k in names})
        obj.validate()
        return obj

    def validate(self) -> None:
        chs = self.get_channels()
        if not chs:
            raise ValueError("fusion_mode selects no channel")
        for ch in chs:
            if ch not in VALID_CHANNELS:
                raise ValueError(f"invalid channel {ch!r}; valid: {VALID_CHANNELS}")
        if self.embedding_type not in ("input", "output"):
            raise ValueError(f"embedding_type must be 'input' or 'output', got {self.embedding_type!r}")
        for name in ("Kr", "Kh", "Kg", "max_topL_cap", "min_topL", "fallback_topL",
                     "seq_len", "chunk_size", "tau_search_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.min_seq_len < 2:
            raise ValueError("min_seq_len must be >= 2 (one prediction needs two tokens)")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("need 0 <= chunk_overlap < chunk_size")
        if not 0.0 < self.topL_cum_prob <= 1.0:
            raise ValueError("topL_cum_prob must be in (0, 1]")
        if not 0.0 < self.tau_min < self.tau_max:
            raise ValueError("need 0 < tau_min < tau_max")
        for name, choices in _COMPAT_CHOICES.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {getattr(self, name)!r}")

    def norm_eps(self) -> float:
        """Epsilon for l2_normalize: sqrt(max(|x|^2, eps)) == max(|x|, sqrt(eps))."""
        return math.sqrt(self.eps) if self.l2_eps == "squared" else self.eps

    # ---- channel helpers (same semantics as the research code) ----
    def get_channels(self) -> List[str]:
        return [x.strip() for x in self.fusion_mode.split("+") if x.strip()]

    def has_channel(self, ch: str) -> bool:
        return ch in self.get_channels()

    def get_effective_Kr(self) -> int:
        ch = self.get_channels()
        if len(ch) == 1 and ch[0] == "rh" and self.rh_only_Kr_boost > self.Kr:
            return self.rh_only_Kr_boost
        return self.Kr

    def dim_rh(self) -> int:
        return self.get_effective_Kr() * self.Kh

    def dim_gh(self) -> int:
        return self.Kg * self.Kh

    def dim_rg(self) -> int:
        return self.get_effective_Kr() * self.Kg

    def get_vector_dim(self) -> int:
        dim = 0
        if self.has_channel("rh"):
            dim += self.dim_rh()
        if self.has_channel("gh"):
            dim += self.dim_gh()
        if self.has_channel("rg"):
            dim += self.dim_rg()
        return dim
