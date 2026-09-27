"""SGLang's own model classes, each with its pooler replaced by ``RiseSGLangPooler``.

SGLang registers an external package's classes by class name, over its built-in ones, so the wrappers
keep the names of the classes they wrap. Architectures this SGLang build lacks are skipped.
"""

import importlib

from ..sglang_head import RiseSGLangPooler

_BASES = (
    ("sglang.srt.models.llama", "LlamaForCausalLM"),
    ("sglang.srt.models.mistral", "MistralForCausalLM"),
    ("sglang.srt.models.qwen2", "Qwen2ForCausalLM"),
    ("sglang.srt.models.qwen3", "Qwen3ForCausalLM"),
    ("sglang.srt.models.olmo2", "Olmo2ForCausalLM"),
    ("sglang.srt.models.olmo3", "Olmo3ForCausalLM"),
)


def _wrap(base):
    class Wrapped(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pooler = RiseSGLangPooler(self.pooler)

    Wrapped.__name__ = Wrapped.__qualname__ = base.__name__
    return Wrapped


EntryClass = []
for _module, _name in _BASES:
    try:
        EntryClass.append(_wrap(getattr(importlib.import_module(_module), _name)))
    except (ImportError, AttributeError):
        pass
