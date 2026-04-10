"""Built-in pipeline plugins.

Importing this module registers bundled plugins into the pipeline registries.
"""

from .llm_ner import LLMBasedNERPlugin, register as register_ner
from .llm_re import LLMBasedREStubPlugin, register as register_re

register_ner()
register_re()

__all__ = ["LLMBasedNERPlugin", "LLMBasedREStubPlugin", "register_ner", "register_re"]