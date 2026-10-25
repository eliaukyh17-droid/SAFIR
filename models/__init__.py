"""Public model API with lazy imports.

Keeping the package initializer lightweight lets individual components (for
example the smoke-aware TIFM block) be audited without importing every
optional backbone dependency. Existing ``from models import TIFDFusion``
calls retain the same public interface.
"""

from importlib import import_module

__all__ = [
    "TIFDFusion",
    "DualStreamEncoder",
    "TIFM",
    "TIFMBlock",
    "FusionDecoder",
    "TaskPriorExtractor",
    "TIFDLoss",
]

_EXPORTS = {
    "TIFDFusion": (".tifd_fusion", "TIFDFusion"),
    "DualStreamEncoder": (".encoder", "DualStreamEncoder"),
    "TIFM": (".tifm", "TIFM"),
    "TIFMBlock": (".tifm", "TIFMBlock"),
    "FusionDecoder": (".decoder", "FusionDecoder"),
    "TaskPriorExtractor": (".prior", "TaskPriorExtractor"),
    "TIFDLoss": (".losses", "TIFDLoss"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
