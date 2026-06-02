"""Plugin registry — instantiates classes from Hydra _target_ strings."""

from __future__ import annotations

import importlib
from typing import Any, TypeVar

T = TypeVar("T")


class PluginRegistry:
    """Resolves dotted-path class names (Hydra _target_ convention) to instances.

    Usage::

        registry = PluginRegistry()
        ingestor = registry.build(cfg.ingestor, Ingestor)
    """

    def build(self, cfg: Any, expected_type: type[T] | None = None) -> T:
        """Instantiate a class from *cfg._target_* and validate against *expected_type*.

        *cfg* is typically an OmegaConf DictConfig with a ``_target_`` key.
        Additional keys become constructor kwargs.
        """
        if hasattr(cfg, "_target_"):
            target_str: str = cfg._target_
            kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
        elif isinstance(cfg, dict):
            target_str = cfg["_target_"]
            kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
        else:
            raise TypeError(f"cfg must have a _target_ attribute, got {type(cfg)}")

        cls = self._import(target_str)
        instance = cls(**kwargs)

        if expected_type is not None and not isinstance(instance, expected_type):
            raise TypeError(
                f"_target_ '{target_str}' resolved to {type(instance).__name__}, "
                f"expected {expected_type.__name__}"
            )
        return instance  # type: ignore[return-value]

    @staticmethod
    def _import(dotted_path: str) -> type:
        """Import and return the class at *dotted_path*."""
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)  # type: ignore[no-any-return]


# Module-level singleton
_registry = PluginRegistry()


def build(cfg: Any, expected_type: type[T] | None = None) -> T:
    """Module-level convenience wrapper around PluginRegistry.build."""
    return _registry.build(cfg, expected_type)
