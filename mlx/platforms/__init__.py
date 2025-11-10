"""
Utilities for routing module execution to the appropriate platform backend.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict, Mapping

ModuleConfig = Dict[str, Any]
ModuleRunner = Callable[[ModuleConfig], Any]


class UnknownModuleError(ValueError):
    """Raised when a module/platform combination is not registered."""


class PlatformRegistry:
    """Keeps a mapping between platforms and their supported modules."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, str]] = {}
        self._fallback_platform = "generic"

    def register(self, platform: str, module_name: str, dotted_path: str) -> None:
        self._registry.setdefault(platform, {})[module_name] = dotted_path

    def register_generic(self, module_name: str, dotted_path: str) -> None:
        self.register(self._fallback_platform, module_name, dotted_path)

    def available(self) -> Mapping[str, Mapping[str, str]]:
        return {platform: modules.copy() for platform, modules in self._registry.items()}

    def resolve(self, platform: str, module_name: str) -> ModuleRunner:
        search_order = [platform]
        if platform != self._fallback_platform:
            search_order.append(self._fallback_platform)

        for candidate in search_order:
            modules = self._registry.get(candidate)
            if not modules:
                continue
            dotted_path = modules.get(module_name)
            if dotted_path:
                module_path, func_name = dotted_path.split(":")
                module = import_module(module_path)
                return getattr(module, func_name)

        raise UnknownModuleError(
            f"Module '{module_name}' is not available for platform '{platform}'."
        )


registry = PlatformRegistry()

# -- Platform registrations -------------------------------------------------
registry.register("openai", "chat", "mlx.platforms.openai.chat:run_chat")
registry.register("torch", "ic-one-shot", "mlx.platforms.torch.ic_one_shot:run_ic_one_shot")
registry.register("ultralytics", "obj-detect", "mlx.platforms.ultralytics.obj_detect:run_obj_detect")
# Provide a generic fallback so existing invocations without --platform still work.
registry.register_generic("ic-one-shot", "mlx.platforms.torch.ic_one_shot:run_ic_one_shot")
registry.register_generic("system", "mlx.platforms.system:run_system")
registry.register_generic("rag", "mlx.modules.rag.run:run")


def run_module(platform: str, module_name: str, config: ModuleConfig) -> Any:
    runner = registry.resolve(platform, module_name)
    return runner(config)


def registered_modules() -> Mapping[str, Mapping[str, str]]:
    """Expose a copy of the registered modules for help messages or debugging."""
    return registry.available()
