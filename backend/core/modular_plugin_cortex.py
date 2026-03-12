"""ModularPluginCortex — runtime plugin discovery and execution.

Scans a plugin directory for Python files exposing a ``Plugin`` class,
instantiates each one, and provides a uniform ``run_plugin`` interface.

FILE: modular_plugin_cortex.py
VERSION: v1.0.0-MPC-GODCORE
NAME: ModularPluginCortex
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Discover, load, and execute modular skills in runtime — plug-and-play brain extensions
LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any


class ModularPluginCortex:
    """
    Runtime plug-and-play brain extensions for Victor AGI.

    Conventions
    -----------
    - Each plugin is a ``.py`` file in *plugin_dir*.
    - Files must expose a top-level ``Plugin`` class with a ``run`` method.
    - Files whose names begin with ``__`` are skipped.
    """

    def __init__(self, plugin_dir: str = "victor_plugins") -> None:
        self.plugin_dir = plugin_dir
        self.plugins: dict[str, Any] = {}
        self.load_plugins()

    # ------------------------------------------------------------------ loading

    def load_plugins(self) -> None:
        """Discover and instantiate every valid plugin in *plugin_dir*."""
        if not os.path.exists(self.plugin_dir):
            print(f"[MPC] Plugin directory '{self.plugin_dir}' not found. Creating it.")
            try:
                os.makedirs(self.plugin_dir, exist_ok=True)
                self._write_dummy_plugin()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[MPC⚠️] Could not create plugin directory or dummy plugin: {exc}"
                )
                return

        for filename in os.listdir(self.plugin_dir):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue
            path = os.path.join(self.plugin_dir, filename)
            name = filename[:-3]
            self._load_plugin_file(name, path)

    def _write_dummy_plugin(self) -> None:
        """Write a demonstrative dummy plugin if the directory was just created."""
        dummy_path = os.path.join(self.plugin_dir, "dummy_plugin.py")
        if os.path.exists(dummy_path):
            return
        try:
            with open(dummy_path, "w", encoding="utf-8") as fh:
                fh.write("class Plugin:\n")
                fh.write("    def run(self, *args, **kwargs):\n")
                fh.write(
                    "        return f'Dummy plugin executed with args: {args},"
                    " kwargs: {kwargs}'\n"
                )
            print(f"[MPC] Created dummy plugin: {dummy_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[MPC⚠️] Could not write dummy plugin: {exc}")

    def _load_plugin_file(self, name: str, path: str) -> None:
        """Load a single plugin file and register it if valid."""
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                print(
                    f"[MPC ⚠️] Could not create spec for plugin '{name}' from '{path}'."
                )
                return
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if hasattr(mod, "Plugin"):
                self.plugins[name] = mod.Plugin()
                print(f"[MPC 🔌] Plugin '{name}' loaded.")
            else:
                print(
                    f"[MPC ⚠️] Plugin file '{name}' does not have a 'Plugin' class."
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[MPC ⚠️] Failed to load plugin '{name}': {exc}")

    # ------------------------------------------------------------------ execution

    def run_plugin(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute the named plugin's ``run`` method."""
        plugin_instance = self.plugins.get(name)
        if plugin_instance is None:
            return f"[MPC ❌] Plugin '{name}' not found or not loaded."
        if not hasattr(plugin_instance, "run") or not callable(plugin_instance.run):
            return f"[MPC 💥] Plugin '{name}' does not have a callable 'run' method."
        try:
            return plugin_instance.run(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return f"[MPC 💥] Plugin '{name}' crashed during execution: {exc}"

    def list_plugins(self) -> list[str]:
        """Return the names of all currently loaded plugins."""
        return list(self.plugins.keys())
