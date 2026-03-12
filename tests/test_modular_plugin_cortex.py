import os
import tempfile

import pytest

from backend.core.modular_plugin_cortex import ModularPluginCortex


def _make_plugin_dir(content: dict[str, str]) -> str:
    """Create a temp directory with the given filename→source mapping."""
    tmpdir = tempfile.mkdtemp()
    for filename, src in content.items():
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
    return tmpdir


VALID_PLUGIN = """\
class Plugin:
    def run(self, *args, **kwargs):
        return "hello from valid_plugin"
"""

PLUGIN_WITH_ARGS = """\
class Plugin:
    def run(self, x, y):
        return x + y
"""

NO_PLUGIN_CLASS = """\
def helper():
    return 42
"""

CRASHING_PLUGIN = """\
class Plugin:
    def run(self, *args, **kwargs):
        raise ValueError("intentional crash")
"""

NO_RUN_METHOD = """\
class Plugin:
    pass
"""


# ------------------------------------------------------------------ loading

class TestPluginLoading:
    def test_loads_valid_plugin(self):
        d = _make_plugin_dir({"my_plugin.py": VALID_PLUGIN})
        mpc = ModularPluginCortex(plugin_dir=d)
        assert "my_plugin" in mpc.list_plugins()

    def test_skips_files_without_plugin_class(self):
        d = _make_plugin_dir({"no_class.py": NO_PLUGIN_CLASS})
        mpc = ModularPluginCortex(plugin_dir=d)
        assert "no_class" not in mpc.list_plugins()

    def test_skips_dunder_files(self):
        d = _make_plugin_dir({"__init__.py": VALID_PLUGIN})
        mpc = ModularPluginCortex(plugin_dir=d)
        assert "__init__" not in mpc.list_plugins()

    def test_loads_multiple_plugins(self):
        d = _make_plugin_dir({
            "plugin_a.py": VALID_PLUGIN,
            "plugin_b.py": PLUGIN_WITH_ARGS,
        })
        mpc = ModularPluginCortex(plugin_dir=d)
        assert "plugin_a" in mpc.list_plugins()
        assert "plugin_b" in mpc.list_plugins()

    def test_nonexistent_dir_creates_it(self):
        tmpdir = tempfile.mkdtemp()
        new_dir = os.path.join(tmpdir, "brand_new_plugins")
        mpc = ModularPluginCortex(plugin_dir=new_dir)
        assert os.path.exists(new_dir)

    def test_nonexistent_dir_creates_dummy_plugin(self):
        tmpdir = tempfile.mkdtemp()
        new_dir = os.path.join(tmpdir, "fresh_plugins")
        mpc = ModularPluginCortex(plugin_dir=new_dir)
        assert "dummy_plugin" in mpc.list_plugins()


# ------------------------------------------------------------------ run_plugin

class TestRunPlugin:
    def test_run_valid_plugin(self):
        d = _make_plugin_dir({"greet.py": VALID_PLUGIN})
        mpc = ModularPluginCortex(plugin_dir=d)
        result = mpc.run_plugin("greet")
        assert result == "hello from valid_plugin"

    def test_run_with_positional_args(self):
        d = _make_plugin_dir({"adder.py": PLUGIN_WITH_ARGS})
        mpc = ModularPluginCortex(plugin_dir=d)
        result = mpc.run_plugin("adder", 3, 4)
        assert result == 7

    def test_run_missing_plugin_returns_error_string(self):
        d = _make_plugin_dir({})
        mpc = ModularPluginCortex(plugin_dir=d)
        result = mpc.run_plugin("nonexistent")
        assert "not found" in result

    def test_run_crashing_plugin_returns_error_string(self):
        d = _make_plugin_dir({"boom.py": CRASHING_PLUGIN})
        mpc = ModularPluginCortex(plugin_dir=d)
        result = mpc.run_plugin("boom")
        assert "crashed" in result

    def test_run_plugin_without_run_method_returns_error_string(self):
        d = _make_plugin_dir({"no_run.py": NO_RUN_METHOD})
        mpc = ModularPluginCortex(plugin_dir=d)
        result = mpc.run_plugin("no_run")
        assert "callable" in result


# ------------------------------------------------------------------ list_plugins

class TestListPlugins:
    def test_list_returns_list(self):
        d = _make_plugin_dir({})
        mpc = ModularPluginCortex(plugin_dir=d)
        assert isinstance(mpc.list_plugins(), list)

    def test_list_empty_when_no_valid_plugins(self):
        d = _make_plugin_dir({"no_class.py": NO_PLUGIN_CLASS})
        mpc = ModularPluginCortex(plugin_dir=d)
        assert mpc.list_plugins() == []
