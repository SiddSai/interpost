"""Phase 0: the package installs, imports, and exposes the planned layout."""

import importlib

SUBMODULES = [
    "interpost.signals",
    "interpost.activations",
    "interpost.activations.hooks",
    "interpost.activations.pooling",
    "interpost.interventions",
    "interpost.trainers",
    "interpost.eval",
    "interpost.config",
]


def test_version_is_a_nonempty_string():
    import interpost

    assert isinstance(interpost.__version__, str)
    assert interpost.__version__


def test_all_submodules_import():
    for name in SUBMODULES:
        importlib.import_module(name)


def test_hookmanager_and_pool_are_exposed():
    from interpost.activations import HookManager, pool

    assert callable(pool)
    assert isinstance(HookManager, type)
