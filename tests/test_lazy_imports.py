"""masknmf's top-level names load on first use and stay the objects the eager star imports gave."""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import masknmf

# the subpackages masknmf/__init__.py star-imported before its names became lazy
STAR_IMPORTED = [
    "masknmf.arrays",
    "masknmf.compression",
    "masknmf.motion_correction",
    "masknmf.demixing",
    "masknmf.visualization",
    "masknmf.diagnostics",
    "masknmf.pipelines",
    "masknmf.pipelines.configs",
]
SUBMODULES = [
    "arrays",
    "compression",
    "demixing",
    "diagnostics",
    "io",
    "motion_correction",
    "multisession",
    "pipelines",
    "utils",
    "visualization",
]


def run_python(code: str) -> str:
    """Run ``code`` in a fresh interpreter from the repo root, so it imports this checkout, and return its stdout."""
    dir_repo = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, "-c", code], cwd=dir_repo, capture_output=True, text=True, check=True).stdout


def test_names_are_the_star_exports_of_each_subpackage():
    expected = {"__version__", "version_info", "display", *SUBMODULES}
    for name_module in STAR_IMPORTED:
        expected |= set(importlib.import_module(name_module).__all__)
    assert set(masknmf.__all__) == expected
    assert set(dir(masknmf)) >= expected


@pytest.mark.parametrize("name_module", STAR_IMPORTED)
def test_each_name_is_the_object_its_subpackage_holds(name_module):
    module = importlib.import_module(name_module)
    for name in module.__all__:
        assert getattr(masknmf, name) is getattr(module, name), name


def test_submodules_and_display_resolve():
    for name in SUBMODULES:
        assert getattr(masknmf, name) is importlib.import_module(f"masknmf.{name}")
    assert masknmf.display is importlib.import_module("masknmf.utils.display").display


def test_no_lazy_name_is_also_a_submodule():
    # NOTE: importing masknmf.<name> sets that attribute to the module, which would hide a lazy name of the same spelling
    attributes = set(masknmf.__all__) - set(SUBMODULES)
    assert [name for name in attributes if importlib.util.find_spec(f"masknmf.{name}") is not None] == []


def test_a_name_reached_after_its_submodule_is_imported_directly():
    out = run_python(
        "from masknmf.demixing.demixing_results import DemixingResults\n"
        "import masknmf\n"
        "print(masknmf.DemixingResults is DemixingResults)"
    )
    assert out.strip() == "True"


def test_unknown_names_raise_attribute_error():
    with pytest.raises(AttributeError):
        masknmf.NotAMasknmfName


def test_importing_masknmf_loads_no_heavy_dependency():
    out = run_python(
        "import sys\n"
        "import masknmf\n"
        "heavy = ['numpy', 'torch', 'h5py', 'fastplotlib', 'pytorch_lightning', 'roicat', 'cvxpy']\n"
        "print([name for name in heavy if name in sys.modules])"
    )
    assert out.strip() == "[]"
