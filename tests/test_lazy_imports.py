"""masknmf's lazy packages load each name on first use and stay the objects their eager imports gave."""

import ast
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


LAZY_SUBPACKAGES = ["masknmf.compression", "masknmf.pipelines", "masknmf.visualization"]


def read_stub(name_package: str) -> tuple[list[str], dict[str, str]]:
    """Read the submodules and the ``{name: submodule}`` attributes a lazy package's __init__.pyi lists."""
    filepath_stub = Path(importlib.import_module(name_package).__file__).with_suffix(".pyi")
    submodules, attributes = [], {}
    for node in ast.parse(filepath_stub.read_text()).body:
        if node.module is None:
            submodules += [alias.name for alias in node.names]
        else:
            attributes.update({alias.name: node.module for alias in node.names})
    return submodules, attributes


def public_submodules(name_package: str) -> list[str]:
    """List the modules and subpackages in a package's folder whose names do not start with an underscore."""
    dir_package = Path(importlib.import_module(name_package).__file__).parent
    return sorted(
        path.stem
        for path in dir_package.iterdir()
        if not path.name.startswith("_") and (path.suffix == ".py" or (path / "__init__.py").exists())
    )


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


@pytest.mark.parametrize("name_package", ["masknmf", *LAZY_SUBPACKAGES])
def test_no_lazy_name_is_also_a_submodule(name_package):
    # NOTE: importing <package>.<name> sets that attribute to the module, which would hide a lazy name of the same spelling
    _, attributes = read_stub(name_package=name_package)
    assert [n for n in attributes if importlib.util.find_spec(f"{name_package}.{n}") is not None] == []


@pytest.mark.parametrize("name_package", LAZY_SUBPACKAGES)
def test_a_lazy_subpackage_lists_every_submodule(name_package):
    # an eager __init__ left each submodule it imported as an attribute, and code reaches some only that way
    submodules, _ = read_stub(name_package=name_package)
    assert sorted(submodules) == public_submodules(name_package=name_package)


@pytest.mark.parametrize("name_package", LAZY_SUBPACKAGES)
def test_each_name_of_a_lazy_subpackage_resolves(name_package):
    package = importlib.import_module(name_package)
    submodules, attributes = read_stub(name_package=name_package)
    for name in submodules:
        assert getattr(package, name) is importlib.import_module(f"{name_package}.{name}")
    for name, name_submodule in attributes.items():
        assert getattr(package, name) is getattr(importlib.import_module(f"{name_package}.{name_submodule}"), name)
    assert set(package.__all__) <= set(attributes)


def test_a_submodule_resolves_without_being_imported_first():
    out = run_python("import masknmf.compression\nprint(masknmf.compression.denoising.__name__)")
    assert out.strip() == "masknmf.compression.denoising"


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
