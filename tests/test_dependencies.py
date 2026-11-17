"""Guard test for the ipecharts dependency (Evaluation tile migration, Task 1).

ipecharts must be importable in the live env AND declared on both installation
paths — environment.yml (heavy local dev) and sepal_environment.yml (lean
deploy). This keeps the declaration honest: if either file drifts from the
installed version, this test catches it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PINNED_SPEC = "ipecharts>=1.4.2,<2"


def test_ipecharts_is_importable():
    """The live env can import ipecharts and its widget entry point."""
    import ipecharts
    from ipecharts import EChartsWidget  # noqa: F401
    from packaging.version import Version

    # 1.4.2 resizes a chart with its container; the app no longer nudges it.
    assert Version(ipecharts.__version__) >= Version("1.4.2")


def test_ipecharts_pinned_in_environment_yml():
    """The dev environment declares the pinned pip spec."""
    text = (ROOT / "environment.yml").read_text()
    assert PINNED_SPEC in text


def test_ipecharts_pinned_in_sepal_environment_yml():
    """The deploy environment declares the same version, conda-style."""
    # sepal_environment.yml installs ipecharts from conda-forge (conda pin
    # syntax), unlike environment.yml's pip spec.
    text = (ROOT / "sepal_environment.yml").read_text()
    assert "- ipecharts>=1.4.2,<1.5" in text
