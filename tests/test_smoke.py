import sys
import harness


def test_python_version() -> None:
    assert sys.version_info >= (3, 11), "This book assumes Python 3.11+"


def test_package_imports() -> None:
    assert harness is not None