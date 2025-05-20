"""Commands to run before build."""

import os
from pathlib import Path

import scipy_openblas64


def main() -> None:
    """Main."""
    path = Path(os.environ["PKG_CONFIG_PATH"]) / "scipy-openblas.pc"
    print(f"Writing file: {path!s}")
    with path.open("wt") as fobj:
        fobj.write(scipy_openblas64.get_pkg_config())


if __name__ == "__main__":
    main()
