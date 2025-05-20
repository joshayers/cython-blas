"""Commands to run before build."""

import os
from pathlib import Path

import scipy_openblas64


def main() -> None:
    """Main."""
    dir_path = os.environ["PKG_CONFIG_PATH"].split(os.pathsep)[-1]
    path = Path(dir_path) / "scipy-openblas.pc"
    print(f"Writing file: {path!s}")
    with path.open("wt") as fobj:
        fobj.write(scipy_openblas64.get_pkg_config())


if __name__ == "__main__":
    main()
