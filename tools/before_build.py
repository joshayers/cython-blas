"""Commands to run before build."""

import os
import platform
from pathlib import Path

import scipy_openblas64

current_dir = Path(__file__).resolve().parent


def _blis_get_pkg_config() -> str:
    if platform.system() == "Windows":
        root_dir = str((current_dir / ".." / "lib" / "win" / "blis").resolve()).replace("\\", "/")
    else:
        raise ValueError
    return (
        f"""prefix={root_dir!s}\n"""
        f"""exec_prefix={root_dir!s}\n"""
        f"""libdir={root_dir!s}/lib\n"""
        f"""includedir={root_dir!s}/include\n"""
        """Name: BLIS\n"""
        """Description: BLAS-like Library Instantiation Software Framework\n"""
        """Version: 3.0-dev\n"""
        """Libs: -L${libdir} -lblis\n"""
        """Libs.private:    -fopenmp\n"""
        """Cflags: -I${includedir}/blis\n"""
    )


def main() -> None:
    """Main."""
    pkg_config_path = os.environ["PKG_CONFIG_PATH"].split(os.pathsep)[-1]
    # scipy-openblas
    path = Path(pkg_config_path) / "scipy-openblas.pc"
    print(f"Writing file: {path!s}")
    with path.open("wt") as fobj:
        fobj.write(scipy_openblas64.get_pkg_config())
    # # blis
    # path = Path(pkg_config_path) / "blis.pc"  # noqa: ERA001
    # print(f"Writing file: {path!s}") # noqa: ERA001
    # with path.open("wt") as fobj:
    #     fobj.write(_blis_get_pkg_config()) # noqa: ERA001


if __name__ == "__main__":
    main()
