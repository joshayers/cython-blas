"""Custom spin commands."""

import platform
import shutil
import subprocess
from pathlib import Path

import click

current_dir = Path(__file__).resolve().parent

root_dir = (current_dir / "..").resolve()
build_dir = root_dir / "build"
build_ninja_path = build_dir / "build.ninja"


def fix_paths(paths: list[str]) -> list[str]:
    """Remove the '$' after the drive letter on Windows paths."""
    if platform.system() != "Windows":
        return [Path(path) for path in paths]
    output = []
    for path in paths:
        if len(path) < 2 or path[1] != "$":
            output.append(Path(path))
            continue
        mod_path = f"{path[0]}{path[2:]}"
        if Path(mod_path).exists():
            output.append(Path(mod_path))
        else:
            output.append(Path(path))
    return output


def get_ninja_build_rules() -> list[tuple[Path, str, list[Path]]]:
    """Parse build.ninja to find all build rules."""
    rules = []
    with build_ninja_path.open("rt") as build_ninja:
        for line in build_ninja:
            line = line.strip()  # noqa: PLW2901
            if line.startswith("build "):
                line = line[len("build ") :]  # noqa: PLW2901
                target, rule = line.split(": ")
                if target == "PHONY":
                    continue
                compiler, *srcfiles = rule.split(" ")
                # target is a path relative to the build directory. We will
                # turn that into an absolute path so that all paths in target
                # and srcfiles are absolute.
                target = build_dir / target
                rule = (target, compiler, fix_paths(srcfiles))
                rules.append(rule)
    return rules


def get_cython_build_rules(ninja_build_rules: list[tuple[Path, str, list[Path]]]) -> list[tuple[Path, Path]]:
    """Parse build.ninja to find all Cython compiler rules."""
    rules = []
    for target, compiler, srcfiles in ninja_build_rules:
        if compiler == "cython_COMPILER":
            assert target.suffix in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")  # noqa: S101
            assert len(srcfiles) == 1  # noqa: S101
            assert srcfiles[0].suffix == ".pyx"  # noqa: S101
            (source_file,) = srcfiles
            rules.append((target, source_file))
    return rules


def get_cpp_build_rules(ninja_build_rules: list[tuple[Path, str, list[Path]]]) -> list[tuple[Path, Path]]:
    """Parse build.ninja to fina all all C and C++ compiler rules."""
    rules = []
    for target, compiler, srcfiles in ninja_build_rules:
        if compiler == "cpp_COMPILER":
            assert target.suffix in (".obj", ".o")  # noqa: S101
            assert len(srcfiles) == 1  # noqa: S101
            assert srcfiles[0].suffix in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")  # noqa: S101
            (source_file,) = srcfiles
            rules.append((target, source_file))
    return rules


def get_link_rules(ninja_build_rules: list[tuple[Path, str, list[Path]]]) -> list[tuple[Path, Path]]:
    """Parse build.ninja to find all linker rules."""
    rules = []
    for target, compiler, srcfiles in ninja_build_rules:
        if compiler == "cpp_LINKER":
            assert target.suffix in (".pyd", ".so")  # noqa: S101
            assert len(srcfiles) >= 1  # noqa: S101
            assert srcfiles[0].suffix in (".obj", ".o")  # noqa: S101
            (source_file, *_) = srcfiles
            rules.append((target, source_file))
    return rules


def copy_compiled_files() -> None:
    """Copy Cython-generated C++ files and compiled extension modules from build directory to the code tree."""
    ninja_build_rules = get_ninja_build_rules()
    cython_build_rules = get_cython_build_rules(ninja_build_rules)
    cpp_build_rules = get_cpp_build_rules(ninja_build_rules)
    link_rules = get_link_rules(ninja_build_rules)

    for cpp_src, pyx_src in cython_build_rules:
        dest_dir = pyx_src.parent
        # find matching C++ compiler rule
        for i, (_, cpp_src2) in enumerate(cpp_build_rules):
            if build_dir / cpp_src2 == cpp_src:
                obj_dest, _ = cpp_build_rules[i]
                break
        # find matching linker rule
        for j, (_, obj_src) in enumerate(link_rules):
            if build_dir / obj_src == obj_dest:
                pyd_src, _ = link_rules[j]
                break
        cpp_suffixes = cpp_src.suffixes
        assert len(cpp_suffixes) == 2  # noqa: S101
        cpp_basename = Path(cpp_src.stem).stem
        cpp_dest = dest_dir / (cpp_basename + cpp_src.suffix)
        shutil.copy(cpp_src, cpp_dest)
        pyd_dest = dest_dir / pyd_src.name
        shutil.copy(pyd_src, pyd_dest)


@click.command
def build() -> None:
    """Run 'python -m build .' to build a source distribution and (optionally) a wheel."""
    cmd = ["python", "-m", "build", "."]
    print(f"Running the following command: \n{' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=root_dir, shell=True)  # noqa: S602


@click.command
def docs() -> None:
    """Run 'python -m docs.build' to build the html documentation."""
    cmd = ["python", "-m", "docs.build"]
    print(f"Running the following command: \n{' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=root_dir, shell=True)  # noqa: S602


@click.command
@click.option("-c", "--coverage", is_flag=True, help="Enable line tracing to allow Cython coverage analysis.")
@click.option("--vecreport", is_flag=True, help="Enable the compiler's auto-vectorization report.")
@click.option("-w", "--warn", type=click.Choice(["0", "1", "2", "3", "4"]), default="2")
def setup_in_place(coverage: bool, vecreport: bool, warn: str) -> None:
    """Run 'meson setup --reconfigure' to reconfigure the build."""
    meson_path = (
        root_dir / ".venv" / "Scripts" / "meson"
        if platform.system() == "Windows"
        else root_dir / ".venv" / "bin" / "meson"
    )
    coverage_cmd = "-Dcoverage=true" if coverage else "-Dcoverage=false"
    vecreport_cmd = "-Dvectorization-report=true" if vecreport else "-Dvectorization-report=false"
    warnlevel = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "everything"}[warn]
    warnlevel_cmd = f"--warnlevel={warnlevel}"
    cmd = [
        str(meson_path),
        "setup",
        f"{build_dir!s}",
        "--buildtype",
        "release",
        "--reconfigure",
        warnlevel_cmd,
        vecreport_cmd,
        coverage_cmd,
    ]
    print(f"Running the following command:\n{' '.join(cmd)}\n")
    subprocess.run(  # noqa: S603
        cmd,
        check=True,
        cwd=root_dir,
    )


@click.command
def in_place() -> None:
    """Create an in-place install.

    This command runs `meson compile build/`.

    The resulting compiled files are then copied to the source directory.
    """
    meson_path = (
        root_dir / ".venv" / "Scripts" / "meson"
        if platform.system() == "Windows"
        else root_dir / ".venv" / "bin" / "meson"
    )
    # Run 'meson compile'
    cmd = [str(meson_path), "compile", f"-C{build_dir!s}"]
    print(f"Running the following command:\n{' '.join(cmd)}\n")
    subprocess.run(  # noqa: S603
        cmd,
        check=True,
        cwd=root_dir,
    )
    copy_compiled_files()


@click.command
def cython_lint() -> None:
    """Run the cython-lint command.

    This command checks all Cython files for linting errors. It does not automatically fix
    them.
    """
    cython_lint_path = (
        root_dir / ".venv" / "Scripts" / "cython-lint"
        if platform.system() == "Windows"
        else root_dir / ".venv" / "bin" / "cython-lint"
    )
    cmd = [str(cython_lint_path), "."]
    print(f"Running the following command:\n{' '.join(cmd)}\n")
    subprocess.run(  # noqa: S603
        cmd,
        check=False,
        cwd=root_dir,
    )


@click.command
def cython_stringfix() -> None:
    """Run the double-quite-cython-strings command.

    This command replaces all quotes with double quotes in Cython .pyx and .pxd files.
    """
    string_fix_path = (
        root_dir / ".venv" / "Scripts" / "double-quote-cython-strings"
        if platform.system() == "Windows"
        else root_dir / ".venv" / "bin" / "double-quote-cython-strings"
    )
    package_dir = root_dir / "rs_cla_model"
    pyx_files = [str(path.relative_to(root_dir)) for path in package_dir.rglob("*.pyx")]
    pxd_files = [str(path.relative_to(root_dir)) for path in package_dir.rglob("*.pxd")]
    cmd = [str(string_fix_path), *pyx_files, *pxd_files]
    print(f"Running the following command:\n{' '.join(cmd)}\n")
    subprocess.run(  # noqa: S603
        cmd,
        check=False,
        cwd=root_dir,
    )
