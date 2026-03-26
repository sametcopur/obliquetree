from setuptools import setup
from Cython.Build import cythonize
import os
import numpy as np
from setuptools.extension import Extension
import sys
import shutil
import subprocess

include_dirs_extra = []
library_dirs = []
extra_compile_args = []
extra_link_args = []


def find_libomp_prefix():
    candidates = []

    env_prefix = os.environ.get("LIBOMP_PREFIX")
    if env_prefix:
        candidates.append(env_prefix)

    brew = shutil.which("brew")
    if brew is not None:
        try:
            brew_prefix = subprocess.check_output(
                [brew, "--prefix", "libomp"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if brew_prefix:
                candidates.append(brew_prefix)
        except (OSError, subprocess.CalledProcessError):
            pass

    candidates.extend(
        [
            "/opt/homebrew/opt/libomp",
            "/usr/local/opt/libomp",
        ]
    )

    seen = set()
    for prefix in candidates:
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        include_dir = os.path.join(prefix, "include")
        lib_dir = os.path.join(prefix, "lib")
        if os.path.isdir(include_dir) and os.path.isdir(lib_dir):
            return prefix, include_dir, lib_dir

    return None, None, None

if sys.platform in ("win32", "linux", "darwin"):  # All platforms
    if sys.platform == "win32":
        extra_compile_args = [
            "/O2",
            "/std:c++17",  # C++17 standard
            "/openmp",  # OpenMP support
        ]
        extra_link_args = ["/OPT:REF", "/OPT:ICF"]
    else:  # linux and darwin (macOS)
        extra_compile_args = [
            "-O3",  # Maximum optimization
            "-funroll-loops",  # Unroll small loops (matvec d=2/3)
            "-ftree-vectorize",  # Enable vectorization
            "-fstack-protector-strong",  # Stack protection
            "-std=c++17",  # C++17 standard
        ]
        extra_link_args = []

        if sys.platform == "linux":
            extra_compile_args.append("-fopenmp")
            extra_link_args.append("-fopenmp")
        elif sys.platform == "darwin":
            libomp_prefix, libomp_include, libomp_lib = find_libomp_prefix()
            if libomp_prefix is not None:
                include_dirs_extra = [libomp_include]
                library_dirs = [libomp_lib]
                extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
                extra_link_args.extend([
                    "-L" + library_dirs[0],
                    "-lomp",
                    "-Wl,-rpath," + library_dirs[0],
                ])
            else:
                raise RuntimeError(
                    "OpenMP (libomp) is required but was not found. "
                    "Install it via 'brew install libomp' or set the LIBOMP_PREFIX "
                    "environment variable to the libomp installation path."
                )

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
include_dirs = [np.get_include(), *include_dirs_extra]


ext_names = ["tree", "oblique", "base", "utils", "metric", "ccp"]
extensions = [
    Extension(
        f"obliquetree.src.{name}",
        [f"obliquetree/src/{name}.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    for name in ext_names
]

setup(
    name="obliquetree",
    packages=["obliquetree", "obliquetree.src"],  # Explicitly list packages
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "nonecheck": False,
            "overflowcheck": False,
        },
    ),
)
