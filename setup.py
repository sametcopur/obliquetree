from setuptools import setup
from Cython.Build import cythonize
import os
import numpy as np
from setuptools.extension import Extension
import sys

include_dirs_extra = []
library_dirs = []
extra_compile_args = []
extra_link_args = []

if sys.platform in ("win32", "linux", "darwin"):  # All platforms
    if sys.platform == "win32":
        extra_compile_args = [
            "/O2",  # Equivalent to -O3
            "/fp:fast",  # Fast floating point model
            "/Ot",  # Favor fast code
            "/Ox",  # Maximum optimization
            "/Oi",  # Enable intrinsic functions
            "/GT",  # Fiber-safe optimizations
            "/std:c++17",  # C++17 standard
            "/openmp",  # OpenMP support
        ]
        extra_link_args = ["/OPT:REF", "/OPT:ICF"]
    else:  # linux and darwin (macOS)
        extra_compile_args = [
            "-O3",  # Maximum optimization
            "-funroll-loops",  # Loop unrolling
            "-ftree-vectorize",  # Enable vectorization
            "-fstrict-aliasing",  # Enable strict aliasing
            "-fstack-protector-strong",  # Stack protection
            "-std=c++17",  # C++17 standard
        ]
        extra_link_args = []

        if sys.platform == "linux":
            extra_compile_args.append("-fopenmp")
            extra_link_args.append("-fopenmp")
        elif sys.platform == "darwin":
            libomp_prefix = os.environ.get("LIBOMP_PREFIX", "/opt/homebrew/opt/libomp")
            if os.path.isdir(libomp_prefix):
                include_dirs_extra = [os.path.join(libomp_prefix, "include")]
                library_dirs = [os.path.join(libomp_prefix, "lib")]
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


extensions = [
    Extension(
        "obliquetree.src.tree",
        ["obliquetree/src/tree.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.oblique",
        ["obliquetree/src/oblique.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.base",
        ["obliquetree/src/base.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.utils",
        ["obliquetree/src/utils.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.metric",
        ["obliquetree/src/metric.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "obliquetree.src.ccp",
        ["obliquetree/src/ccp.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
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
