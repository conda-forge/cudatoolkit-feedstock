# Originally forked from https://github.com/AnacondaRecipes/cudatoolkit-feedstock
# Distributed under the BSD-2-Clause license
# Copyright (c) 2017, Continuum Analytics, Inc. All rights reserved.

import os
import platform
import sys

from numba.cuda.cudadrv.libs import test, get_cudalib
from numba.cuda.cudadrv.nvvm import NVVM
from packaging import version


def run_test():
    nvvm = NVVM()
    print("NVVM version:", nvvm.get_version())
    print("Platform:", sys.platform)
    print("Machine:", platform.machine())

    # on windows only nvvm is available to numba
    if sys.platform.startswith("win"):
        return nvvm.get_version() is not None

    libc_ver = version.parse(platform.libc_ver()[1])
    print("GLIBC version:", libc_ver)
    # aarch64 requires GLIBC >= 2.27
    if platform.machine() == "aarch64" and libc_ver < version.parse("2.27"):
        print("WARNING: Skipping runtime tests on aarch64 as GLIBC version is lower than 2.27")
        return nvvm.get_version() is not None

    # Skip this test as it looks for the `libcuda.so` driver library,
    # which is not included in the Docker image used here.
    #if not test():
    #    return False

    extra_lib_tests = (
        "cublas",  # check pkg version matches lib pulled in
        "cufft",  # check cufft b/c cublas has an incorrect version in 10.1 update 1
        "cupti",  # check this is getting included
    )
    found_paths = [get_cudalib(lib) for lib in extra_lib_tests]
    print(*zip(extra_lib_tests, found_paths), sep="\n")

    return all(extra_lib_tests)


sys.exit(0 if run_test() else 1)
