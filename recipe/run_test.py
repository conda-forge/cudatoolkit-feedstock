# Originally forked from https://github.com/AnacondaRecipes/cudatoolkit-feedstock
# Distributed under the BSD-2-Clause license
# Copyright (c) 2017, Continuum Analytics, Inc. All rights reserved.

import sys
import os
from numba.cuda.cudadrv.libs import test, get_cudalib
from numba.cuda.cudadrv.nvvm import NVVM


def run_test():
    # on windows only nvvm is available to numba
    if sys.platform.startswith("win"):
        nvvm = NVVM()
        print("NVVM version", nvvm.get_version())
        return nvvm.get_version() is not None
    if not test():
        return False
    nvvm = NVVM()
    print("NVVM version", nvvm.get_version())

    extra_lib_tests = (
        "cublas",  # check pkg version matches lib pulled in
        "cufft",  # check cufft b/c cublas has an incorrect version in 10.1 update 1
        "cupti",  # check this is getting included
    )
    found_paths = [get_cudalib(lib) for lib in extra_lib_tests]
    print(*zip(extra_lib_tests, found_paths), sep="\n")

    return all(extra_lib_tests)


sys.exit(0 if run_test() else 1)
