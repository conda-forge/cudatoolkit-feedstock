# Originally forked from https://github.com/AnacondaRecipes/cudatoolkit-feedstock
# Distributed under the BSD-2-Clause license
# Copyright (c) 2017, Continuum Analytics, Inc. All rights reserved.

# To accommodate nvtoolsext not being present as a DLL in the installer PE32s on windows,
# the windows variant of this script supports assembly directly from a pre-installed
# CUDA toolkit. The environment variable "NVTOOLSEXT_INSTALL_PATH" can be set to the
# installation path of the CUDA toolkit's NvToolsExt location (this is not the user
# defined install directory) and the DLL will be taken from that location.
import os
import sys
import shutil
import tarfile
import fnmatch
import platform
import itertools
from pathlib import Path
from subprocess import call, check_call, CalledProcessError
from argparse import ArgumentParser
from tempfile import TemporaryDirectory as tempdir


class Extractor(object):
    """Extractor base class, platform specific extractors should inherit
    from this class.
    """

    def __init__(self, plat, version, version_patch, runfile):
        """Base class for extracting cudatoolkit

        Parameters
        ----------
        plat : str
            Normalized platform name of system arch, e.g. "linux" or "windows"
        version : str
            Full version sting for cudatoolkit in X.Y.Z form, i.e. 11.0.3
        version_patch : str
            Extra version patch number, e.g. 450.51.06
        runfile : str
            Downloaded local runfile (blob) filename. This is just the basename,
            not the full path.

        Attributes
        ----------
        cuda_libraries : list of str
            The shared libraries to copy in.
        cuda_static_libraries : list of str
            The static libraries to copy in.
        libdevice_versions : list of str
            The library device versions supported (.bc files)
        runfile : str
            The name of the downloaded file to extract, for linux this is the .run file
        embedded_blob : str or None
            CUDA 11 installer has channed, there are no embedded blobs
        patches : list of str
            A list of the patch files for the blob, they are applied in order
        cuda_lib_fmt : str
            String format for the cuda libraries
        nvvm_lib_fmt : str
            String format for the nvvm libraries
        libdevice_lib_fmt : str
            string format for the libdevice.compute bitcode file
        version_patch_underscore : str
            An "_" stting if version_patch is non-empty. An empty string otherwise.
        """
        self.version = version
        version_parts = version.split(".")
        if len(version_parts) == 2:
            self.major, self.minor = version_parts
        elif len(version_parts) > 2:
            self.major, self.minor, self.micro = version_parts[:3]
        else:
            raise ValueError(f"{version!r} not a valid version string")
        self.major_minor = (int(self.major), int(self.minor))
        self.runfile = runfile

        # set attrs
        self.cuda_libraries = [
            "cublas",
            "cudart",
            "cufft",
            "cufftw",
            "cupti",
            "curand",
            "cusolver",
            "cusparse",
            "nppc",
            "nppial",
            "nppicc",
            "nppidei",
            "nppif",
            "nppig",
            "nppim",
            "nppist",
            "nppisu",
            "nppitc",
            "npps",
            "nvToolsExt",
            "nvblas",
            "nvrtc",
            "nvrtc-builtins",
        ]
        if (getplatform() == "linux" and self.major_minor == (10, 0)) or (self.major_minor >= (10, 1)):
            self.cuda_libraries.append("nvjpeg")
        if self.major_minor >= (10, 1):
            self.cuda_libraries.append("cublasLt")
        if self.major_minor < (11, 0):
            self.cuda_libraries.append("nppicom")
        if self.major_minor >= (11, 0):
            self.cuda_libraries.append("cusolverMg")
        self.cuda_static_libraries = ["cudadevrt"]
        self.libdevice_versions = [self.major]
        self.libdevice_lib_fmt = "libdevice.10.bc"
        self.patches = []
        self.prefix = os.environ["PREFIX"]
        self.src_dir = os.environ["SRC_DIR"]
        self.debug_install_path = os.environ.get("DEBUG_INSTALLER_PATH")

    def post_init(self):
        """Additional prep, after init."""
        self.output_dir = os.path.join(self.prefix, self.libdir)
        os.makedirs(self.output_dir, exist_ok=True)

    def copy(self, *args):
        """The method to copy extracted files into the conda package platform
        specific directory. Platform specific extractors must implement.
        """
        raise RuntimeError("Must implement")

    def extract(self, *args):
        """The method to extract files from the cuda binary blobs.
        Platform specific extractors must implement.
        """
        raise RuntimeError("Must implement")

    def get_paths(self, libraries, dirpath, template):
        """Gets the paths to the various cuda libraries and bc files"""
        pathlist = []
        for libname in libraries:
            filename = template.format(libname)
            paths = fnmatch.filter(os.listdir(dirpath), filename)
            if not paths:
                msg = "Cannot find item: %s, looked for %s" % (libname, filename)
                raise RuntimeError(msg)
            if (not self.symlinks) and (len(paths) != 1):
                msg = "Aliasing present for item: %s, looked for %s" % (
                    libname,
                    filename,
                )
                msg += ". Found: \n"
                msg += ", \n".join([str(x) for x in paths])
                raise RuntimeError(msg)
            pathsforlib = []
            for path in paths:
                tmppath = os.path.join(dirpath, path)
                assert os.path.isfile(tmppath), "missing {0}".format(tmppath)
                pathsforlib.append(tmppath)
            if self.symlinks:  # deal with symlinked items
                # get all DSOs
                concrete_dsos = [x for x in pathsforlib if not os.path.islink(x)]
                # find the most recent library version by name
                target_library = max(concrete_dsos)
                # remove this from the list of concrete_dsos
                # all that remains are DSOs that are not wanted
                concrete_dsos.remove(target_library)
                # drop the unwanted DSOs from the paths
                [pathsforlib.remove(x) for x in concrete_dsos]
            pathlist.extend(pathsforlib)
        return pathlist

    def copy_files(self, cuda_lib_dir, nvvm_lib_dir, libdevice_lib_dir, cupti_dir):
        """Copies the various cuda libraries and bc files to the output_dir"""
        filepaths = []
        # nvToolsExt and cupti are different to the rest of the cuda libraries,
        # they follow different naming conventions or locations, this accommodates...
        cudalibs = [x for x in self.cuda_libraries if x not in ("nvToolsExt", "cupti")]
        filepaths += self.get_paths(cudalibs, cuda_lib_dir, self.cuda_lib_fmt)
        if "nvToolsExt" in self.cuda_libraries:
            filepaths += self.get_paths(
                ["nvToolsExt"], cuda_lib_dir, self.nvtoolsext_fmt
            )
        if "cupti" in self.cuda_libraries:
            filepaths += self.get_paths(
                ["cupti"], cupti_dir, self.cupti_fmt
            )
        filepaths += self.get_paths(
            self.cuda_static_libraries, cuda_lib_dir, self.cuda_static_lib_fmt
        )
        filepaths += self.get_paths(["nvvm"], nvvm_lib_dir, self.nvvm_lib_fmt)
        filepaths += self.get_paths(
            self.libdevice_versions, libdevice_lib_dir, self.libdevice_lib_fmt
        )

        for fn in filepaths:
            if os.path.islink(fn):
                # replicate symlinks
                symlinktarget = os.readlink(fn)
                targetname = os.path.basename(fn)
                symlink = os.path.join(self.output_dir, targetname)
                print("linking %s to %s" % (symlinktarget, symlink))
                os.symlink(symlinktarget, symlink)
            else:
                print("copying %s to %s" % (fn, self.output_dir))
                shutil.copy(fn, self.output_dir)


class WindowsExtractor(Extractor):
    """The windows extractor"""

    def __init__(self, plat, version, version_patch, runfile):
        super().__init__(plat, version, version_patch, runfile)
        self.embedded_blob = None
        self.symlinks = False
        if self.major_minor > (11, 1):
            self.cuda_lib_fmt = "{0}64_1*.dll"
            self.nvvm_lib_fmt = "{0}64_40_0.dll"
        elif self.major_minor > (9, 2):
            self.cuda_lib_fmt = "{0}64_1*.dll"
            self.nvvm_lib_fmt = "{0}64_33_0.dll"
        else:
            self.cuda_lib_fmt = "{0}64_92.dll"
            self.nvvm_lib_fmt = "{0}64_32_0.dll"
        self.libdevice_lib_fmt = "libdevice.10.bc"
        self.cuda_static_lib_fmt = "{0}.lib"
        self.nvtoolsext_fmt = "{0}64_1.dll"
        self.cupti_fmt = "{0}64_*.dll"

        pfs = ["Program Files", "Program Files (x86)"]
        nvidias = ["NVIDIA Corporation", "NVIDIA GPU Computing Toolkit"]
        nvtxs = ["NVToolsExt", "NvToolsExt", "nvToolsExt"]
        for pf, nvidia, nvtx in itertools.product(pfs, nvidias, nvtxs):
            nvt_path = os.path.join("C:" + os.sep, pf, nvidia, nvtx, "bin")
            print(f"Looking for {nvt_path}...", end="")
            if os.path.exists(nvt_path):
                self.nvtoolsextpath = nvt_path
                print("found!")
                break
            else:
                print("absent.")
        else:
            self.nvtoolsextpath = None
        self.libdir = "Library/bin"
        self.post_init()

    def copy(self, *args):
        (store,) = args
        self.copy_files(cuda_lib_dir=store, nvvm_lib_dir=store, libdevice_lib_dir=store, cupti_dir=store)

    def extract(self):
        try:
            with tempdir() as tmpd:
                extract_name = "__extracted"
                extractdir = os.path.join(tmpd, extract_name)
                os.mkdir(extract_name)

                check_call(
                    [
                        "7za",
                        "x",
                        "-o%s" % extractdir,
                        os.path.join(self.src_dir, self.runfile),
                    ]
                )
                for p in self.patches:
                    check_call(
                        [
                            "7za",
                            "x",
                            "-aoa",
                            "-o%s" % extractdir,
                            os.path.join(self.src_dir, p),
                        ]
                    )

                nvt_path = os.environ.get(
                    "NVTOOLSEXT_INSTALL_PATH", self.nvtoolsextpath
                )
                print("NvToolsExt path: %s" % nvt_path)
                if nvt_path is not None:
                    if not Path(nvt_path).is_dir():
                        msg = "NVTOOLSEXT_INSTALL_PATH is invalid " "or inaccessible."
                        raise ValueError(msg)

                # fetch all the dlls into DLLs
                store_name = "DLLs"
                store = os.path.join(tmpd, store_name)
                os.mkdir(store)
                for path, dirs, files in os.walk(extractdir):
                    if (
                        "jre" not in path and "GFExperience" not in path
                    ):  # don't get jre or GFExperience dlls
                        for filename in fnmatch.filter(files, "*.dll"):
                            if not Path(os.path.join(store, filename)).is_file():
                                shutil.copy(os.path.join(path, filename), store)
                        for filename in fnmatch.filter(files, "*.lib"):
                            if (
                                path.endswith("x64")
                                and not Path(os.path.join(store, filename)).is_file()
                            ):
                                shutil.copy(os.path.join(path, filename), store)
                        for filename in fnmatch.filter(files, "*.bc"):
                            if not Path(os.path.join(store, filename)).is_file():
                                shutil.copy(os.path.join(path, filename), store)
                if nvt_path is not None:
                    for path, dirs, files in os.walk(nvt_path):
                        for filename in fnmatch.filter(files, "*.dll"):
                            if not Path(os.path.join(store, filename)).is_file():
                                shutil.copy(os.path.join(path, filename), store)
                self.copy(store)
        except PermissionError:
            # TODO: fix this
            # cuda 8 has files that refuse to delete, figure out perm changes
            # needed and apply them above, tempdir context exit fails to rmtree
            pass


class LinuxExtractor(Extractor):
    """The linux extractor"""

    def __init__(self, plat, version, version_patch, runfile):
        super().__init__(plat, version, version_patch, runfile)
        self.embedded_blob = None
        self.symlinks = True
        # need globs to handle symlinks
        self.cuda_lib_fmt = "lib{0}.so*"
        self.cuda_static_lib_fmt = "lib{0}.a"
        self.nvvm_lib_fmt = "lib{0}.so*"
        self.nvtoolsext_fmt = "lib{0}.so*"
        self.nvtoolsextpath = None
        self.cupti_fmt = "lib{0}.so*"
        self.libdir = "lib"
        self.cuda_lib_reldir = "lib64"
        self.machine = platform.machine()
        self.post_init()

    def copy(self, basepath):
        self.copy_files(
            cuda_lib_dir=os.path.join(basepath, self.cuda_lib_reldir),
            nvvm_lib_dir=os.path.join(basepath, "nvvm", "lib64"),
            libdevice_lib_dir=os.path.join(basepath, "nvvm", "libdevice"),
            cupti_dir=os.path.join(basepath, "extras", "CUPTI", "lib64")
        )

    @staticmethod
    def run_extract(cmd, check=True):
        """Run the extract command"""
        print(f"Extract command: {' '.join(cmd)}")
        caller = check_call if check else call
        try:
            caller(cmd)
        except CalledProcessError as e:
            logfile = "/tmp/cuda-installer.log"
            if os.path.isfile(logfile):
                with open(logfile) as f:
                    log = f.read()
                print(f"CUDA-INSTALLER LOG (/tmp/cuda-installer.log):\n\n{log}")
            else:
                print("No log file found")
            raise

    def extract(self):
        os.chmod(self.runfile, 0o777)
        with tempdir() as tmpd:
            if self.embedded_blob is not None:
                with tempdir() as tmpd2:
                    cmd = [
                        os.path.join(self.src_dir, self.runfile),
                        "--extract=%s" % (tmpd2,),
                        "--nox11",
                        "--silent",
                    ]
                    check_call(cmd)
                    # extract the embedded blob
                    cmd = [
                        os.path.join(tmpd2, self.embedded_blob),
                        "-prefix",
                        tmpd,
                        "-noprompt",
                        "--nox11",
                    ]
                    check_call(cmd)
            else:
                # Current Nvidia's Linux based runfiles don't use embedded runfiles
                #
                # "--installpath" runfile command is used to install the toolkit to a specified
                #     directory with the contents and layout similar to an install to
                #     '/usr/local/cuda`
                # "--override" runfile command to disable the compiler check since we are not
                #     installing the driver here
                # "--nox11" runfile command prevents desktop GUI on local install
                cmd = [
                    os.path.join(self.src_dir, self.runfile),
                    "--silent",
                    "--override",
                    "--nox11",
                    "--toolkit",
                ]
                check = True
                # add toolkit install args
                if self.major_minor >= (10, 2):
                    cmd.append(f"--installpath={tmpd}")
                elif self.major_minor == (10, 1):
                    # v10.1
                    cmd.extend([
                        f"--toolkitpath={tmpd}",
                        f"--librarypath={tmpd}",
                    ])
                    if self.machine == "ppc64le":
                        # cublas headers are not available, though the runfile
                        # thinks that they are.
                        check = False
                else:
                    # <=10.0
                    cmd.append(f"--toolkitpath={tmpd}")
                self.run_extract(cmd, check=check)
            for p in self.patches:
                os.chmod(p, 0o777)
                cmd = [
                    os.path.join(self.src_dir, p),
                    "--installdir",
                    tmpd,
                    "--accept-eula",
                    "--silent",
                ]
                check_call(cmd)
            self.copy(tmpd)


def getplatform():
    plt = sys.platform
    if plt.startswith("linux"):
        return "linux"
    elif plt.startswith("win"):
        return "windows"
    else:
        raise RuntimeError("Unknown platform")


DISPATCHER = {"linux": LinuxExtractor, "windows": WindowsExtractor}


def make_parser():
    p = ArgumentParser("build.py")
    p.add_argument("--version", dest="version")
    p.add_argument("--version-patch", dest="version_patch")
    p.add_argument("--runfile", dest="runfile")
    return p


def main():
    print("Running build")
    p = make_parser()
    ns = p.parse_args()

    # get an extractor & extract
    plat = getplatform()
    extractor_impl = DISPATCHER[plat]
    extractor = extractor_impl(plat, ns.version, ns.version_patch, ns.runfile)
    extractor.extract()


if __name__ == "__main__":
    main()
