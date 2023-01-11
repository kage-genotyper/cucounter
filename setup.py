import os
import sys
import re
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


CUDA_ERROR_CHECK = True
USE_COOPERATIVE_GROUPS = False
COOPERATIVE_GROUP_SIZE = 4
USE_MURMUR_HASH = True


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)",
                                         out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                      "-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        global CUDA_ERROR_CHECK 
        cmake_args += ["-DCUDA_ERROR_CHECK=" + str(CUDA_ERROR_CHECK)]
        global USE_COOPERATIVE_GROUPS 
        cmake_args += ["-DUSE_COOPERATIVE_GROUPS=" + str(USE_COOPERATIVE_GROUPS)]
        global COOPERATIVE_GROUP_SIZE
        if COOPERATIVE_GROUP_SIZE in [1, 2, 4, 8, 16, 32]:
            cmake_args += ["-DCOOPERATIVE_GROUP_SIZE=" + str(COOPERATIVE_GROUP_SIZE)]
        global USE_MURMUR_HASH
        cmake_args += ["-DUSE_MURMUR_HASH=" + str(USE_MURMUR_HASH)]

        env = os.environ.copy()
        env["CXXFLAGS"] = "{} -DVERSION_INFO=\\'{}\\'".format(
            env.get("CXXFLAGS", ""),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=self.build_temp)


setup(
    name="cucounter",
    version="1.0.0",
    author="Jørgen Henriksen",
    description="A hashtable based counter implemented in CUDA for high throughput.",
    long_description="",
    packages=["cucounter"],
    ext_modules=[CMakeExtension("cucounter_backend")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)

