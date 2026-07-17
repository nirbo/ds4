import os
import sys

from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    cmake_args = os.environ.get("CMAKE_ARGS", "")
    if "Python_EXECUTABLE=" not in cmake_args:
        cmake_args = f"{cmake_args} -DPython_EXECUTABLE={sys.executable}".strip()
        os.environ["CMAKE_ARGS"] = cmake_args
    setup(
        name="ornith35-mlx-kv-cache",
        version="0.0.0",
        description="Ornith-35 append-only MLX K/V cache primitive",
        ext_modules=[extension.CMakeExtension("ornith35_mlx_kv_cache._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["ornith35_mlx_kv_cache"],
        package_data={
            "ornith35_mlx_kv_cache": ["*.so", "*.dylib", "*.metallib"],
        },
        zip_safe=False,
        python_requires=">=3.10",
    )
