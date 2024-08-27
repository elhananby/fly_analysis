from setuptools import setup, find_packages

setup(
    name="fly_analysis",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "opencv-python",
        "pyarrow",
        "tqdm",
        "spatialmath-python",
        "pybind11",
    ],
    author="Elhanan Buchsbaum",
)
