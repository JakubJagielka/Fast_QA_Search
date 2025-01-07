from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "cython_files.Tokenizer",  # Module 2
        [r"cython_files\Tokenizer.pyx"],
        language="c++"
    ),
    Extension(
        "cython_files.Data_Struct",  # Module 1
        [r"cython_files\Data_Struct.pyx"],
        language="c++"
    ),

]

setup(
    ext_modules=cythonize(extensions),
)