from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "SearchLand.cython_files.Tokenizer", 
        ["SearchLand/cython_files/Tokenizer.pyx"],
        language="c++"
    ),
    Extension(
        "SearchLand.cython_files.Data_Struct",
        ["SearchLand/cython_files/Data_Struct.pyx"],
        language="c++"
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)