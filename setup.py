from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "SearchEngine.cython_files.Tokenizer", 
        ["SearchEngine/cython_files/Tokenizer.pyx"],
        language="c++"
    ),
    Extension(
        "SearchEngine.cython_files.Data_Struct",
        ["SearchEngine/cython_files/Data_Struct.pyx"],
        language="c++"
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)