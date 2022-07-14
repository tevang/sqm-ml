import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('dist_matrix')
    config.add_extension("_criterion",
                         sources=["_criterion.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    from Cython.Build import cythonize
    config.ext_modules[-1] = cythonize(config.ext_modules[-1])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
