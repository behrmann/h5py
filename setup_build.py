
"""
    Implements a custom Distutils build_ext replacement, which handles the
    full extension module build process, from api_gen to C compilation and
    linking.
"""

try:
    from setuptools import Extension
except ImportError:
    from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import sys
import os
import os.path as op
from functools import reduce
import api_gen


def localpath(*args):
    return op.abspath(reduce(op.join, (op.dirname(__file__),)+args))


MODULES =  ['defs','_errors','_objects','_proxy', 'h5fd', 'h5z',
            'h5','h5i','h5r','utils',
            '_conv', 'h5t','h5s',
            'h5p',
            'h5d', 'h5a', 'h5f', 'h5g',
            'h5l', 'h5o',
            'h5ds', 'h5ac']


EXTRA_SRC = {'h5z': [ localpath("lzf/lzf_filter.c"),
              localpath("lzf/lzf/lzf_c.c"),
              localpath("lzf/lzf/lzf_d.c")]}


if sys.platform.startswith('win'):
    COMPILER_SETTINGS = {
        'libraries'     : ['h5py_hdf5', 'h5py_hdf5_hl'],
        'include_dirs'  : [localpath('lzf'), localpath('windows')],
        'library_dirs'  : [],
        'define_macros' : [('H5_USE_16_API', None), ('_HDF5USEDLL_', None)] }

else:
    COMPILER_SETTINGS = {
       'libraries'      : ['hdf5', 'hdf5_hl'],
       'include_dirs'   : [localpath('lzf')],
       'library_dirs'   : ['/opt/local/lib', '/usr/local/lib'],
       'define_macros'  : [('H5_USE_16_API', None)] }


class h5py_build_ext(build_ext):

    """
        Custom distutils command which encapsulates api_gen pre-building,
        Cython building, and C compilation.

        Also handles making the Extension modules, since we can't rely on
        NumPy being present in the main body of the setup script.
    """

    @staticmethod
    def _make_extensions(config):
        """ Produce a list of Extension instances which can be passed to
        cythonize().

        This is the point at which custom directories, MPI options, etc.
        enter the build process.
        """
        import numpy
        import pkgconfig

        settings = COMPILER_SETTINGS.copy()

        settings['include_dirs'] += [numpy.get_include()]

        if config.mpi:
            import mpi4py
            settings['include_dirs'] += [mpi4py.get_include()]

            # Since pkgconfig sometimes has unreliable information we brute
            # force finding mpi.h, which mpi4py solves by including it in its
            # package
            include_search = ['/include', '/usr/include',
                              '/usr/local/include', '/opt/local/include']
            mpi_include = set()

            for d in include_search:
                # Following of links needs to be true since Debian hides the openmpi
                # header files in weird places
                for dirpath, dirs, files in os.walk(d, followlinks=True):
                    if 'mpi.h' in files:
                        mpi_include.add(dirpath)
                        break

            if len(mpi_include) == 0:
                raise IOError('mpi.h not found, cannot build mpi-enabled h5py!')

            settings['include_dirs'].extend(mpi_include)

        # Ensure a custom location appears first, so we don't get a copy of
        # HDF5 from some default location in COMPILER_SETTINGS
        if config.hdf5_libdir is not None:
            settings['library_dirs'] += config.hdf5_libdir

        if config.hdf5_includedir is not None:
            settings['include_dirs'] += config.hdf5_includedir

        if config.hdf5_libname is not None:
            settings['libraries'] = config.hdf5_libname

        if config.hdf5_define_macros is not None:
            settings['define_macros'].extend(config.hdf5_define_macros)

        # TODO: should this only be done on UNIX?
        if os.name != 'nt':
            settings['runtime_library_dirs'] = settings['library_dirs']

        def make_extension(module):
            sources = [localpath('h5py', module+'.pyx')] + EXTRA_SRC.get(module, [])
            return Extension('h5py.'+module, sources, **settings)

        return [make_extension(m) for m in MODULES]


    def run(self):
        """ Distutils calls this method to run the command """

        from Cython.Build import cythonize

        # Provides all of our build options
        config = self.distribution.get_command_obj('configure')
        config.run()

        defs_file = localpath('h5py', 'defs.pyx')
        func_file = localpath('h5py', 'api_functions.txt')
        config_file = localpath('h5py', 'config.pxi')

        # Rebuild low-level defs if missing or stale
        if not op.isfile(defs_file) or os.stat(func_file).st_mtime > os.stat(defs_file).st_mtime:
            print("Executing api_gen rebuild of defs")
            api_gen.run()

        # Rewrite config.pxi file if needed
        if not op.isfile(config_file) or config.rebuild_required:
            with open(config_file, 'wb') as f:
                s = """\
# This file is automatically generated by the h5py setup script.  Don't modify.

DEF MPI = %(mpi)s
DEF HDF5_VERSION = %(version)s
"""
                s %= {'mpi': bool(config.mpi),
                      'version': tuple(int(x) for x in config.hdf5_version.split('.'))}
                s = s.encode('utf-8')
                f.write(s)

        # Run Cython
        self.extensions = cythonize(self._make_extensions(config),
                            force=config.rebuild_required or self.force)

        # Perform the build
        build_ext.run(self)

        # Mark the configuration as built
        config.reset_rebuild()
