
"""
    Implements a new custom Distutils command for handling library
    configuration.

    The "configure" command here doesn't directly affect things like
    config.pxi; rather, it exists to provide a set of attributes that are
    used by the build_ext replacement in setup_build.py.

    Options from the command line and environment variables are stored
    between invocations in a pickle file.  This allows configuring the library
    once and e.g. calling "build" and "test" without recompiling everything
    or explicitly providing the same options every time.

    This module also contains the auto-detection logic for figuring out
    the currently installed HDF5 version.
"""

from distutils.cmd import Command
import os
import os.path as op
import sys
import pickle
import re
import platform

try:
    import pkgconfig
    NOPKGCONFIG = False
except ImportError:
    NOPKGCONFIG = True


def loadpickle():
    """ Load settings dict from the pickle file """
    try:
        with open('h5config.pkl','rb') as f:
            cfg = pickle.load(f)
        if not isinstance(cfg, dict): raise TypeError
    except Exception:
        return {}
    return cfg


def savepickle(dct):
    """ Save settings dict to the pickle file """
    with open('h5config.pkl','wb') as f:
        pickle.dump(dct, f, protocol=0)


def validate_version(s):
    """ Ensure that s contains an X.Y.Z format version string, or ValueError.
    """
    try:
        tpl = tuple(int(x) for x in s.split('.'))
        if len(tpl) != 3: raise ValueError
    except Exception:
        raise ValueError("HDF5 version string must be in X.Y.Z format")


def hdf5_names_to_try(whichmpi=None):
    # choose mpich if it is explicitly requested, choose openmpi else for mpi
    if whichmpi == 'mpich':
        return ['hdf5-mpich', 'hdf5']
    elif whichmpi:
        return ['hdf5-openmpi','hdf5']
    else:
        return ['hdf5-serial', 'hdf5']


def autodetect_libdirs(hdf5_dir=None, hdf5_libdir=None, mpi=None, fallback=False):
    """
    Detect the lib directories of the wanted hdf5 library.

    Intended for Unix-ish platforms (Linux, OS X, BSD).
    Does not support Windows.

    hdf5_dir   : optional HDF5 install directory to look in (containing "lib")
    hdf5_libdir: optional directory where to look for libhdf5
    mpi        : optional switch whether to look for parallel library version
    """
    # given parameters get precedence
    if hdf5_libdir is not None:
        libdirs = {hdf5_libdir}
    elif hdf5_dir is not None:
        libdirs = {op.join(hdf5_dir, 'lib')}
    else:
        if fallback:
            try:
                # first try to get information the canonical way
                import subprocess

                # get path of ldconfig, e.g.
                # ldconfig: /sbin/ldconfig /sbin/ldconfig.real /usr/share/man/man8/ldconfig.8.gz
                ldconfigpath = subprocess.check_output(['whereis', 'ldconfig']).split()[1]

                if sys.platform.startswith('linux'):
                    listswitch = '-p'
                elif (sys.platform.startswith('freebsd') or
                      sys.platform.startswith('darwin')):
                    listswitch = '-r'

                ldconfig_out = subprocess.check_output([ldconfigpath, listswitch])
                if sys.version_info.major == 3:
                    ldconfig_out = ldconfig_out.decode('utf-8')
                libdirs = set(op.dirname(line) for line in ldconfig_out.split()
                              if 'libhdf5' in line and op.dirname(line) != '')

            except OSError:
                # try pkgconfig as the last fall back, since its information
                # can be inaccurate, e.g. it does not find mpi-enabled libhdf5
                # on debian systems
                pass
        else:
            for package in hdf5_names_to_try(mpi):
                if pkgconfig.exists(package):
                    libdirs = pkgconfig.parse(package)['library_dirs']
                    break
            else:
                libdirs = {'/usr/local/lib', '/opt/local/lib'}

    # resolve symlinks so that later we can look for include directories in the
    # right place.
    # This is important on Fedora Linux where libs are found in /lib64 which
    # links to /usr/lib64, but which has only /usr/include but no /include
    libdirs = {op.realpath(path) for path in libdirs}

    return libdirs


def autodetect_libname(hdf5_libname=None, mpi=None, fallback=False):
    """
    Get namelibrary file of hdf5 library.

    Intended for Unix-ish platforms (Linux, OS X, BSD).
    Does not support Windows.

    hdf5_dir: optional HDF5 library name
    mpi     : optional switch whether to look for parallel library version
    """

    # getting the name of the librayr

    def fallback_libname(mpi):
        libname = hdf5_names_to_try(mpi)[0].replace('-','_')
        if not sys.platform.startswith('win'):
            return libname
        else:
            return 'h5py_' + libname

    if hdf5_libname is None:
        if fallback:
            hdf5_libname = fallback_libname(mpi)
        else:
            for package in hdf5_names_to_try(mpi):
                if pkgconfig.exists(package):
                    hdf5_libnames = pkgconfig.parse(package)['libraries']
                    # filter out the correct library name, relevant for mpi cases
                    for name in hdf5_libnames:
                        if 'hdf5' in name:
                            hdf5_libname = name
                    break
            else:
                hdf5_libname = fallback_libname(mpi)

    # which extension should it have?

    if sys.platform.startswith('darwin'):
        libextension = r'.dylib'
    elif sys.platform.startswith('win'):
        libextension = r'.dll'
    else:
        libextension = r'.so'

    # output

    libname = [hdf5_libname, hdf5_libname + '_hl']
    libnameregexp = re.compile(r'lib' + hdf5_libname + libextension)

    return libname, libnameregexp


def autodetect_includedirs(hdf5_dir=None, hdf5_includedir=None,
                           libdirs=None, mpi=None, fallback=False):
    """
    Detect the include directories of the wanted hdf5 library.

    Intended for Unix-ish platforms (Linux, OS X, BSD).
    Does not support Windows. Raises an eror if mpi header files cannot be found,
    but an mpi-enabled h5py is supposded to be built

    hdf5_dir       : optional HDF5 install directory to look in (containing "include")
    hdf5_includedir: optional directory where to look for libhdf5
    libdirs        : optional directories list from autodetect_libdirs
    mpi            : optional switch whether to look for parallel library version
    """
    fallback_include = {'/usr/local/include', '/opt/local/include'}

    def lib_to_include(*libs):
        try:
            if sys.version_info >= (3,3,):
                multiarch = sys.implementation._multiarch
            else:
                multiarch = sys._multiarch
        except AttributeError:
            multiarch = ''
        finally:
            # this regex matches directory names of the form
            # /lib
            # /lib64                    (e.g. Fedora)
            # /usr/lib/x86_64-linux-gnu (e.g. Debian)
            # and similar to catch all the places libraries could be and get
            # appropriate possible include paths
            regexp = re.compile(r'((lib(\b|\d{2}))|(lib/' +
                                multiarch +
                                r'))/?$')
            base = (re.sub(regexp, r'include', d) for d in libs)
            includes = {d for d in base if op.isdir(d)}
        return includes

    def fallback_to_pkgconfig(fallback_include, mpi):
        for package in hdf5_names_to_try(mpi):
            if pkgconfig.exists(package):
                pkgc_inc = pkgconfig.parse(package)['include_dirs']
                includes = pkgc_inc if len(pkgc_inc) > 0 else fallback_include
                break
        else:
            includes = fallback_include


        return includes

    # getting the hdf5 includedir

    if hdf5_includedir is not None:
        includedirs = {hdf5_includedir}
    elif hdf5_dir is not None:
        includedirs = {op.join(hdf5_dir, 'include')}
    elif fallback and libdirs is not None:
        include_search = lib_to_include(*libdirs).union(fallback_include)

        possible_includedirs = set()
        for d in include_search:
            for dirpath, dirs, files in os.walk(d):
                if 'hdf5.h' and 'hdf5_hl.h' in files:
                    possible_includedirs.add(dirpath)

        if len(possible_includedirs) > 1:
            if mpi:
                includedirs = {dir for dir in possible_includedirs if
                               'mpi' in dir}
            else:
                includedirs = {dir for dir in possible_includedirs if
                               not 'mpi' in dir}
        elif len(possible_includedirs) == 1:
            includedirs = possible_includedirs
        else:
            includedirs = fallback_include
    else:
        includedirs = fallback_to_pkgconfig(fallback_include, mpi)

    # getting the mpi4py incluedir

    if mpi:
        import mpi4py
        includedirs.add(mpi4py.get_include())

    # getting the mpi headers
    # we will assume that we only need to search the mpi headers if we're on
    # fallback, because it will be pkg-configs job to give us the right headers
    # in the normal case

    if mpi and fallback:
        include_search = ['/include', '/usr/include',
                          '/usr/local/include', '/opt/local/include']
        mpi_include = set()

        for d in include_search:
            # Following of links needs to be true since Debian hides the openmpi
            # header files in weird places
            for dirpath, dirs, files in os.walk(d, followlinks=True):
                if 'mpi.h' in files:
                    mpi_include.add(dirpath)

        if mpi == 'openmpi' or 'mpich':
            mpi_include = {path for path in mpi_include if mpi in path}

        if len(mpi_include) == 0:
            raise IOError('mpi.h not found, cannot build mpi-enabled h5py!')
        else:
            includedirs.union(mpi_include)

    # getting the numpy headers while we're at it.

    import numpy
    includedirs.add(numpy.get_include())

    includedirs = {op.realpath(path) for path in includedirs}

    return includedirs


def autodetect_version(libdirs, libnameregexp, mpi=None, hdf5_version=None):
    """
    Detect the current version of HDF5, and return X.Y.Z version string and path

    Intended for Unix-ish platforms (Linux, OS X, BSD).
    Does not support Windows. Raises an exception if anything goes wrong.

    libdirs: list of candidate directories where to look for hdf5 library
    mpi    : optional switch whether to look for parallel library version
    """
    import ctypes
    from ctypes import byref

    for d in libdirs:
        try:
            candidates = [x for x in os.listdir(d) if libnameregexp.match(x)]
        except Exception:
            continue   # Skip invalid entries

        if len(candidates) != 0:
            candidates.sort(key=lambda x: len(x))   # Prefer libfoo.so to libfoo.so.X.Y.Z
            librarypath = op.abspath(op.join(d, candidates[0]))
            break

    lib = ctypes.cdll.LoadLibrary(librarypath)

    major = ctypes.c_uint()
    minor = ctypes.c_uint()
    release = ctypes.c_uint()

    lib.H5get_libversion(byref(major), byref(minor), byref(release))

    version = "{0}.{1}.{2}".format(int(major.value), int(minor.value), int(release.value))

    if hdf5_version is not None and not sys.platform.startswith('win'):
        assert hdf5_version == version

    return version


def autodetect_define_macros(mpi=None, fallback=False):
    '''
    Get the hdf5 define macros if applicable.

    This function tries to get the define_macros for the hdf5 library from
    pkgconfig.
    '''
    dmacros = set()

    if not fallback:
        for package in hdf5_names_to_try(mpi):
            if pkgconfig.exists(package):
                dmacros = pkgconfig.parse(package)['define_macros']
                break

    dmacros.add(('H5_USE_16_API', None))
    if sys.platform.startswith('win'):
        dmacros.add(('_HDF5USEDLL_', None))

    return dmacros


def autodetect_hdf5(hdf5_dir=None, hdf5_libdir=None, hdf5_libname=None,
                    hdf5_includedir=None, hdf5_version=None, mpi=None,
                    fallback=False):
    """
    Detect library and include path as well as version of libhdf5.

    Intended for Unix-ish platforms (Linux, OS X, BSD).
    Does not support Windows. Raises an exception if anything goes wrong.

    hdf5_dir: optional HDF5 install directory to look in (containing "include")
    mpi     : optional switch whether to look for parallel library version
    """

    if NOPKGCONFIG:
        fallback = True
    else:
        fallback = bool(fallback)

    print(fallback)
    libdirs = autodetect_libdirs(hdf5_dir, hdf5_libdir, mpi, fallback)
    print(libdirs)
    libname, libnameregexp = autodetect_libname(hdf5_libname, mpi, fallback)
    print(libname, libnameregexp)
    version = autodetect_version(libdirs, libnameregexp, mpi, hdf5_version)
    print(version)
    includedirs = autodetect_includedirs(hdf5_dir, hdf5_includedir, libdirs, mpi, fallback)
    print(includedirs)
    macros = autodetect_define_macros(mpi, fallback)
    print(macros)
    return (list(libdirs), list(includedirs), version, libname, list(macros))


class EnvironmentOptions(object):

    """
        Convenience class representing the current environment variables.
    """

    def __init__(self):
        self.hdf5 = os.environ.get('HDF5_DIR')
        self.hdf5_version = os.environ.get('HDF5_VERSION')
        self.hdf5_libdir = os.environ.get('HDF5_LIB')
        self.hdf5_libname = os.environ.get('HDF5_LIBNAME')
        self.hdf5_includedir = os.environ.get('HDF5_INCLUDE')
        self.mpi = os.environ.get('HDF5_MPI')
        self.fallback = bool(os.environ.get('HDF5_FALLBACK'))
        if self.hdf5_version is not None:
            validate_version(self.hdf5_version)


class configure(Command):

    """
        Configure build options for h5py: custom path to HDF5, version of
        the HDF5 library, and whether MPI is enabled.

        Options come from the following sources, in order of priority:

        1. Current command-line options
        2. Old command-line options
        3. Current environment variables
        4. Old environment variables
        5. Autodetection

        When options change, the rebuild_required attribute is set, and
        may only be reset by calling reset_rebuild().  The custom build_ext
        command does this.s
    """

    description = "Configure h5py build options"

    user_options = [('hdf5=', 'h', 'Custom path to HDF5 (parent of lib and include)'),
                    ('hdf5-libdir=', 'l', 'Custom path to HDF5 library directory'),
                    ('hdf5-libname=', 'n', 'Custom HDF5 library name'),
                    ('hdf5-includedir=', 'i','Custom path to HDF5 include directory'),
                    ('hdf5-version=', '5', 'HDF5 version "X.Y.Z"'),
                    ('mpi=', 'm', 'Enable MPI building'),
                    ('reset', 'r', 'Reset config options'),
                    ('fallback', 'f', 'Do not use pkgconfig')]

    def initialize_options(self):
        self.hdf5 = None
        self.hdf5_libdir = None
        self.hdf5_libname = None
        self.hdf5_includedir = None
        self.hdf5_version = None
        self.mpi = None
        self.reset = None
        self.fallback = None

    def finalize_options(self):
        if self.hdf5_version is not None:
            validate_version(self.hdf5_version)

    def reset_rebuild(self):
        """ Mark this configuration as built """
        dct = loadpickle()
        dct['rebuild'] = False
        savepickle(dct)

    def run(self):
        """ Distutils calls this when the command is run """

        env = EnvironmentOptions()

        # Step 1: determine if settings have changed and update cache

        oldsettings = {} if self.reset else loadpickle()
        dct = oldsettings.copy()

        # Only update settings which have actually been specified this
        # round; ignore the others (which have value None).
        if self.hdf5 is not None:
            dct['cmd_hdf5'] = self.hdf5
        if env.hdf5 is not None:
            dct['env_hdf5'] = env.hdf5

        if self.hdf5_libdir is not None:
            dct['cmd_hdf5_libdir'] = self.hdf5_libdir
        if env.hdf5_libdir is not None:
            dct['env_hdf5_libdir'] = env.hdf5_libdir

        if self.hdf5_libname is not None:
            dct['cmd_hdf5_libname'] = self.hdf5_libname
        if env.hdf5_libname is not None:
            dct['env_hdf5_libname'] = env.hdf5_libname

        if self.hdf5_includedir is not None:
            dct['cmd_hdf5_includedir'] = self.hdf5_includedir
        if env.hdf5_includedir is not None:
            dct['env_hdf5_includedir'] = env.hdf5_includedir

        if self.hdf5_version is not None:
            dct['cmd_hdf5_version'] = self.hdf5_version
        if env.hdf5_version is not None:
            dct['env_hdf5_version'] = env.hdf5_version

        if self.mpi is not None:
            dct['cmd_mpi'] = self.mpi
        if env.mpi is not None:
            dct['env_mpi'] = env.mpi

        self.rebuild_required = dct.get('rebuild') or dct != oldsettings

        # Corner case: rebuild if options reset, but only if they previously
        # had non-default values (to handle multiple resets in a row)
        if self.reset and any(loadpickle().values()):
            self.rebuild_required = True

        dct['rebuild'] = self.rebuild_required

        savepickle(dct)

        # Step 2: update public config attributes according to priority rules

        if self.hdf5 is None:
            self.hdf5 = oldsettings.get('cmd_hdf5')
        if self.hdf5 is None:
            self.hdf5 = env.hdf5
        if self.hdf5 is None:
            self.hdf5 = oldsettings.get('env_hdf5')

        if self.hdf5_libdir is None:
            self.hdf5_libdir = oldsettings.get('cmd_hdf5_libdir')
        if self.hdf5_libdir is None:
            self.hdf5_libdir = env.hdf5_libdir
        if self.hdf5_libdir is None:
            self.hdf5_libdir = oldsettings.get('env_hdf5_libdir')

        if self.hdf5_libname is None:
            self.hdf5_libname = oldsettings.get('cmd_hdf5_libname')
        if self.hdf5_libname is None:
            self.hdf5_libname = env.hdf5_libname
        if self.hdf5_libname is None:
            self.hdf5_libname = oldsettings.get('env_hdf5_libname')

        if self.hdf5_includedir is None:
            self.hdf5_includedir = oldsettings.get('cmd_hdf5_includedir')
        if self.hdf5_includedir is None:
            self.hdf5_includedir = env.hdf5_includedir
        if self.hdf5_includedir is None:
            self.hdf5_includedir = oldsettings.get('env_hdf5_includedir')

        if self.hdf5_version is None:
            self.hdf5_version = oldsettings.get('cmd_hdf5_version')
        if self.hdf5_version is None:
            self.hdf5_version = env.hdf5_version
        if self.hdf5_version is None:
            self.hdf5_version = oldsettings.get('env_hdf5_version')

        if self.mpi is None:
            self.mpi = oldsettings.get('cmd_mpi')
        if self.mpi is None:
            self.mpi = env.mpi
        if self.mpi is None:
            self.mpi = oldsettings.get('env_mpi')

        if self.hdf5_version is None:
            try:
                versioninfo = autodetect_hdf5(self.hdf5,
                                              self.hdf5_libdir,
                                              self.hdf5_libname,
                                              self.hdf5_includedir,
                                              self.hdf5_version,
                                              self.mpi,
                                              self.fallback)
                self.hdf5_libdir = versioninfo[0]
                self.hdf5_includedir = versioninfo[1]
                self.hdf5_version = versioninfo[2]
                self.hdf5_libname = versioninfo[3]
                self.hdf5_define_macros = versioninfo[4]
                print("Autodetected HDF5 %s" % self.hdf5_version)
            except Exception as e:
                sys.stderr.write("Autodetection skipped [%s]\n" % e)
                self.hdf5_libdir = '???'
                self.hdf5_includedir = '???'
                self.hdf5_version = '???'

        # Step 3: print the resulting configuration to stdout

        print('*' * 80)
        print(' ' * 23 + "Summary of the h5py configuration")
        if self.fallback or NOPKGCONFIG:
            print('Used fallback configuration search.')
        else:
            print('Used pkg-config for configuration search.')
        print('')
        print("Path to HDF5 library: " + repr(self.hdf5_libdir))
        print("  Using headers from: " + repr(self.hdf5_includedir))
        if self.hdf5_libname is not None:
            print("   HDF5 library name: " + repr(self.hdf5_libname[0]))
        print("        HDF5 Version: " + repr(self.hdf5_version))
        print("         MPI Enabled: " + repr(bool(self.mpi)))
        print("    Rebuild Required: " + repr(bool(self.rebuild_required)))
        print('')
        print('*' * 80)
