import sys, os, stat, site
from distutils.core import setup
from distutils.extension import Extension
# Check for Cython
try:
    from Cython.Distutils import build_ext
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
except:
    print('Cython is required for this install')
    sys.exit(1)

# ==============================================================
# scan a directory for extension files, converting them to
# extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

# ==============================================================
# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [site.PREFIXES[0]+'/include', '.'],   # adding the '.' to include_dirs is CRUCIAL!!
        library_dirs = [site.PREFIXES[0]+'/lib'],
        extra_compile_args = ["-O3", "-fopenmp", "-Wno-maybe-uninitialized", "-std=c99"],
        extra_link_args = ["-fopenmp"],
        # libraries = ['/Users/yue/anaconda/lib/libpython2.7.dylib'],
        # libraries = ['python2.7'],
        )

# ==============================================================
# get the list of extensions
extNames = scandir("imseg")
# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]
for ext in extensions:
    ext.cython_directives = {}
    ext.cython_directives["boundscheck"] = False
    ext.cython_directives["wraparound"] = False
    ext.cython_directives["cdivision"] = True
    ext.cython_directives["embedsignature"] = True
    ext.cython_directives["profile"] = False
# ==============================================================
setup(
    name="imseg",
    version='0.1.0',
    author='Yue Sun',
    author_email='y@u.northwestern.edu',
    packages=['imseg',
              'imseg.segment',
              'imseg.threshold',
              'imseg.imtools',
              'imseg.imtools.imclip',
              'imseg.preprocess',
              'imseg.regions',
              'imseg.update_sdf',
              ],
    url=None,
    license='LICENSE.txt',
    description='Multi-dimensional image segmentation using SDF',
    long_description=open('README.md').read(),
    ext_modules=extensions,
    cmdclass = {'build_ext': build_ext},
    requires=['numpy', 'cython']  #, 'skimage']
)

