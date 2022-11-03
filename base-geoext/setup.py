from setuptools import setup, find_packages

VERSION = '0.0.5' 
DESCRIPTION = 'Helpful geographic functions'
LONG_DESCRIPTION = 'Helpful geographic functions for rasters, shapefiles, and HDF5 files'

# Setting up
setup(
    name="geoext", 
    version=VERSION,
    author="Jacob Daniels",
    author_email="jacobdaniels2@my.unt.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # Included so .egg file is not created, as the .egg file
    # is not compatible with some editor's documentation viewers
    zip_safe=False,
    packages=find_packages(),
    # add any additional packages that 
    # needs to be installed along with your package.
    install_requires=[], 
)