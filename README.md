# GeoExt
This project contains functionality for handling geographic data in Python including rasters and shapefiles, integrating with common geographic libraries like rasterio, geopandas, and rioxarray, and HDF5 integration. Also available as a (very WIP) anaconda package: conda install -c jedaniels000 geoext.

# Code Structure
## base-geoext
Contains the 'setup.py' file used in creating a conda package as well as the **main code folder, 'geoext'**. 
### geoext
- geo.py: Functionality to aid in the processing of geographic data such as tif and shapefiles and integrating with common geographic libraries like rasterio, geopandas, rioxarray, etc.
- h5geo.py: Functionality for integrating geographic data with the HDF5 workflow.

## geoext-conda-build
Contains other files used in creating a conda package. No functionality code here.

# Usage
## Creating a local build of the Anaconda package
- Navigate to the directory: geoext/geoext-conda-build
- Run the following command: `conda-build .`
- Alternatively, from the root directory, the package can be built by running `conda-build geoext-conda-build/`