"""

Functionality for integrating geographic data with the HDF5 workflow.

TODO: In df_to_ds, add option to not replace (throw error instead)

"""

import numpy as np
import pandas as pd
import rasterio
import affine
import h5py

def df_to_ds(
    df: pd.DataFrame, 
    h5: h5py.File, 
    ds_name: str
) -> h5py.Dataset:
    """ 
    Convert dataframe to numpy record, and then save this in the HDF5
    file under the specified name. Numpy records are used as this 
    allows us to name the columns in the HDF5 dataset. If a dataset 
    with the same name as that that is provided exists in the HDF5 
    file, it will be replaced.

    Parameters
    ----------
    df: pd.DataFrame
        the Pandas DataFrame to be converted to a new dataset in the
        HDF5 file
    h5: h5py.File
        the HDF5 file to add the new dataset to
    ds_name: str
        the name of the new dataset that will be created

    Returns
    -------
    h5py.Dataset
        the newly created HDF5 dataset object
    """

    # Extract information to be used to create numpy record.
    # We use numpy records as these provide an easy method
    # to assign column headers (or column labels) to h5 data
    # We must also replace objects with their corresponding
    # string representations
    col_names = df.columns
    col_vals = [ df[col].to_numpy() for col in col_names ]
    col_dtypes = [
        array.dtype if array.dtype != 'O'
        else array.astype(str).dtype.str.replace(
            '<U','S'
        ) for array in col_vals
    ]

    # Convert the previously extracted data to a numpy record
    rec_array = np.rec.fromarrays(col_vals, dtype={
        'names': col_names, 'formats': col_dtypes
    })

    # If the dataset already exists, delete it
    if '/' + ds_name in h5:
        del h5['/' + ds_name]

    # Create the dataset in the hdf5 file; we use gzip
    # compression as it provides decent compression and 
    # is lossless
    ds = h5.create_dataset(
        ds_name, data=rec_array, compression="gzip"
    )

    return ds

def df_from_ds(
    ds: h5py.Dataset
) -> pd.DataFrame:
    """ Parse the data from the HDF5 dataset into a pd dataframe """
    # Retrieve the column or field names from the hdf5 file
    col_names = np.array(list(ds.dtype.fields.keys()))
    # Put the file data into a dataframe
    df = pd.DataFrame(ds[()], columns=col_names)

    # Decode any strings to bytes
    df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

    return df

def raster_to_ds(
    h5File: h5py.File, 
    ds_name: str, 
    arr: np.ndarray, 
    metadata: dict,
    compression: str = "gzip"
) -> h5py.Dataset:
    """ 
    Convert a raster object to a dataset in the specified HDF5 file.

    Parameters
    ----------
    h5File: h5py.File
        the HDF5 file to save the new dataset in
    ds_name: str
        the name of the new dataset to be created
    arr: np.ndarray
        the raster to convert to a dataset (and store in the HDF5 file)
    metadata: dict
        a dictionary of the raster's geographic metadata
    compression: str
        the type of compression to use when storing the new dataset

    Returns
    -------
    h5py.Dataset
        the newly created HDF5 dataset object
    """

    def affine_to_geotrans(aff) -> np.ndarray:
        """ 
        Convert an affine transform object into a GDAL GeoTransform
        in the form of a numpy array.
        """
        # x offset, or left-most x coordinate
        x_off = aff[2]
        # y offset, or top-most y coordinate
        y_off = aff[5]
        # Pixel width (x-direction) or west-east pixel resolution
        px_w = aff[0]
        # Pixel height (y-direction) or north-south pixel resolution
        px_h = aff[4]
        # Row rotation
        row_rot = aff[1]
        # Column rotation
        col_rot = aff[3]
        geotransform = np.array([
            x_off, px_w, row_rot,
            y_off, col_rot, px_h
        ])

        return geotransform

    if '/' + ds_name in h5File:
        del h5File['/' + ds_name]
    ds = h5File.create_dataset(
        ds_name, data=arr, compression=compression
    )
        
    # Create an attribute for this dataset corresponding to each
    # of the items in the raster's metadata
    meta = metadata.copy()
    for name in meta:
        if name == 'crs':
            meta[name] = meta[name].to_string()
        if name == 'transform':
            meta[name] = affine_to_geotrans(meta[name])
        if name == 'nodata':
            if meta[name] == None:
                meta[name] = ''

        ds.attrs.create(name, meta[name])

    return ds

def ds_to_raster(
    ds: h5py.Dataset
) -> tuple[np.ndarray, dict]:
    """
    Convert an HDF5 dataset back to raster.
    Assumes that the attributes are properly formatted so that we can
    use them for metadata (CRS, transform, etc.)
    """

    data = ds[()]

    meta = {
        'count': ds.attrs['count'],
        'crs': rasterio.crs.CRS.from_string(ds.attrs['crs']),
        'driver': ds.attrs['driver'],
        'dtype': ds.attrs['dtype'],
        'height': ds.attrs['height'],
        'nodata': ds.attrs['nodata'] if ds.attrs['nodata'] != '' else None,
        'transform': affine.Affine.from_gdal(*ds.attrs['transform']),
        'width': ds.attrs['width']
    }

    return data, meta

def recursive_datasets(
    h5_group: h5py.Group
):
    """
    Takes an HDF5 file group as input and recursively 
    retrieves all datasets within that group.
    """
    class DS_list:
        def __init__(self):
            # Store an empty list for datasets
            self.datasets = []

        def __call__(self, name, h5obj):
            if isinstance(h5obj, h5py.Dataset):
                self.datasets.append(h5obj)

    ds_list = DS_list()

    h5_group.visititems(ds_list)

    return ds_list.datasets