"""
Functionality to aid in the processing of geographic data 
such as tif and shapefiles and integrating with common geographic
libraries like rasterio, geopandas, rioxarray, etc.

TODO: Update 'align' to handle non single-band rasters
TODO: Change zonal_stats to compute metrics other than mean

"""

import os
from typing import Union
import numpy as np
import rasterio
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask as rasterio_mask
import tempfile
from shapely.geometry import Polygon
import shapely

def align(
    raster: Union[rasterio.Band, np.ndarray], 
    raster_meta: dict, 
    ref_fp: str,
    resampling=rasterio.warp.Resampling.nearest
) -> tuple[np.ndarray, dict]:
    """ 
    Aligns the input raster with the projection and extent of the
    reference raster.

    Parameters
    ----------
    raster: Union[rasterio.Band, np.ndarray]
        raster to be aligned
    raster_meta: dict
        rasterio metadata for raster to be aligned
    ref_fp: str
        the filepath to the raster to be aligned to
    resampling: rasterio.warp.Resampling
        the resampling method to be used for alignment

    Returns
    -------
    tuple[np.ndarray, dict]
        a tuple of the aligned raster data (in numpy array format)
        and the corresponding geographic metadata
    """

    # From the raster to be aligned to, we need the transform, crs, 
    # and shape for reprojection
    with rasterio.open(ref_fp) as ref_raster:
        ref_trans = ref_raster.transform
        ref_crs = ref_raster.crs
        # The count is the number of bands, and the shape is 
        # the x and y dimensions
        band_ct, ref_row_ct, ref_col_ct = (ref_raster.count, *ref_raster.shape)

    # This variable will hold the new metadata for the aligned raster
    aligned_meta = raster_meta.copy()

    # --- Reproject --- #
    aligned = np.zeros((ref_row_ct, ref_col_ct), dtype=raster.dtype)
    rasterio.warp.reproject(
        raster,
        aligned,
        src_transform=raster_meta['transform'],
        src_crs=raster_meta['crs'],
        dst_transform=ref_trans,
        dst_crs=ref_crs,
        resampling=resampling)
    # --- Update metadata to reflect reprojection --- #
    aligned_meta.update({
        'width': ref_col_ct,
        'height': ref_row_ct,
        'crs':ref_crs,
        'transform': ref_trans
    })

    return aligned, aligned_meta


def rasterize(
    gdf: gpd.GeoDataFrame,
    col: str,
    raster_fp: str,
    fill = None,
    dtype = None,
    merge_alg = rasterio.enums.MergeAlg('REPLACE')
) -> tuple[np.ndarray, dict]:
    """ 
    Rasterize the input GeoDataFrame using the reference raster and
    save at the specified output path. Partially based on: 
    https://v.gd/YsvGO6.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame from which to pull geometries and values
    col: str
        name of the column containing the values of the geometries
    raster_fp: str
        file path to the raster to use as reference (for shape, 
        transform, etc) 
    fill
        the fill value (nodata value) to use on the new raster
    dtype
        the data type to use on the new raster
    merge_alg: rasterio.enums.MergeAlg
        the algorithm used when shapes overlap: https://v.gd/AQDtTQ

    Returns
    -------
    tuple[np.ndarray, dict]
        a tuple of the rasterized data (in numpy array format)
        and the corresponding geographic metadata
    """

    # --- Get metadata from reference raster --- #
    with rasterio.open(raster_fp) as raster:
        meta = raster.meta.copy()
        
        # Add the lossless lzw compression if it isn't already set
        meta.update(compress='lzw')
        if dtype is not None: meta.update(dtype=dtype)
        # Set no data value
        if fill is not None: meta.update(nodata=fill)

        # For reprojecting before rasterization
        ref_crs = raster.crs

        # For rasterization
        ref_shp = raster.shape
        ref_transform = raster.transform

    # --- Match the projections between gdf and raster --- #
    gdf = gdf.to_crs(ref_crs)

    # --- Rasterize gdf with reference raster --- #
    # Create generator of (geometry, value) pairs for each shape in
    # the GeoDataFrame
    shapes = ((geom,value) for geom, value in zip(gdf.geometry, gdf[col]))
    
    # Rasterize the GeoDataFrame
    rasterized = rasterio.features.rasterize(
        shapes=shapes, 
        fill=fill, 
        out_shape=ref_shp, 
        transform=ref_transform,
        merge_alg=merge_alg
    )

    return rasterized, meta


def zonal_stats(
    shapefile: gpd.GeoDataFrame, 
    raster: rasterio.io.DatasetReader
) -> list:
    """ 
    Calculates the aggregated raster values within each shape/polygon 
    contained within the shapefile. The aggregation method is
    determined based on the corresponding parameter.

    Parameters
    ----------
    shapefile: gpd.GeoDataFrame
        GeoDataFrame from which to pull shapes within which the 
        corresponding raster values will be computed
    raster: rasterio.io.DatasetReader
        raster values to use in computing zonal statistics

    Returns
    -------
    list
        a list of the specified stat for each shape in GeoDataFrame
    """

    # The shapefile and raster must have the same crs in order to 
    # compute the zonal statistics, so we raise an error if they
    # don't match
    if shapefile.crs != raster.crs:
        raise Exception('CRS does not match.')

    # The nodata value is obtained to make sure the nodata value is not
    # used in computing statistics
    nodata_val = raster.nodata

    # For each shape, mask the raster to its extent (remove pixels
    # outside shape bounds) then the remaining pixel values (after
    # removing nodata values) are used in computing the statistics
    shapes = shapefile.geometry
    means = []
    for shape in shapes:
        # Crop=True removes the raster pixels rather than just setting
        # them to the nodata value
        cropped_np, affine = rasterio_mask(raster, [shape], crop=True)

        mean = cropped_np[cropped_np!=nodata_val].mean()
        means.append(mean)

    return means


def mask_raster(
    var: xr.DataArray, 
    shpPath: str, 
    xpad: int = 0, 
    ypad: int = 0, 
    native: bool = False
) -> xr.DataArray:
    """ 
    Mask a raster based on the box-extent of a shapefile. The function
    determines the maximum extent in each direction for all shapes in
    the shapefile, creates a corresponding box, and masks the raster
    using this box.

    Parameters
    ----------
    var: xr.DataArray
        raster data in the form of a rioxarray
    shpPath: str
        file path to shapefile to use as mask
    xpad: int
        the x-padding in each direction to use when masking 
    ypad: int
        the y-padding in each direction to use when masking
    native: bool
        if True, will use the crs of the raster when masking. This
        limits reprojection errors.

    Returns
    -------
    xr.DataArray
        the masked rioxarray
    """

    # Get shapefile data
    shpFile = gpd.read_file(shpPath)
    
    if native: shpFile = shpFile.to_crs(crs=var.rio.crs)

    # Get the crs of the shapefile
    shpCrs = shpFile.crs

    # Get all of the feature geometries in the shapefile
    shpGeoms = [feature["geometry"] for i, feature in shpFile.iterrows()]

    x = []
    y = []

    for shpGeom in shpGeoms:
        x.append(shpGeom.exterior.coords.xy[0])
        y.append(shpGeom.exterior.coords.xy[1])

    # Flatten list
    x_flat = [item for sublist in x for item in sublist]
    y_flat = [item for sublist in y for item in sublist]

    # Get the min and max x and y values
    xMax = max(x_flat) + xpad
    xMin = min(x_flat) - xpad
    yMax = max(y_flat) + ypad
    yMin = min(y_flat) - ypad

    polygon = [
        Polygon([(xMin, yMax), (xMax, yMax), (xMax, yMin), (xMin, yMin)])
    ]

    var_crpd = var.rio.clip(polygon, shpCrs, all_touched=True, drop=True)

    return var_crpd


def rxr_to_rio(
    data_rxr: xr.DataArray
) -> tuple[np.ndarray, dict]:
    """ Convert rioxarray to rasterio """
    # Create a temporary file so we can save the raster
    # data to it then reopen as rasterio dataset object
    tmp_f_dir = tempfile.gettempdir()
    tmp_f_name = os.urandom(24).hex() + '.tif'
    tmp_f_path = os.path.join(tmp_f_dir, tmp_f_name)

    # Save the rioxarray as a tif file
    data_rxr.rio.to_raster(tmp_f_path)

    # Open the raster as rasterio dataset object
    # and extract data and metadata
    with rasterio.open(tmp_f_path) as src:
        arr = src.read()
        meta = src.meta

    # Delete the temporary file to keep space free
    os.remove(tmp_f_path)

    return arr, meta


def df_to_gdf_pts(
    df: pd.DataFrame,
    coord_crs: str = 'epsg:4326',
    xy_col_names: list = ['Longitude', 'Latitude']
) -> gpd.GeoDataFrame :
    """ 
    Converts pandas DataFrame to GeoDataFrame using the X and Y 
    (i.e. Longitude and Latitude) coordinate columns specified.
    Only works for points, not other geometries (e.g. polygons).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data to be converted to GeoDataFrame.
        Must have both X and Y coordinate columns (or equivalent).
    coord_crs: str
        The CRS (Coordinate Reference System) of the geometry.
    xy_col_names: list
        A list of shape (1, 2) with the names of the X and Y 
        (longitude and latitude) columns, respectively.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame corresponding to the DataFrame once the
        geometry column has been created.
    """

    x_col = df[xy_col_names[0]]
    y_col = df[xy_col_names[1]]

    geometry = gpd.points_from_xy(x_col, y_col)

    return gpd.GeoDataFrame(df, geometry=geometry).set_crs(coord_crs)


def patch_rxr(
    raster: xr.DataArray,
    side_len: int,
    drop_ends: bool = True,
    overlap_px: int = 0,
    return_index: bool = False
) -> Union[list, tuple[list, list]]:
    """
    Patch a raster (i.e. tile the raster or break it into sub-rasters).

    Parameters
    ----------
    raster: xr.DataArray
        The raster (rioxarray) to be patched.

    side_len: int
        The side length of the patches (in pixels). The patches are 
        square, so the side length is both the length and width values.

    drop_ends: bool
        If True, when the width and height of the raster are not
        evenly divisible by the specified side_len, the last M pixels
        (where M is width % side_len or height % side_len) are dropped
        and a corresponding patch is not created. If False, the end
        patches will be created but overlap with the N-1 patch along
        the uneven dimensions (width and/or height).

    overlap_px: int
        The number of pixels to overlap between patches. By default,
        this value is 0 (meaning no overlap, except in the case where
        drop_ends is False and the image is not evenly divisible by
        side_len).

    return_index: bool
        If True, returns the indices of the patch positions as a list
        as (row_index, col_index). For example, the top left patch 
        would be (0, 0), the next would be (0, 1), etc.

    Returns
    -------
    Union[list, tuple[list, list]] 
        A list of the raster patches. If return_index is True, returns
        a tuple of (<list_of_raster_patches>, <list_of_xy_indices>) 
        where the list_of_xy_indices is the xy-position indices (e.g. 
        the top left patch would be (0, 0))
    """
    def box_from_bounds(
        raster: xr.DataArray, 
        dest_crs=None
    ):
        """ Creates a shapely box from the bounds of a rioxarray"""
        if dest_crs is not None:
            bounds = raster.rio.reproject(dest_crs).rio.bounds()
        else:
            bounds = raster.rio.bounds()

        return shapely.geometry.box(*bounds)

    def segment_ct_in_range(
        range_width: int,
        segment_width: int,
        overlap_width: int
    ) -> float:
        """
        Calculate the number of fixed-size segments that fit within a
        specified range (i.e. [0, N)) when the segments overlap. 
        For example:
        With the range 0-9 (i.e. a range with a width of 10), segments
        of length 3, and an overlap of 1, there are 4.667 segments:
        - [0:3]
        - [2:5]
        - [4:7]
        - [6:9]
        - [8:10]  # Non-full segment (2/3 or 0.667 of full segment)
        """

        # To determine the number of overlapping segments that can fit
        # within the range, we must first determine the number of 
        # unique sub-segments (i.e. those not shared by more than one
        # segment) that can be subtracted from the range before only 
        # one segment-width or less is left (i.e. the remainder, r, is
        # <= segment width but not negative). We do this by first 
        # subtracting one segment width from the range width, and then
        # taking the ceiling of 
        # <remaining_range_width> / <unique_width>. 
        unique_width = segment_width - overlap_width
        ct_initial = np.ceil((range_width - segment_width) / (unique_width))
        # We then find the remainder of the range (in range units) 
        # after allocating the computed unique ranges
        range_remainder = range_width \
            - ct_initial * (segment_width - overlap_width)
        # And, finally, find the total number of overlapping segments
        # that will fit within the range by adding the initial count
        # of the segments to the number of segments that can be created
        # from the remainder of the range (will be between (0.0, 1.0])
        count = ct_initial + range_remainder / segment_width

        return count

    # Height and width of the raster
    _, h, w = raster.shape
    # Number of sub-images along x (width) and y (height) axes
    x_ct = segment_ct_in_range(w, side_len, overlap_px)
    y_ct = segment_ct_in_range(h, side_len, overlap_px)
    # Floor the number of patches if drop ends, and take the ceiling
    # if the ends are to be kept.
    x_ct = int(np.floor(x_ct)) if drop_ends else int(np.ceil(x_ct))
    y_ct = int(np.floor(y_ct)) if drop_ends else int(np.ceil(y_ct))
    
    # The subdivided rasters
    patches = []
    # For each image index (i.e. x and y pair), extract the 
    # corresponding sub-raster from the original raster
    for y in range(y_ct): 
        for x in range(x_ct):
            last_y = y == (y_ct - 1)
            last_x = x == (x_ct - 1)

            x_offset = x * (side_len - overlap_px)
            y_offset = y * (side_len - overlap_px)

            # If the ends are to be kept, and this is the last column
            # or row, replace the offset with the ends of the image.
            if not(drop_ends):
                if last_x: x_offset = w - side_len
                if last_y: y_offset = h - side_len

            # Get the 'window' from which to extract the sub-raster
            window = rasterio.windows.Window(
                x_offset, y_offset, 
                side_len, side_len
            )
            # Extract the sub-raster at window and add to sub-raster list
            patches.append(
                (raster.rio.isel_window(window), (y, x))
            )

    # The actual patch is the first element in the patch tuple, so if the 
    # indices of the patch in the original image are not required, just 
    # return the list of the first elements (patches)
    return patches if return_index  else [patch[0] for patch in patches]

