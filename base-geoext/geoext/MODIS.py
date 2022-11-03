"""

Description:
Functionality for processing MODIS HDF-EOS products. Currently, 
the products available are:
- MCD19A2
- MOD21A1D

"""

import os
from typing import List, Union
import rioxarray as rxr
import xarray as xr
import rasterio as rio
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

class HDF_EOS:
    def __init__(
        self, 
        filepath: str, 
        desired_vars: List[str] = None,
        group_index: int = 0
    ):
        self.xd = self.open(filepath, desired_vars, group_index)

    @staticmethod
    def open(
        fp: str, 
        variables: List[str] = None,
        group_index: int = 0,
        hasty: bool = True
    ) -> Union[xr.DataArray, xr.Dataset]:
        """ 
        Open the HDF-EOS file as a rioxarray.

        Parameters
        ----------
        fp: str
            The filepath to the HDF-EOS file
        variables: List[str]
            The variables to extract from the file, as usually
            many variables/layers are contained in a single file
        group_index: int
            These files are sometimes (maybe all the time,
            I'm not sure) divided into groups of variables.
            This specifies which group of variables to use.
        hasty: bool
            Whether or not speed should be prioritized over
            safety checks

        Returns
        -------
        Union[xr.Datarray, xr.Dataset]
            the rioxarray Dataset or DataArray for the HDF-EOS file
        """

        # If safety checks > speed, validate the requested variables
        # exist in the file
        if not(hasty): 
            HDF_EOS.validate_variables(fp, variables, group_index)

        # Open the HDF file using the specified variables and filepath
        # using rioxarray
        hdf_file = rxr.open_rasterio(fp, variable=variables)
        
        # We convert to list if not already a list
        # as this represents the 'groups'; this will just
        # result in a single group
        if not(isinstance(hdf_file, list)): hdf_file = [hdf_file]
        hdf_file = hdf_file[group_index]

        return hdf_file

    @staticmethod
    def validate_variables(
        # Filepath to the file
        fp: str,                
        # Variable list to validate
        variables: List[str],   
        # The group number; some files may have groups of variables
        group_index: int = 0    
    ):
        """ 
        Validate if specified variables exist within file and group (if
        specified. If not, default is group 0).
        Throws exception if invalid.

        Parameters
        ----------
        fp: str
            Filepath to the file
        variables: List[str]
            Variable list to validate
        group_index: int
            The group number; some files may have groups of variables

        Returns
        -------
        None
        """
        hdf_vars = HDF_EOS.variables(fp)['Group ' + str(group_index)]
        
        if variables is not None:
            for var in variables:
                if not(var in hdf_vars):
                    raise Exception(
                        f"\'{var}\' not in Group {str(group_index)}'s "
                        f"variable list for HDF file.\n"
                        f"Hint: Use the 'variables' function to get a "
                        f"list of the available variables by group."
                    )

    @staticmethod
    def variables(
        filepath: str,
    ) -> dict:
        """ Returns a list of variable in HDF-EOS file """
        hdf_file = rxr.open_rasterio(filepath)
        # We convert to list if not already a list
        # as this represents the 'groups'; this will just
        # result in a single group
        if not(isinstance(hdf_file, list)): hdf_file = [hdf_file]
        
        var_dict = {}
        for i, group in enumerate(hdf_file):
            var_dict['Group ' + str(i)] = [var for var in group.data_vars]

        return var_dict

class MCD19A2(HDF_EOS):

    # Constant water body pixel value
    WATER_BODY_PXL_VALUE = -101

    def __init__(
        self, 
        filepath: str, 
        nlcd_fp: str,
        QA_mask_method: str = 'best',
        desired_vars: List[str] = ["Optical_Depth_055", "AOD_QA"],
        group_index: int = 0
    ) -> None:
        """ 
        Initialize a MCD19A2 object based on the raw MODIS MCD19A2 
        HDF-EOS file passed. Performs the following operations after
        opening:
        - Masks the raster
        - Takes a daily average (as multiple orbit overpasses occur per 
        day)
        - Masks out water pixels (as the AOD from MCD19A2 over water 
        is not typically reliable) and replaces them with constant
        value defined in the main class body
        - Performs post-processing, which just consists of replacing
        negative values (indicating very clear) with 0
        - Computes the percentage of valid (non-corrupted) pixels

        Parameters
        ----------
        filepath: str
            path to raw MCD19A2 HDF-EOS file
        nlcd_fp: str
            path to NLCD for masking water bodies. MUST cover entire
            area for MCD19A2 file.
        QA_mask_method: str
            which QA mask filter to apply to data
        desired_vars: List[str]
            variables to extract from HDF-EOS file. This probably
            shouldn't change.
        group_index: int
            HDF-EOS files have variable groups. This should almost
            definitely never change.

        Returns
        -------
        None
        """

        super().__init__(filepath, desired_vars, group_index)

        self.mask_QA(method=QA_mask_method)
        self.daily_avg()
        self.mask_water_bodies(nlcd_fp)
        self.post_processing()

        self.calc_valid_data()

    def mask_QA(
        self, 
        method: str = 'best', 
        filter_band: str = 'Optical_Depth_055', 
        verbose: bool = False
    ):
        """ 
        Mask (replace pixels with no data values) the raster
        according the the QA (quality assurance) layer

        Parameters
        ----------
        method: str
            the mask to use in QA masking. Options are 'best', 
            'noClouds', 'urban', and 'urban_mod'
        filter_band: str
            the band to filter using the QA layer
        verbose: bool
            whether or not to output the number of removed pixels

        Returns
        -------
        None
        """
        def split_qa(qa_arr):
            """ 
            Split the QA mask into its sub masks.
            These include:
                - Cloud mask
                - Adjacency mask
                - QA mask
            """
            cloud_bitmask       = int('0000000000000111', 2)
            adjacency_bitmask   = int('0000000011100000', 2)
            qa_bitmask          = int('0000111100000000', 2)

            cld_mask    = qa_arr & cloud_bitmask
            adj_mask    = qa_arr & adjacency_bitmask
            qa_mask     = qa_arr & qa_bitmask

            return cld_mask, adj_mask, qa_mask

        ### Constants corresponding to various values of QA mask
        ## Cloud mask
        # Cloud mask clear
        CLD_CLEAR       = int('0000000000000'+'001', 2)
        # Cloud mask possibly cloud
        CLD_POSSIBLE    = int('0000000000000'+'010', 2)
        ## Adjacency mask
        # Adjacency mask clear
        ADJ_CLEAR       = int('00000000'+'000'+'00000', 2)
        # Adjacency mask single cloudy pixel
        ADJ_SINGLE      = int('00000000'+'011'+'00000', 2)
        ## QA mask
        QA_BEST         = int('0000'+'0000'+'00000000', 2)

        # Split the cloud mask into its 3 sub-masks
        cld_mask, adj_mask, qa_mask = split_qa(self.xd['AOD_QA'].to_numpy())

        # Convert the rioxarray to a numpy ndarray so we can perform
        # masking more easily
        band_np = self.xd[filter_band].to_numpy()
        nodata = self.xd[filter_band].rio.nodata
        # Create a copy of the variable data for the new filtered
        # version
        band_masked = np.copy(band_np)

        if method == 'best':
            band_masked[~(qa_mask == QA_BEST)] = nodata

        elif method == 'noClouds':
            band_masked[~(cld_mask == CLD_CLEAR)] = nodata

        elif method == 'urban':
            # Allow cloud masks clear and possibly cloudy.
            # Possibly cloudy allowed bc this erroneously erases 
            # non-cloudy data in urban areas (https://v.gd/omo3NG)
            # Allow adjacency masks clear and single adjacent cloud
            # since single_adjacent_cloud is often false positive.
            band_masked[~(
                ((cld_mask == CLD_CLEAR) | (cld_mask == CLD_POSSIBLE)) \
                    & ((adj_mask == ADJ_CLEAR) | (adj_mask == ADJ_SINGLE))
            )] = nodata

        elif method == 'urban_mod':
            # This method is the same as the previous, but we do not 
            # account for adjacent cloudy pixels. This is because areas 
            # around bodies of water are removed when the adjacent 
            # masks are used.
            band_masked[~(
                ((cld_mask == CLD_CLEAR) | (cld_mask == CLD_POSSIBLE))
            )] = nodata

        else:
            raise Exception("QA filter method not recognized.")

        if verbose:
            # Compute the total number of masked pixels and the total 
            # number of pixels so we can determine how many pixels
            # are masked (as a percentage)
            masked_ct = len(band_np[np.where(~(band_np==band_masked))])
            total_ct = len(band_np[~(band_np==nodata)])

            percent_removed = masked_ct / total_ct * 100 if total_ct else 0
            print(
                f"Removed {round(percent_removed, 2)}% of non-null data "
                f"as it did not meet the specified QA critera."
            )

        self.xd = self.xd[filter_band].copy(data=band_masked)

    def calc_valid_data(self):
        """
        Calculate the amount (as a proportion of the total) of 
        data that is available (i.e. not missing) in the xarray 

        TODO: Make this more generalized (i.e. not only for 
        "Optical_Depth_055")
        """

        if isinstance(self.xd, xr.Dataset):
            data_np = self.xd["Optical_Depth_055"].to_numpy()
            nodata = self.xd["Optical_Depth_055"].rio.nodata
        else:
            data_np = self.xd.to_numpy()
            nodata = self.xd.rio.nodata
        null_px_ct = len(data_np[data_np == nodata])
        total_px_ct = data_np.size
        self.valid_data = (total_px_ct - null_px_ct) / total_px_ct

        return self.valid_data

    def daily_avg(self):
        """
        Take the daily average of MCD19A2 (as there are multiple orbit)
        overpasses per day.

        TODO: Add functionality for using this function pre-QA-masking (i.e.
        when the QA mask is still within the objects data)
        """
        var = self.xd.to_numpy()

        daily_avg = np.true_divide(
            var.sum(axis=0, where=var!=self.xd.rio.nodata), 
            (var!=self.xd.rio.nodata).sum(axis=0),
            out=np.ones_like(
                var.sum(axis=0, where=var!=self.xd.rio.nodata), dtype=float
            )*self.xd.rio.nodata,
            where=(var!=self.xd.rio.nodata).any(axis=0)
        ).astype(var.dtype)

        # # Get the first band of xarray (since our new values only have a 
        # # single band) & replace the values with our
        # # new averaged values
        self.xd = self.xd.isel(band=0).copy(data=daily_avg)

    def post_processing(self):
        """
        Negative values for MCD19A2 are classified as "very clear";
        they can just be set to 0 if this information is not needed,
        so here we do just that.
        """

        data_np = self.xd.to_numpy()

        data_np[
            (
                data_np < 0
            ) & (
                data_np != self.xd.rio.nodata
            ) & (
                data_np != MCD19A2.WATER_BODY_PXL_VALUE
            )
        ] = 0

        self.xd = self.xd.copy(data=data_np)

    def mask_water_bodies(self, nlcd_fp):
        """
        Using the NLCD data, mask the water body pixels using the
        water body pixel value defined in the class body. We do this
        as MCD19A2 is unreliable over water.
        """
        nlcd = rxr.open_rasterio(nlcd_fp)

        # Reproject NLCD to match the transform of the input
        # xarray. 
        # Resampling = 6 means resample using Mode
        # Resampling options here: https://v.gd/iMCNCF
        nlcd_reproj = nlcd.rio.reproject_match(self.xd, resampling=6)

        aod = self.xd.to_numpy()
        nlcd_vals = nlcd_reproj.to_numpy().squeeze()

        aod[nlcd_vals == 11] = MCD19A2.WATER_BODY_PXL_VALUE

        self.xd = self.xd.copy(data=aod)

    @staticmethod
    def get_corruption_mask(data):
        """ Get binary mask of corruption (where 1 is corrupt) """
        # Where the data values are the same as the nodata values, set
        # the binary mask to 1; everywhere else, set the value to 0
        mask = xr.where(data==data.rio.nodata, 1, 0).astype(np.uint8)
        mask_np = mask.to_numpy()
        data_np = data.to_numpy()

        corrupted_px_ct = len(mask_np[mask_np==1])
        non_water_px_ct = data_np[data_np!=MCD19A2.WATER_BODY_PXL_VALUE].size

        corrupted_ratio = corrupted_px_ct/non_water_px_ct

        return mask, corrupted_ratio

class MOD21A1D:

    @staticmethod
    def mask_QA(
        data_rxr: xr.DataArray, 
        filter_band: str ='LST_1KM'
    ) -> xr.DataArray:
        """ 
        Mask (replace pixels with no data values) the rioxarray
        according the the QA (quality assurance) layer

        Parameters
        ----------
        data_rxr: xr.DataArray
            the DataArray to mask with the QA layer
        filter_band: str
            the band to filter using the QA layer

        Returns
        -------
        xr.DataArray
            the QA-masked DataArray
        """
        
        def parse_QA(qa_arr):
            """ Extract QA summary flags from QA """
            qa_flag_bitmask = int('0000000000000011', 2)
            qa_flag_mask    = qa_arr & qa_flag_bitmask

            return qa_flag_mask

        ## Constants corresponding to various values of QA mask 
        # QA flags indicate best quality
        QA_GOOD = int('00000000000000'+'00', 2)

        # Used to replace the masked pixel values
        null_val = data_rxr[filter_band].rio.nodata

        qa_flag_mask = parse_QA(data_rxr.QC.to_numpy())

        masked = np.copy(data_rxr[filter_band].to_numpy())
        # Where the QA is not good quality,
        # set the pixel value to no data
        masked[~(qa_flag_mask == QA_GOOD)] = null_val

        return data_rxr[filter_band].copy(data=masked)

    @staticmethod
    def data_availability(data_rxr):
        """
        Calculate the amount (as a proportion of the total) of data
        that is available (i.e. not missing) in the data
        """

        data_np = data_rxr.to_numpy()
        null_px_count = len(data_np[data_np == data_rxr.rio.nodata])
        total_px_count = data_np.size
        amountDataNonNull = (total_px_count - null_px_count) / total_px_count

        return amountDataNonNull

    @staticmethod
    def get_cloud_mask(data_rxr):
        """ Obtain a binary mask of clouds where 1 is cloudy """
        # Anywhere where the data is equivalent to the no data value(i.e. a  
        # null reading), set the cloud mask to 1
        cloud_mask_xr = xr.where(data_rxr == data_rxr.rio.nodata, 1, 0)
        # Convert to uint8 to save space since only options are 1 and 0
        cloud_mask_xr = cloud_mask_xr.astype(np.uint8)

        return data_rxr.copy(data=cloud_mask_xr)

def get_orbit_times(filename: str) -> list:
    """ 
    Get the timestamps at which MCD19A2 recorded the data in the 
    provided HDF file
    """

    hdfFile = rxr.open_rasterio(filename)

    # Get the attributes from the hdf file
    attrs = hdfFile[0].attrs

    # Return the timestamps in list format
    return attrs["Orbit_time_stamp"].split()