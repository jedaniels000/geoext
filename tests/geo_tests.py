import os, sys
from re import S
import rioxarray as rxr
import numpy as np
import rasterio
from pyproj import CRS
import tempfile

# The path to the directory of the file being executed
exec_dir = os.path.dirname(os.path.realpath(__file__))
# Path to the directory containing the functionality being tested
fun_dir = os.path.join(exec_dir, '../base-geoext/geoext/')
# Append the directory containing the functionality being tested 
# to the system path 
sys.path.append(fun_dir)
import geo

def test_patch_rxr():
    def create_diagonal_gradient():
        """ Creates a 2D diagonal gradient as a numpy array """
        arr = np.empty((100, 100), dtype=np.uint8)
        for x in range(100):
            for y in range(100):
                arr[x, y] = x + y

        return arr
    def save_test_raster(arr, fp):
        """ Save a new raster w/ arbitrary geographical features """
        with rasterio.open(
            fp, 'w',
            driver='GTiff', width=100, height=100, count=1,
            dtype=np.uint8,
            crs=CRS.from_user_input(4326),
            transform=rasterio.Affine(0.01, 0, 0, 0, -0.01, 0)
        ) as dst:
            dst.write(arr[np.newaxis, ...])
    
    def test_basic_patch(rxr_arr, drop_ends):
        """
        Tests evenly divisible patching.
        """
        print(f"Testing basic patching w/ drop_ends = {drop_ends} ...")

        # Test with ends dropped
        patches = geo.patch_rxr(
            rxr_arr,
            side_len=50,
            drop_ends = drop_ends,
            return_index = False
        )
        # With an original side length of 100 and a patched side length
        # of 50, we should have 4 patches
        assert len(patches) == 4
        
        # We then check that the values are the same between the 
        # original matrix and the patch (using numpy slices)
        orig_np = rxr_arr.to_numpy().squeeze()
        patches_np = [patch.to_numpy().squeeze() for patch in patches]
        assert np.array_equal(orig_np[0:50, 0:50], patches_np[0])
        assert np.array_equal(orig_np[0:50, 50:100], patches_np[1])
        assert np.array_equal(orig_np[50:100, 0:50], patches_np[2])
        assert np.array_equal(orig_np[50:100, 50:100], patches_np[3])
        print("Passed\n")

    def test_non_even_patch(rxr_arr, drop_ends):
        """
        Tests non-evenly divisible patching.
        """
        print(f"Testing non-even patching w/ drop_ends = {drop_ends} ...")

        # Test with ends dropped
        patches = geo.patch_rxr(
            rxr_arr,
            side_len=40,
            drop_ends = drop_ends,
            return_index = False
        )
        if drop_ends:
            # With an original side length of 100, a patched side
            # length of 40, and drop_ends=True, should have 4 patches
            assert len(patches) == 4
            
            # We then check that the values are the same between the 
            # original matrix and the patch (using numpy slices)
            orig_np = rxr_arr.to_numpy().squeeze()
            patches_np = [patch.to_numpy().squeeze() for patch in patches]
            assert np.array_equal(orig_np[0:40, 0:40], patches_np[0])
            assert np.array_equal(orig_np[0:40, 40:80], patches_np[1])
            assert np.array_equal(orig_np[40:80, 0:40], patches_np[2])
            assert np.array_equal(orig_np[40:80, 40:80], patches_np[3])
        else:
            # With an original side length of 100, a patched side
            # length of 40, and drop_ends=False, should have 9 patches
            assert len(patches) == 9
            
            # We then check that the values are the same between the 
            # original matrix and the patch (using numpy slices)
            orig_np = rxr_arr.to_numpy().squeeze()
            patches_np = [patch.to_numpy().squeeze() for patch in patches]

            assert np.array_equal(orig_np[0:40, 0:40], patches_np[0])
            assert np.array_equal(orig_np[0:40, 40:80], patches_np[1])
            assert np.array_equal(orig_np[0:40, 60:], patches_np[2])
            assert np.array_equal(orig_np[40:80, 0:40], patches_np[3])
            assert np.array_equal(orig_np[40:80, 40:80], patches_np[4])
            assert np.array_equal(orig_np[40:80, 60:], patches_np[5])
            assert np.array_equal(orig_np[60:, 0:40], patches_np[6])
            assert np.array_equal(orig_np[60:, 40:80], patches_np[7])
            assert np.array_equal(orig_np[60:, 60:], patches_np[8])
        
        print("Passed\n")
    
    def test_overlapping_patches(
        rxr_arr, 
        drop_ends
    ):
        """
        Tests overlapping patching.
        """
        print(f"Testing overlapping patching w/ drop_ends = {drop_ends} ...")

        # Test with ends dropped
        patches = geo.patch_rxr(
            rxr_arr,
            side_len=30,
            drop_ends = drop_ends,
            return_index = False,
            overlap_px=2
        )

        if drop_ends:
            assert len(patches) == 9
            
            # We then check that the values are the same between the 
            # original matrix and the patch (using numpy slices)
            orig_np = rxr_arr.to_numpy().squeeze()
            patches_np = [patch.to_numpy().squeeze() for patch in patches]
            
            assert np.array_equal(orig_np[0:30, 0:30], patches_np[0])
            assert np.array_equal(orig_np[0:30, 28:58], patches_np[1])
            assert np.array_equal(orig_np[0:30, 56:86], patches_np[2])

            assert np.array_equal(orig_np[28:58, 0:30], patches_np[3])
            assert np.array_equal(orig_np[28:58, 28:58], patches_np[4])
            assert np.array_equal(orig_np[28:58, 56:86], patches_np[5])

            assert np.array_equal(orig_np[56:86, 0:30], patches_np[6])
            assert np.array_equal(orig_np[56:86, 28:58], patches_np[7])
            assert np.array_equal(orig_np[56:86, 56:86], patches_np[8])
        else:
            assert len(patches) == 16
            
            # We then check that the values are the same between the 
            # original matrix and the patch (using numpy slices)
            orig_np = rxr_arr.to_numpy().squeeze()
            patches_np = [patch.to_numpy().squeeze() for patch in patches]

            assert np.array_equal(orig_np[0:30, 0:30], patches_np[0])
            assert np.array_equal(orig_np[0:30, 28:58], patches_np[1])
            assert np.array_equal(orig_np[0:30, 56:86], patches_np[2])
            assert np.array_equal(orig_np[0:30, 70:], patches_np[3])

            assert np.array_equal(orig_np[28:58, 0:30], patches_np[4])
            assert np.array_equal(orig_np[28:58, 28:58], patches_np[5])
            assert np.array_equal(orig_np[28:58, 56:86], patches_np[6])
            assert np.array_equal(orig_np[28:58, 70:], patches_np[7])

            assert np.array_equal(orig_np[56:86, 0:30], patches_np[8])
            assert np.array_equal(orig_np[56:86, 28:58], patches_np[9])
            assert np.array_equal(orig_np[56:86, 56:86], patches_np[10])
            assert np.array_equal(orig_np[56:86, 70:], patches_np[11])

            assert np.array_equal(orig_np[70:, 0:30], patches_np[12])
            assert np.array_equal(orig_np[70:, 28:58], patches_np[13])
            assert np.array_equal(orig_np[70:, 56:86], patches_np[14])
            assert np.array_equal(orig_np[70:, 70:], patches_np[15])
        
        print("Passed\n")
        

    tmp_f_dir = tempfile.gettempdir()
    tmp_f_name = os.urandom(24).hex() + '.tif'
    fp = os.path.join(tmp_f_dir, tmp_f_name)

    arr = create_diagonal_gradient()
    save_test_raster(arr, fp)

    # The raster file to be used in testing the patching
    rxr_arr = rxr.open_rasterio(fp)

    test_basic_patch(rxr_arr, drop_ends=True)
    test_basic_patch(rxr_arr, drop_ends=False)
    test_non_even_patch(rxr_arr, drop_ends=True)
    test_non_even_patch(rxr_arr, drop_ends=False)
    test_overlapping_patches(rxr_arr, drop_ends=True)
    test_overlapping_patches(rxr_arr, drop_ends=False)

    os.remove(fp)

test_patch_rxr()