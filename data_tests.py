import unittest
from dataset import *

class DataTests(unittest.TestCase):
    def setUp(self):
        # Init folders
        self.dicom_files_folder = "final_data/dicoms/SCD0000101"
        self.contour_files_folder = "final_data/contourfiles/SC-HF-I-1/i-contours/"
        self.previously_saved_mask_file = "final_data/IM-0001-0048-icontour-manual-mask.npy"
        self.contour_file_mask_format="IM-0001-{:04}-icontour-manual.txt"
        self.contour_file_name = os.path.join(self.contour_files_folder, self.contour_file_mask_format.format(48))

    def testReadSingleDicomFile(self):
        file_name = os.path.join(self.dicom_files_folder, "1.dcm")
        self.assertTrue(os.path.exists(file_name), "File '{}' not found. Please review test data folders".format(file_name))
        pixels = parse_dicom_file(file_name)
        self.assertIsNotNone(pixels, "Could not find any pixels data in dicom file '{}'".format(file_name))
        self.assertIsInstance(pixels, np.ndarray, "The returned pixels in '{}' are not a numpy array".format(file_name))
        self.assertTrue(pixels.dtype == np.int16, "Pixels data type is '{}'. Expected: int16".format(pixels.dtype))
        # Check image size
        self.assertTrue(pixels.shape == (256, 256), "Expected image size: 256x256. Image size: {}".format(pixels.shape))
        # Check this is not a "null" image by testing a known pixel value
        self.assertTrue(pixels[100, 100] == 32, "Data in '{}' is not as expected. Corrupt image?".format(file_name))

    def testParseContourFile(self):
        # Use for the test the first contours file
        self.assertTrue(os.path.exists(self.contour_file_name), "File '{}' not found. Please review test data folders".
                        format(self.contour_file_name))
        coords = parse_contour_file(self.contour_file_name)
        self.assertIsNotNone(coords, "Coords not found in '{}'".format(self.contour_file_name))
        # Check we have the expected number of coords
        self.assertTrue(len(coords) == 150, "Expected number of coords: 150. Found: {}".format(len(coords)))
        # Check first and last known values in the contours file
        self.assertTrue(coords[0][0] == 120.50 and coords[0][1] == 137.50 \
                        and coords[149][0] == 120.50 and coords[149][1] == 138.00)

    def testParseMaskSyntetic(self):
        # Build a squared mask
        coords = [(0, 0), (0, 7), (7, 7), (7,0)]
        mask = poly_to_mask(coords, 10, 10)
        # Make sure the inner content is as expected
        self.assertTrue(mask[1:7, 1:7].sum() == 36)
        # Make sure there is no any other positive pixel in the mask
        self.assertTrue(mask.sum() == 36)

    def testCompareWithKnownMask(self):
        file_name = os.path.join(self.contour_files_folder, self.contour_file_mask_format.format(48))
        coords = parse_contour_file(file_name)
        # Get a mask from the contour
        mask1 = poly_to_mask(coords, 256, 256)
        # Load a previously saved mask
        mask2 = np.load(self.previously_saved_mask_file)
        # Compare masks
        self.assertTrue(np.array_equal(mask1, mask2), "Previously calculated mask does not match with the current results")

    def testLoadFullStudy(self):
        self.fail("Not implemented!")

def main():
    unittest.main()

if __name__ == '__main__':
    main()