import unittest
from study_parser_parallel import *

class DataTests(unittest.TestCase):
    def setUp(self):
        """
        Initialize all the settings needed to run the tests
        """
        # Init folders
        self.dicom_files_folder = "final_data/dicoms/SCD0000101"
        self.contour_files_folder = "final_data/contourfiles/SC-HF-I-1/i-contours/"
        self.contour_file_mask_format="IM-0001-{:04}-icontour-manual.txt"
        self.contour_file_name = os.path.join(self.contour_files_folder, self.contour_file_mask_format.format(48))
        self.previously_saved_mask_file = os.path.join(self.contour_files_folder, "IM-0001-0048-icontour-manual-mask.npy")
        self.total_images_in_study = 240

    def testReadSingleDicomFile(self):
        """
        Check the first dicom image in the study
        """
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
        """
        Read a contour file and get a list of coords
        """
        self.assertTrue(os.path.exists(self.contour_file_name), "File '{}' not found. Please review test data folders".
                        format(self.contour_file_name))
        coords = parse_contour_file(self.contour_file_name)
        self.assertIsNotNone(coords, "Coords not found in '{}'".format(self.contour_file_name))
        # Check we have the expected number of coords
        self.assertTrue(len(coords) == 150, "Expected number of coords: 150. Found: {}".format(len(coords)))
        # Check first and last known values in the contours file
        self.assertTrue(coords[0][0] == 120.50 and coords[0][1] == 137.50 \
                        and coords[149][0] == 120.50 and coords[149][1] == 138.00)

    def testParseMaskSynthetic(self):
        """
        Test the method to get a boolean mask from a list of coords, using a small synthetic dataset
        """
        # Build a squared mask
        coords = [(0, 0), (0, 7), (7, 7), (7,0)]
        mask = poly_to_mask(coords, 10, 10)
        # Make sure the inner content is as expected
        self.assertTrue(mask[1:7, 1:7].sum() == 36)
        # Make sure there is no any other positive pixel in the mask
        self.assertTrue(mask.sum() == 36)

    def testCompareWithKnownMask(self):
        """
        Test the method to get a boolean mask from a list of coords, using as a reference a mask numpy array that
        was previously validated
        """
        coords = parse_contour_file(self.contour_file_name)
        # Get a mask from the contour
        mask1 = poly_to_mask(coords, 256, 256)
        # Load a previously saved mask
        mask2 = np.load(self.previously_saved_mask_file)
        # Compare masks
        self.assertTrue(np.array_equal(mask1, mask2), "Previously calculated mask does not match with the current results")

    def testLoadFullStudy(self):
        i = 0
        for im, mask in process_study(self.dicom_files_folder, self.contour_files_folder,
                                        contour_file_mask_format=self.contour_file_mask_format):
            self.assertIsNotNone(im)
            self.assertIsNotNone(mask)
            self.assertIsInstance(im, np.ndarray, "Error in image {}".format(i))
            self.assertIsInstance(mask, np.ndarray, "Error in image {}".format(i))
            self.assertTrue(im.dtype == np.int16, "Error in image {}".format(i))
            self.assertTrue(mask.dtype == np.bool, "Error in image {}".format(i))
            i += 1
        self.assertTrue(i == self.total_images_in_study, "Expected a total of {} images, but got {}".
                        format(self.total_images_in_study, i))

def main():
    unittest.main()

if __name__ == '__main__':
    main()