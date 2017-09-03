import unittest
import os
import numpy as np
import math

# import study_parser_v1 as study_parser        # Use this import to test the first version of study_parser
import study_parser_v2_parallel as study_parser
from full_pipeline_v3 import DataManager, StudyDataset


class DataTests(unittest.TestCase):
    def setUp(self):
        """
        Initialize all the settings needed to run the tests
        """
        # Init settings
        self.dicom_files_folder = "final_data/dicoms/SCD0000101"
        self.contour_files_folder = "final_data/contourfiles/SC-HF-I-1/i-contours/"
        self.num_images_study1 = 240
        self.dicom_files_folder_2 = "final_data/dicoms/SCD0000201"
        self.contour_files_folder_2 = "final_data/contourfiles/SC-HF-I-2/i-contours/"
        self.num_images_study2 = 260
        self.img_size = (256, 256)
        self.contour_mask_file_format= "IM-0001-{:04}-icontour-manual.txt"
        self.contour_file_name = os.path.join(self.contour_files_folder, self.contour_mask_file_format.format(48))
        self.previously_saved_mask_file = os.path.join(self.contour_files_folder, "IM-0001-0048-icontour-manual-mask.npy")
        self.total_images_in_study = 240

    def testReadSingleDicomFile(self):
        """
        Check the first dicom image in the study
        """
        file_name = os.path.join(self.dicom_files_folder, "1.dcm")
        self.assertTrue(os.path.exists(file_name), "File '{}' not found. Please review test data folders".format(file_name))
        pixels = study_parser.parse_dicom_file(file_name)
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
        coords = study_parser.parse_contour_file(self.contour_file_name)
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
        mask = study_parser.poly_to_mask(coords, 10, 10)
        # Make sure the inner content is as expected
        self.assertTrue(mask[1:7, 1:7].sum() == 36)
        # Make sure there is no any other positive pixel in the mask
        self.assertTrue(mask.sum() == 36)

    def testCompareWithKnownMask(self):
        """
        Test the method to get a boolean mask from a list of coords, using as a reference a mask numpy array that
        was previously validated
        """
        coords = study_parser.parse_contour_file(self.contour_file_name)
        # Get a mask from the contour
        mask1 = study_parser.poly_to_mask(coords, 256, 256)
        # Load a previously saved mask
        mask2 = np.load(self.previously_saved_mask_file)
        # Compare masks
        self.assertTrue(np.array_equal(mask1, mask2), "Previously calculated mask does not match with the current results")

    def testLoadFullStudy(self):
        i = 0
        for im, mask in study_parser.process_study(self.dicom_files_folder, self.contour_files_folder,
                                                   contour_file_mask_format=self.contour_mask_file_format):
            self.assertIsNotNone(im)
            self.assertIsNotNone(mask)
            self.assertIsInstance(im, np.ndarray, "Error in image {}".format(i))
            self.assertIsInstance(mask, np.ndarray, "Error in image {}".format(i))
            self.assertTrue(im.dtype == np.int16, "Error in image {}".format(i))
            self.assertTrue(mask.dtype == np.bool, "Error in image {}".format(i))
            i += 1
        self.assertTrue(i == self.total_images_in_study, "Expected a total of {} images, but got {}".
                        format(self.total_images_in_study, i))

    def testLoadFullStudyParallel(self):
        i = 0
        for im, mask in study_parser.process_study(self.dicom_files_folder, self.contour_files_folder,
                                                   contour_file_mask_format=self.contour_mask_file_format):
            self.assertIsNotNone(im)
            self.assertIsNotNone(mask)
            self.assertIsInstance(im, np.ndarray, "Error in image {}".format(i))
            self.assertIsInstance(mask, np.ndarray, "Error in image {}".format(i))
            self.assertTrue(im.dtype == np.int16, "Error in image {}".format(i))
            self.assertTrue(mask.dtype == np.bool, "Error in image {}".format(i))
            i += 1
        self.assertTrue(i == self.total_images_in_study, "Expected a total of {} images, but got {}".
                        format(self.total_images_in_study, i))

    def testFullPipeline(self):
        # Use a test dataset here. For demo purposes I am using the first two studies
        data_manager = DataManager(randomize_images=False)
        study1 = StudyDataset(self.dicom_files_folder, self.contour_files_folder,
                               contour_mask_file_format=self.contour_mask_file_format)
        self.assertTrue(study1.num_images == self.num_images_study1)
        data_manager.add_study(study1)
        study2 = StudyDataset(self.dicom_files_folder_2, self.contour_files_folder_2,
                              contour_mask_file_format=self.contour_mask_file_format)
        self.assertTrue(study2.num_images == self.num_images_study2)
        data_manager.add_study(study2)

        batch_size = 10
        # Calculate the total number of steps per epoch based on the based size
        total_num_images = data_manager.total_images
        self.assertTrue(total_num_images == study1.num_images + study2.num_images, "The total number of images does not match")

        steps_per_epoch = int(math.floor(total_num_images / float(batch_size)))

        num_step = 0
        for ims, masks in data_manager.get_next_batch(batch_size, self.img_size):
            # Size tests
            self.assertTrue(ims.shape[0] == batch_size)
            self.assertTrue(masks.shape[0] == batch_size)
            self.assertTrue(ims.shape[1:] == self.img_size)
            self.assertTrue(masks.shape[1:] == self.img_size)
            # Type tests (the batch should be floats in 0-1 range)
            self.assertTrue(ims.dtype == np.float32, "Wrong array type: {}".format(ims.dtype))
            self.assertTrue(masks.dtype == np.float32, "Wrong array type: {}".format(masks.dtype))
            self.assertTrue(ims.min() >= 0.0, "Min dataset value: {}. Expected: >= 0".format(ims.min()))
            self.assertTrue(ims.max() <= 1.0, "Max dataset value: {}. Expected: <= 1.0".format(ims.max()))
            self.assertTrue(masks.min() >= 0.0, "Min dataset value: {}. Expected: 0".format(masks.min()))
            self.assertTrue(masks.max() <= 1.0, "Max dataset value: {}. Expected: <= 1.0".format(masks.max()))
            expected_epoch = int(num_step / steps_per_epoch)
            self.assertTrue(expected_epoch == data_manager.num_epochs,
                    "Num step: {}. Expected epoch: {}. Real epoch: {}".format(num_step, expected_epoch,
                                                                              data_manager.num_epochs))
            num_step += 1

            if num_step > (steps_per_epoch * 2):
                return  # Test passed


def main():
    unittest.main()

if __name__ == '__main__':
    main()