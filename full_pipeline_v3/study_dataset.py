import os
import dicom
import warnings
import glob
import numpy as np

from PIL import Image, ImageDraw
from scipy.misc import imresize

class StudyDataset(object):
    def __init__(self, dicom_files_folder, contour_files_folder, contour_mask_file_format=None):
        """
        :param dicom_files_folder:
        :param contour_files_folder:
        :param contour_mask_file_format: string format for the contour file names. Default: IM-0001-{:04}-icontour-manual.txt
        """
        self.__dicom_files_folder__ = dicom_files_folder
        self.__contour_files_folder__ = contour_files_folder
        self.__contour_mask_file_format__ = contour_mask_file_format if contour_mask_file_format is not None \
                                            else "IM-0001-{:04}-icontour-manual.txt"
        self._dicom_images_paths_ = None

    # Read only properties
    @property
    def dicom_files_folder(self):
        return self.__dicom_files_folder__

    @property
    def contour_files_folder(self):
        return self.__contour_files_folder__

    @property
    def contour_mask_file_format(self):
        return self.__contour_mask_file_format__

    # Lazy load properties
    @property
    def dicom_images_paths(self):
        if self._dicom_images_paths_ is None:
            self._dicom_images_paths_ = glob.glob(os.path.join(self.__dicom_files_folder__, "*.dcm"))
        return self._dicom_images_paths_

    @property
    def num_images(self):
        return len(self.dicom_images_paths)


    def _parse_contour_file_(self, filename):
        """Parse the given contour filename

        :param filename: filepath to the contourfile to parse
        :return: list of tuples holding x, y coordinates of the contour
        """
        coords_lst = []

        with open(filename, 'r') as infile:
            for line in infile:
                coords = line.strip().split()

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                coords_lst.append((x_coord, y_coord))

        return coords_lst

    def _parse_dicom_file_(self, filename):
        """Parse the given DICOM filename

        :param filename: filepath to the DICOM file to parse
        :return: image data
        """
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image * slope + intercept
        return dcm_image

    def _poly_to_mask_(self, polygon, width, height):
        """Convert polygon to mask

        :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
         in units of pixels
        :param width: scalar image width
        :param height: scalar image height
        :return: Boolean mask of shape (height, width)
        """
        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
        mask = np.array(img).astype(bool)
        return mask


    def _process_dicom_entry_(self, dicom_file, contour_file):
        try:
            # Read the dicom file
            dicom_pixel_array = self._parse_dicom_file_(dicom_file)
            # Get the image dimensions
            im_width = dicom_pixel_array.shape[0]
            im_height = dicom_pixel_array.shape[1]

            # Default mask in case that there is not a valid contour: blank mask
            mask = np.zeros([im_width, im_height], dtype=np.bool)

            # Try to search for the img in the contour folder
            # If the file does not exist, return a blank mask
            if os.path.exists(contour_file):
                coords = self._parse_contour_file_(contour_file)
                mask = self._poly_to_mask_(coords, im_width, im_height)
            return dicom_pixel_array, mask
        except Exception as ex:
            if dicom_file is None:
                warnings.warn("Error in image: {}".format(ex))
            else:
                warnings.warn("Error in image '{}': {}".format(dicom_file, ex))
            # Return None to make it more robust, as this method will be run in parallel
            return None

    def get_image_and_mask(self, index, normalize=False, resize_shape=None):
        """
        Get a numpy array and a mask for the dicom image specified by "index"
        :param index: dataset image index
        :param normalize:
        :param resize_shape:
        :return:
        """
        try:
            dicom_file = self.dicom_images_paths[index]
            image_ix = int(os.path.basename(dicom_file)[:-4])
            # Build the full path to the contour file, based on the contour images file name format
            contour_file = os.path.join(self.contour_files_folder, self.__contour_mask_file_format__.format(image_ix))
            img, mask = self._process_dicom_entry_(dicom_file, contour_file)
            #if normalize:
            # Normalize to a 0-1 range
            img, mask = self.normalize(img, mask, resize_shape=resize_shape)
            return img, mask
        except Exception as ex:
            return None
            pass

    def normalize(self, img, mask, resize_shape=None):
        low_threshold = float(img.min())
        high_threshold = float(img.max())
        img = (img - low_threshold) / (abs(high_threshold - low_threshold) + 0.0000001)
        mask = mask.astype(np.float32)
        if resize_shape is not None and img.shape != resize_shape:
            # Resize image (chosen strategy: deform)
            img = imresize(img, resize_shape, mode='F')
            # Resize mask (make sure we always return 0.0 or 1.0 values)
            mask = np.around(imresize(mask, resize_shape, mode='F'), decimals=0)

        return img, mask

    # def process_study(self):
    #     """
    #     In a study, receive a folder that contains the DICOM images and other folder that contains contour files.
    #     Optionally, we can specify the file name format of the contour files. In this case we have set by default the
    #     original data, just to respect the API of the problem
    #     :param dicom_images_folder: path to the folder that contains the dicom images
    #     :param contour_files_folder: path to the folder that contains the contour files
    #     :param contour_mask_file_format: file name format for the contour files. By default, we will use the original
    #     data format, ex: IM-0001-0048-icontour-manual.txt
    #     :return: iterator that return tuples with (numpy_image, contour_mask)
    #     """
    #     # Check all the files that must be read
    #     files = glob.glob(os.path.join(dicom_images_folder, "*.dcm"))
    #     params = []
    #     for dicom_file in files:
    #         # Get the image index from the image name
    #         image_ix = int(os.path.basename(dicom_file)[:-4])
    #         # Build the full path to the contour file, based on the contour images file name format
    #         contour_file = os.path.join(contour_files_folder, contour_mask_file_format.format(image_ix))
    #         params.append((dicom_file, contour_file))
    #
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    #         futs = [executor.submit(process_dicom_entry, dicom_file, contour_file) for (dicom_file, contour_file) in params]
    #         for future in concurrent.futures.as_completed(futs):
    #             result = future.result()
    #             if result is not None:
    #                 yield result[0], result[1]
