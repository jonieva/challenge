#########################################################################################
# StudyDataset will contain the operations to extract images from a single DICOM study
# Author: Jorge Onieva
#########################################################################################

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
        """
        Given a DICOM file and a contour file, return the image as a numpy array and a numpy boolean mask
        :param dicom_file: path to the DICOM file
        :param contour_file: path to the DICOM contour file. If the file does not exist, the method returns a blank mask
        :return: tuple wity numpy array image and mask
        """
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
            # Return None to make it more robust, as this method may be run in parallel
            return None

    def get_image_and_mask(self, index, normalize=False, resize_shape=None):
        """
        Get a numpy array and a mask for the dicom image specified by "index"
        :param index: dataset image index
        :param normalize: when True, normalize the images and mask to a 0.0-1.0 range to work with Deep Learning models
        :param resize_shape: Tuple with the desired size for images-masks
        :return:
        """
        dicom_file = self.dicom_images_paths[index]
        image_ix = int(os.path.basename(dicom_file)[:-4])
        # Build the full path to the contour file, based on the contour images file name format
        contour_file = os.path.join(self.contour_files_folder, self.__contour_mask_file_format__.format(image_ix))
        img, mask = self._process_dicom_entry_(dicom_file, contour_file)
        if normalize:
            # Normalize to a [0.0-1.0] range and resize if needed
            img, mask = self.normalize(img, mask, resize_shape=resize_shape)
        return img, mask

    def normalize(self, img, mask, resize_shape=None):
        """
        Normalize an image and a mask to float32 in a [0.0-1.0] range to work with Deep Learning models
        If resize_shape is not None, the images-masks will be resized
        :param img: numpy array with the image
        :param mask: numpy array with the mask
        :param resize_shape: tuple with the shape of the images or None
        :return: tuple with (image,mask) normalized
        """
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