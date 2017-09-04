import os
import dicom
import warnings
import glob
import numpy as np
from PIL import Image, ImageDraw

import concurrent.futures

def parse_contour_file(filename):
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

def parse_dicom_file(filename):
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

def poly_to_mask(polygon, width, height):
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

def mask_from_contour_file(file_name, width, height):
    """
    From a contour file, get a boolean mask
    :param file_name: name of the contour file that encodes a polygon
    :param width: image width
    :param height: image height
    :return: numpy array of boolean that will be used as a mask
    """
    coords = parse_contour_file(file_name)
    return poly_to_mask(coords, width, height)

def process_dicom_entry(dicom_file, contour_file):
    """
    Given a DICOM file and a contour file, return the image as a numpy array and a numpy boolean mask
    :param dicom_file: path to the DICOM file
    :param contour_file: path to the DICOM contour file. If the file does not exist, the method returns a blank mask
    :return: tuple with numpy array image and mask
    """
    try:
        # Read the dicom file
        dicom_pixel_array = parse_dicom_file(dicom_file)
        im_width = dicom_pixel_array.shape[0]
        im_height = dicom_pixel_array.shape[1]

        # Default mask in case that there is not a valid contour: blank mask
        mask = np.zeros([im_width, im_height], dtype=np.bool)

        # Try to search for the img in the contour folder
        if os.path.exists(contour_file):
            coords = parse_contour_file(contour_file)
            mask = poly_to_mask(coords, im_width, im_height)
        return dicom_pixel_array, mask
    except Exception as ex:
        if dicom_file is None:
            warnings.warn("Error in image: {}".format(ex))
        else:
            warnings.warn("Error in image '{}': {}".format(dicom_file, ex))
        return None

def process_study(dicom_images_folder, contour_files_folder,
                  contour_file_mask_format="IM-0001-{:04}-icontour-manual.txt",
                  num_images=10):
    """
    In a study, receive a folder that contains the DICOM images and other folder that contains contour files.
    Optionally, we can specify the file name format of the contour files. In this case we have set by default the
    original data, just to respect the API of the problem
    :param dicom_images_folder: path to the folder that contains the dicom images
    :param contour_files_folder: path to the folder that contains the contour files
    :param contour_file_mask_format: file name format for the contour files. By default, we will use the original
    data format, ex: IM-0001-0048-icontour-manual.txt
    :param num_images: number of images that are going to be returned
    :return: iterator that return a list of tuples with (numpy_image, contour_mask)
    """
    # Check all the files that must be read
    files = glob.glob(os.path.join(dicom_images_folder, "*.dcm"))
    params = []
    for dicom_file in files:
        # Get the image index from the image name
        image_ix = int(os.path.basename(dicom_file)[:-4])
        # Build the full path to the contour file, based on the contour images file name format
        contour_file = os.path.join(contour_files_folder, contour_file_mask_format.format(image_ix))
        params.append((dicom_file, contour_file))

    ix = 0
    while ix < len(files):
        results = [None] * num_images
        j = 0
        # Load num_images asynchronously
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futs = [executor.submit(process_dicom_entry, dicom_file, contour_file)
                    for (dicom_file, contour_file) in params[ix:ix+num_images]]
            for future in concurrent.futures.as_completed(futs):
                result = future.result()
                results[j] = result
                j += 1
            ix += num_images
            yield results
