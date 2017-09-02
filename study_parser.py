import os
import dicom
import warnings
import glob
import numpy as np
from PIL import Image, ImageDraw

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

def process_study(dicom_images_folder, contour_files_folder, contour_file_mask_format="IM-0001-{:04}-icontour-manual.txt"):
    """
    In a study, receive a folder that contains the dicom images and other folder that contains contour files.
    Optionally, we can specify the file name format of the contour files. In this case we have set by default the
    original data, just to respect the API of the problem
    :param dicom_images_folder: path to the folder that contains the dicom images
    :param contour_files_folder: path to the folder that contains the contour files
    :param contour_file_mask_format: file name format for the contour files. By default, we will use the original
    data format, ex: IM-0001-0048-icontour-manual.txt
    :return: iterator that return tuples with (numpy_image, contour_mask)
    """
    # Create index to iterate over all the dicom files
    files = glob.glob(os.path.join(dicom_images_folder, "*.dcm"))
    index = range(len(files))

    for i in index:
        try:
            # Read the dicom file
            dicom_pixel_array = parse_dicom_file(files[i])
            im_width = dicom_pixel_array.shape[0]
            im_height = dicom_pixel_array.shape[1]

            # Process the contour file
            # Get the image index from the image name
            image_ix = int(os.path.basename(files[i])[:-4])
            # Build the full path to the contour file, based on the contour images file name format
            contour_file = os.path.join(contour_files_folder, contour_file_mask_format.format(image_ix))

            # Default mask in case that there is not a valid contour: blank mask
            mask = np.zeros([im_width, im_height], dtype=np.bool)

            # Try to search for the img in the contour folder
            if os.path.exists(contour_file):
                try:
                    coords = parse_contour_file(contour_file)
                    mask = poly_to_mask(coords, im_width, im_height)
                except Exception as ex:
                    # In case there was any problem with the mask, just return a blank mask
                    warnings.warn("Contour could not be parsed in file '{}'".format(contour_file))
        except Exception as ex:
            warnings.warn("Error processing DICOM image '{}': {}".format(files[i], ex))
            # There was some fatal error. Search for the next image
            continue
        # Return image, mask tuple in an iterator
        yield dicom_pixel_array, mask

