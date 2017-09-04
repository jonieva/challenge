#################################################################################################################
# The DataManager will coordinate all the operations in the pipeline, extracting randomized data from different
# studies asynchronously
# Author: Jorge Onieva
#################################################################################################################

import numpy as np
import threading
import concurrent.futures


class DataManager(object):
    def __init__(self, randomize_images=True):
        """

        :param randomize_images:
        """
        # List of tuples with (Study-Index), where Index will contain the number of images where the study ends.
        # Example, if we have 2 studies with 10 and 20 images, this list will contain [(Study1, 10), (Study2, 30)]
        # This will be used to retrieve images from several studies. Ex: the image 15 in this case would be the
        # image 5 of Study2
        self.__studies_indexes__ = []
        self.__total_images__ = 0                   # Total number of images
        self.__randomized_index_list__ = None       # List with (typically) randomized indexes to retrieve images randomly
        self.__current_image_index__ = None         # Global image index
        self.randomize_images = randomize_images    # Bool that indicate if the DataManager should return the images in a random order
        self.lock = threading.Lock()                # Object used to make safe multithreading ops
        self.__num_epochs__ = 0                     # Current epoch number

    # Read only properties
    @property
    def total_images(self):
        return self.__total_images__

    @property
    def num_epochs(self):
        return self.__num_epochs__

    def add_study(self, study_ds):
        """
        Add a new study to the global dataset
        :param study_ds: StudyDataset object
        """
        # Increase the total number of images
        self.__total_images__ += study_ds.num_images
        # Save the index where the image index for this study will end
        self.__studies_indexes__.append((study_ds, self.__total_images__))

    def reset_indexes(self):
        """
        Initialize the list of indexes that will be used to retrieve images from the dataset.
        Please note that we are updating internal variables in the dataset, so if we are loading the images in parallel
        mode, this method should be invoked in a safe way (use self.lock)
        """
        # Generate an index for all the images in the dataset
        self.__randomized_index_list__ = np.arange(self.__total_images__)
        if self.randomize_images:
            # Shuffle the index
            np.random.shuffle(self.__randomized_index_list__)
        self.__current_image_index__ = 0

    def _get_study_image_index_(self, global_index):
        """
        Get the study and the image index for that study, given a global index.
        Ex: If we have 2 studies with 10 and 20 images, __studies_indexes__ will contain [(Study1, 10), (Study2, 30)]
        # If global_index==15, this would correspond to the image 5 of Study2, so the method would return (Study2, 5)
        :param global_index:
        :return: tuple with StudyDataset-Image_index in the study
        """
        if len(self.__studies_indexes__) == 0:
            raise Exception("No studies available. Please load some study")

        for i in range(len(self.__studies_indexes__)):
            study_index = self.__studies_indexes__[i][1]
            if global_index < study_index:
                if i == 0:
                    # First study, just return the index searched for
                    return self.__studies_indexes__[i][0], global_index
                # Return the difference between the searched index and rhe previous study limit
                return self.__studies_indexes__[i][0], global_index - self.__studies_indexes__[i - 1][1]

    def get_next_batch(self, batch_size, im_size):
        """
        Iterator that gets a new batch of images and masks.
        Both arrays will be normalized to a 0.0-1.0 range to work with Deep Learning models
        :param batch_size: number of images-masks
        :param im_size: shape of the images-masks. The images will be resized to this shape
        :return: This method return as tuple of images-masks in an iterator
        """
        # Randomize the images (if needed)
        self.reset_indexes()

        while True:
            with self.lock:
                # Check that we didn't loop over all the images in the dataset
                # This operation must be thread-safe in case that two threads get to the end at the same time
                if self.__current_image_index__ + batch_size > self.__total_images__:
                    # End of epoch. Shuffle the dataset
                    self.reset_indexes()
                    self.__num_epochs__ += 1

                # Get the indexes of the images that we are processing in this batch.
                # This still has to be single-threaded not to process the same indexes in different threads
                indexes = self.__randomized_index_list__[self.__current_image_index__:(self.__current_image_index__ + batch_size)]
                self.__current_image_index__ += batch_size

            # Calculate the images that will be loaded for each study
            study_image_index = []
            for index in indexes:
                study_image_index.append(self._get_study_image_index_(index))

            # Initialize the arrays
            ims = np.zeros([batch_size, im_size[0], im_size[1]], dtype=np.float32)
            masks = np.zeros([batch_size, im_size[0], im_size[1]], dtype=np.float32)

            # Load the data asynchronously. The number of workers could be parametrized
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                for i in range(batch_size):
                    # Run studyDataset.get_image_and_mask with normalized images and a specified image size
                    futs = [executor.submit(study.get_image_and_mask, index, True, im_size)
                            for study, index in study_image_index]
                i = 0
                for future in concurrent.futures.as_completed(futs):
                    result = future.result()
                    if result is not None:
                        # If result is None for a system error, just return a blank image-mask
                        ims[i] = result[0]
                        masks[i] = result[1]
            # Return the final results in an iterator
            yield ims, masks

