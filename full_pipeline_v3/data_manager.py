import numpy as np
import threading
import concurrent.futures
import warnings
from .study_dataset import StudyDataset

class DataManager(object):
    def __init__(self, randomize_images=True):
        self._studies_ = []
        self.__total_images__ = 0
        self.__current_image_index__ = None
        self.__randomized_index_list__ = None
        self.lock = threading.Lock()
        self.randomize_images = randomize_images
        self.__num_epochs__ = 0

    @property
    def total_images(self):
        return self.__total_images__

    @property
    def num_epochs(self):
        return self.__num_epochs__

    def add_study(self, study_ds):
        # Get the number of images
        num_images = study_ds.num_images
        self.__total_images__ += num_images
        self._studies_.append((study_ds, self.__total_images__))
        # Save the index where the images for this study will start

    def reset_indexes(self):
        """
        Initialize the DataManager. All the studies must be added by now
        :return:
        """
        # Generate a random index for all the images in the dataset
        # This operation must be thread-safe!
        print("New epoch starting...")
        #with self.lock:
        self.__randomized_index_list__ = np.arange(self.__total_images__)
        if self.randomize_images:
            np.random.shuffle(self.__randomized_index_list__)
        self.__current_image_index__ = 0

    def _get_study_image_index_(self, global_index):
        """
        Get the study and the image index for that study, given a global index
        :param global_index:
        :return: tuple with Study-Image_index in the study
        """
        if len(self._studies_) == 0:
            raise Exception("No studies available. Please load some study")

        # if len(self._studies_) == 1:
        #     # Only one study. No need to keep track of more indexes
        #     return global_index

        for i in range(len(self._studies_)):
            study_index = self._studies_[i][1]
            if global_index < study_index:
                if i == 0:
                    return self._studies_[i][0], global_index
                return self._studies_[i][0], global_index - self._studies_[i-1][1]


    def get_next_batch(self, batch_size, im_size):
        """
        Get a new batch of images and masks.
        Both arrays will be normalized to a 0.0-1.0 range to work with Deep Learning models
        :param batch_size:
        :return:
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

            # Load the data asynchronously
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                for i in range(batch_size):
                    futs = [executor.submit(study.get_image_and_mask, index, False, im_size)
                            for study, index in study_image_index]
                i = 0
                for future in concurrent.futures.as_completed(futs):
                    result = future.result()
                    if result is not None:
                        # Otherwise just leave it blank
                        ims[i] = result[0]
                        masks[i] = result[1]
            # Return the final results in an iterator
            yield ims, masks

