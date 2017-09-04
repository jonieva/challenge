## Code challenge 2017/09/03

### General structure of the code:
* study_parser_v1.py. First version of the Part 1 requirements
* study_parser_v2.py. Parallel version of the Part 1 requirements. Still a simple implementation
* full_pipeline_v3. Part 2. Full pipeline in a more complex structure, with an object that handles a single DICOM study and a DataManager that coordinates the operations in a asynchronous way
* tests.py. Unit tests

All the tests assume that there is a folder named **final_data** with the same structure as the images that were provided for the challenge

### Python packages used.
The code has been implemented using Python 3.6.2 based on a Miniconda distribution.
Packages used:
* pillow
* pydicom (downloaded from conda-forge channel)
* scipy
* matplotlib (optional, just for internal visualization purposes)

### Part 1
#### 1.1
I used a small synthetic mask for tests where I could control all the values.

        coords = [(0, 0), (0, 7), (7, 7), (7,0)]
        mask = study_parser.poly_to_mask(coords, 10, 10)
        # Make sure the inner content is as expected
        self.assertTrue(mask[1:7, 1:7].sum() == 36)
        # Make sure there is no any other positive pixel in the mask
        self.assertTrue(mask.sum() == 36)

I also used matplotlib to visualize the image and the mask together:

    import matplotlib.pyplot as plt
    im = parse_dicom_file("final_data/dicoms/SCD0000101/48.dcm")
    mask = mask_from_contour_file("final_data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt", 256, 256)
    plt.imshow(im, 'gray', interpolation='none')
	plt.imshow(mask, 'Reds', interpolation='none', alpha=0.1)

#### 1.2
 I modified the method 'parse_dicom_file':
 * Instead of returning a dictionary, for efficiency in this example I think we just need the array image (the dictionary is not used at all)
 *
 * Also, personally I don't like very much the idea of returning None when a DICOM image cannot be read. I think this may lead to unexpected behaviour. If the developer wants to be robust to this kind of error, in my opinion it should be controlled from the outside in the method that is reading all the DICOM images.

#### 1.3
There is a second version of the pipeline that, instead of loading a single image-mask, loads a number of them in parallel. This could speed up the process, specially in environments where IO operations are expensive (ex: the data are in the cloud or in a NFS disk).

#### 1.4
As in any parallel execution, it is important to watch for deadlocks. In this case there shouldn't be any problems because the method that is extracting images and masks do not use any shared resource.

There could be issues tough in console operations because of the "warnings".

As an addition, I decided to make the "process_dicom_entry" robust to errors, in order not to crash the whole pipeline if there's an error in a single image.


### Part 2
#### 2.1.
For the full pipeline, I used the ProcessPoolExecutor, because I compared to the ThreadPoolExecutor and my code run faster in the former case. However, this can happen because all my data are stored locally and the IO operations are not expensive.

If the operations that are going to be run in parallel make intensive use of the CPU, the ProcessPoolExecutor should perform better. In a staging scenario, depending on the environment configuration, the ProcessPoolExecutor could be replaced by the ThreadPoolExecutor, more efficient when the bottleneck is in the I/O operations.

There are another options for multithreading like manual Threads and wait signals (kind of complex to manage) and asyncio for I/O operations.

#### 2.2.
I made use of a threading.Lock object to prevent that several processes access at the same time to variables that impact the behaviour of the full pipeline. For instance, when we are selecting the next batch of images.

#### 2.3.
In Part 1, parallel execution was embedded in the method that retrieves the images/masks. In this part, as we need to handle multiple studies, I decided that it made more sense to create a class that is in charge to calling asynchronously to the method that provides the images/masks, and that coordinates all the studies at the same time.

For a better code organization, I also created a class to handle the operations of a single study, pretty much the functionality in Part 1.

#### 2.4.
I created a unit test for the full pipeline ('testFullPipeline'), that goes over more than one epoch, to simulate the behaviour of a Deep Learning training.

#### 2.5.
We could possibly add a second level of parallelism: one for reading files that contain the contour images and another one to build the boolean mask. However, my intuition is that the overhead to create the parallel threads would be bigger than the improvement in the performance.

We could also adjust better the number of processes used during parallelism based on the system configuration.
