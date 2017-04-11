# Vehicle Detection

This document describes the implamentation of the project.
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)
[image1]: ./output_images/carnocar_1.png
[image2]: ./output_images/color_spaces_2.png
[image3]: ./output_images/car_hub_3.png
[image4]: ./output_images/car_windows_4.png
[image5]: ./output_images/car_windows_5.png
[image6]: ./output_images/car_windows_6.png
[image7]: ./output_images/car_windows_7.png
[image8]: ./output_images/car_windows_8.png
[image9]: ./output_images/car_detection_video_9.png
[image10]: ./output_images/car_detection_video_10.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This project contains the following main files:
* CarDetection.ipynb
* window_slider.py
* search_classify.py
* heat_maps.py
* cardetection.py
* lesson_functions.py

The notebook contains step by step code to generate the supported images as well as videos.  It may use the code in the py files or show it in the notebook directly without using the functions created in those py files as a way to clearly show some of the procedures used directly on the notebook.

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines #6 through #23 of the file called `lesson_functions.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of the different color spaces for `car` and `no car` images.

![alt text][image2]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, specially different number of orientations.  I found that 9 orientations for HUB was showing the best results.
Other parameters were also tested and tuned accordingly to final results when using SVG classifier.

#### 3. Training the classifier.

I trained a linear SVM using sklearn `LinearSVC` and `StandardScaler` to scale the images properly.

```
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
...
svc = LinearSVC()
svc.fit(X_train, y_train)
````

The features used were concatenated from the previous steps using different set of feature sources,  hog features, spatial and histogram features were used, making a total of `8412` features per sample.

The number of samples available were:
* Cars: 8792 images
* No Cars: 8968 images

### Sliding Window Search

#### 1. Windows scales

The function `slide_window` in `window_slider.py` creates the different windows based on size and range positions.

Different sizes were tested trying to accomodate the expected sizes of the cars given the example images.  Only some regions of the window were used per window size to avoid searching in images regions where no cars are expected.

This image shows the window positions without sliding:

![alt text][image4]

This image shows the window positions with sliding:

![alt text][image5]


#### 2. Pipeline

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Pipeline

Pipeline is using all channels from YCrCB conversion as well as HOG and spatial features.  This features are obtained for the different sliding windows defined above.  For each window we obtain a result based on SVM model.
The windows are then use to create a heatmap which was then thresholded to detect the false positives from the model.  As this is applied from each frame, the previous windows from last hot windows from last 15 frames are also stored to even reduce even more those false positives by getting all the hot windows from last frames and increasing the threshold levels.

I this picture we can see some of the windows detected by the model.  Many of those are false positives.

![alt text][image6]

Those windows are used to create a heat map of regions with high probability of finding a car.  These heatmap regions allow also to recreate the windows covering those warmed areas
![alt text][image7]

Finally appliying thresholding by last 15 frames we can drop the false positives and accurately find the car location.

![alt text][image8]

In this picture we see the final 2 windows in blue (after the heatmap) and the initial original windows found by the model.
![alt text][image9]

Below an snapshot of the video submission, with the embeded hotmap.

#### 2. Video Output

![alt text][image10]

Here's a [link to my video result](./project_video_output.mp4)


---

### Discussion

* Even using several frames to reduce the false positives there's still sometimes that those happen.  Additional models (convnets?) would definitively help in reducing those false positives.  Additionally have some smarter region clipping to avoid detecting cars from the opposite directions.

* Performance is not good if it would need realtime processing.  Some part of the code should be further optimize.



