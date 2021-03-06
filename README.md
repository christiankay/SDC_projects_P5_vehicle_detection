##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/cars.PNG "Typical no car and car images"
[image2]: ./output_images/car_nocar_hog.png "car and no car HOG features"
[image3]: ./output_images/MajorityVote.png "MajorityVote to SVM comparision"
[image4]: ./output_images/SVM_07_standarthog.png "SVM with HOG features"
[image5]: ./output_images/Majorityvote_07_standarthog.png "Majority vote classifier based on HOG features"
[image6]: ./output_images/learning_curve.png "Learning curves"
[image7]: ./output_images/test_all_scales.png "Searching windows"
[image8]: ./output_images/heat_without_thresh.png "heat map without threshold"
[image9]: ./output_images/heat_thresh.png "heat map after thresholding"
[image10]: ./output_images/SVM_08_para4.png "vehicle detection applied on test images"

[video1]: ./output_images/

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 74 through 91 of the file called `obj_detect.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally I found the parameter above used in YUV colorspace with all color channels. To reach high accuracy of the classifier as well as low prediction times in order to spend more computing for expensive sliding windows. 
The tables below show several run with different parameters and feature vector sizes. The perfomance of the trained classifier was evaluated by metrics like precicion, recall and f1-score (sk.metrics).

## Parameter 1
Feature vector length: (17760, 2112)
Using: 11 orientations 12 pixels per cell and 2 cells per block
33.72 Seconds to train SVC...
Test Accuracy of SVC =  0.9716

|             | precision | recall | f1-score | support |
|:-----------:|:---------:|:------:|:--------:|---------|
| not cars    | 0.9771    | 0.9681 | 0.9726   | 1852    |
| cars        | 0.9656    | 0.9753 | 0.9704   | 1700    |
| avg / total | 0.9716    | 0.9716 | 0.9716   | 3552    |


## Parameter 2
Feature vector length: (17760, 2112)
Using: 11 orientations 8 pixels per cell and 1 cells per block
37.28 Seconds to train SVC...
Test Accuracy of SVC =  0.9727

|             | precision | recall | f1-score | support |
|:-----------:|:---------:|:------:|:--------:|---------|
| not cars    | 0.9724    | 0.9740 | 0.9732   | 1811    |
| cars        | 0.9730    | 0.9713 | 0.9721   | 1741    |
| avg / total | 0.9727    | 0.9727 | 0.9727   | 3552    |

## Parameter 3 (video_1_hog_only)
Number of HOG feature features 6468
Feature vector length: (17760, 6468)
Using: 11 orientations 8 pixels per cell and 2 cells per block
#### 114.05 Seconds to train SVC...
#### Test Accuracy of SVC =  0.9834

|             | precision | recall | f1-score | support |
|:-----------:|:---------:|:------:|:--------:|---------|
| not cars    | 0.9839    | 0.9833 | 0.9836   | 1798    |
| cars        | 0.9829    | 0.9835 | 0.9832   | 1754    |
| avg / total | 0.9834    | 0.9834 | 0.9834   | 3552    |



## Parameter 4 (video_2)
Number of spatial features 768
Number of histogram features 96
Number of HOG feature features 972
Feature vector length: (17760, 1836)
Using: 9 orientations 16 pixels per cell and 2 cells per block
#### 19.49 Seconds to train SVC...
#### Test Accuracy of SVC =  0.993

|             | precision | recall | f1-score | support |
|:-----------:|:---------:|:------:|:--------:|---------|
| not cars    | 0.9918    | 0.9945 | 0.9932   | 1830    |
| cars        | 0.9942    | 0.9913 | 0.9927   | 1722    |
| avg / total | 0.9930    | 0.9930 | 0.9930   | 3552    |


Parameter 3 was found first but the prediction times were very slow due to a high amount of features (HOG) that needs to be calculated which also results in a complex SVM model. Using histogram features (32 bins) as well as color binned features (resize to (16,16)) a much faster classifier with even higher accuracy were found.


##### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear support vector machine using HOG features (line 421 trough 504 within the function `training()`), histogram and spatial features with high accuracy and fast prediction rates. I've also trained a majority vote classifier based on three different classifiers. The MVC was composed of a logistic regression classifier, a linear SVM and an k-nearest-neightbor classifier. The performance can be seen in the following test images:

The metric values might increase a little but on the other hand the system was approxemately two times slower than the single linear-SVM approach. Evaltion of both are shown in the images below 

Confusion matrix for SVM and MVC:

Confusion matrix
![alt text][image3]

Linear SVM:
![alt text][image4]

Majority classifier:
![alt text][image5]

The following image shows the accuracy over the amount of training data used. It shows that it might be possible to slightly increase the prediction accuracy with more training data. Due to the small differences between the performance on training data and test data, I didn't recognize an overfitting problem of the model at this point. 

![alt text][image6]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To find cars in an image that has different perspectives were cars could appear in one need to search for trained objects in several positions and scales. In this project the follwing setup including 309 windows per image with an overlap of 50% was used. The implemention and corresponding y ranges can be seen in line 722 through 852 in file `obj_detecct.py`. This approach is an adaption of the basic idea by (jeremy-shannon) on github.


| ###Searching of | 39 windows completed### @ scale 1.0 (64x64 pixel) |
|-----------------|---------------------------------------------------|
| ###Searching of | 39 windows completed### @ scale 1.0               |
| ###Searching of | 39 windows completed### @ scale 1.0               |
| ###Searching of | 39 windows completed### @ scale 1.0               |

| ###Searching of | 25 windows completed### @ scale 1.5 |
|-----------------|-------------------------------------|
| ###Searching of | 25 windows completed### @ scale 1.5 |
| ###Searching of | 25 windows completed### @ scale 1.5 |

| ###Searching of | 19 windows completed### @ scale 2.0 |
|-----------------|-------------------------------------|
| ###Searching of | 19 windows completed### @ scale 2.0 |
| ###Searching of | 19 windows completed### @ scale 2.0 |

| ###Searching of | 10 windows completed### @ scale 3.5 |
|-----------------|-------------------------------------|
| ###Searching of | 10 windows completed### @ scale 3.5 |

Overall 309 windows

![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. At the I summed up all found car rectangles and produced a heat map based on udacity's suggestion. After that I applied a threshold function based on maximum heatmap values:

Raw heat map showing all found car bounderies
![alt text][image8]

Threshold (line 835) applied on heat map before: max(( np.max(heat) / 2 , 2))) * 0.9
![alt text][image9]

Here are some example images of the final pipeline outcome:

![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap based on the last 15 frames and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected (line 821 through 850).


---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Detection of cars in camera images in changing conditions is a difficult task. The sliding window method is expensiveand a massive parallelization would be neccassary for real-time applications. And this method is likely to find a lot of false positives, even averaging frames. Maybe more sofiscticated "hand crafted" features are needed or even deep learning feature extraction.

The described method is likely to fail in cases where it wasn’t trained for: a bicycle, a pedestrian, trucks, or even in offroad environment.

More cameras (sensor) and other technology like car-2-car communication would help to detect vehicles close to the car.

I would like to thank Udacity for providing this high level challenge and avaluable guidance on this term. 

