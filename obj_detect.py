# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:43:26 2017

@author: Chris
"""


import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
import pickle
import os
import csv
import sklearn
import warnings
from scipy.ndimage.measurements import label
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from moviepy.editor import VideoFileClip


#if Version(sklearn_version) < '0.18':
from sklearn.learning_curve import learning_curve
#else:
#    from sklearn.model_selection import learning_curve

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Define a function to return HOG features and visualization
class filter_detection():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_rects = [] 
        self.keep_last = 15
        self.rect_mean = []
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > self.keep_last:
            self.prev_rects = self.prev_rects[len(self.prev_rects)-self.keep_last:]
           
            
    
def convert_color(image, color_space='RGB'):
        # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image) 
    
    return feature_image

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):

    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(16, 16),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        single_image = False):
    
    
        
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    if single_image is True:
        
        imgs = [imgs]
        
    
    for file in imgs:
        
        file_features = []
#        print ("shape file", file.shape)
        image = mpimg.imread(file)
      #  print("image prop", image.shape, image.dtype)
     #   print("Max in img before color", np.max(image))
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image) 
        
        #feature_image = feature_image.astype(np.float32)/360
        #print("Max in img", np.max(feature_image))
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
         #   print("file_features",len(file_features))
        features.append(np.concatenate(file_features))
    print("Number of spatial features",len(spatial_features))
    print("Number of histogram features",len(hist_features))
    print("Number of HOG feature features",len(hog_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#def hog_extractor(img):
#    orient = 9
#    pix_per_cell = 8
#    cell_per_block = 2
#    
#    feature_array = hog(img, orientations=orient, 
#                        pixels_per_cell=(pix_per_cell, 
#                                         pix_per_cell), 
#                                         cells_per_block=(cell_per_block, cell_per_block), 
#                                         visualise=False, 
#                                         feature_vector=False)
                    
 #   return feature_array                    

def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles=False):
    
    # array of rectangles where cars were detected
    rectangles = []
    
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)   
    
    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # select colorspace channel for HOG 
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]
       
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)   
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    count = 0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            count = count + 1

            
            ################ ONLY FOR BIN_SPATIAL AND COLOR_HIST ################

            # Extract the image patch
           
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
           # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
           # test_prediction = svc.predict(test_features)
            
            ######################################################################
           # features = X_scaler.transform(hog_features) 
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    print("###Searching of ", count, "windows completed###" )          
    return rectangles




def generator(samples, batch_size=128, training_patch_size=(96,96)):
    num_samples = len(samples)
    print ('Number of samples for training', num_samples)
    print ('Generator batch size', batch_size)
    while 1: # Loop forever so the generator never terminates
        #sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            img_patches = []
            labels = []
            
            for batch_sample in batch_samples:
                 
                if batch_sample[5] == "Car":
                    
                    current_path = "D:/Data_vehicle_tracking/object-detection-crowdai/" + batch_sample[4]
                   # print("path", current_path)
                    image = cv2.imread(current_path)
                    #image =np.asarray(image, dtype = np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
#                    if batch_sample[2] > batch_sample[3] :
#                        ymin = int(batch_sample[3])
#                        ymax = int(batch_sample[2])
#                    else:    
                    ymin = int(batch_sample[1])
                    ymax = int(batch_sample[3])
                        
                        
#                    if batch_sample[0] > batch_sample[1] :
#                        xmin = int(batch_sample[1])
#                        xmax = int(batch_sample[0])
#                    else:    
                    xmin = int(batch_sample[0])
                    xmax = int(batch_sample[2])    
                        
                

                    patch_sel = image.copy()
                    patch_sel = image[ymin:ymax, xmin:xmax] ## get boundery box image
                    no_car_image_newa = image[:ymin  , :xmin]
                    no_car_image_newb = image[ymax:  , xmax:]
   
                    if no_car_image_newa.shape[0]*no_car_image_newa.shape[1] > no_car_image_newb.shape[0]*no_car_image_newb.shape[1]:
                        
                        no_car_image_sel = no_car_image_newa[:ymax-ymin,:xmax-xmin]
                    else:
                        no_car_image_sel = no_car_image_newb[:ymax-ymin,:xmax-xmin]
                        
                    draw = draw_boxes(image, bboxes=[((int(batch_sample[0]), int(batch_sample[1])),(int(batch_sample[2]),int(batch_sample[3])))], color=(0, 0, 255), thick=6)
                    patch = cv2.resize(patch_sel,training_patch_size)    ## resize boundery to 64x64 image
                   # print("patch", patch)
                    no_car_image_sel = cv2.resize(no_car_image_sel,training_patch_size)
                    ## add car image and label to training data
                    img_patches.append(patch)
                    labels.append(1)
                    
                    img_patches.append(no_car_image_sel)
                    labels.append(0)
                    
                    ### add also car data flipped over vertical axis
                    flip_patch = patch.copy()
                    flip_patch=cv2.flip(flip_patch,1) 
                    flip_no_car_image = no_car_image_sel.copy()
                    flip_no_car_image = cv2.flip(no_car_image_sel,1)
                    
                    img_patches.append(flip_patch)
                    labels.append(1)
                    
                    img_patches.append(flip_no_car_image)
                    labels.append(0)
#                    plt.figure()
#                    plt.imshow(draw)
#                    img_patches.append(draw)
#                    labels.append(5)
                    ### add also car data with random noise 
#                    mean = (0,0,0)
#                    sig = (2.2,2.2,2.2)
#                   
#                    noise_patch = cv2.randn(image.copy(), mean, sig) + image.copy()
#                    noise_no_car_image = no_car_image_sel.copy()
#                    noise_no_car_image = cv2.randn(noise_no_car_image, mean, sig) 
#                    
#                    img_patches.append(noise_patch)
#                    labels.append(1)    
#                    
#                    img_patches.append(noise_no_car_image)
#                    labels.append(0)  
                    
                    
                   
                    
                    

            # trim image to only see section with road
            X_train = np.array(img_patches)
            y_train = np.array(labels)
            
#            print ("All data successfully loaded into memory")
            print ("X_data shape", X_train.shape)
            print ("y_data shape", y_train.shape)
            return X_train, y_train




def training(sample_limit, color_space, orient,pix_per_cell,cell_per_block,hog_channel,spatial_size,hist_bins, spatial_feat,hist_feat, hog_feat, training_patch_size):

#    images = glob.glob('*.jpeg')
#    cars = []
#    notcars = []
#    for image in images:
#        if 'image' in image or 'extra' in image:
#            notcars.append(image)
#        else:
#            cars.append(image)
    
    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
 
        # Read in cars and notcars
#    samples = []
#    with open('D:\\Data_vehicle_tracking\\object-detection-crowdai\\labels_crowdai.csv') as csvfile:
#        reader = csv.reader(csvfile)
#        for line in reader:
#
#            samples.append(line)   
#                
#    samples.pop(0) ##remove firt line    
#        
#    data = samples[:sample_limit]    
    #X_train, y_train = generator(data, batch_size=len(data), training_patch_size=training_patch_size)
    
    
    imagescar = glob.glob('D:/Data_vehicle_tracking/vehicles/**/*.png', recursive=True)
    imagesnocar = glob.glob('D:/Data_vehicle_tracking/non-vehicles/**/*.png' ,recursive=True)
    cars = []
    notcars = []
    
    for image in imagescar:
            cars.append(image)
    for image in imagesnocar:
            notcars.append(image)        
  
            
    print ("notcars", len(notcars))
    print ("cars", len(cars))
    example_img = mpimg.imread(cars[0])
    print("Image size: ", example_img.shape)
    print("Image type: ", example_img.dtype)
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
#    print('car feat',len(car_features) )
#    print('no_car',len(notcar_features) )
    #X = np.vstack((car_features)).astype(np.float64) 
    X = np.vstack((car_features, notcar_features)).astype(np.float64) 
    print('Feature vector length:', X.shape)                     
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    #print("scaled_X",scaled_X.shape)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))#np.hstack(y_train)
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    
    # Use a linear SVC 
    svc = SVC(kernel = 'linear', C=1.0, probability= True)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    
    ## calculate classification report
    y_pred = svc.predict(X_test)
    target_names = ['not cars', 'cars']
    report = classification_report(y_test, y_pred, target_names=target_names, digits = 4 )
    print(report)
    ## calculate confusion matrix
    
    
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    np.set_printoptions(precision=4)
    print(confmat)
    ## plot conf matrix
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.GnBu, alpha=0.2)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    
    
#    ## plot performance over training data
#    train_sizes, train_scores, test_scores =\
#                learning_curve(estimator=svc,
#                               X=X_train,
#                               y=y_train,
#                               train_sizes=np.linspace(0.1, 1.0, 10),
#                               cv=10,
#                               n_jobs=1)
#
#    train_mean = np.mean(train_scores, axis=1)
#    train_std = np.std(train_scores, axis=1)
#    test_mean = np.mean(test_scores, axis=1)
#    test_std = np.std(test_scores, axis=1)
#    
#    plt.figure('learning curves')
#    plt.plot(train_sizes, train_mean,
#             color='blue', marker='o',
#             markersize=5, label='training accuracy')
#    
#    plt.fill_between(train_sizes,
#                     train_mean + train_std,
#                     train_mean - train_std,
#                     alpha=0.15, color='blue')
#    
#    plt.plot(train_sizes, test_mean,
#             color='green', linestyle='--',
#             marker='s', markersize=5,
#             label='validation accuracy')
#    
#    plt.fill_between(train_sizes,
#                     test_mean + test_std,
#                     test_mean - test_std,
#                     alpha=0.15, color='green')
#    
#    plt.grid()
#    plt.xlabel('Number of training samples')
#    plt.ylabel('Accuracy')
#    plt.legend(loc='lower right')
#    plt.ylim([0.9, 1.05])
#    plt.tight_layout()
#    plt.savefig('learning_curve.png', dpi=300)
#    plt.show()
    
    ### test majority vote
    clf1 = LogisticRegression(penalty='l2', random_state=rand_state)
    clf1.fit(X_train, y_train)

    clf2 = svc
    
    clf3 = KNeighborsClassifier(n_neighbors=1,
                                p=2,
                                metric='minkowski')
    
    clf3.fit(X_train, y_train)
    

    
    clf_labels = ['Logistic Regression', 'SVM', 'KNN']
    
#    print('10-fold cross validation:\n')
#    for clf, lab in zip([clf1, clf2, clf3], clf_labels):
#        scores = cross_val_score(estimator=clf,
#                                 X=X_train,
#                                 y=y_train,
#                                 cv=10,
#                                 scoring='roc_auc')
#        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
#              % (scores.mean(), scores.std(), lab))
        
    # Majority Rule (hard) Voting 
    mv_clf = VotingClassifier(estimators=[
                ('Logistic Regression', clf1), ('SVM', clf2), ('KNN', clf3)], voting='hard', n_jobs = -1)
    t=time.time()
    mv_clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train Majority vote classifier...')
    
    clf_labels += ['Majority Voting']
    all_clf = [clf1, clf2, clf3, mv_clf]
    print('10-fold cross validation:\n')
    for clf, lab in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc',
                                 n_jobs = -1)
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), lab)) 
        
    y_pred =  mv_clf.predict(X_test) 
    
    
    
    # Check the score of the MVC
    print('Test Accuracy of MVC = ', round(mv_clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    
    ## calculate classification report
    y_pred = mv_clf.predict(X_test)
    target_names = ['not cars', 'cars']
    report = classification_report(y_test, y_pred, target_names=target_names, digits = 4 )
    print(report)
    
    
    
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    np.set_printoptions(precision=4)
    print(confmat)
    ## plot conf matrix
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.GnBu, alpha=0.2)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()     
#        
#        
        
        
        
    dist_pickle = {"svc" : svc, 
                   "scaler": X_scaler, 
                   "orient": orient, 
                   "pix_per_cell" : pix_per_cell, 
                   "cell_per_block": cell_per_block,
                   "spatial_size": spatial_size,
                   "hist_bins": hist_bins,
                   "color_space" : color_space,
                   "hog_channel" : hog_channel,
                   "hist_feat": hist_feat,
                   "hog_feat" : hog_feat,
                   "spatial_feat" : spatial_feat,
                   "training_patch_size" : training_patch_size
                   }
    
    with open("svc_pickle_mv_clf.p", "wb" ) as file:
        pickle.dump(dist_pickle, file)
        
    print("svc_pickle.p has been saved!")    

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        
        
    # Return the image
    return img

        
def load_and_find(image):
    
    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"]
    hog_channel = dist_pickle["hog_channel"]
    hist_feat = dist_pickle["hist_feat"]
    hog_feat = dist_pickle["hog_feat"]
    spatial_feat = dist_pickle["spatial_feat"]
    training_patch_size = dist_pickle["training_patch_size"] 
    
    


    
    bboxes = []
    ystart = 400
    ystop = 464
    scale = 1.0
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
    ystart = 416
    ystop = 480
    scale = 1.0
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
    
    ystart = 404
    ystop = 468
    scale = 1.0
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
    
    ystart = 408
    ystop = 472
    scale = 1.0
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
#
#        
    ystart = 400
    ystop = 496
    scale = 1.5
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
 
    
    ystart = 432
    ystop = 528
    scale = 1.5
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
    
    ystart = 460
    ystop = 556
    scale = 1.5
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))

    
    ystart = 400
    ystop = 528
    scale = 2.0
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
    
    
    ystart = 404
    ystop = 532
    scale = 2.0
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
    ystart = 432
    ystop = 560
    scale = 2.0
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,True))
    ystart = 400
    ystop = 624
    scale = 3.5
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))
    ystart = 464
    ystop = 688
    scale = 3.5
    bboxes.append(find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True))





    ### 
    bboxes = [item for sublist in bboxes for item in sublist]
    print ("###" , len(bboxes), "objects found!###" )
    fil.add_rects(bboxes)

    # Add heat to each box in box list
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    print ('lenght prev rect',len(fil.prev_rects))
    for rect_set in fil.prev_rects:
        heat = add_heat(heat, rect_set)
#    plt.figure(1)
#    plt.imshow(heat) 
       
    # Apply threshold to help remove false positives
    tresh = (max(( np.max(heat) / 2 , 2))) * 0.9
    print("current heat threshold", tresh)
    heat = apply_threshold(heat,tresh)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
#    plt.figure(2)
#    plt.imshow(heat)
#    
#    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    ## enable to draw all boxes
   # print("bboxes", bboxes)
#    for box in bboxes:
#         cv2.rectangle(image, box[0], box[1], (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)), 6)
        
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    plt.figure(3)
    plt.imshow(image)
    

    return draw_img
        
if __name__ is "__main__":
    
    sample_limit = 50
    color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 16# HOG pixels per cell highe leads to less windows
    cell_per_block = 2 # no influens to windows
    # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    training_patch_size = (64,64) # size of image used for training 
#
   # training(sample_limit, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,  
#                                                                spatial_feat, hist_feat,hog_feat, training_patch_size)
  
#    input("Press Enter to continue...")
    fil = filter_detection()  
    current_path = "D:/SDC_projects/CarND_Vehicle-Detection-P5/test_images/test5.jpg"
    #current_path = "D:/Data_vehicle_tracking/object-detection-crowdai/" + link[4]
    image = mpimg.imread(current_path)

    load_and_find(image)
   


#    test_out_file = 'project_video_out.mp4'
#    clip_test = VideoFileClip('project_video.mp4')
#    clip_test_out = clip_test.fl_image(load_and_find)
#    clip_test_out.write_videofile(test_out_file, audio=False)    
###HOG feature visualisation
#    car_img = mpimg.imread("D:/SDC_projects/CarND_Vehicle-Detection-P5/test_images/15.png")
#    _, car_dst = get_hog_features(car_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
#    noncar_img = mpimg.imread("D:/SDC_projects/CarND_Vehicle-Detection-P5/test_images/nonvehicle.png")
#    _, noncar_dst = get_hog_features(noncar_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
#    
#    # Visualize 
#    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
#    f.subplots_adjust(hspace = .4, wspace=.2)
#    ax1.imshow(car_img)
#    ax1.set_title('Car Image', fontsize=16)
#    ax2.imshow(car_dst, cmap='gray')
#    ax2.set_title('Car HOG', fontsize=16)
#    ax3.imshow(noncar_img)
#    ax3.set_title('Non-Car Image', fontsize=16)
#    ax4.imshow(noncar_dst, cmap='gray')
#    ax4.set_title('Non-Car HOG', fontsize=16)
  