import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

from skimage.feature import hog


from imp import reload

from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

import lesson_functions as lf
import heat_maps as hm
import window_slider as ws
reload(ws)
reload(lf)

class CarDetection():
    '''
    CarDetection is used to train a classifier and implement bounding boxes on the detected cars
    '''

    def __init__(self):

        #SVC model
        self.svc = None  
        self.X_scaler = None

        #windows
        self.windows = None
        self.hot_windows = []
        #Params
        self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 16    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off

  
        # Params boxes
        self.threshold_heat = 1
        self.lastFrames = 15

        self.addHeatMapView = False
        self.addBoxesFound = False

    def buildModel(self, cars, not_cars):
        '''
        buildModel: Trains a svc classifier from the supplied training set.
        It uses a 20% of the supplied set as test set while the 80% is used a training set.
        '''
        car_features = lf.extract_features(cars, color_space=self.color_space, 
            spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
            orient=self.orient, pix_per_cell=self.pix_per_cell, 
            cell_per_block=self.cell_per_block, 
            hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
            hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        notcar_features = lf.extract_features(not_cars, color_space=self.color_space, 
            spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
            orient=self.orient, pix_per_cell=self.pix_per_cell, 
            cell_per_block=self.cell_per_block, 
            hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
            hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        self.X_scaler = X_scaler

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',self.orient,'orientations',self.pix_per_cell,
            'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)

        self.svc = svc

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))



    def getImageWindows(self, image):
        '''
        getImageWindows:  returns a list of sliding windows
        '''
        windows1 = ws.slide_window(image, x_start_stop=[200, 1200], y_start_stop=[350, 550], 
                            xy_window=(70, 70), xy_overlap=(0.8, 0.8))
                            
        windows2 = ws.slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 650], 
                            xy_window=(120, 120), xy_overlap=(.7, .7))
                 
        windows3 = ws.slide_window(image, x_start_stop=[20, None], y_start_stop=[350, None], 
                            xy_window=(160,160), xy_overlap=(0.6, 0.6))

        windows = windows1 + windows2 + windows3

        return windows

    def processImage(self, image):
        '''
        processImage:   process given image and overlays the bounding box of the detected cars
        '''
        if self.windows is None: 
            self.windows = self.getImageWindows(image)

        #ret_image = np.copy(image)
        cimage = image.astype(np.float32)/255

        hot_windows = lf.search_windows(cimage, self.windows, self.svc, self.X_scaler, color_space=self.color_space, 
            spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
            orient=self.orient, pix_per_cell=self.pix_per_cell, 
            cell_per_block=self.cell_per_block, 
            hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
            hist_feat=self.hist_feat, hog_feat=self.hog_feat)      

        self.hot_windows.append(hot_windows)
        self.hot_windows = self.hot_windows[-self.lastFrames:]
        
        hot_windows = [ht for frames in self.hot_windows for ht in frames]
        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = hm.add_heat(heat,hot_windows)
        # Apply threshold to help remove false positives
        heat = hm.apply_threshold(heat,self.threshold_heat)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)


        draw_img = hm.draw_labeled_bboxes(image, labels)
        if self.addBoxesFound:
            draw_img = ws.draw_boxes(draw_img, hot_windows, color=(255,0,0), thick=1)

        if self.addHeatMapView:
            # Add heatmaped image
            x_offset=960
            y_offset=20
            norm = plt.Normalize(vmin=heatmap.min(), vmax=heatmap.max())
            norm_heatmap = norm(heatmap)
            norm_heatmap = cv2.resize(norm_heatmap, (300,150))
            draw_img[y_offset:y_offset+norm_heatmap.shape[0], x_offset:x_offset+norm_heatmap.shape[1]] = np.dstack((norm_heatmap*255, norm_heatmap, norm_heatmap)).astype(np.uint8)
            

        return draw_img


