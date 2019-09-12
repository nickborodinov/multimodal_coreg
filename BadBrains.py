# -*- coding: utf-8 -*-

#%% Dependencies

import scipy
import time
import pickle
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import h5py
from scipy import ndimage as ndi
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['savefig.dpi']=600
pgf_with_rc_fonts = {
    "font.family": "Arial",
    "font.serif": [], 
     "font.size"   : 20,
    "axes.titlesize" : 20,
    "font.sans-serif": ["Times New Roman"], # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)
import matplotlib.pyplot as plt
import cv2
from scipy import signal, stats
from sklearn.decomposition import NMF
from numpy import genfromtxt
from scipy import stats

#%% Helper Functions

def cartesian_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    distance = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

def crawldir(topdir=[], ext='sxm'):
    fn = dict()
    for root, dirs, files in os.walk(topdir):
              for name in files:
              
                if len(re.findall('\.'+ext,name)):
                    addname = os.path.join(root,name)

                    if root in fn.keys():
                        fn[root].append(addname)

                    else:
                        fn[root] = [addname]    
    return fn
        
def gamma_correlation(array, feature, window_size):
    
    size = feature.size
    
    in_array=array-np.min(array)
    in_array=in_array/np.max(in_array)
    
    arr = feature.arr
    arr=cv2.resize(arr,(2*window_size,2*window_size))

    try:
        AA=np.array(stats.spearmanr(np.ndarray.flatten(arr),np.ndarray.flatten(in_array)))
    except ValueError:

        AA=0
    return AA

#%% function_ls class
    
class function_ls:
    import numpy
    import warnings
    warnings.filterwarnings("ignore")
    def __init__(self,function,target,feature,epochs=500,alpha=0.1,decay=0.1,verbose=0):
        ### function fitter that performs optimization for a multivariate function
        self.target=target
        self.function=function
        self.parameters='None'
        self.guesses='None'
        self.epochs=epochs
        self.alpha=alpha
        self.decay=decay
        self.verbose=verbose
    
    def load_guesses(self,guesses):
        self.guesses=guesses
        
    def calc_rmse(self,parameters):
        return np.sum((self.function(*parameters)-self.target)**2)**0.5
    
    def gradient_descent_step(self):
        if self.parameters=='None':
            self.parameters=self.guesses
        for i in range(len(self.parameters)):
            altered_parameters_p=np.copy(self.parameters)
            altered_parameters_p[i]=altered_parameters_p[i]*(1+self.alpha)
            
            altered_parameters_m=np.copy(self.parameters)
            altered_parameters_m[i]=altered_parameters_m[i]*(1-self.alpha)
            
            altered_parameters=np.copy(self.parameters)
            altered_parameters[i]=altered_parameters[i]*(1)
            
            min_idx=np.argmin(np.array([self.calc_rmse(altered_parameters_m),self.calc_rmse(altered_parameters),self.calc_rmse(altered_parameters_p)]))
            self.parameters=np.array([altered_parameters_m,altered_parameters,altered_parameters_p])[min_idx]
        self.alpha=self.alpha*(1-self.decay)
        return self.parameters
    def fit(self):
        if self.parameters=='None':
            self.parameters=self.guesses
        for iteration in range(self.epochs):
           
            stage=np.copy(self.parameters)
            history=self.gradient_descent_step()
#             if all(history==stage)==True:
#                 print(history)
#                 break
            if self.verbose==1:
                print(iteration)
#         print(history)
                
#%% Feature class
                
class feature(object):
    '''
    '''
    def __init__(self,size,template=None,rotation=0,flip=False):
        self.arr = None
        self.size = size
        self.target = None
        self._corner_array_()
        if template != None:
            if template.upper() == 'CORNER':
                self._corner_array_()
            elif template.upper() == 'SQUARE':
                self._square_array_()
            elif template.upper() == 'GAMMA':
                self._gamma_array_()
            else:
                print('Feature template not recognized, defaulting to corner.')
                self._corner_array_()
        self.rotate(rotation)
        if flip:
            self.flip()
        return
        
    def _corner_array_(self):
        size = self.size
        arr_1=np.ones([size,size])
        arr_0=np.zeros([size,size])
        arr_00=np.hstack([arr_0,arr_0,arr_0])
        arr_0_0=np.vstack([arr_0,arr_0,arr_0,arr_0,arr_0])
        arrt=np.hstack([arr_0,arr_1,arr_1])
        arrm=np.hstack([arr_0,arr_1,arr_0])
        arrb=np.hstack([arr_0,arr_0,arr_0])
        arr=np.vstack([arr_00,arrt,arrm,arrb,arr_00])
        arr=np.hstack([arr_0_0,arr,arr_0_0])
        self.arr = arr
        height, length = arr.shape
        self.center = (height/2, length/2)
        self.target = (height*0.5996, length*0.3996)
        correction = (self.target[0]-self.center[0], self.target[1] - self.center[1])
        self.correction = (correction[0], correction[1])
        return
        
    def _square_array_(self):
        size = self.size
        arr_1=np.ones([size,size])
        arr_0=np.zeros([size,size])
        arr_00=np.hstack([arr_0,arr_0,arr_0])
        arr_0_0=np.vstack([arr_0,arr_0,arr_0,arr_0,arr_0])
        arrt=np.hstack([arr_0,arr_0,arr_0])
        arrm=np.hstack([arr_0,arr_1,arr_0])
        arrb=np.hstack([arr_0,arr_0,arr_0])
        arr=np.vstack([arr_00,arrt,arrm,arrb,arr_00])
        arr=np.hstack([arr_0_0,arr,arr_0_0])
        self.arr = arr
        height, length = arr.shape
        self.center = (height/2, length/2)
        self.target = self.center
        correction = (self.target[0]-self.center[0], self.target[1] - self.center[1])
        self.correction = (correction[0], correction[1])
        return
    
    def _gamma_array_(self):
        size = self.size
        arr_1=np.ones([size,size])
        arr_0=np.zeros([size,size])
        arr_00=np.hstack([arr_0,arr_0,arr_0])
        arr_0_0=np.vstack([arr_0,arr_0,arr_0,arr_0,arr_0])
        arrt=np.hstack([arr_0,arr_1,arr_1])
        arrm=np.hstack([arr_0,arr_1,arr_0])
        arrb=np.hstack([arr_0,arr_1,arr_0])
        arr=np.vstack([arr_00,arrt,arrm,arrb,arr_00])
        arr=np.hstack([arr_0_0,arr,arr_0_0])
        self.arr = arr
        height, length = arr.shape
        self.center = (height/2, length/2)
        self.target = (height*0.5996, length*0.3996)
        correction = (self.target[0]-self.center[0], self.target[1] - self.center[1])
        self.correction = (correction[0], correction[1])
        return
        
    def _set_target_(self):
        pass
    
    def import_array(self, array):
        self.arr = array
        height, length = self.arr.shape
        self.center = (height/2, length/2)
        self._set_target_()
        
    def _invert_(self):
        arr_flat = self.arr.flatten()
        og_shape = self.arr.shape
        inverted_arr = np.empty_like(arr_flat)
        for i, j in enumerate(arr_flat):
            if j == 1:
                inverted_arr[i] = 0
            elif j == 0:
                inverted_arr[i] = 1
        inverted_arr = inverted_arr.reshape(og_shape)
        self.arr = inverted_arr
        
    def rotate(self, degrees):
        '''
        Function to rotate feature array in integer multiples of 90 degrees counterclockwise.
        
        Input:
        --------
        degrees : int
            desired rotation in degrees counterclockwise. Must be an integer multiple of 90 degrees!
        '''
        
        #Rotate the feature array itsef
        if degrees % 90 != 0:
            raise ValueError('degrees must be an integer multiple of 90 degrees')
        rotations = degrees/90
        self.arr = np.rot90(self.arr, rotations)
        
        #Apply rotation to target point coordinates
        degrees = 360-degrees
        sin = np.sin(np.deg2rad(degrees))
        cos = np.cos(np.deg2rad(degrees))
        x_t, y_t = self.target
        x_c, y_c = self.center
        x_t -= x_c
        y_t -= y_c
        x_n = x_t * cos - y_t * sin
        y_n = x_t * sin + y_t * cos
        x_t = x_n + x_c
        y_t = y_n + y_c
        self.target = (x_t, y_t)
        correction = (self.target[0]-self.center[0], self.target[1] - self.center[1])
        self.correction = (correction[0], correction[1])
        return
    
    def flip(self):
        '''
        Flips feature array along its vertical axis, essentially providing a
        mirror image of the original feature
        '''
        #Flip the feature array itself
        self.arr = np.flip(self.arr, 1)
        #Flip coordinates of target point
        x_t, y_t = self.target
        x_c, y_c = self.center
        diff_x = x_c - x_t
        x_n = x_c + diff_x
        self.target = (x_n, y_t)
        correction = (self.target[0]-self.center[0], self.target[1] - self.center[1])
        self.correction = (correction[0], correction[1])
        return
    
#%% FeatureFinder class
        
class FeatureFinder(object):
    '''
    '''
    def __init__(self, function=None, feature=None, window_size=10):
        self.function = function
        self.feature = feature
        self.window_size = window_size
        
    def wiggle(self, rotation, scaling, shift_x, shift_y):
        arr = cv2.resize(self.feature.arr,(self.window_size*2, self.window_size*2),interpolation=cv2.INTER_CUBIC)
        rotation=-30+rotation*60
        scaling=0.7+scaling*0.6
        shift_x=int(-5+shift_x*10)
        shift_y=int(-5+shift_y*10)

        rotation_matrix = cv2.getRotationMatrix2D((self.window_size, self.window_size), rotation, scaling)
        altered_not_shifted = cv2.warpAffine(arr, rotation_matrix, (2*self.window_size, 2*self.window_size))
        shifted = np.roll(np.roll(altered_not_shifted,shift_x,axis=1),shift_y,axis=0)

        return shifted 
    
    def sliding_function(self,array,function,feature,window_size=32,step=1):
        a1=array
        x_1=window_size
        y_1=window_size
        a2=a1[x_1-window_size:x_1+window_size,y_1-window_size:y_1+window_size]
        output=np.ndarray.flatten(function(a2,feature,window_size))
        x_dim=len(np.arange(window_size,a1.shape[0]-window_size,step))
        y_dim=len(np.arange(window_size,a1.shape[1]-window_size,step))
        transformed=np.zeros([x_dim,y_dim,len(output)],dtype='float64')
        #print(x_dim,y_dim)
        x_count=0
        y_count=0
        for i in range(window_size,a1.shape[0]-window_size,step):
            for j in range(window_size,a1.shape[1]-window_size,step):

                x_1=i
                y_1=j
                a2=a1[x_1-window_size:x_1+window_size,y_1-window_size:y_1+window_size]

                a3=function(a2, feature, window_size)
                try:
                    transformed[y_count,x_count]=np.ndarray.flatten(a3)
                except TypeError:
                    transformed[y_count,x_count]=0
                x_count=x_count+1
            x_count=0
            y_count=y_count+1
        return transformed
    
    def search_image(self, image, f_count, feature=None):
        if feature == None:
            feature = self.feature
            
        return fit_parameters
    
    def generate_heatmap(self, image, step, window_size):
        DD = self.sliding_function(image,self.function,self.feature,window_size,step)
        heatmap = np.real(DD[:,:,0])
        return heatmap