import keras
from keras.models import *
from keras.layers import *
from keras import backend as k
from keras.optimizers import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pickle
import cv2
from sklearn.utils import shuffle
import glob
from PIL import Image

#url open to get image
import urllib.request
from urllib.request import urlopen

#for data augmentation
import imgaug as ia
from imgaug import augmenters as iaa

#in case you have images without need to do other specific preprocessing (i.e. not put black partout) you can simply use:
'''test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary') ... '''

###################################################################################################
###################################### download data from www #####################################
###################################################################################################

#download image with url if not already in the folder
def get_image(url, path, name):
    k = 0
    #if we dont already have it, download it
    if len(glob.glob(path))<1:
        #download until get image or until 10 trials of same image
        while k<10:
            try:
                response = urlopen(url)
                img = Image.open(response)
                img.save(path)
                k = 10
            except KeyboardInterrupt:
                    raise
            except Exception as e:
                if response.getcode() in [200, 404]: 
                    print('Not able to SAVE image for species %s and url %s, \
                    due to: %s ,  lets STOP'%(name,str(url),e))
                    print(response.getcode())
                    k = 10
                else:
                    print('Not able to SAVE image for species %s and url %s, \
                    due to: %s ,  lets RETRY'%(name,str(url),e))
                    print(response.getcode())
                    k = k+1 #typical example: "HTTP Error 504: Gateway Time-out"
                    time.sleep(5)


                    
###################################################################################################
################################### preprocessing fct for image ###################################
###################################################################################################

def data_augmentation_remove_reflect(img):
    '''r: Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.'''
    #convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #should be more robust gray = cv2.GaussianBlur(gray, (41, 41), 0) #flou une image en utilisant un filtre gaussian
    #keep only the brightess part
    mask_ = cv2.inRange(gray, 0, 150)
    #put zero pixel to non-zero pixels and vise versa so that we can use inpaint
    mask_ = np.array([[1 if j==0 else 0 for j in i] for i in mask_]).astype('uint8')
    #image inpainting is used. The basic idea is simple: Replace those bad marks with its neighbouring pixels 
    #non-zero pixels corresponds to the area which is to be inpainted.
    #Radius of a circular neighborhood of each point inpainted that is considered by the algorithm. : 3
    result = cv2.inpaint(img,mask_,5,cv2.INPAINT_TELEA)
    return result

#from an image and mask, keep only the part of the image that intersect with the mask
def KeepMaskOnly(image, mask, debug_text):
    try:
        #instead of one channel produce 3 exact same channel
        mask = np.stack((mask.reshape([mask.shape[0], mask.shape[1]]),)*3, -1)
        #outside the mask put to black color
        image[~mask]=0
    except:
        print(mask.shape)
        print(debug_text)
    return(image)


#replace black pixel by smoothing with adequate color to keep all info, removing shape 
def remove_shape_keep_all_info(img):
    #create mask (s.t. Non-zero pixels indicate the area that needs to be inpainted)
    mask_ = np.array([[1 if j==0 else 0 for j in i] for i in cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_ = cv2.dilate(mask_, kernel, iterations=4)
    result = cv2.inpaint(img,mask_,5,cv2.INPAINT_TELEA) #,cv2.INPAINT_TELEA, INPAINT_NS
    return (result)


#resize the image regarding either width or height, keeping ratio
def image_resize_keeping_prop(image, width = None, height = None, inter = cv2.INTER_AREA):

    #initialize
    dim = None
    (h, w) = image.shape[:2]

    #if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    
    #if both is not nonw return error
    if (width is not None) & (height is not None):
        print('ERROR: you should give either the width or the height, not both')
        sys.exit()

    #if width is None, then height is not None
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    #otherwise the height is None and width is not None
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    #resize the image in convenient manner and return it
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
'''or use 
import imutils
img = imutils.resize(image, width=500) '''


#function that resize image keeping the aspect raito and adding less possible black pixel. In other words:
#function that takes as input an image, change the magnitude of the image to fit better the new dimension (n1, n2) and
# finaly resize it to fit exactly the dimension keeping the intiial aspect ration and hence adding black pixel where 
#its needed.
def adjust_size(image, h, w):
    
    #initialize
    dim = None
    (hi, wi) = image.shape[:2]
    
    #change image resolution
    #as we dont want to remove some pixel, we will resize the image keeping the initial aspect ratio, making sur that
    #both width and height stay smaller than the one wanted (w,h)
    #if the initial image is more 'flat-rectangle' than the target dimension, then resize w.r.t.  the width
    if (hi/wi)<(h/w):
        image = image_resize_keeping_prop(image, width=w)
    #if the nitial image is more 'height-rectangle' than the target dimension, then resize w.r.t.  the height
    else:
        image = image_resize_keeping_prop(image, height=h)
    
    #change dimension
    (hi, wi) = image.shape[:2]
    #finally resize to fit the exact target dimension by adding black pixel where its needed
    l = int((w-wi)/2)
    r = w-(wi+l)
    t = int((h-hi)/2)
    b = h-(hi+t)
    image = cv2.copyMakeBorder(image,t,b,l,r,cv2.BORDER_CONSTANT) #top, bottom, left, right #,value=[0,0,0]
    
    return(image)

#to use the preprocessing step used in inceptionv3 training for other purpose as well (in testing for example)
#augmenta should be used in training essentially
def image_augmentation_with_maskrcnn(ID, n1, n2, image_path, mask_path, augmentation=None, normalize=False, preprocessing=None,
                                    plot_3_image=False):
    
    #downlaod image and mask
    P = os.path.join(mask_path,'mask_output_'+ID+'.pkl') #in the next version dont save with mask_output in front
    mask = pickle.load(open(P, 'rb'))
    mask = mask['unique_binary_mask']
    image = cv2.imread(os.path.join(image_path, ID+'.jpg'))
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    if plot_3_image==True:
        image1 = image.copy()
    
    #keep only mask info
    image = KeepMaskOnly(image,mask,str(ID))
    
    #remove black area at maximum by zoomimg keeping aspect ratio
    boxes = pickle.load(open(P, 'rb'))['rois']
    y1, x1, y2, x2 = boxes[0]
    image = image[y1:y2, x1:x2]
    image = adjust_size(image, n1, n2)
    if plot_3_image==True:
        image2 = image.copy()

    
    #replace black pixel by smoothing with adequate color to keep all info, removing shape 
    image = remove_shape_keep_all_info(image)
    if plot_3_image==True:
        image3 = image.copy()

    #normalize
    if normalize == True:
        normalizedImg = np.zeros((200, 200))
        normalizedImg = cv2.normalize(image, normalizedImg, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX) 
        image = normalizedImg
        
    #preprocessing
    if preprocessing is not None:
        image = preprocessing(image)
        
    #augment image
    if augmentation is not None:
        image = augmentation.augment_image(image)
        
    #for debuging   
    if plot_3_image==True:
        return(image1, image2, image3)
    
    return(image)


###################################################################################################
########################################## datagenerator ##########################################
###################################################################################################

#copy from internet then small modifications
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, image_path, mask_path, batch_size, n_rows, n_cols, n_channels, n_classes, 
                 augmentation=None, preprocessing=None, shuffle=True, normalize=False, age=None):
        self.n1 = n_rows
        self.n2 = n_cols
        self.batch_size = batch_size
        self.image_path = image_path
        self.mask_path = mask_path
        self.labels = labels
        self.age = age
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.normalize = normalize
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.on_epoch_end()

    def __len__(self):
        'number of step per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        return(self.__data_generation(list_IDs_temp))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels) where n_sampled=batch_size
        # Initialization
        X = np.empty((self.batch_size, self.n1, self.n2, self.n_channels))
        a = np.empty((self.batch_size), dtype=int)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):     
            #handle image
            image = image_augmentation_with_maskrcnn(ID=ID, n1=self.n1,n2=self.n2, 
                                                     image_path=self.image_path, mask_path=self.mask_path,
                                                     augmentation=self.augmentation,
                                                     normalize=self.normalize,
                                                     preprocessing=self.preprocessing)
            X[i,] = image
            #handle class
            y[i] = self.labels[ID]
            #handle age
            if self.age is not None:
                #handle age
                a[i] = self.age[ID]

        #handle age
        if self.age is not None:
            return [X,a], keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
    
    