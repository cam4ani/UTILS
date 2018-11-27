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
import time
import inflect
import requests
import json

#url open to get image
import urllib.request
from urllib.request import urlopen

#for data augmentation
import imgaug as ia
from imgaug import augmenters as iaa



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
                response = None
                response = urlopen(url)
                img = Image.open(response)
                img.save(path)
                k = 10
                del img
            except KeyboardInterrupt:
                    raise
            except Exception as e:
                if response is not None:
                    if response.getcode() in [200, 404]: 
                        print('Not able to SAVE image for species %s and url %s, lets STOP due to: \n %s'%(name,str(url),e))
                        print(response.getcode())
                        k = 10
                    else:
                        print('Not able to SAVE image for species %s and url %s, lets RETRY due to: \n %s'%(name,str(url),e))
                        print(response.getcode())
                        k = k+1 
                        time.sleep(5)
                else: 
                        print('Not able to SAVE image for species %s and url %s, lets STOP due to: \n %s'%(name,str(url),e))
                        k = 10  #HTTP Error 404: Not Found

                        
#search wikipedia translation of the title
#more parameter at: https://www.mediawiki.org/w/api.php?action=help&modules=query%2Blanglinks
def search_wikipedia_laguage(text, language='en'):
    #wiki query with properties 'langlinks' to ask for all opssible language translation given on that page with limit to 500 language
    #(default is 10, 500 is maximum)
    url = 'https://'+language+'.wikipedia.org/w/api.php?action=query&format=json&prop=langlinks&lllimit=500&llprop=langname|autonym&titles=%s&redirects=1'% requests.utils.quote(text)
    while True:
        try:
            #call API
            content = requests.get(url).content
            content = json.loads(content)
            #content[1][0].upper()==text.upper(): #if exact match in the title
            return(content)
        except KeyError:
            print('species %s failed, try again' % text)
#for more general info (sentences, url ...)
#url = 'https://'+language+'.wikipedia.org/w/api.php?action=opensearch&search=%s&limit=1&namespace=0&format=json&redirects=resolve&prop=langlinks' % requests.utils.quote(text)
#note that we have put 'limit=1' as we are searching for the exact amtch (car pour le reste on va faire avec d'autre
#technique comme la distance entre le smots etc)
#to search wiki sumamry: wikipedia.summary(x)

    
#used for allrecipes.com for example    
# Get HTML content if not already existing in our files (not to request it two times)
def get(id, url, path_):
    # Check if file html was already dowloaded
    cache_path = os.path.join(path_, 'cache')
    path = os.path.join(cache_path, '%d.html' % id)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            content = file.read()
    # Otherwise get page content
    else:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        while True:
            try:
            # Write cache
                with open(path, 'wb') as file:
                    page = requests.get(url+str(id), timeout=5)
                    content = page.content
                    file.write(content)
                break
            except KeyboardInterrupt:
                raise
            except:
                print('page %d failed, try again' % id)
    return content

def parse(content,end_of_title,encoding='utf-8'):
    # Check if page is valid
    try:
        tree = html.fromstring(content.decode(encoding))
        title = tree.xpath('head/title')[0].text
        if not title.endswith(end_of_title):
            return None
    except :
        print('error')
        tree=None
    return tree    
    
    
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

#in case you have images without need to do other specific preprocessing (i.e. not put black partout) you can simply use:
'''test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary') ... '''

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
    
    
###################################################################################################
########################################### other - old ###########################################
###################################################################################################
    
#join several dico together without duplicate info but with all possible info
def join_dico(li_s):
    l = len(li_s)
    if l==1:
        return(li_s[0])
    else:
        s1 = li_s[0]
        s2 = li_s[1]
        s = s1.copy()
        for k,v in s2.items():
            if k in s:
                s[k] = s[k] + ' /-/ ' + s2[k]
                s[k] = ' /-/ '.join(list(set([i.strip() for i in s[k].split('/-/')])))
            else:
                s[k] = s2[k]
        r = li_s[2:]
        if len(r)>0:
            return(join_dico(li_s = r+[s]))
        else:
            return(s)
#small example 
#s1 = {'parent':'afdsf','a':'12'}
#s2 = {'a':'sdfsdf /-/ df /-/12','cac':'q'}
#s3 = {'a':'sdfsdf1213'}
#s4 = {'hello':'new'}
#join_dico([s1,s2,s3,s4]) 


#take two strings as input. rule pour enlever les pluriels/singulier: guarder celui qui permettra de retrouver l'autre (car pas forcément le cas
#dans les deux sens) ou alors garder celui qui est ordonner alphabetiquement le premier (si les deux peuvent induire l'autre)
#example d'utilisation: si initialement dans notre liste d'ingredient il y a une forme qui ne permet pas de retourner 
#à son autre form, alors il faudra l'updater avec l'autre
engine = inflect.engine()
def keep_goodone_singplu(x1,x2):
    x1_s = engine.plural(x1)
    x2_s = engine.plural(x2)
    if (x1_s==x2) and (x2_s==x1):
        return([sorted([x1,x2])[0]])
    elif (x1_s==x2):
        return([x1])
    elif (x2_s==x1):
        return([x2])
    else:
        return([x1,x2])
#e.g.
#keep_goodone_singplu('pie cakes','pie cake')
#the plural are: pie cakess, pie cakes -->'pie cake'


#output a itertools of tuples of all possible combinations of 2 elements form the list 'li'
def all_subsets(li):
    return chain(*map(lambda x: combinations(li, x), range(2, 3)))
#for subset in all_subsets([1,3,5]):
#    print(subset)
#-->:
#(1, 3)
#(1, 5)
#(3, 5)


#It simply says that a tree is a dict whose default values are trees.
def tree(): return defaultdict(tree)
#from tree to dict
def dicts(t): return {k: dicts(t[k]) for k in t}
#iteration
def add(t, path):
    for node in path:
        t = t[node]
        
        
#put the values of all keys except the one in li_ke together (they should be list) (used in below fct)
def all_except_keys(dico,li_ke):
    r = []
    dico_ = dico.copy()
    for k in li_ke:
        dico_.pop(k,None)
    r = list(dico_.values())
    r = [i for sublist in r for i in sublist]
    return(set(r))        
        
        
        