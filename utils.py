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
import glob
from PIL import Image
import time
import inflect
import requests
import json
import random
import shutil
import sys
from operator import itemgetter
import tqdm
import colorsys
import operator
import re
#to match substring in string
import fuzzysearch
from fuzzysearch import find_near_matches

#video
import skvideo.io 
#from skimage import color
#import scipy.misc
import moviepy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips

#models
from scipy import stats
import sklearn
from sklearn import *
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

#for eucledian distance computation (for centroids)
from scipy.spatial import distance

#to tranform binary mask to VGG polygons: does not support windows!!
try:
    from pycocotools import mask
except:
    print('you are on windows, so pycocotools can not be installed')

#parallel computing
from multiprocessing import Pool

#structured data from text
from pdf2image import convert_from_path

#url open to get image
import urllib.request
from urllib.request import urlopen

#compute simple similarity between two images
from skimage import measure
from skimage.measure import compare_ssim

#for data augmentation
#import imgaug as ia
#from imgaug import augmenters as iaa
#data augmentation
#import local version of the library imgaug
sys.path.append('C:\\Users\\camil\\Desktop\\animals_code\\imgaug')
import imgaug as ia
from imgaug import augmenters as iaa

#videos
import imageio
from skimage import color

#plot
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.patches as patches

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
                        k = 10  #e.g. HTTP Error 404: Not Found

                        
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
#get HTML content if not already existing in our files (not to request it two times)
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
#################################### structured data from text ####################################
###################################################################################################

#take as input an image ( np.array or PIL image)
def frompng2images(img, path, page_id=0, plot_=0):
    
    #convert to numpy if its not
    if type(img) is not np.ndarray:
        img = np.asarray(img)
    
    #some operation will directly impact the input image, so we must keep a copy of it
    imCopy = img.copy()
    
    ### convert to grayscale ###
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ### find contour ###
    #for better accuracy we use binary images before finding contours, applying threshold: 
    #if pixel is above 200 (first value, reducing to 160 may lead to to much images) we assign 255 (second value), 
    #below we assign 0 (third value).
    ret,thresh = cv2.threshold(imgray,200,255,0)
    image, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #create a list of rectangle which may correspond to an image
    li_bbox = []
    for contour in contours:
        poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, False), False)
        x, y, w, h = cv2.boundingRect(poly)
        #remove if its really small compared to initial (i.e. smaller than 10%) image or equal to the initial image (or half the page
        #in case the book was scanned two pages at a time (horizontally or vertically))
        hi, wi, ci = imCopy.shape
        #avoid: not to smalle image (bad quality or can even be logo etc)
        #avoid: equal to the hole page
        #avoid: equal to half page when scanned with two page on the width
        #avoid: equal to half page when scanned with two page on the height
        if (h>(hi*0.1)) & (w>(wi*0.1)) & \
        ((h<(hi*0.95))|(w<(wi*0.95))) & \
        ((h<(hi*0.95))|(w>(wi*0.55))|((wi*0.45)>w)) & \
        ((w<(wi*0.95))|(h>(hi*0.55))|((hi*0.45)>h)):
            li_bbox.append((x,y,w,h))
    
    #remove images included in another image
    li_bbox = remove_embedded_bbox(li_bbox)
    
    if plot_==1:
        for bbox in li_bbox:
            x,y,w,h = bbox
            # Create figure and axes
            fig,ax = plt.subplots(1)
            # Display the image
            ax.imshow(imCopy)
            # Create a Rectangle patch
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.show()
    
    #create directory if not existing
    if not os.path.exists(path):
        os.makedirs(path)
    
    #save
    for image_id,bbox in enumerate(li_bbox):
        x,y,w,h = bbox
        img_to_save = Image.fromarray(imCopy[y:y+h,x:x+w])
        img_to_save.save(os.path.join(path,'p'+str(page_id)+'_i'+str(image_id)+'.png'))  
        del img_to_save
      
    #TODO: verify if useful:
    del img
    del imCopy
    del imgray
        
        
def from_path_scannpdf_book_2image(path, path_save, nbrp=2, plot_=0):
    pages = convert_from_path(path)
    print('There is %d pages in the book'%len(pages))
    for i,page in enumerate(pages):
        frompng2images(img=page, path=path_save, page_id=i, plot_=plot_)    
    del pages
    
###################################################################################################
################################### preprocessing fct for image ###################################
###################################################################################################

#concatenate images one next to the other (i.e. to make nicer plot)
def concat_images(img1, img2, g=15):
    """ Combines two (colored or black and white) images ndarrays side-by-side """
    
    #from black and white convert to 3 channel
    if len(img1.shape)!=3:
        img1 = skimage.color.gray2rgb(imga)
    if len(img2.shape)!=3:
        img1 = skimage.color.gray2rgb(imgb)
        
    #concat with white gap
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    new_img = np.zeros(shape=(np.max([h1, h2]), w1+w2+g, 3))
    new_img[:h1,:w1]=img1
    new_img[:h2,w1+g:w1+w2+g]=img2
    new_img[:h2,w1:w1+g]=255
    return(new_img)

def concat_n_images(li_img, g=15):
    """ Combines N color images from a list of image """
    output = None
    for i, img in enumerate(li_img):
        if i==0:
            output = img
        else:
            output = concat_images(output, img, g=15)
    return(output.astype(int))

#il regarde la ou cest bleu

#################### construct a list of consecutive blak and white image (if 3 image then result is a image, if  more
#then its a pickle list)
#fct that construct one image from a list of images (one images as one channel, the first one is the 1st channel etc)
def construct_list_of_bw_image(li_images, li_index_to_keep=None):
    
    #convert images into black and white
    li_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in li_images]
    h = max([img.shape[0] for img in li_images])
    w = max([img.shape[1] for img in li_images])
    
    #if no index where given, then we will keep all
    if li_index_to_keep is None: 
        li_index_to_keep = range(0,len(li_images))
    
    #create new image made of black and white images
    new_img = np.zeros(shape=(h, w, len(li_index_to_keep)))
    for i,ind in enumerate(li_index_to_keep):
        new_img[:,:,i] = li_images[ind]
        
    return(new_img.astype(np.uint8))


#################### construct a channel image
def threeimg_to_1(h,w,img1,img2,img3):
    new_img = np.zeros(shape=(h, w, 3))
    new_img[:,:,0] = img1
    new_img[:,:,1] = img2
    new_img[:,:,2] = img3
    return(new_img)

def reduce_li_img(li):
    #convert images into black and white
    li = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) for img in li]
    h = max([img.shape[0] for img in li])
    w = max([img.shape[1] for img in li])
    #create a smaller list reducing each set of three images to one image
    li_smaller = []
    for i in range(0,len(li),3):
        li_smaller.append(threeimg_to_1(h, w, li[i], li[i+1], li[i+2]))    
    return(li_smaller)

def construct_image_3channels(li_images):
    '''fct that construct one image from a list of x images (one images as one channel, the first one is the 1st channel etc)'''
    #### check if its a power of 3, otherwise remove some images to have a power of 3
    li_power3 = [np.power(3,x) for x in range(1,len(li_images))]
    l = max([x for x in li_power3 if x<=len(li_images)])
    li =  li_images[0:l]
    #if l!=len(li_images):
    #    print('WARNING: we are not able to use %d images, as we need a power of 3 images'%(len(li_images)-l))
    
    #### now that we have a list of images of length a power of 3, we will recursively create images
    while len(li)!=1:
        li = reduce_li_img(li)
        
    return(li[0].astype(np.uint8))

def construct_3image_3channels(li_images):
    '''fct that construct one image from a list of 3 images (one images as one channel, the first one is the 1st channel etc)'''
    
    #convert images into black and white
    li = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) for img in li_images]
    h = max([img.shape[0] for img in li])
    w = max([img.shape[1] for img in li])
    
    #create a smaller list reducing each set of three images to one image
    new_img = np.zeros(shape=(h, w, 3))
    new_img[:,:,0] = li[0]
    new_img[:,:,1] = li[1]
    new_img[:,:,2] = li[2]

    return(new_img.astype(np.uint8))


################################

def binary_bbox_to_bbox(binary_mask):
    '''from a binary bbox (i.e. true where the rectangle-mask is and false elsewhere, to x,y,w,h bbox'''
    #first row with True: y1
    y1 = [i for i in range(len(binary_mask)) if True in binary_mask[i]][0]
    #last row with true: y2
    y2 = [i for i in range(len(binary_mask)) if True in binary_mask[i]][-1]
    h = y2-y1
    #first column with true: x1
    x1 = list(binary_mask[y1]).index(True)
    #last column with true: x2
    x2 = len(binary_mask[y1])-list(reversed(binary_mask[y1])).index(True)
    w = x2-x1
    return(x1,y1,w,h)


#from a bbox (x,y,w,h) output a polygons (can be hence used as a mask in mask-rcnn for e.g.)
def from_bbox_get_polygon(bbox):
    x,y,w,h = bbox
    all_points_x = [x, x + w, x + w, x]
    all_points_y = [y, y, y + h, y + h]  
    return(all_points_x, all_points_y)

def from_vggbbox_get_vggpolygon(x):
    x_poly = []
    for r in x:
        p = r['shape_attributes']
        all_points_x, all_points_y = from_bbox_get_polygon((p['x'], p['y'], p['width'], p['height']))
        x_poly.append({'shape_attributes':{'all_points_x':all_points_x,
                                           'all_points_y':all_points_y,
                                           'name':'polygon'}})
    return x_poly


#takes a list of bbox ([(x,y,w,h),...]) with the associated image and plot on the image
def plot_bboxes(li_bboxes, image, li_text=None):
    '''li_bboxes=[(x,y,w,h),...], while in maskrcnn rois its given like this: y1, x1, y2, x2'''
    
    #create plot with the bbox
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for bbox in li_bboxes:
        x,y,w,h = bbox
        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
    #add text
    if li_text!=None:
        for (x,y,s) in li_text:
            plt.text(x, y,s)   
            
    #show the plot        
    plt.show()


#remove from a list of rectangles (tuples: (x,y,w,h)), the rectangle embedded in another one
#note that the (0,0) point in an image is up left.
def remove_embedded_bbox(li_bbox, plot_bbox=0):
    
    #sort (smaller to bigger) list of rectangles by the highest height (to make things more efficient)
    li_bbox = sorted(li_bbox,key=itemgetter(3))
    
    #initialize list of rectangle to return
    li_bbox_r = li_bbox.copy()
    
    #remove all rectangle when its included in another one. Hence we will compare each rectangle with the one having
    #a higher height only (for efficiency). as soon as we see that the rectangle is included in another one we will remove it and pass 
    #to the next one
    for i,bbox in enumerate(li_bbox):
        for bbox2 in li_bbox[i+1:]:
            x1, y1, w1, h1 =  bbox
            x2, y2, w2, h2 =  bbox2
            if (w1<w2) & (x1>x2) & (y1>y2) & (x1+w1<x2+w2) & (y1+h1<y2+h2):
                li_bbox_r.remove(bbox)

                #plot (to debug)
                if plot_bbox==1:
                    #print(x1, y1, w1, h1)
                    #print('is included in :')
                    #print(x2, y2, w2, h2)
                    # Create figure and axes
                    fig,ax = plt.subplots(1)
                    # Display the image
                    ax.imshow(np.zeros(shape=(max(y1+h1,y2+h2)+50,max(x1+w1,x2+w2)+50)))
                    # Create a Rectangle patch
                    rect = patches.Rectangle((x1,y1),w1,h1,linewidth=1,edgecolor='r',facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)

                    rect = patches.Rectangle((x2,y2),w2,h2,linewidth=1,edgecolor='r',facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    plt.show()
                break
    return(li_bbox_r)
#small test for embeded images
#li_bbox = [(1281, 79, 933, 1425), (1557, 600, 282, 396)]
#remove_embedded_bbox(li_bbox,plot_bbox=1)

#take an image and return the image withou reflect
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
    return(result)

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
    return(result)


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


#function that resize image keeping the aspect ratio and adding less possible black pixel. In other words:
#function that takes as input an image, change the magnitude of the image to fit better the new dimension (n1, n2) and
# finaly resize it to fit exactly the dimension keeping the initial aspect ration and hence adding black pixel where 
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

#to use the preprocessing step used in inceptionv3
def image_augmentation_with_maskrcnn(ID, n1, n2, image_path, mask_path, augmentation=None, normalize=False, preprocessing=None,
                                    plot_3_image=False):
    
    #download image and mask
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


#function that from mask output of the mask-RCNN model give the VGG mask anotation: li_all_points_x, li_all_points_y of the length of the number of masks (R should be of format of mask output from mask rcnn)
def convert_binary_mask_to_VGG_polygons(R):
    R = R.astype(np.uint8)
    l, c, channel = R.shape
    li_R = []
    for i in range(channel):
        li_R.append(R[:,:,i])
    len(li_R)
    li_seg = []
    for r in li_R: 
        contours = measure.find_contours(r, 0.5)
        seg = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            seg.extend(segmentation) #havent check when their is several contours in one mask
        li_seg.append(seg)
    li_all_points_x = []
    li_all_points_y = []
    for i in range(channel):
        li_all_points_x.append([li_seg[i][j] for j in range(len(li_seg[i])) if j%2==0]) #all even numbers
        li_all_points_y.append([li_seg[i][j] for j in range(len(li_seg[i])) if j%2!=0]) #all odd numbers
    return(li_all_points_x, li_all_points_y)
#all_points_x, all_points_y = convert_binary_mask_to_VGG_polygons(r['masks'])
#create bbox out of binary mask
#l, c = R.shape[0:2]
#R = R.astype(np.uint8)
#R.resize([l, c])
#fortran_R = np.asfortranarray(R)
#encoded_ground_truth = mask.encode(fortran_R)
#ground_truth_area = mask.area(encoded_ground_truth)
#ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)

#taken from: https://stackoverflow.com/questions/2573997/reduce-number-of-points-in-line
def _vec2d_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _vec2d_sub(p1, p2):
    return (p1[0]-p2[0], p1[1]-p2[1])

def _vec2d_mult(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1]

def ramerdouglas(line, dist):
    """Does Ramer-Douglas-Peucker simplification of a curve with `dist`
    threshold.
    line: is a list-of-tuples, where each tuple is a 2D coordinate
    """
    if len(line) < 3:
        return line

    (begin, end) = (line[0], line[-1]) if line[0] != line[-1] else (line[0], line[-2])

    distSq = []
    for curr in line[1:-1]:
        tmp = (
            _vec2d_dist(begin, curr) - _vec2d_mult(_vec2d_sub(end, begin), _vec2d_sub(curr, begin)) ** 2 / (0.001+_vec2d_dist(begin, end)))
        distSq.append(tmp)

    maxdist = max(distSq)
    if maxdist < dist ** 2:
        return [begin, end]

    pos = distSq.index(maxdist)
    return (ramerdouglas(line[:pos + 2], dist) + 
            ramerdouglas(line[pos + 1:], dist)[1:])


###################################################################################################
########################################## datagenerator ##########################################
###################################################################################################

#in case you have images in train/test/val folder and which does not need other specific preprocessing (i.e. not put black partout) you can simply use:
'''test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary') ... '''

#if you have images in one folder and csv fro training/testing/val (and images without mask)
class DataGenerator_simple(keras.utils.Sequence):
    '''Generates data for Keras
    labels: class names'''
    def __init__(self, list_IDs, labels, image_path, batch_size, n_rows, n_cols, n_channels, n_classes, 
                 augmentation=None, preprocessing=None, shuffle=True, normalize=False):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.batch_size = batch_size
        self.image_path = image_path
        self.labels = labels
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
        X = np.empty((self.batch_size, self.n_rows, self.n_cols, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp): 
            
            #handle image
            image = cv2.imread(os.path.join(self.image_path, ID+'.jpg'))
            b,g,r = cv2.split(image)
            image = cv2.merge([r,g,b])
            #augment image
            if self.augmentation is not None:
                image = self.augmentation.augment_image(image)
            #adapt to desired size
            image = adjust_size(image, self.n_rows, self.n_cols)
            X[i,] = image
            
            #handle class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    


#with mask (for inceptionv3 for example)
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
        #self.age = age
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
            image = image_augmentation_with_maskrcnn(ID=ID, n1=self.n1, n2=self.n2, 
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
########################################### other & old ###########################################
###################################################################################################

#removes element from two lists based on the entry of the first one
def lists_remove_in1(li1,li2,s):
    #remove s until no more s
    if s in li1:
        ind = li1.index(s)
        del li1[ind]
        del li2[ind] 
        return(lists_remove_in1(li1,li2,s))
    else:
        return(li1,li2)
#small example
#li1_, li2_ = lists_remove_in1(['MA', 'A', 'fhsgdf','MA'], [12,34,'s',46],'MA')
#print(li1_)
#print(li2_)


#round to the closest number with base 5
def myround(x, base=5):
    return int(base * round(float(x)/base))

#this fct was taken from internet then modified. It generate a list og random color
def random_colors(N, bright=True):
    """
    To get visually distinct colors, generate them in HSV space then convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    #convert from 0-1 to 0-255 integers
    colors = [ (int(i[0]*255), int(i[1]*255), int(i[2]*255)) for i in colors]
    return(colors    )
    
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


#taken from: 
#from:https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
def histogram(ratings):
    """Returns the counts of each type of rating that a rater made"""
    min_rating = min(ratings)
    max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings
def quadratic_weighted_kappa(rater_a, rater_b):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    min_rating = min(min(rater_a), min(rater_b))
    max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, 0, 4)
    hist_rater_b = histogram(rater_b, 0, 4)
    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


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


#from a string x withoutwhitespace, and a list of whitespace index, it return the string wiht the adequate whitespace
def from_string_without_whitespace_to_string_withwithespace(x, li_index):
    initial_length = len(x)+len(li_index)
    for i in range(initial_length):
        if i in li_index:
            x = x[0:i] + ' ' + x[i:]
    return(x)
#small example
#x = 'ab asd whfjzf gdzf  fuj'
#x_ = ''.join(x.split(' '))
#li_index = [m.start() for m in re.finditer(' ', x)]
#from_string_without_whitespace_to_string_withwithespace(x_, li_index)


#given a list of text without any whitespace and a list of whitespace index corresponding to its original whitespace
#places, it will outputa list with whitespace at the coret places (not at end or begning of entries if their was any)
def from_string_without_whitespace_to_string_withwithespace(li_x, li_index):
    
    #removing space at end and begining
    li_x = [x.strip() for x in li_x]
    
    #initialisation
    x = ''.join(li_x)
    li_x_r = []
    initial_length = len(x)+len(li_index)
    li_split_index = [len(x) for x in li_x]
    li_split_index = [sum(li_split_index[0:i])-1 for i in range(1,len(li_split_index))]
    last_split_index = -1
    
    #pass through each index
    for i in tqdm.tqdm(range(initial_length)):
        #if it should have a whitespace at this index
        if i in li_index:
            x = x[0:i] + ' ' + x[i:]
            li_split_index = [i+1 for i in li_split_index]
            #print('whitespace',i,li_split_index)
            
        #if it should be splitted at this place
        if i in li_split_index:
            #print('splitted',i,li_split_index)
            li_x_r.append(x[last_split_index+1:i+1])
            last_split_index = i
            
    #add last part and return
    li_x_r.append(x[last_split_index+1:])
    return([i.strip() for i in li_x_r if i!=' '])
#small example: from a text and a list of title, without taking into account whitespace, we want to split it, keeping
#at the end the whitespace too
#text = 'hello snake1 and goodbye snake  2b jkjk labla snake3 '
#li_title = ['snake1', 'snake 2', 'snake3']
#text_nws = ''.join(text.split(' '))
#li_title_nws = [''.join(x.split(' ')) for x in li_title]
#print(li_title_nws)
#pattern = ''
#for p in li_title_nws:
#    pattern = pattern+'|'+p
#pattern = pattern.strip('|')
#print(pattern)
#pattern = re.compile(r'(%s)'%pattern)
#li_text_nws = pattern.split(text_nws)
#print(li_text_nws)
#li_ws_index = [m.start() for m in re.finditer(' ', text)]
#print(li_ws_index)
#from_string_without_whitespace_to_string_withwithespace(li_text_nws, li_ws_index)


#from doxc file extract all the bold text , outputing one string. the idea is to extract letter by letter and when one is not bold, we will not add to the ouput (except when its a whitespace and before was a bold letter
def extract_bold_text(document):
    li_bolds = []
    for para in document.paragraphs:
        last_was_bold = 0
        li_bolds.append(' ')
        for run in para.runs:
            if run.bold:
                li_bolds.append(run.text)
                last_was_bold = 1
            elif (last_was_bold==1) & (run.text==' '):
                li_bolds.append(run.text)
                last_was_bold = 0
    return(''.join(li_bolds))


#from a text (chapter text) and with a list of bold-title to find, we will output the text splitted with the titles
#or closest matched titles
def from_chapter_to_structured_data(text, li_title):
    
    #remove all whitespace as these are not equally ditributed in the bold or in the text outptu
    text_nws = ''.join(text.split(' '))
    li_title_nws = [''.join(x.split(' ')) for x in li_title]
    
    #get index of whitespace in original text
    li_ws_index = [m.start() for m in re.finditer(' ', text)]
    
    #create a list of titles which all match 
    li_matched_title = []
    title_not_matched = []
    li_distance = []
    for i,title in enumerate(li_title):
        r = find_near_matches(title, text_nws, max_deletions=max(int(0.10*len(title)),1), 
                              max_insertions=max(int(0.05*len(title)),1), max_substitutions=0)
        if len(r)==1:
            li_matched_title.append(text[r[0][0]:r[0][1]])
            li_distance.append(r[0][2])
        #keep track of non-matched title (to add rules perhaps or allow more flexibility: TODO)
        elif len(r)==0:
            print(title)
            title_not_matched.append(title)
        else:
            print(r)

    #create a list from text by splitting it with the titles
    pattern = ''
    for p in li_matched_title:
        pattern = pattern+'|'+p.replace('(','\(').replace(')','\)').replace('|','\|') #caractere not supp in regex without backslash
    pattern = pattern.strip('|')
    pattern = re.compile(r'(%s)'%pattern)
    li_text_nws = pattern.split(text_nws)
    
    #compute and return the splited list with adequate whitespace
    r = from_string_without_whitespace_to_string_withwithespace(li_text_nws, li_ws_index)
    return(r, title_not_matched, li_matched_title, li_distance)



###################################################################################################
####################################### manage data  for ML #######################################
###################################################################################################

def split_test_train_within_cat(df, p_test, category_to_split_within, id_to_split_with):
    
    #create lists (one test one train) of id within each category 
    li_test = []
    li_train = []
    for i,j in df.groupby([category_to_split_within]):
        li = list(j[id_to_split_with].unique())
        #shuffle list
        random.shuffle(li)
        n1 = int(len(li)*p_test)
        li_test.extend(li[0:n1])
        li_train.extend(li[n1:])
        
    #create associated dataframes    
    df_test = df[df[id_to_split_with].isin(li_test)]
    df_train = df[df[id_to_split_with].isin(li_train)]
    return(df_test, df_train)


#for regression models
def df_to_arrays(df, names=None):
    """ Transforms the columns or arrays from the input into an output consisting of two arrays.
    Inputs:
    df - a dataframe, a numpy.ndarray with 2 columns or a list with 2 elements (either of type numpy.array or pandas.core.series.Series)
    names - list of names for the columns if the first argument is a dataframe. names[0] should be all features names, names[1 should be the value to predict
    Outputs: the resulting arrays
    """
    if isinstance(df, pd.core.frame.DataFrame):
        if not names:
            raise ValueError('Names of columns to be analyzed in the df need to be given')
        else:
            if len(names[1])!=1:
                raise ValueError('the second element from names var should be only the predicted var name, i.e. of lenght 1')
            #remove nan
            df = df.dropna(subset=names[0]+names[1])
            x = df.loc[:, names[0]]; y = df.loc[:, names[1]]
            
    elif isinstance(df, numpy.ndarray):
        if df.shape[1] != 2:
            raise ValueError('The array needs to have eaxctly 2 columns')
        else:
            x = df[:,0]; y = df[:,1]
            
    else:    
        if len(df) != 2:
            raise ValueError('List of exactly 2 elements (pandas.core.series.Series or numpy.ndarray) needs to be given')
        else:
            x = df[0]; y = df[1]
    return(x, y)

###################################################################################################
############################################### plot ##############################################
###################################################################################################

# will be used in the next fct. Its is used to create a list of length x for the explode parameter in the donut plot
def list_same_number_with_threshold(x, v, nbr_set, nbr_without_explode):
    if x<nbr_without_explode:
        return(np.repeat(0,x))
    else:
        li = list(np.repeat(0,nbr_without_explode))
        r = x-nbr_without_explode
        n = int(r/nbr_set)
    for i in range(nbr_set):
        li.extend(list(np.repeat(v*(i+1), n)))
    if len(li)<x:
        li.extend(list(np.repeat(v*nbr_set,x-len(li))))
    return(li)    


#create a donut plot based on two lists, one for the names, and one for the associated quantity
def donut_plot(li_labels, li_sizes, path, min_val=None, v=0.3, nbr_without_explode=50, fontsize_=6, circle_hole=0.75, nbr_set=5):
    
    #sort list of tuples according to second value
    t = [(li_labels[i], li_sizes[i]) for i in range(len(li_labels))]
    t.sort(key=lambda x: x[1])
    t.reverse()
    if min_val==None:
        li_labels = [i[0] for i in t]
        li_sizes = [i[1] for i in t]
    else:
        li_labels = [i[0] for i in t if i[1]>=min_val]
        li_sizes = [i[1] for i in t if i[1]>=min_val]

    #plot
    fig1, ax1 = plt.subplots()
    li_ = list_same_number_with_threshold(len(li_labels),v=v, nbr_set=nbr_set, nbr_without_explode=nbr_without_explode)
    ax1.pie(li_sizes, labels=li_labels, startangle=90, explode=li_, rotatelabels=True, textprops={'fontsize': fontsize_})
    
    #circle
    centre_circle = plt.Circle((0,0),circle_hole,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')  #ensures that pie is drawn as a circle
    plt.tight_layout()
    
    #save and show
    plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
    plt.show()        
    
    
###################################################################################################
######################################### work with videos ########################################
###################################################################################################

def remove_isolated_index(li, isol):
    '''
    function that takes a list of '0' and '1' entries and replace to much isolated '1' by '0'
    ----li: list of '0' and '1' entries
    ----isol: nbr of entries to the left and to the right which must be 1 in order to keep the particular entry to 1
    note that in the case where we will replace one entry which should not have been replaced if we new the previous or next
    value (outside from this list), it also means that it wont have any impact on the ones of the list, so no importance 
    e.g. [...,1][0,1,0,0,0,1,0,0,1,1,1,0,0,0,1,0,0,0,...]'''
    li_index = [i for i in range(len(li)) if ((sum(li[max(i-isol,0):i+isol+1])-li[i])>=1) & (li[i]==1)]
    return([1 if i in li_index else 0 for i in range(len(li))])
#small example
#li = [0,1,1,0,0,0,0,0,0,0,1,0,0,0,0]
#li = [0,1,1,0,0,0,0,1,0,0,1,0,0,0,1]
#remove_isolated_index(li,3)

#GPU. 1-7%, processeur: 40-50%, mémoire GPU dédié: 3,3/4, mémoire GPU partagé: 0,1/7,9
#we will save the video to have a visual, but later the purpose is to save representative images of detected fish
#sorted(li_video_paths, reverse=True)
def reduce_video_size(path_initial_video, path_treated_info, algo_name, model, img_cols, img_rows, batch_size, save_video=True,
                      save_images_3in1=False, save_images_lonely=False, save_full_video_with_text=False, debug=False,
                      nbr_frames_ba=1, careful_index=1, img_end='.jpg', width=600, height=480, crf=10):
    
    '''    
    ----path_initial_video: path to the video to reduce size
    ----path_treated_info: path to the folder where the treated info will be saved, it will create an images and a videos folder
    ----algo_name: string representing the name of the algo, to save the images and video produced thanks to this algo
    ----model: model to use for fish detection.It must have been trained on 3 channels images created from 3 consecutives images
    ----imgcols, img_rows: img dimension going with the model
    ----batch_size: corresponding batchsize of the model
    ----save_images_3in1: if True, it will save each images used as input in the algo, where a fish was detected with the 
        certainty in the name, even if we did not used the image in the video (e.g. if the image was alone in the set of 
        consecutives images). Hence, the purpose of these images is to understand better the waeknesses of the algorithme to 
        help improvement, it may be usefull to retrain the fast algo on these images outputed from video without fish 
    ----save_images_lonely: same as save_images_3in1 except that it will save the three consecutvies images instead of the 
        3in1, it may be usefull to train the heavier algo on these images outputed from video without fish
    ----save_video: if true, will save the video in purpose to give it to human or heavier algo (i.e. saving 3 images next to
        prediction not saving when only one prediction in the next x frames, saving list of same size as nbr of frames saved, 
        given the info of the exact time in the initial video)
    ----save_full_video_with_text: if True, save_video must be True as well. In this case its asking to save the full video with
        text 'fish'/'no_fish' on the frames. Otherwise, it will save the smaller video with no text but only part with
        fish detections
    ----nbr_frames_ba: number of set of 3frames to save in smaller video, next to the one where a fish was detected
    ----careful_index: number of frames before and after that must have at least one other fish prediction to take into account 
        the actual fish pred. If==0: we always count the fish pred, if =10: 10*3 frames before and 10*3 frames after must have 
        at list one other fish prediction to save it. 
    ----img_end: format to save images: png jpg...
    ----width, height, crf: parameters given in the FFmpegWriter fct, parameter to save video properly
    ----debug: if True (save_video must be true and save_full_video_with_text must be false) it will save all images used in 
        smaller film for verification and all images from the initial video with their predictions
    '''
    
    #start timer
    start = time.time()
    
    #small test: are the parameters coherent?
    if (save_video==False) & (save_full_video_with_text==True):
        print('ERROR: you can not ask not to save video, but to save the full video with text')
        sys.exit()
        
    #small test: does the impact of previous/next frames is possible with the actual batchsize?
    #maximum number of previous and next frames we need to look at to decide weither we must save or not a particular frame
    first_set_starting = max(careful_index, nbr_frames_ba)
    if (first_set_starting*2+1) > batch_size:
        print('max(careful_index*2,nbr_frames_ba) parameter must be smaller or equal to half of the batchsize')
        sys.exit()
        
    #small test:  check if video exists
    if len(glob.glob(path_initial_video))!=1:
        print('the video does not exist at your path')
        sys.exit()
    
    #create path to save images and create name of smaller video
    path_img_treated = os.path.join(path_treated_info, 'images', algo_name+'_'+path_initial_video.split('\\')[-1].split('.')[0])
    path_img_treated_3in1 = os.path.join(path_img_treated, '3in1')
    path_img_treated_lonely = os.path.join(path_img_treated, 'lonely')
    path_img_debuginit = os.path.join(path_img_treated, 'debug','init')
    path_img_debugsaved = os.path.join(path_img_treated, 'debug','saved')
    path_img_debugcorrectindex = os.path.join(path_img_treated, 'debug','correctindex')
    path_vid_treated = os.path.join(path_treated_info, 'videos')
    if not os.path.exists(path_img_treated_3in1):
        os.makedirs(path_img_treated_3in1)
    if not os.path.exists(path_img_treated_lonely):
        os.makedirs(path_img_treated_lonely)
    if not os.path.exists(path_img_debuginit):
        os.makedirs(path_img_debuginit)
    if not os.path.exists(path_img_debugsaved):
        os.makedirs(path_img_debugsaved)
    if not os.path.exists(path_img_debugcorrectindex):
        os.makedirs(path_img_debugcorrectindex)
    if not os.path.exists(path_vid_treated):
        os.makedirs(path_vid_treated)

    #read video and output info on the video
    video = cv2.VideoCapture(path_initial_video)
    fps = video.get(cv2.CAP_PROP_FPS)      
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = int(frameCount)/(fps+0.00000001) #if empty video
    minutes = int(duration/60)
    seconds = duration%60
    print('INITIAL VIDEO:  seconds=%d (=%dmin %dsec), fps=%d, number of frames=%d '%(duration,int(minutes),int(seconds),fps,
                                                                                     int(frameCount)))
    if fps==0:
        return

    #info
    dico_classid_classname = {0:'no-fish', 1:'fish'}
    dico_class_color = {'no-fish':(139, 0, 0), 'fish':(0, 139, 0)}

    #define writer to save the annotated video
    #take same nbr of fps as the initial video
    writer = skvideo.io.FFmpegWriter(os.path.join(path_vid_treated, 
                                                  'complexer_'+algo_name+'_'+path_initial_video.split('\\')[-1]),
                inputdict={'-r': str(fps), '-s':'{}x{}'.format(width,height)},
                outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf), '-preset': 'ultrafast', '-pix_fmt':'yuvj420p'}
    ) #yuv444p, crf=0:lossless: (no loss in compression) constant rate factor, '-vcodec': 'libx264': use the h.264 codec

    #initialisation
    #number of frames taken from video, might be bigger than the total available number of frames, if its not divisible by 
    #batchsize
    k = 0 
    something_saved = False
    pred_old = None
    li_index_savedimg = []
    li_sec_savedimg = []
    nbr_to_let = 5
    
    #loop over frames from the video file stream
    while True:

        #grab the same nbr of images as the batchsize*3 (as it will then be reduced to a list of batchsize images)
        li_images = []
        for i in range(batch_size*3):
            #take frames & check to see if we have reached the end of the video in which case we go out of the for loop
            #but continue by adding necessary missing mages to have the expected batchsize nbr 
            (grabbed, image) = video.read()
            if not grabbed:
                break    
                print('break')
            k = k+1
            li_images.append(image)

        ##################################################### detect fish #####################################################
        #add images to complete batch
        try:
            if (batch_size*3)!=len(li_images):
                li_images = li_images+[li_images[-1] for i in range(0,(batch_size*3)-len(li_images))]
                #adapt k as well
                print('instead of %d we will use %d'%(k,75*math.ceil(k/75)))
                k = batch_size*3*math.ceil(k/(batch_size*3)) #round to the next highest multiple of 75

            #construct one colored image from 3 consecutives images
            li_img = [construct_3image_3channels(li_images[(i*3):(i*3+3)]) for i in range(batch_size)]
            #detect fish (first put into adequate format)
            images = [cv2.resize(img,(img_cols,img_rows),interpolation=cv2.INTER_CUBIC) for img in li_img]
            images = [np.reshape(img,[1,img.shape[0],img.shape[1],3]) for img in images]
            images = np.vstack(images)
            pred = model.predict(images)
            
        except Exception as e:
            print('ERROR in predictiong fish: ', e)
            print(k)
            print(len(li_images))
            pred = None
            sys.exit()
        
        ################################################ save smaller video #################################################
        if (save_full_video_with_text==False) & (save_video):                   
            
            #if first iteration treat all within the first first_set_starting frames
            #otherwise treat from first_set_starting to batch_size+first_set_starting using possibly all others value as well
            if pred_old is None:
                
                #create list of class
                li_pred_all = [np.argmax(pred[m]) for m in range(len(pred))]
                #lets remove the 'fish' frames when they are isolated (i.e. with repsect to def of careful_index)
                li_pred_all = remove_isolated_index(li_pred_all, careful_index)

                #print(li_pred_all) #verified
                #if last batch (i.e. when video has even less than ~batchsize*3 frames), then we must as well consider last 
                #first_set_starting images for saving
                until = first_set_starting
                if k>=frameCount-nbr_to_let:
                    until = len(li_pred_all)
                #select the correct images to save
                for p in range(0, until):
                    if sum(li_pred_all[max(p-nbr_frames_ba,0):min(p+nbr_frames_ba+1,len(li_pred_all))])>0:
                        #print('save:', p)
                        writer.writeFrame(li_images[p*3])
                        writer.writeFrame(li_images[p*3+1])
                        writer.writeFrame(li_images[p*3+2])
                        #print(k, p)
                        k1 = k-(batch_size*3-(p*3+1))
                        k2 = k-(batch_size*3-(p*3+2))
                        k3 = k-(batch_size*3-(p*3+3))
                        li_index_savedimg.extend([k1, k2, k3])
                        li_sec_savedimg.extend([k1/fps, k2/fps, k3/fps])
                        something_saved = True
                        #save all images used in smaller film for verification
                        if debug:
                            imageio.imwrite(os.path.join(path_img_debugsaved, str(k1)+'_'+str(round(k1/fps,2))+img_end), 
                                            li_images[p*3])                             
                            imageio.imwrite(os.path.join(path_img_debugsaved, str(k2)+'_'+str(round(k2/fps,2))+img_end), 
                                            li_images[p*3+1])  
                            imageio.imwrite(os.path.join(path_img_debugsaved, str(k3)+'_'+str(round(k3/fps,2))+img_end), 
                                            li_images[p*3+2])                        
            else:
                li_all_images = li_images_old+li_images
                pred_all = list(pred_old.copy())
                pred_all.extend(list(pred))
                #create list of class
                li_pred_all = [np.argmax(pred_all[m]) for m in range(len(pred_all))]
                #lets remove the 'fish' frames when they are isolated (i.e. with respect to def of careful_index)
                li_pred_all = remove_isolated_index(li_pred_all, careful_index)

                #small verification of size
                if len(li_all_images)!=2*3*batch_size:
                    print(len(li_all_images))
                    print('ERROR: not enough data!')
                    sys.exit()
                    
                #print(li_pred_all)  #verified
                #select the correct images to save
                #if last batch, then we must as well consider last first_set_starting images for saving
                until = batch_size+first_set_starting
                if k>=frameCount-nbr_to_let:
                    until = len(li_pred_all)
                    
                for p in range(first_set_starting, until):
                    if sum(li_pred_all[p-nbr_frames_ba:min(p+nbr_frames_ba+1,len(li_pred_all))])>0:
                        #print('save:', p, k) 
                        writer.writeFrame(li_all_images[p*3])
                        writer.writeFrame(li_all_images[p*3+1])
                        writer.writeFrame(li_all_images[p*3+2])
                        k1 = k-(batch_size*3*2-(p*3+1))
                        k2 = k-(batch_size*3*2-(p*3+2))
                        k3 = k-(batch_size*3*2-(p*3+3))
                        sec1 = k1/fps
                        sec2 = k2/fps
                        sec3 = k3/fps
                        li_index_savedimg.extend([k1,k2,k3])
                        li_sec_savedimg.extend([sec1, sec2, sec3])
                        something_saved = True
                        
                        #save all images used in smaller film for verification
                        if debug:
                            imageio.imwrite(os.path.join(path_img_debugsaved, str(k1)+'_'+str(round(sec1,2))+img_end), 
                                            li_all_images[p*3])                             
                            imageio.imwrite(os.path.join(path_img_debugsaved, str(k2)+'_'+str(round(sec2,2))+img_end), 
                                            li_all_images[p*3+1])  
                            imageio.imwrite(os.path.join(path_img_debugsaved, str(k3)+'_'+str(round(sec3,2))+img_end), 
                                            li_all_images[p*3+2])
                            
        #################################### save images and full video with annotation #####################################
        for i in range(len(li_img)):
            
            #find predictions output
            pred_class = dico_classid_classname[np.argmax(pred[i])]
            k1 = str(k-(len(li_img)-i)*3+1)
            k2 = str(k-(len(li_img)-i)*3+2)
            k3 = str(k-(len(li_img)-i)*3+3)
            proba_t = str('-'.join([str(round(l,3)) for l in pred[i]]))
            
            #save the image used to detect fish (i.e. three images into one)
            if (pred_class=='fish') & (save_images_3in1):
                imageio.imwrite(os.path.join(path_img_treated_3in1, k1+'_'+ k2+'_'+ k3+'_P'+proba_t+img_end), li_img[i])
            
            if (pred_class=='fish') & (save_images_lonely):
                imageio.imwrite(os.path.join(path_img_treated_lonely, k1+'_P'+proba_t+img_end), li_images[i*3])                   
                imageio.imwrite(os.path.join(path_img_treated_lonely, k2+'_P'+proba_t+img_end), li_images[i*3+1])   
                imageio.imwrite(os.path.join(path_img_treated_lonely, k3+'_P'+proba_t+img_end), li_images[i*3+2])   
            
            #save absolutely all images one by one for verification (specifically for smaller video creation)
            if debug:
                imageio.imwrite(os.path.join(path_img_debuginit, k1+'_P'+proba_t+img_end), li_images[i*3])                   
                imageio.imwrite(os.path.join(path_img_debuginit, k2+'_P'+proba_t+img_end), li_images[i*3+1])   
                imageio.imwrite(os.path.join(path_img_debuginit, k3+'_P'+proba_t+img_end), li_images[i*3+2])   
                
            #save the complete video with text
            if save_full_video_with_text & save_video:
                #add text: add fish/nofish class on image
                cv2.putText(li_images[i*3], pred_class, (150,155), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            dico_class_color[pred_class], 1, cv2.LINE_AA)
                cv2.putText(li_images[i*3+1], pred_class, (150,155), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            dico_class_color[pred_class], 1, cv2.LINE_AA)
                cv2.putText(li_images[i*3+2], pred_class, (150,155), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            dico_class_color[pred_class], 1, cv2.LINE_AA)
                #add frame to video
                writer.writeFrame(li_images[i*3])
                writer.writeFrame(li_images[i*3+1])
                writer.writeFrame(li_images[i*3+2])
                something_saved = True 

        ###################################### stop or continue loop, save old info ######################################
        #does not worth looking again into the last few frames as it will add some images to attain batchsize
        if k>=frameCount-nbr_to_let:            
            #go out of the while loop
            break
            
        #keep in memory the predictions & the original list of images in case we need to save some due to fish detection in the
        #next batch note that as the text happen only when we save the full video, in which case we wont use the old list
        pred_old = pred.copy()
        li_images_old = li_images.copy()
           
    #close videos
    try:
        if something_saved:
            writer.close()
        else:
            print('empty video, no video saved')
    except Exception as e:
        print('ERROR: in closing the video: ', e)

    end = time.time()
    sec = end-start
    print('--> video analysed during %.2f percent of its total time (total running time: %.2fmin)'%(sec/duration*100,sec/60))
    
    #give info on smaller video if it was asked to save the smaller video
    if (save_full_video_with_text==False) & (save_video): 
        smaller_video = cv2.VideoCapture(os.path.join(path_vid_treated, 
                                                      'complexer_'+algo_name+'_'+path_initial_video.split('\\')[-1]))
        fps = smaller_video.get(cv2.CAP_PROP_FPS)      
        frameCount = int(smaller_video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration2 = frameCount/(fps+0.000001)
        minutes = int(duration2/60)
        seconds = duration2%60
        print('SMALLER VIDEO:  seconds=%d (=%dmin %dsec), fps=%d, number of frames=%d '%(duration2,int(minutes),int(seconds),
                                                                                         fps,frameCount))
        print('--> it lasts %d percent of the total time of the initial one'%(duration2/duration*100))
    print('------------Finish \n')
    #save seconds of saved frames
    pickle.dump(li_index_savedimg, open(os.path.join(path_vid_treated,
                'li_index_savedimg_complexer_'+algo_name+'_'+path_initial_video.split('\\')[-1].replace('.mp4','.pkl')), 'wb'))
    pickle.dump(li_sec_savedimg, open(os.path.join(path_vid_treated,
                'li_sec_savedimg_complexer_'+algo_name+'_'+path_initial_video.split('\\')[-1].replace('.mp4','.pkl')), 'wb'))

    if debug:
        video = cv2.VideoCapture(path_initial_video)
        k = 0 #number of frames taken from video
        while True:
            (grabbed, image) = video.read()
            if not grabbed:
                break    
            k = k+1
            imageio.imwrite(os.path.join(path_img_debugcorrectindex, str(k)+img_end), image)                   



#from a video save the most dissimilar images and several cosnecutively
def create_1dissimilar_consecutive_frames(video_path, video_name, path_save_images, image_name_init='', 
                                          nbr_consec=12, video_change_to_file=None):
    '''save the nbr_consev first images'''
    #initialise video path
    vp = os.path.join(video_path, video_name)
    
    #check if video exists
    if len(glob.glob(vp))!=1:
        print('the video does not exist at your path: %s'%vp)
        sys.exit()

    #read video (create a threaded video stream)
    video = cv2.VideoCapture(vp)
        
    #loop over frames from the video file stream
    for n in range(nbr_consec):

        #take frames and check if we have reached the end of the video
        (grabbed, image) = video.read()
        if not grabbed:
            break

        #put into black and white
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageio.imwrite(os.path.join(path_save_images, 
                                     image_name_init+video_name.split('.')[0]+'_'+str(n+1)+'.jpg'), image)

    #close video
    video.release()
    
    #when all is finish put video in the 'done' folder (and remove from the other folder)
    if video_change_to_file is None:
        video_change_to_file = os.path.join(video_path, 'done') 
    #os.rename(vp, os.path.join(video_change_to_file, video_name) )
    shutil.move(vp, os.path.join(video_change_to_file, video_name) )
    
    
    
#from a video save the most dissimilar images and several cosnecutively
def create_dissimilar_consecutive_frames_3consimg(video_path, video_name, path_save_images, gap, sim_index, 
                                                  image_name_init='', nbr_consec=3, first_number_frames_to_consider=100000,
                                                  video_change_to_file=None, save_img_on_first_it=False):
    '''save the maximum of non-similar nbr_consev images '''
    #initialise video path
    vp = os.path.join(video_path, video_name)
    
    #check if video exists
    if len(glob.glob(vp))!=1:
        print('the video does not exist at your path: %s'%vp)
        sys.exit()

    #read video (create a threaded video stream)
    video = cv2.VideoCapture(vp)
        
    #loop over frames from the video file stream
    k = 0
    id_ = 0
    while True:

        #take frames and check if we have reached the end of the video
        (grabbed, image) = video.read()
        if not grabbed:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        id_ = id_+1
        
        if save_img_on_first_it:
            if id_ == 1:
                imageio.imwrite(os.path.join(path_save_images, 
                             image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_1.jpg'), 
                image)

                for n in range(nbr_consec-1):

                    #take frames, check if we have reached the end of the video, if not put in black and white and update the id_
                    (grabbed, image) = video.read()
                    if not grabbed:
                        break
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    id_ = id_+1

                    imageio.imwrite(os.path.join(path_save_images, 
                                                 image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_'+str(n+2)+'.jpg'), 
                                    image)
        
        if image is not None: 
            
            #if no benchmarking image yet create one
            if k==0:
                #last image for comparaison
                if image is not None:
                    im_compared = image.copy()    
                k = 1

            #see if image should be save, i.e. if enough dissimilar from the last annotated one
            elif k==1:

                #compute similarity between two possible consecutive annotated images
                sim = compare_ssim(im_compared, image, multichannel=False)

                #if not that similar from last annotation image, save with the next 'nbr_consec-1' frames as one image 
                #& updated benchmarking image
                if sim<sim_index:
                    imageio.imwrite(os.path.join(path_save_images, 
                                                 image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_1.jpg'), 
                                    image)
                    
                    for n in range(nbr_consec-1):
                        
                        #take frames, check if we have reached the end of the video, if not put in black and white 
                        #note that we wont update the id_ for image retrieval
                        (grabbed, image) = video.read()
                        if not grabbed:
                            break
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                      
                        imageio.imwrite(os.path.join(path_save_images, 
                                                     image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_'+str(n+2)+'.jpg'), 
                                        image)

   
                    #last image for comparaison
                    if image is not None:
                        im_compared = image.copy() 
                    k = k+1

                #if similar save for detection and continue until find one not similar to save
                else:
                    k = 1
            else:
                k = k+1

            if k%(gap+2)==0:
                k = 1
            
            id_ = id_+1
            #to be verified exactly
            if id_>=first_number_frames_to_consider:
                break

    #close video
    video.release()
    
    #when all is finish put video in the 'done' folder (and remove from the other folder)
    if video_change_to_file is None:
        video_change_to_file = os.path.join(video_path, 'done') 
    #os.rename(vp, os.path.join(video_change_to_file, video_name) )
    shutil.move(vp, os.path.join(video_change_to_file, video_name) )
    
    
#from a video save the most dissimliar images
def create_dissimilar_consecutive_frames(video_path, video_name, path_save_images, gap, sim_index, reverse_rgb=False, image_name_init=''):
    
    #initialise video path
    video_path = os.path.join(video_path, video_name)
    
    #check if video exists
    if len(glob.glob(video_path))!=1:
        print('the video does not exist at your path: %s'%video_path)
        sys.exit()

    #read video 
    video = cv2.VideoCapture(video_path)
        
    # loop over frames from the video file stream
    k = 0
    id_ = 0
    while True:

        #take frames and check if we have reached the end of the video
        (grabbed, image) = video.read()
        if not grabbed:
            break

        if reverse_rgb:
            b,g,r = cv2.split(image)           
            image = cv2.merge([r,g,b])
         
        if image is not None: 
            #if no benchmarking image yet create one
            if k==0:
                im_compared = image.copy()
                k = 1

            #see if image should be savec, i.e. if enough dissimilar from the last annotated one
            elif k==1:

                #compute similarity between two possible consecutive annotated images
                sim = compare_ssim(im_compared, image, multichannel=True)

                #if not that similar from last annotation image, save & updated benchmarking image
                if sim<sim_index:
                    im_compared = image.copy()
                    imageio.imwrite(os.path.join(path_save_images, image_name_init+video_name.split('.')[0]+'_'+str(id_)+'.jpg'), image)
                    k = k+1

                #if similar save for detection and continue until find one not similar to save
                else:
                    k = 1
            else:
                k = k+1

            if k%(gap+2)==0:
                k = 1

            #update id of frame
            id_ = id_+1

    #close video
    video.release()

#from a video it create consecutives frames with a csv file indicating which one should be annotated and which one should be predicted
def create_consecutive_frames(video_path, video_name, path_image, gap, sim_index, reverse_rgb=False, need_save=True, save_in_folder=True):
    
    #initialise video path
    video_path = os.path.join(video_path, video_name)
    
    #check if video exists
    if len(glob.glob(video_path))!=1:
        print('the video does not exist at your path: %s'%video_path)
        sys.exit()

    #create directories if not existing to save images
    if save_in_folder:
        path_save_images = os.path.join(path_image,'consecutives_frames_'+video_name.split('.')[0])
        if not os.path.exists(path_save_images):
            os.makedirs(path_save_images)
    else:
        path_save_images = path_image

    #read video 
    video = cv2.VideoCapture(video_path)

    #initialize list of annotation information
    li_annotation_info = []
        
    # loop over frames from the video file stream
    k = 0
    id_ = 0
    while True:

        #take frames and check if we have reached the end of the video
        (grabbed, image) = video.read()
        if not grabbed:
            break

        if reverse_rgb:
            b,g,r = cv2.split(image)           
            image = cv2.merge([r,g,b])
            
        #save the image
        imageio.imwrite(os.path.join(path_save_images, 'consecutives_frames_'+video_name.split('.')[0]+'_'+str(id_)+'.jpg'), image)
        
        ##############################################################
        ### see if its an 'annotated-image' or a 'predicted-image' ###
        #if no benchmarking image yet create one and save the image
        if k==0:
            im_compared = image.copy()
            annotation_type = 'detection'
            k = 1

        #see if image should be annotated, i.e. if enough dissimilar from the last annotated one
        elif k==1:

            #compute similarity between two possible consecutive annotated images
            sim = compare_ssim(im_compared, image, multichannel=True)

            #if not that similar from last annotation image, save for annotation & updated benchmarking image
            if sim<sim_index:
                im_compared = image.copy()
                annotation_type = 'annotation'
                k = k+1

            #if similar save for detection and continue until find one not similar to save
            else:
                annotation_type = 'detection'
                k = 1

        else:
            #otherwise save for detection
            annotation_type = 'detection'
            k = k+1

        if k%(gap+2)==0:
            k = 1

        #save info
        li_annotation_info.append({'id':str(id_)+'jpg', 'annotation_type': annotation_type})
        #update id of frame
        id_ = id_+1

    #close video
    video.release()
    
    #save as a csv file
    if need_save:
        df = pd.DataFrame(li_annotation_info)
        df.to_csv(os.path.join(path_save_images, 'annotation_info.csv'), index=False, sep=';')
    
    
#from an initial dictionary of ids-object with their corresponding bbox it output a dico with the new ids& associated bboxes
#max_used_id: bigger object id already used
def simple_bbox_tracker(dico0_id_bboxes_mask, li_t_bboxes_mask, max_used_id, smaller_dist=100):
    
    #TODO: if two new object has the same id
        
    #if no old object return the new object with new ids
    if len(dico0_id_bboxes_mask)==0:
        return({max_used_id+i+1:li_t_bboxes_mask[i] for i in range(len(li_t_bboxes_mask))})
        
    #compute centroids of old objects
    dico0_id_centroids = {id_:(x + int(w/2), y + int(h/2)) for id_,((x,y,w,h),mask) in dico0_id_bboxes_mask.items()}    
    
    #initialise
    dico_id_bboxes_mask = {}
    
    #pass through each new object
    for new_object, mask in li_t_bboxes_mask:
        
        #compute centroids of the new object
        x,y,w,h = new_object
        xmid = x + int(w/2)
        ymid = y + int(h/2)
        
        #compute the euclidean distance with centroids of old objects
        dico_id0_distance = {k:distance.euclidean((xmid,ymid), (xmid0,ymid0)) for k,(xmid0,ymid0) in dico0_id_centroids.items()}
        
        #if the smaller distance is smaller than 'smaller_dist', then use this id for the new obj, otherwise assign a new id
        possible_corresponding_id, d = min(dico_id0_distance.items(), key=lambda x: x[1])         
        if d<smaller_dist:
            dico_id_bboxes_mask[possible_corresponding_id] = (new_object, mask)
        else:
            max_used_id = max_used_id+1
            dico_id_bboxes_mask[max_used_id] = (new_object, mask)
    
    return dico_id_bboxes_mask
    
#givn an old dico-result and the new dico_id_bboxes with the X_LINE from which we need to count object, it will
#update the dico-results of the form:
#results = {'object_count':5,
#           'frame_count':1,
#           'dico_last_objects':{}}#{1:(61, 326, 175, 84), 2:(415, 145, 129, 87)}}
def update_results(results, dico_id_bboxes_mask, X_LINE):
    
    #update number of frames saw
    results['frame_count'] = results['frame_count']+1
    
    #update the objects count
    nbr_new_obj = len([i for i in list(dico_id_bboxes_mask.keys()) if i not in list(results['dico_last_objects'].keys())])
    results['object_count'] = results['object_count']+nbr_new_obj
    
    #update the last saw objects
    results['dico_last_objects'] = dico_id_bboxes_mask
    
    #see if an object pass the line
    for id_, (bbox,mask) in dico_id_bboxes_mask.items():
        x,y,w,h = bbox
        xline = x+w*0.95
        if xline>X_LINE:
            results['li_id_passed_object'].append(id_)
            results['li_id_passed_object'] = list(set(results['li_id_passed_object']))
            results['object_pass'] = len(results['li_id_passed_object'])
            
    return results    
  
#return the image with the (not several) mask on it   
#color must be a tuple with integer from 0 to 255
def apply_mask(image, mask, color, alpha=0.5):
    """return the image with the mask on it"""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, 
                                  image[:, :, c] *(1 - alpha) + alpha *color[c], 
                                  image[:, :, c])
    return(image)

#simple example
#r = li_results[0]
#image = cv2.imread(r['filename'])
#masked_image = image.copy()
#Mask
#masks = r['masks']
#for i in range(len(masks[0][0])):
#    mask = masks[:, :, i]
#    apply_mask(masked_image, mask)
#plt.imshow(masked_image);    

    
#take an image with the associated dico_id_bbox and annotate (label, bbox, line) the image with one color per label and save it
#to add more details: https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
def label_image(dico_id_bboxes_mask, image, li_text, X_LINE, FRAME_HEIGHT, dico_id_color, has_bbox, has_label, has_line, has_masks, alpha):
        
    #define font for text and nbr of color we have
    font = cv2.FONT_HERSHEY_SIMPLEX
    nbr_color = len(dico_id_color)
   
    #draw rectangles with label, masks and associated color
    for id_, d in dico_id_bboxes_mask.items():
        bbox, mask = d
        x,y,w,h = bbox
        thickness = 3
        color = dico_id_color[id_%nbr_color]
        if has_bbox:
            cv2.rectangle(image, (x,y), (x+w,y+h), color, thickness)
        if has_label:
            cv2.putText(image, str(id_), (x,y), font, 1, color, thickness, cv2.LINE_AA)
        if has_masks:
             image = apply_mask(image, mask, color, alpha)
   
    #draw the end line (thickness 2)
    if has_line:
        cv2.line(image, (X_LINE,0), (X_LINE, FRAME_HEIGHT), (255,0,0), 2)
    
    #draw text count
    if li_text!=None:
        for (x,y,s) in li_text:
            cv2.putText(image, s, (x,y), font, 1, (139, 0, 0), 1, cv2.LINE_AA)
    
    #return
    return(image)
    
###################################################################################################
########################################### basic models ##########################################
###################################################################################################

#from internship S
"""NOTE: IBM Watson works with decision tree instead of Random Forest and hence can visualize the tree and also give information about its leaves and the exact decisions"""

def filter_df(df, func_dict, logical_op = operator.and_):
    """ Function that filters rows of a dataframe based on a set of criteria for them.
    Inputs: 
    df - the dataframe to be filtered
    func_dict - a dictionary with {key:value} = {column_name:filtering_function}. Filtering functions may receive just one argument: a dataframe column. See is0() below for an example.
                Other examples: pandas.isnull or similar.
    logical_op -  the logical operator to be used to connect the criteria.
    Outputs: 
    the filtered dataframe
    """
    res_bool = None
    for col, operation in func_dict.items():
        if res_bool is None:
            res_bool = operation(df[col])
        else: 
            res_bool = logical_op(res_bool, operation(df[col]))
    return df[res_bool]
  
    
def dfcols_to_arrays(df, names=None):
    """ Transforms the columns or arrays from the input into an output consisting of two arrays.
    Inputs:
    df - a dataframe, a numpy.ndarray with 2 columns or a list with 2 elements (either of type numpy.array or pandas.core.series.Series)
    names - list of names for the columns if the first argument is a dataframe
    Outputs: the resulting arrays
    """
    if isinstance(df, pd.core.frame.DataFrame):
        if not names:
            raise ValueError('Names for the columns to be analyzed in the dataframe need to be given.')
        else:
            x = df.loc[:, names[0]]; y = df.loc[:, names[1]]
    elif isinstance(df, np.ndarray):
        if df.shape[1] != 2:
            raise ValueError('The array needs to have eaxctly 2 columns')
        else:
            x = df[:,0]; y = df[:,1]
    else:    
        if len(df) != 2:
            raise ValueError('List of exactly two elements (pd.core.series.Series or numpy.ndarray) needs to be given')
        else:
            x = df[0]; y = df[1]
    return x, y    


def random_forest_regressor_parameter_tunning(df, targetnames, varnames = None):  
    
    #create variables if not given
    if varnames is None:
        varnames = list(set(df.columns) - set(targetnames)) # automatically infer varnames

    # remove Nan values from target columns
    #filter_criteria = {x:pd.notnull for x in targetnames} 
    #df = filter_df(df, filter_criteria)

    # transform the data into format suitable for training the forest
    X, y = dfcols_to_arrays(df, [varnames, targetnames])
    c, r = y.shape
    y = y.values.reshape(c,)
    
    #Crossvalidation:
    np.random.seed(5)
    rfc = sklearn.ensemble.RandomForestRegressor(n_estimators=50) 
    param_grid = {'n_estimators': [2,3,5,10,50,100],'max_features': ['auto', 'sqrt', 'log2'],'min_samples_leaf':[1,3,5,10,20,30,50]}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
    CV_rfc.fit(X, y)
    print (CV_rfc.best_params_)

    #creat and train the forest
    forest = sklearn.ensemble.RandomForestRegressor(n_estimators=CV_rfc.best_params_['n_estimators'],
                                                    max_features=CV_rfc.best_params_['max_features'],
                                                    min_samples_leaf=CV_rfc.best_params_['min_samples_leaf'])
    forest = forest.fit(X, y)

    #save the tree
    #tree.export_graphviz(forest,out_file='tree.dot')
    #i=0
    #for tree_in_forest in forest.estimators_:
    #    if i<1:
    #        export_graphviz(tree_in_forest,feature_names=X.columns,filled=True,rounded=True,out_file='tree.dot')
    #        i=i+1
    
    # standard deviations of feature importances 
    std_importances = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # sort the features by importance
    sorted_importances = np.sort(forest.feature_importances_)[::-1]
    sorted_importance_indices = np.argsort(forest.feature_importances_)[::-1]
    sorted_importance_std = std_importances[sorted_importance_indices]
    sorted_importance_featnames = [varnames[i] for i in sorted_importance_indices]

    return sorted_importance_featnames, sorted_importances, sorted_importance_std


def random_forest_classifier_parameter_tunning(df, targetnames, varnames = None):  
    if varnames is None:
        varnames = list(set(df.columns) - set(targetnames)) # automatically infer varnames

    # remove Nan values from target columns
    filter_criteria = {x:pd.notnull for x in targetnames} 
    df = filter_df(df, filter_criteria)   

    # convert numerical target columns into strings
    df.loc[:,targetnames] = df.loc[:, targetnames].astype(str)  

    # transform the data into format suitable for training the forest
    X, y = dfcols_to_arrays(df, [varnames, targetnames])
    c, r = y.shape
    y = y.values.reshape(c,)
    
    #Crossvalidation:
    np.random.seed()
    rfc = sklearn.ensemble.ExtraTreesClassifier(n_estimators=50) 
    param_grid = { 'n_estimators': [5,10,50,100],'max_features': ['auto', 'sqrt', 'log2'],
                 'min_samples_leaf':[1,3,5,10,20]}
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
    CV_rfc.fit(X, y)
    print (CV_rfc.best_params_)

    #create and train the forest
    forest = sklearn.ensemble.ExtraTreesClassifier(n_estimators=CV_rfc.best_params_['n_estimators'],
                                                   max_features=CV_rfc.best_params_['max_features'],
                                                   min_samples_leaf=CV_rfc.best_params_['min_samples_leaf'],
                                                   random_state=5)
    forest = forest.fit(X, y)

    # standard deviations of feature importances 
    std_importances = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # sort the features by importance
    sorted_importances = np.sort(forest.feature_importances_)[::-1]
    sorted_importance_indices = np.argsort(forest.feature_importances_)[::-1]
    sorted_importance_std = std_importances[sorted_importance_indices]
    sorted_importance_featnames = [varnames[i] for i in sorted_importance_indices]

    return sorted_importance_featnames, sorted_importances, sorted_importance_std



###################################################################################################
################################## plot for deep learning models ##################################
###################################################################################################
    
#took partly from internet, but i dont remember where from...

##################### Class Activation Maps    
def plot_heatmap(layer_, li_image, model, img_heatmap_separate=False, save_path=None):
    fig = plt.figure(figsize=(15,15))
    c = 4
    l = int(len(li_image)/4) #round inf
    img_w = li_image[0].shape[1]
    img_h = li_image[0].shape[0]
    fig = plt.figure(figsize=(int(c*img_w/100), int(l*img_h/100)))
    for k,img in enumerate(li_image[0:c*l]):

        #predict
        x = np.reshape(img,[1,img.shape[0],img.shape[1],3])
        preds = model.predict(x)
        class_ = np.argmax(preds[0])

        ###### compute heatmap ######
        class_output = model.output[:, class_]
        last_conv_layer = model.get_layer(layer_)
        #gradient of the class_ with regard to the output feature map of layer_
        grads = K.gradients(class_output, last_conv_layer.output)[0]

        #vector of shape (192,), where each entry is the mean intensity of the gradient over a specific feature map channel
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # This function allows us to access the values of the quantities we just defined:
        # `pooled_grads` and the output feature map of `conv2d_94`, given a sample image
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])

        #We multiply each channel in the feature map array by "how important this channel is" with regard to the 
        #predicted class. If the value is positif it means that this associated weight should be increased to be more sure 
        #for this class prediction
        for i in range(int(pooled_grads.shape[0])):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        #The channel-wise mean of the resulting feature map is our heatmap of class activation
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        #up-sampled to the input image resolution using bi-linear interpolation (INTER_LINEAR: by default)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        #we convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        #we apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


        ###### compute uided Backpropagation ######
        #TODO
        #GBP = GuidedBackprop(pretrained_model)
        # Get gradients
        #guided_grads = GBP.generate_gradients(prep_img, target_class)

        ###### fuse Guided Backpropagation and Grad-CAM visualizations via point- wise multiplication ######
        #TODO
        #cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
        
        #convert to gray then add other channels to have 3 dimensions (so that one can add with map later)
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        #add with map
        superimposed_img = cv2.addWeighted(gray, 0.6, heatmap, 0.3, 0.5)
        
        if img_heatmap_separate:
            images = [img, heatmap, superimposed_img]
            fig = plt.figure(figsize=(10,10))
            plt.imshow(concat_n_images(images));
            if save_path is not None: 
                plt.savefig(save_path+str(k)+'.png',dpi=300,format='png',bbox_inches='tight')
            plt.show()
            
        else:
            plt.subplot(l,c,k+1)
            plt.tight_layout()
            plt.xticks([]) #remove xlabel annotations
            plt.yticks([])
            plt.title('bg:'+str(round(preds[0][0],3))+' fish:'+str(round(preds[0][1],3)))
            plt.imshow(superimposed_img.astype(np.uint8));
    if (save_path is not None) & (img_heatmap_separate==False): 
        plt.savefig(save_path+'.png',dpi=300,format='png',bbox_inches='tight')
    plt.show()    
    
    
    
##################### Intermediate activations
def inter_activation(img, activation_model, layer_outputs, save_path=None, images_per_row=16):
    #from below plus small modification
    #https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
    # predict one one random input sample
    #small example on noise
    #img = np.random.rand(1, img_rows, img_cols, 3)
    #activations = activation_model.predict([np.reshape(img,[1,img.shape[1],img.shape[2],3])])

    activations = activation_model.predict([np.reshape(img,[1,img.shape[0],img.shape[1],3])])

    #names of the layers for plot title
    layer_names = []
    for layer in layer_outputs:
        layer_names.append(layer.name)

    #plot feature maps
    for layer_name, layer_activation in zip(layer_names, activations):

        n_features = layer_activation.shape[-1] #nbr of channel/features in the feature map 
        img_row = layer_activation.shape[1]
        img_col = layer_activation.shape[2]
        nbr_row = n_features // images_per_row #temps pis si cest pas un multiple de 16 on ne prendra pas en compte tout
        display_grid = np.zeros((img_row * nbr_row, images_per_row * img_col))

        #we'll tile each filter into this big horizontal grid
        for r in range(nbr_row):
            for c in range(images_per_row):
                channel_image = layer_activation[0, :, :, r * images_per_row + c]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[r * img_row : (r + 1) * img_row,
                             c * img_col : (c + 1) * img_col] = channel_image

        #display the grid
        scale = 1. / img_row
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig(save_path+'_'+layer_name.split('/')[0]+'.png',dpi=300,format='png',bbox_inches='tight')    
        plt.show()
    
##################### Filters    
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x

def generate_pattern(layer_name, filter_index, model, img_rows, img_cols):
    
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index]) #to be maximized, i.e. higher pixel mean 

    #compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    #normalize gradient: to ensures that the magnitude of the updates done to the input image is always 
    #within a same range.
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    #this function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    #We start from a gray image with some noise
    input_img_data = np.random.random((1, img_rows, img_cols, 3)) * 20 + 128.

    #run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step 

    img = input_img_data[0]
    return deprocess_image(img)
#small example
#gen_img = generate_pattern(layer_name='block_15_project', filter_index=10)
#plt.imshow(gen_img)
#plt.show()    
    
    
    