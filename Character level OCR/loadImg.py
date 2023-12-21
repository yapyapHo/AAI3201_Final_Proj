from PIL import Image
import numpy as np
from utility import *

def loadImg(path):
    ''' load image and resize. convert to np array. '''
    try:
        img = Image.open(path)
        img = img.resize((32,32))
        img = np.array(img, 'uint8')
        img = Normal(img) # normalize
        img = np.ascontiguousarray(img)
        return img
    
    except:
        FileNotFoundError
    
    