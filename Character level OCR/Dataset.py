from loadImg import loadImg
from utility import *

import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import time


class load_Dataset(torch.utils.data.Dataset):
    '''
    load datas from given path. append them to list one by one.
    uses folder name as class name.
    '''
    def __init__(self, data_path, syllable_to_idx, transform=None, imgSize=32*32):
        self.data_path = data_path
        self.class_names = os.listdir(data_path)
        self.imgSize = imgSize
        self.syllable_to_idx = syllable_to_idx
        self.transform = transform
        self.data_array = self.preprocessing() # [class_name, img]
        self.ToTensor = transforms.ToTensor()
        print("length of data array:", len(self.data_array))
        
    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, index):
        if index >= self.__len__():  
            raise IndexError

        label, img = self.data_array[index]
        img = self.ToTensor(img)
        if self.transform is not None:
            img = self.transform(img)
        
        return [label, img]

    def preprocessing(self):
        ''' data into array. '''
        data_array = []
        t_start = time.time()

        for class_name in tqdm(self.class_names):
            class_dir = os.path.join(self.data_path, class_name)
            for filename in os.listdir(class_dir):
                img = os.path.join(class_dir, filename)
                img = loadImg(img) # converted to np array, and normalized -> float32 np array
                label = self.syllable_to_idx[class_name] # label -> int
                data_array.append((label, img))

        print("======== DATA PREPROCESSING TIME %.2f sec ========" % (time.time() - t_start))
        
        return data_array
    
    # not used
    def padding(self, data, arraySize):
        ''' 
        Make the array the same size as arraySize.
        no padding when larger than arraySize.
        '''
        if max(data.shape) < arraySize:
            pad_sequence = []
            
            for l in data.shape:
                pad_before = max(0, (arraySize - l)//2)
                pad_after = max(0, arraySize - l - pad_before)
                pad_sequence.append((pad_before, pad_after))
            data = np.pad(data, pad_sequence, mode='constant', constant_values=0)
        
        return data