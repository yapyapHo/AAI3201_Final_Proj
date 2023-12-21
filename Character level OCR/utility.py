import os
import torch
import shutil
import numpy as np
from tqdm import tqdm


def create_dir(dir, opts=None):
    '''
    creates directory for results
    '''
    try:
        if os.path.exists(dir):
            if opts == 'del':
                shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)
    except OSError:
        print("Error: Failed to create the directory.")
        

def divide_dataset(ratio, dataset_path):
    ''' 
    divide the dataset into given ratio.
    ratio = [a, b, c] where a: train, b: val, c: test ratio, a+b+c = 1
    returns new data path (splitted).
    '''
    
    # Create new directories
    base_dir = os.path.dirname(dataset_path) # C:/Users/wyrtr/Desktop/Final_Project/code
    new_dir = os.path.join(base_dir, "splitted_dataset")
    os.makedirs(new_dir, exist_ok=True)
    
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(new_dir, subset), exist_ok=True)
        
    print("===================== DATASET SPLITTING STARTED! =====================")
    
    for class_name in tqdm(os.listdir(os.path.join(base_dir, 'hangul_dataset'))):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            files = os.listdir(class_dir)
            np.random.shuffle(files)  # shuffle the data
            
            # Split the files according to the given ratio
            a_count = int(len(files) * ratio[0])
            b_count = a_count + int(len(files) * ratio[1])
            train, val, test = files[:a_count], files[a_count:b_count], files[b_count:]
            
            for subset, file_names in zip(['train', 'val', 'test'], [train, val, test]):
                subset_dir = os.path.join(new_dir, subset, class_name)
                os.makedirs(subset_dir, exist_ok=True)
                
                for file_name in file_names:
                    src = os.path.join(class_dir, file_name)
                    dst = os.path.join(subset_dir, file_name)
                    shutil.copy2(src, dst)  # copy the file to the new directory
                    
    print("===================== DATASET SPLITTING COMPLETED! =====================")
    
    return new_dir
        

def txt_to_dict(filepath, filename):
    ''' 
    for reading KS X 1001 file. makes it to dictionary. 
    each word is a key or value of dictionary.
    
    returns (key, syllable) and (syllable, key), two dicts.
    '''
    with open(os.path.join(filepath, filename), 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        
    idx_to_syllable = {} # (0 : 가)
    syllable_to_idx = {} # (가 : 0)
    key = 0
    for line in lines:
        line_no_enter = line[0:-1] # remove '\n'
        words = line_no_enter.split()
        for word in words:
            idx_to_syllable[key] = word
            syllable_to_idx[word] = key
            key += 1
    return [idx_to_syllable, syllable_to_idx]
        

def call_optimizer(optim_name, net_parameters, lr):
    '''
    calls optimizer by name. Ex) optim = call_optimizer('sgd')
    '''
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(net_parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optim_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(net_parameters, lr=lr, alpha=0.9)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(net_parameters, lr=lr)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(net_parameters, lr=lr)
    elif optim_name == 'sparseadam':
        optimizer = torch.optim.SparseAdam(net_parameters, lr=lr)
        
    return optimizer


def save(ckpt_dir, net, optim, epoch):
    '''
    to save results in some path per every epoch. can stop training!
    train_continue -> off (default) (main.py)
    '''
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


def load(ckpt_dir, net, optim, set_epoch=None):
    '''
    resume training, or use the trained weight for test.
    train_continue -> on (main.py)
    '''
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    if set_epoch is None:
        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    else:
        dict_model = torch.load('%s/%s' % (ckpt_dir, 'model_epoch'+str(set_epoch)+'.pth'))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    if set_epoch is None:
        epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    else:
        epoch = set_epoch

    return net, optim, epoch


def Normal(data):
    ''' data(image) normalize. '''
    data = data.astype(np.float32)
    data = data/255.0
    
    return data


def Denorm(data):
    ''' data(image) denormalize. '''
    data = (data * 255).round().astype(np.uint8)  # Scale to [0, 255] for saving as PNG
        
    return data