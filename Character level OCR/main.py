from Hangeul_OCR import HangeulClassification
import os
import torch
import argparse
import utility

# for colab
# from google.colab import drive
# drive.mount('/content/drive')

'''
AAI3201
Team: YapYapHo
Hangeul grapheme-level OCR Project
'''

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.tqdm_disable = False
    
    if args.data_split == True:
        data_dir = utility.divide_dataset([0.7, 0.2, 0.1], args.data_dir)
    else:
        data_dir = args.splitted_dir
    
    idx_to_syllable, syllable_to_idx = utility.txt_to_dict(args.save_dir, 'KSX1001.txt')
    
    step = HangeulClassification(args, data_dir)
    step.do(idx_to_syllable, syllable_to_idx)


if __name__ == '__main__':
    
    # defining arguments. used in class, like __init__(self, args) -> self.data_dir = args.data_dir 
    parser = argparse.ArgumentParser(description='Deep learning parameters')
    parser.add_argument("--project", default="Hangeul OCR", type=str)
    parser.add_argument("--mode", default="test", choices=["train", "test"], type=str)
    parser.add_argument("--data_split", default=False, choices=[True, False], type=bool) # set True if split needed
    parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str)
    parser.add_argument('--model_name', default='CNN', type=str)
    parser.add_argument("--data_aug", default=False, choices=[True, False], type=bool)
    parser.add_argument("--save_png", default=True, choices=[True, False], type=bool)
    parser.add_argument("--window", default=[0., 1.], type=list)
    
    # need to be tuned: grid search..?
    parser.add_argument("--crop_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--epoch_start", default=0, type=int)
    parser.add_argument("--num_epoch", default=10, type=int)
    parser.add_argument("--optimizer", default="adam", choices=["sgd", "rmsprop", "adam", "adamw", "sparseadam"], type=str)

    # for your own desktop
    parser.add_argument('--data_dir', type=str, default = 'hangul_dataset')
    parser.add_argument("--splitted_dir", type=str, default = 'splitted_dataset')
    parser.add_argument('--save_dir', type=str, default = '')
    
    # for Vessl experiment
    # parser.add_argument('--data_dir', type=str, default = "./../input/")
    # parser.add_argument("--splitted_dir", type=str, default = './../input/')
    # parser.add_argument('--save_dir', type=str, default = "./../output/") 
     
    args = parser.parse_args()
    main()

