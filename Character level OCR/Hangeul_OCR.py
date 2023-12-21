from model import *
from Dataset import *
from utility import *

import torch
import time
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
# import vessl

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class HangeulClassification:
    def __init__(self, args, data_dir):
        self.args = args
        self.device = self.args.device
        # ETC Settings
        self.cmap = 'gray'

        # Address Settings
        self.data_dir = data_dir
        self.save_dir = args.save_dir
        self.tensorboard_dir = os.path.join(args.save_dir, 'Tensorboard', args.model_name)
        self.model_dir = os.path.join(args.save_dir, 'TrainModel', args.model_name) 
        self.results_dir = os.path.join(args.save_dir, 'TrainResult', args.model_name) 

        create_dir(os.path.join(self.tensorboard_dir,'train'), opts='del')
        create_dir(os.path.join(self.tensorboard_dir, 'val'), opts='del')
        create_dir(os.path.join(self.tensorboard_dir, 'test'), opts='del')
        create_dir(self.model_dir)

        # SummaryWriter settings for Tensorboard
        self.writer_train = SummaryWriter(log_dir= os.path.join(self.tensorboard_dir, 'train'))
        self.writer_val = SummaryWriter(log_dir=os.path.join(self.tensorboard_dir, 'val'))
        self.writer_test = SummaryWriter(log_dir=os.path.join(self.tensorboard_dir, 'test'))
        
    
    def do(self, idx_to_syllable, syllable_to_idx):
        # Initializing
        batch_size = self.args.batch_size
        device = self.device
        learning_rate = self.args.learning_rate
        num_epoch = self.args.num_epoch

        train_continue = self.args.train_continue
        # crop_size = self.args.crop_size (maybe has a problem. not used)
        save_png = self.args.save_png
        window = self.args.window
        cmap = self.cmap
        
        # Split data into train, val, test
        # Dataloader setting
        t_start = time.time()
        print("==================== STARTING DATA SETUP ====================")
        
        if self.args.data_aug:  # data augmentation 'True'.
            transform_train = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]) # fix needed: noramlize mean/std values, crop size..

            transform_val = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]) # should not be same. let's discuss
        else:
            transform_train = None
            transform_val = None
        
        # to use augumented ones, and original ones both.
        if self.args.mode == 'train':
            nonaug_dataset_train = load_Dataset(os.path.join(self.data_dir, 'train'), syllable_to_idx)
            nonaug_dataset_val = load_Dataset(os.path.join(self.data_dir, 'val'), syllable_to_idx)
            aug_dataset_train = load_Dataset(os.path.join(self.data_dir, 'train'), syllable_to_idx, transform=transform_train)
            aug_dataset_val = load_Dataset(os.path.join(self.data_dir, 'val'), syllable_to_idx, transform=transform_val)
        
            dataset_train = nonaug_dataset_train + aug_dataset_train
            dataset_val = nonaug_dataset_val + aug_dataset_val
            
            loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            
            num_data_train = dataset_train.__len__()
            num_data_val = dataset_val.__len__()
            num_batch_train = np.ceil(num_data_train / batch_size)
            num_batch_val = np.ceil(num_data_val / batch_size)
            
            train_losses = []; avg_train_accuracy = []
            val_losses = []; avg_val_accuracy = []
            
        elif self.args.mode == 'test':
            dataset_test = load_Dataset(os.path.join(self.data_dir, 'test'), syllable_to_idx)
            loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
            num_data_test = dataset_test.__len__()
            num_batch_test = np.ceil(num_data_test / batch_size)
            
        print("=========== DATA SET-UP TIME %.2f sec ===========" % (time.time() - t_start))
        
        ######################################-- train/test! --######################################
        
        net = CNN().to(device)
        # Optimizer
        optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

        # Loss function
        fn_loss = nn.CrossEntropyLoss()  # nn.MSE()  # nn.L1Loss() # what else?
        fn_loss = fn_loss.to(device)   

        st_epoch = 0
        if self.args.mode == 'train':
            if train_continue == "on":
                print("=========== TRAIN CONTINUE ===========")
                net, optim, st_epoch = load(ckpt_dir=self.model_dir, net=net, optim=optim)

            for epoch in tqdm(range(st_epoch + 1, num_epoch + 1), desc = 'TRAINING START!'):
                
                # train
                net.train()
                train_correct = 0
                train_loss = 0.0
                tstart = time.time()
                for label, input in loader_train:
                    # forward pass
                    label, input = label.to(device), input.to(device)
                    output = net.forward(input)
                    
                    # backward pass
                    optim.zero_grad()                    
                    loss = fn_loss(output, label)
                    _, predicted = torch.max(output.data, 1)
                    train_correct += (predicted == label).sum().item()
                
                    loss.backward()
                    optim.step()

                    # compute loss
                    train_loss += loss.item()
                    
                train_losses.append(train_loss / len(loader_train))
                self.writer_train.add_scalar('loss', train_loss / len(loader_train), epoch)
                avg_train_accuracy.append(100*train_correct / len(dataset_train))

                # validation
                val_correct = 0
                val_loss = 0.0
                with torch.no_grad():
                    net.eval()
                    for label, input in loader_val:
                        # forward pass
                        label, input = label.to(device), input.to(device)
                        output = net.forward(input)

                        loss = fn_loss(output, label)
                        _, predicted = torch.max(output.data, 1)
                        val_correct += (predicted == label).sum().item()
                        
                        val_loss += loss.item()
                        
                for predicted, label in zip(predicted, label):
                    print(f"PREDICTED: {idx_to_syllable[predicted.item()]} | REAL CLASS: {idx_to_syllable[label.item()]}")
                    
                val_losses.append(val_loss / len(loader_val))
                self.writer_val.add_scalar('loss', val_loss / len(loader_val), epoch)
                avg_val_accuracy.append(100*val_correct / len(dataset_val))
                
                print("EPOCH %04d / %04d | TRAIN LOSS %.6f | VAL LOSS %.6f | TIME %.2f sec" %
                      (epoch, num_epoch, train_loss / len(loader_train), val_loss / len(loader_val), time.time() - tstart))
                print("TRAIN ACCURACY: %.6f | VALIDATION ACCURACY: %.6f" % (avg_train_accuracy[-1], avg_val_accuracy[-1]))
                if epoch % 10 == 0:
                    print('============= EPOCH %04d COMPLETED! =============' % epoch)
                    save(ckpt_dir=self.model_dir, net=net, optim=optim, epoch=epoch)

            self.writer_train.close()
            self.writer_val.close()
        
        # TEST MODE
        elif self.args.mode == 'test':
            net, optim, st_epoch = load(ckpt_dir=self.model_dir, net=net, optim=optim)
            test_correct = 0
            test_loss = 0.0
            
            with torch.no_grad():
                net.eval()
                for label, input in tqdm(loader_test, desc="TEST START!"):
                    label, input = label.to(device), input.to(device)
                    output = net(input)
                    
                    loss = fn_loss(output, label)
                    _, predicted = torch.max(output.data, 1)
                    test_correct += (predicted == label).sum().item()

                    test_loss += loss.item()
                    # for predicted, label in zip(predicted, label):
                    #     print(f"PREDICTED: {idx_to_syllable[predicted.item()]} | REAL CLASS: {idx_to_syllable[label.item()]}")
                                           
            test_loss = test_loss / len(loader_test)
            test_accuracy = 100*test_correct / len(dataset_test)
            print("TEST LOSS: %.6f | TEST ACCURACY: %.6f" % (test_loss, test_accuracy))
