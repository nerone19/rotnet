# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:47:11 2021

@author: gabri
"""
from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,models
import cv2 
from PIL import Image
import random
from skimage.transform import rotate,resize
from imgaug import augmenters as iaa
import math
from sklearn.metrics import mean_squared_error
import pretrainedmodels
from tqdm import tqdm
import time
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import wandb
import pdb
import logging
from learning_rate_scheduler import LRFinder
from early_stopping import EarlyStopping

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
transform_seq = iaa.Sequential([
    
 sometimes(iaa.Multiply(0.5)),
 sometimes(iaa.LogContrast(gain=(0.6, 1.4) )),
 sometimes(iaa.GaussianBlur(sigma=1))
 ], random_order= True)



val_perc = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device {} used for training model'.format(device) )
#-------------------------------------------------------------------------------------------------------------------------------------------
def init_normalization_ranges(root_dir, image_list):
    
    max = 0
    for image in image_list: 
        image = cv2.imread( os.path.join(root_dir,image))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        m = image.max()
        if( m > max):
            max = m
            
    os.environ['max-value'] = str(max)
    print('maximum pixel value found: {}'.format(max))
    

def create_train_val_sets(root_dir,configuration = True):
    
    trainImageList,valImageList = [],[] 
    
    for i in range(20):
        sub_folder_path = os.path.join(root_dir,str(i))
        
        filterFiles = map(lambda key: [key,i%10], \
                          filter(lambda el: os.path.isfile(os.path.join(sub_folder_path,el)), os.listdir(sub_folder_path)))
        image_list = list(filterFiles)
        
        
        random.shuffle(image_list)
        
        limit = math.floor( (1-val_perc)*len(image_list))
        train = image_list[:limit]
        valid = image_list[limit:]
        
        trainImageList.extend(train)
        valImageList.extend(valid)
    
    random.shuffle(trainImageList)
    random.shuffle(valImageList)

    filterFiles = filter(lambda el: os.path.isfile(os.path.join(root_dir+'/all',el)), os.listdir(root_dir+'/all'))
    image_list = list(filterFiles)
    if(configuration): 
            init_normalization_ranges(root_dir+'/all', image_list)


    return trainImageList, valImageList

def get_data(train_set,val_set,batch_size = 32):
     trn_dl = DataLoader(train_set, batch_size=batch_size, collate_fn=train_set.collate_fn,\
                            shuffle=True,drop_last=True)

     val_dl = DataLoader(val_set, batch_size=batch_size,collate_fn=val_set.collate_fn,
                         shuffle=True,drop_last=True)
     return trn_dl, val_dl
#-------------------------------------------------------------------------------------------------------------------------------------------
     
def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data, mode='fan_in',nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)
      

def conv_layer(ni,no,kernel_size,stride=1):
     return nn.Sequential(
         nn.Conv2d(ni, no, kernel_size, stride),
         nn.ReLU(),
         nn.BatchNorm2d(no),
         nn.MaxPool2d(2)
         )



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(107648, 256)
        self.linear2 = nn.Linear(256, 180)
        
        # nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')
        # # nn.init.kaiming_normal_(self.conv1.weight)
        # # nn.init.kaiming_normal_(self.conv2.weight)
        # nn.init.xavier_normal_(self.linear1.weight.data)
        # nn.init.xavier_normal_(self.linear2.weight.data)
        
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self,image):
        bs, c, h, w = image.size()

        x = self.relu(self.conv1(image))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(bs, -1) 
        x = self.relu(self.linear1(x)) 
        x = self.linear2(x)
        x = nn.LogSoftmax(dim=1)(x)
        
        return x 


def get_pretrained_model(pretrained=True):
    if pretrained:
        model = models.resnet18(pretrained=True)
        #freeze resnet layer for further finetuning where needed
        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = models.resnet18(pretrained=False)
        
    #let's froze layers in the pretrained model
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 180)
    return model    



def get_model(pretrained=False):
     # model = nn.Sequential(
     #     conv_layer(1, 64, 3),
     #     conv_layer(64, 128, 3),
     #     conv_layer(128, 256, 3),
     #     nn.Flatten(),
     #     nn.Linear(9216, 180),
     #     nn.LogSoftmax(dim=1),
     # ).to(device)

    
     
     
     if(pretrained):
         model = get_pretrained_model()
     else: 
         model = Network()
         model.apply(initialize_weights)
         
     
     model.to(device)
     loss_fn = nn.CrossEntropyLoss()
     optimizer = Adam(model.parameters(), lr=1e-9)
     # optimizer = SGD(model.parameters(), lr=3e-4)
     # scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
     
     scheduler = ReduceLROnPlateau(optimizer,
         factor=0.5,patience=0,
         threshold = 0.001,
         verbose=True,
         min_lr = 1e-5,
         threshold_mode = 'abs'
     )

     
     return model, loss_fn, optimizer,scheduler

    
def train(data_loader, model, optimizer, scheduler,device,wandb):
    
    model.train()
    running_loss = 0.0
    tk0 = tqdm(data_loader, total=int(len(data_loader)))
    counter = 0

    try:
        if(device == 'cuda'):
            scaler = torch.cuda.amp.GradScaler()

        for data in  tk0:
            images,labels = data
            # pdb.set_trace()
            inputs =images.to(device,dtype = torch.float)
            labels =labels.to(device,dtype = torch.long)
            
            optimizer.zero_grad()

            if(device  == 'cuda'):
                #run in half precision for faster training
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    # loss = F.nll_loss(outputs, labels.view(inputs.size(0)))
                    loss = nn.CrossEntropyLoss()(outputs, labels.view(inputs.size(0)))
            
                #we expect gradient to be in half precision, We have to properly handle these data
                scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

            else: 
                outputs = model(inputs)
                # loss = F.nll_loss(outputs, labels.view(inputs.size(0)))
                loss = nn.CrossEntropyLoss()(outputs, labels.view(inputs.size(0)))
                loss.backward()
                
                optimizer.step()
            
            running_loss += loss.item()
            counter +=1
            tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))
            
        epoch_loss = running_loss / len(data_loader)
        print('Training Loss: {:.4f}'.format(epoch_loss))
        wandb.log({'Train Loss': epoch_loss})
    except:
        pass



    
def evaluate(data_loader,model,device,epoch,wandb):
    
    model.eval()
    
    final_labels = []
    final_outputs = []
    running_loss = 0.0
    tk0 = tqdm(data_loader, total=int(len(data_loader)))
    counter = 0
    with torch.no_grad():
        
        try:
            for data in tk0:
                images,labels = data
                inputs =images.to(device,dtype = torch.float)
                labels =labels.to(device,dtype = torch.long)    
                labels = labels.view(inputs.size(0))

                if(device == 'cuda'):
                    #run in half precision for faster training
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels)

                else: 
                    outputs = model.forward(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                outputs = outputs.detach().cpu().numpy()
                predictions =  np.argmax(outputs,axis =1)
                
                final_labels.extend(labels.tolist())
                final_outputs.extend(predictions.tolist())

                running_loss += loss.item()
                counter +=1
                tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))

            epoch_loss = running_loss / len(data_loader)
            scheduler.step(epoch_loss)
            print('Validation Loss: {:.4f}'.format(epoch_loss))
            wandb.log({'Valid Loss': epoch_loss})

            if(epoch_loss < float(os.environ['best_loss']) ):
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, os.path.join(os.environ['log_path'], 'model.pt'))
        except: 
            pass

        return final_outputs, final_labels,epoch_loss
        

def test_set_prediction(model,test_path,test_image_list, epoch):

    model.eval()

    if( os.path.isfile(os.path.join(os.environ['log_path'],'test_predictions.txt')) ):
        mode = 'w'
    else: 
        mode = 'a'

    with torch.no_grad():

        with open( os.path.join(os.environ['log_path'],'test_predictions.txt'),mode ) as log_prediction_file:
            log_prediction_file.writelines('--------------------------PREDICTION FOR EPOCH {} ------------------------------------------------\n'.format(epoch))
            for image_name in test_image_list:
                image = cv2.imread( os.path.join(test_path,image_name) )
                image = image.astype(np.float32)
                image = resize(image, (64,64))
                image = torch.tensor(image).unsqueeze(0).permute(0,3,1,2).to(device)/float(os.environ['max-value'])
                # labels = torch.tensor(labels).to(device)
                prediction = model(image)
                prediction = np.argmax(prediction.cpu().detach().numpy(),axis=1)
                if(prediction > 90):
                    prediction = 90 - prediction
                log_prediction_file.writelines(image_name + ' :  ' + str(prediction) + '\n')
                




#-------------------------------------------------------------------------------------------------------------------------------------------

class RotatedDigitsDataset(Dataset):

    
    
    def __init__(self, root_dir, image_list, transform=None):

        self.root_dir = root_dir
        self.transform = transform 
        self.image_list = image_list
        
        #for each digit type (0,..10) we create a random list with  [0,180] where each element cover different rotations
        self.rotation_list = [ list(range(180)) for el in range(10) ]
        for sublist in self.rotation_list:
            #we shuffle every possible rotation for every digit
            random.shuffle(sublist)
        #we keep count of every idx in every sublist in rotation_list. We need to do this in order to cover every rotation for every digit.
        #when all rotations were covered and training keeps going, we reshuffle again the possible rotations and zeroed the relative idx
        self.idxs = [0 for el in  range(10)]
        if(transform):
            self.transform = transform


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_name,digit_type = os.path.join(self.root_dir,
                                self.image_list[idx][0]),self.image_list[idx][1]
        

        image = cv2.imread(img_name)
        # pdb.set_trace()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32)

        #random rotation for the current number associated to the image
        randomRotation = self.rotation_list[digit_type][self.idxs[digit_type]]
        #we finished to cover every combination for the current number. We can reset the random rotation 
        if(self.idxs[digit_type] + 1 == 180):
            self.idxs[digit_type] = 0
            random.shuffle(self.rotation_list[digit_type])
        #we still have to cover other rotations for the number type
        else: 
            self.idxs[digit_type] += 1
        
        #since we are dealing with a classification task, we have 180 classes.However, we want to map values bigger than 90 degreees to 
        #counterclockwise values
        if(randomRotation > 90):
            degree = 90 - randomRotation
        else: 
            degree = randomRotation
        # pdb.set_trace()
        image = rotate(image, degree, resize=True,mode = 'constant',cval = np.mean(image)+ (np.std(image)) )
        image = resize(image, (64,64))
        
        image = np.stack([image for channel in range(3)],axis=2)
        #normalize images in [-1,1]
        # image = (image - (float(os.environ['max-value'])/2)) / (float(os.environ['max-value'])/2) 
        #normalize image in [0,1]
        # image = image/float(os.environ['max-value'])
        # if(self.transform):
        #     image = self.transform(image)

        # pdb.set_trace()
        # torch.tensor(image).unsqueeze(0).to(device),torch.tensor(randomRotation).to(device) 
        # return  torch.tensor(image).permute(2,0,1).to(device),torch.tensor(randomRotation).to(device) 
        return np.array(image),randomRotation

    def collate_fn(self, batch):
          'logic to modify a batch of images'
          ims, labels = list(zip(*batch))

          # transform a batch of images at once
          if self.transform: ims=self.transform.augment_images(images=ims)
          ims = torch.tensor(ims).permute(0,3,1,2).to(device)/float(os.environ['max-value'])
          # labels = torch.tensor(labels).to(device)
          return ims, torch.tensor(labels).to(device)


image_path = 'D:/additional datasets/OCR/rotation dataset/small'
test_path = 'D:/additional datasets/OCR/rotation dataset/test/all'

#home pc
test_path = 'D:/work/office/OCR/rotnet/test'
image_path = 'D:/work/office/OCR/rotnet/dataset'


trainImageList, valImageList = create_train_val_sets(image_path)
#get test images
filterFiles = map(lambda key: key, \
                          filter(lambda el: os.path.isfile(os.path.join(test_path,el)), os.listdir(test_path)))
testImageList = list(filterFiles)


train_set = RotatedDigitsDataset( os.path.join(image_path,'all'),trainImageList,transform_seq)
val_set = RotatedDigitsDataset(os.path.join(image_path,'all'),valImageList)


# for i in range(5):
#     image,label =  train_set[i]
#     if(label.cpu().detach().numpy() > 90):
#        label = 90 - label.cpu().detach().numpy()
#     else: label =  label.cpu().detach().numpy()
#     plt.imshow(image.permute(1,2,0).cpu().detach().numpy())
#     plt.title(label)
#     plt.show()



mode = 'train'
trn_dl, val_dl = get_data(train_set,val_set,32)
model, loss_fn, optimizer,scheduler = get_model(True)

#log dir for storing model training
curr_dir = os.getcwd()
log_dir = os.path.join(curr_dir,'logs')
logging.basicConfig(filename = log_dir+'/log.log')

if(os.path.exists( log_dir ) == False):
    os.mkdir(log_dir)

os.environ['log_path'] = str(log_dir)
os.environ['best_loss'] = str(math.inf)

wandb.login()
wandb.init(project='pytorchw_b')
wandb.watch(model, log='all')
since = time.time()

early_stopping = EarlyStopping()
if(mode == 'finder'):
    lr_finder = LRFinder(model, optimizer, device, loss_fn)
    lr_finder.range_test(trn_dl, end_lr=10, num_iter=100, logwandb=True)
    exit()


for epoch in range(1000):
    
    try:
        print('starting epoch {}'.format(epoch))
        train(trn_dl,model,optimizer,scheduler,device,wandb)
        predictions, valid_labels,val_epoch_loss = evaluate(val_dl,model,device,epoch,wandb)

        mse_metric = mean_squared_error(predictions, valid_labels)/len(valid_labels)
        print('MSE {}'.format(mse_metric))

        if( (epoch % 10 == 0) and epoch > 0):
            test_set_prediction(model,test_path,testImageList,epoch)
        wandb.log({'MSE': mse_metric})

        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break

    except:
        print('something happened')
        logging.error("exception ",exc_info=1) #or
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(os.environ['log_path'], 'model.pt'))
        break
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

