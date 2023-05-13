import os
import numpy as np
import torch
torch.cuda.empty_cache()
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
import torchvision.models as models
from model import Custom_AlexNet, Custom_VGG16, Custom_ResNet34
from dataloader import MyJP2Dataset, NFDataset, FLDataset
from evaluation import sklearn_Compatible_preds_and_targets

# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 
import torch.nn.functional as F

# For all Optimization algorithms, SGD, Adam, etc.
import torch.optim as optim

# Loading and Performing transformations on dataset
import torchvision
import torchvision.transforms as transforms 
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

#Labels in CSV and Inputs in Fits in a folder
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

#For Confusion Matrix
from sklearn.metrics import confusion_matrix

#Warnings
import warnings
warnings.simplefilter("ignore", Warning)

#Time Computation
import timeit
import argparse

saving_directory = "/scratch/cpandey1/acm_results"
os.makedirs(saving_directory, exist_ok=True)
job_id = os.getenv('SLURM_JOB_ID')

saving_directory = os.path.join(saving_directory, str(job_id))
os.makedirs(saving_directory, exist_ok=True)

model_dir = os.path.join(saving_directory, "trained_models")
os.makedirs(model_dir, exist_ok=True)

results_dir = os.path.join(model_dir, "results")
os.makedirs(results_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="fullDiskModeltrainer")
parser.add_argument("--fold", type=int, default=1, help="Fold Selection")
parser.add_argument("--epochs", type=int, default=51, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.001, help="regularization parameter")
parser.add_argument("--max_lr", type=float, default=0.00001, help="MAX learning rate")
# parser.add_argument("--factor", type=float, default=0.1, help="lr scheduler reduction factor")
parser.add_argument("--models", type=str, default='alex', help="enter alex, vgg, resnet for alexnet, vgg16, and resnet resp")
opt = parser.parse_args()


# Random Seeds for Reproducibility
def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(40)


# CUDA for PyTorch -- GPU SETUP
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# device= torch.device('cpu')
torch.backends.cudnn.benchmark = True
# print(device)

def dataloading():
    #Define Transformations
    #General transformation which resizes the input to specified size. Default=256
    #This transformation is for majority class NF and also for entire validation set.
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    #We augment only the flaring instances
    #Augmentation1: Rotating full-disk magnetogram images within +/- 5degrees.
    rotation = transforms.Compose([
        transforms.RandomRotation(degrees=(-5,5)),
        transforms.ToTensor()
    ])

    #Augmentation2: Flipping full-disk magnetogram images horizontally.
    hr_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ])

    #Augmentation3: Flipping full-disk magnetogram images vertically.
    vr_flip = transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor()
    ])

    nf_augment = transforms.Compose([
    transforms.RandomChoice([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomRotation(degrees=(-5,5))]),
    transforms.ToTensor()
    ])

    #Specifying image and labels location
    image_dir = '/scratch/cpandey1/hmi_jpgs_512/'
    if opt.fold==1:
        csv_file_train = '/scratch/cpandey1/data_labels/Fold1_train.csv'
        csv_file_val = '/scratch/cpandey1/data_labels/Fold1_val.csv'

    elif opt.fold==2:
        csv_file_train = '/scratch/cpandey1/data_labels/Fold2_train.csv'
        csv_file_val = '/scratch/cpandey1/data_labels/Fold2_val.csv'

    elif opt.fold==3:
        csv_file_train = '/scratch/cpandey1/data_labels/Fold3_train.csv'
        csv_file_val = '/scratch/cpandey1/data_labels/Fold3_val.csv'

    else:
        csv_file_train = '/scratch/cpandey1/data_labels/Fold4_train.csv'
        csv_file_val = '/scratch/cpandey1/data_labels/Fold4_val.csv'

    #Loading Dataset -- Trimonthly partitioned with defined augmentation
    ori_nf = NFDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = transformations)
    aug_nf = NFDataset(csv_file = csv_file_train, 
                             root_dir = image_dir,
                             transform = nf_augment)

    ori_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = transformations)
    hr_flip_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = hr_flip)
    vr_flip_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = vr_flip)
    rotation_fl = FLDataset(csv_file = csv_file_train, 
                                root_dir = image_dir,
                                transform = rotation)


    imbalance_ratio = 1-((2*len(ori_nf)) / ((4*len(ori_fl)) + (2*len(ori_nf))))
    
    
    train_set = ConcatDataset([ori_fl, hr_flip_fl, vr_flip_fl, rotation_fl, ori_nf, aug_nf])

    val_set = MyJP2Dataset(csv_file = csv_file_val, 
                                root_dir = image_dir,
                                transform = transformations)

    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=16, pin_memory = True, shuffle = True)

    val_loader = DataLoader(dataset=val_set, batch_size=opt.batch_size, num_workers=16, pin_memory = True, shuffle = False)

    return train_loader, val_loader, imbalance_ratio


def visualize_batch_sample(data_loader):
    cmap = plt.get_cmap('Greys_r')
    dataiter = iter(data_loader)
    #dataiter.next()
    images, labels = dataiter.next()
    flare_types = {0: 'Non_flare', 1: 'Flare'}
    fig, axis = plt.subplots(4, 4, figsize=(20, 20))
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            image, label = images[i], labels[i]
            ax.imshow(image.permute(1,2,0), cmap=cmap, vmin=0, vmax=1) # add image
            ax.set(title = f"{flare_types[label.item()]}")


def train():
    train_loader, val_loader, imbalance_ratio = dataloading()
    # visualize_batch_sample(train_loader)

    #Model Configuration
    learning_rate = opt.lr
    num_epochs = opt.epochs
    weight_decay = opt.weight_decay
    max_lr = opt.max_lr
    # factor = opt.factor
    # patience = opt.patience
    device_ids = [0, 1]
    # print("Model Selected: ", opt.models)

    if opt.models=="alex":
        net = Custom_AlexNet(ipt_size=(512, 512), pretrained=True).to(device)
        print("Model Selected: ", opt.models)
        # print(net)
    elif opt.models=="vgg":
        net = Custom_VGG16(ipt_size=(512, 512), pretrained=True).to(device)
        print("Model Selected: ", opt.models)
        # print(net)
    elif opt.models=="resnet":
        net = Custom_ResNet34(ipt_size=(512, 512), pretrained=True).to(device)
        print("Model Selected: ", opt.models)
        # print(net)
    else:
        print("Model Selected: ", opt.models)
        print('Invalid Model')
        exit()
    model = nn.DataParallel(net, device_ids=device_ids).to(device)

    # Loss and optimizer

    criterion = nn.NLLLoss(weight=torch.tensor([imbalance_ratio, 1-imbalance_ratio])).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=1e-6, verbose=True)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                       max_lr = max_lr, # Upper learning rate boundaries in the cycle for each parameter group
                       steps_per_epoch = len(train_loader), # The number of steps per epoch to train for.
                       epochs = num_epochs-1, # The number of epochs to train for.
                       anneal_strategy = 'cos') # Specifies the annealing strategy

    # Training Network
    print(f"Training in Progress Fold {opt.fold}...")
    train_loss_values = []
    val_loss_values = []
    train_tss_values = []
    val_tss_values = []
    train_hss_values = []
    val_hss_values = []
    train_time = []
    val_time = []
    learning_rate_values = []
    best_tss = 0.
    best_hss = 0.
    best_epoch = 0
    for epoch in range(1, num_epochs):

        # if epoch%5==0:
        #     for g in optimizer.param_groups:
        #         g['lr'] = g['lr']/2
        
        #Timer for Training one epoch
        start_train = timeit.default_timer() 
        
        # setting the model to train mode
        model.train()
        train_loss = 0
        train_tss = 0.
        train_hss = 0.
        train_prediction_list = []
        train_target_list = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            train_target_list.append(targets)
            
            # forward prop
            scores = model(data)
            loss = criterion(scores, targets)
            _, predictions = torch.max(scores,1)
            train_prediction_list.append(predictions)
            
            # backward prop
            optimizer.zero_grad()
            loss.backward()
            
            # Adam step
            optimizer.step()
            scheduler.step()
            
            # accumulate the training loss
            #print(loss.item())
            train_loss += loss.item()
            #train_acc+= acc.item()
            
            
        stop_train = timeit.default_timer()
        print(stop_train-start_train)
        train_time.append(stop_train-start_train)
        # Validation: setting the model to eval mode
        model.eval()
        start_val = timeit.default_timer()
        val_loss = 0.
        val_tss = 0.
        val_hss = 0.
        val_prediction_list = []
        val_target_list = []
        # Turning off gradients for validation
        with torch.no_grad():
            for d, t in val_loader:
                # Get data to cuda if possible
                d = d.to(device=device)
                t = t.to(device=device)
                val_target_list.append(t)
                
                # forward pass
                s = model(d)
                #print("scores", s)
                                    
                # validation batch loss and accuracy
                l = criterion(s, t)
                _, p = torch.max(s,1)
                #print("------------------------------------------------")
                #print(torch.max(s,1))
                #print('final', p)
                val_prediction_list.append(p)
                
                # accumulating the val_loss and accuracy
                val_loss += l.item()
                #val_acc += acc.item()
                del d,t,s,l,p
                torch.cuda.empty_cache()
        # scheduler.step()
        stop_val = timeit.default_timer()
        val_time.append(stop_val-start_val)
        learning_rate_values.append(optimizer.param_groups[0]['lr'])
        # print("Learning Rate Values: ", learning_rate_values)
        #Epoch Results
        train_loss /= len(train_loader)
        train_loss_values.append(train_loss)
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
        train_tss, train_hss = sklearn_Compatible_preds_and_targets(train_prediction_list, train_target_list)
        train_tss_values.append(train_tss)
        train_hss_values.append(train_hss)
        val_tss, val_hss = sklearn_Compatible_preds_and_targets(val_prediction_list, val_target_list)
        val_tss_values.append(val_tss)
        val_hss_values.append(val_hss)
        if val_tss>best_tss and (val_hss-best_hss)>=-0.02:
            print('New best model found!')
            best_tss = val_tss
            best_hss = val_hss
            best_epoch = epoch
            best_model = model.module.state_dict()
            best_path = f'{model_dir}/Model_{opt.models}_Epoch_{epoch}_fold{opt.fold}.pth'
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, best_path)
        print(f'Epoch: {epoch}/{num_epochs-1}')
        print(f'Training--> loss: {train_loss:.4f}, TSS: {train_tss:.4f}, HSS2: {train_hss:.4f} | Val--> loss: {val_loss:.4f}, TSS: {val_tss:.4f} | HSS2: {val_hss:.4f} ')
    PATH = f'{model_dir}/Model_{opt.models}_fold{opt.fold}.pth'
    print("**********************************************************************************")
    print("Best Model Selected", 'TSS: ', best_tss, "HSS: ", best_hss, 'Epoch: ', best_epoch)
    print("**********************************************************************************")
    torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, PATH)

    results = {
        'train_tss_values':train_tss_values,
        'val_tss_values':val_tss_values,
        'train_hss_values':train_hss_values,
        'val_hss_values':val_hss_values,
        'train_loss_values':train_loss_values,
        'val_loss_values':val_loss_values,
        'learning_rate': learning_rate_values,
        'train_time': train_time,
        'val_time': val_time
    }
    df = pd.DataFrame(results, columns=['train_tss_values','val_tss_values', 'train_hss_values', 'val_hss_values', 'train_loss_values', 'val_loss_values', 'learning_rate', 'train_time', 'val_time' ])
    df.to_csv(f'{results_dir}/Model_{opt.models}_fold{opt.fold}.csv', index=False, header=True)

if __name__ == "__main__":
    train()
