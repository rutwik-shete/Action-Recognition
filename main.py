
from args import argument_parser
from data_manager import BlockFrameDataset
from data_split import split_data
from AverageMeter import AverageMeter
from utils.createDataset import createDataset
from utils.loggers import (Logger,RankLogger)
from utils.torchtools import (save_checkpoint,resume_from_checkpoint)
import time
from datetime import datetime
import progressbar
import pandas as pd
from utils.iotools import (check_isfile,mkdir_if_missing)


import numpy as np
import os.path as osp

import sys
from sys import platform

import warnings

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torchinfo import summary

from model import getModel


# global variables
parser = argument_parser()
args = parser.parse_args()

import wandb
import random

from Constants import CATEGORY_INDEX

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="aml",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": args.model,
    "dataset": args.source_names,
    "epochs": args.epochs,
    },
    name=args.run_name,
)

# Main Function Code

def main():

    # Code For GPU Selection

    has_mps = None
    has_cuda = None

    if platform == "darwin": #MAC, check mps
        has_mps = True if torch.has_mps else False
    else: # linux or windows, check cuda
        has_cuda = True if torch.has_cuda else False

    device = torch.device('cpu')
    if has_cuda :
        device = torch.device('cuda')
        print("Using Cuda GPU")

    elif has_mps :
        device = torch.device('mps')
        print("Using MPS GPU")

    else :
        warnings.warn("Using SPU , GPU is Recommended")

    print("Creating Custom Data If Needed")
    customDatasetPath = createDataset(args.dataset_path,args.block_size,args.home_path)


    print("Initializing Dataloader")

    #Split Dataset

    train_split_path = osp.join(args.home_path,"splitInfo_BlockSize_"+str(args.block_size),"train.csv")
    val_split_path = osp.join(args.home_path,"splitInfo_BlockSize_"+str(args.block_size),"val.csv")
    test_split_path = osp.join(args.home_path,"splitInfo_BlockSize_"+str(args.block_size),"test.csv")

    if not (check_isfile(train_split_path) and check_isfile(val_split_path) and check_isfile(test_split_path)):

        train_data,val_data,test_data = split_data(customDatasetPath)

        train_data_df = pd.DataFrame({"VideoId":train_data})
        val_data_df = pd.DataFrame({"VideoId":val_data})
        test_data_df = pd.DataFrame({"VideoId":test_data})

        mkdir_if_missing(osp.join(args.home_path,"splitInfo_BlockSize_"+str(args.block_size)))

        train_data_df.to_csv(train_split_path,index=False)
        val_data_df.to_csv(val_split_path,index=False)
        test_data_df.to_csv(test_split_path ,index=False)
    
    else :

        train_data = pd.read_csv(train_split_path)['VideoId'].values.tolist()
        val_data = pd.read_csv(val_split_path)['VideoId'].values.tolist()
        test_data = pd.read_csv(test_split_path)['VideoId'].values.tolist()

    val_dataset = BlockFrameDataset(customDatasetPath,val_data,block_size=args.block_size)
    train_dataset = BlockFrameDataset(customDatasetPath,train_data,block_size=args.block_size)
    test_dataset = BlockFrameDataset(customDatasetPath,test_data,block_size=args.block_size)

    val_loader = DataLoader(val_dataset,shuffle=True,batch_size=args.val_batch_size,num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=args.train_batch_size,num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset,shuffle=True,batch_size=args.test_batch_size,num_workers=1, pin_memory=True)

    # Procuring the pretrained model

    model,processor = getModel(args)
    
    summary(model, input_size=(args.train_batch_size, args.block_size, 3, 224, 224))
    
    model.to(device)

    wandb.watch(model, log="all")
    
    learning_rate = args.lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)

    if args.resume and osp.isdir(args.resume):
        startEpochs = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )
    else:
        startEpochs = 0

    ranklogger = RankLogger(args.source_names, args.target_names)

    for epoch in range(startEpochs,args.epochs):
        print("\nStarting Epoch",str(epoch+1),"......")
        avg_train_acc,avg_train_loss = train(model,processor,train_loader,val_loader,optimizer,device,epoch+1)
        print("\nTraining Epoch",str(epoch+1),"Average Loss :",avg_train_loss,"Average Accuracy :",avg_train_acc)

        if(epoch+1) % args.eval_freq == 0:
            print("Saving Checkpoint .......")

            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "Avg_Train_Loss": avg_train_loss,
                    "epoch": epoch+1 ,
                    "optimizer": optimizer.state_dict(),
                },
                args.save_dir,
            )

            print("\nValidation Started ....................")
            avg_val_acc,avg_val_loss = test(model,val_loader,device,is_test=False)
            print("Validation Ended ....................\n")

            wandb.log({
                "TrainAccuracy":avg_train_acc,
                "TrainLoss":avg_train_loss,
                "ValidationAccuracy":avg_val_acc,
                "ValidationLoss":avg_val_loss,
                "Epoch":epoch
            })

            for name in args.target_names:
                ranklogger.write(name, epoch + 1, avg_val_acc)
                
        scheduler.step()
        
    print("\nTest Started ....................")
    test(model,test_loader,device)
    print("Test Ended ....................\n")

    ranklogger.show_summary()


def train(model, processor, data_loader, val_loader, optimizer, device, epoch):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    correct = 0
    model.train()

    bar = progressbar.ProgressBar(maxval=len(data_loader)).start()
    for batch_idx, data in enumerate(data_loader):
        bar.update(batch_idx+1)
        frame, label = data[0], data[1]
        frame = torch.squeeze(frame)
        frame, label = frame.to(device), label.to(device)
        output = model(frame)

        if(args.model == "timesformer400"):
            logits = output.logits
        elif(args.model == "resnet18WithAttention"):
            logits = output

        pred = logits.argmax(dim=1, keepdim=True)
        correct_this = pred.eq(label.view_as(pred)).sum().item()
        correct += correct_this
        acc_this = correct_this / label.shape[0] * 100.0

        loss_this = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss_this.backward()
        optimizer.step()
        
        acc_meter.update(acc_this, label.shape[0])
        loss_meter.update(loss_this.item(), label.shape[0])
        
        # print("Epoch {:02d} Batch [{:02d}/{:02d}] --> Avg Loss : {:f}".format(epoch,batch_idx+1,len(data_loader),loss_meter.avg))
    
    return acc_meter.avg, loss_meter.avg
    


def test(model, data_loader, device, is_test=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    correct = 0
    model.eval()
    bar = progressbar.ProgressBar(maxval=len(data_loader)).start()
    
    all_preds = np.empty((0))
    all_labels = np.empty((0))
    
    all_labels_pr = np.empty((0))
    all_preds_pr = np.empty((0,25))

    for batch_idx, data in enumerate(data_loader):
        bar.update(batch_idx+1)
        frame, label = data[0], data[1]
        frame = torch.squeeze(frame)
        frame, label = frame.to(device), label.to(device)
        with torch.no_grad():
            output = model(frame)
        
        if(args.model == "timesformer400"):
            logits = output.logits
        elif(args.model == "resnet18WithAttention"):
            logits = output

        loss_this = F.cross_entropy(logits, label)
        pred = logits.argmax(dim=1, keepdim=True)
        correct_this = pred.eq(label.view_as(pred)).sum().item()
        correct += correct_this
        acc_this = correct_this / label.shape[0] * 100.0

        acc_meter.update(acc_this, label.shape[0])
        loss_meter.update(loss_this.item(), label.shape[0])
        
        logits = logits.cpu().numpy()
        for i in range(logits.shape[0]):
            all_preds_pr = np.append(all_preds_pr, [logits[i]],axis=0)
        all_labels_pr = np.append(all_labels_pr, label.cpu().numpy())
        all_labels = np.append(all_labels, label.cpu().numpy())
        all_preds = np.append(all_preds, pred.cpu().numpy())
            
    classes = [*CATEGORY_INDEX]

    all_labels_pr = all_labels_pr.astype(int)

    if is_test:
        wandb.log({"pr" : wandb.plot.pr_curve(all_labels_pr, all_preds_pr, labels=classes)})
        wandb.log({"conf_mat" : wandb.sklearn.plot_confusion_matrix(all_labels, all_preds, classes)})

    #print("{:s} Batch [{:02d}/{:02d}] --> Batch Accuracy : {:.2f}%".format("Test" if is_test else "Validation",batch_idx+1,len(data_loader),acc_this))

    if is_test:
        print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct, len(data_loader.dataset), acc_meter.avg))
    
    else:
        print('\nValidation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct, len(data_loader.dataset), acc_meter.avg))

    return acc_meter.avg,loss_meter.avg

if __name__ == "__main__":
    start_time = datetime.now()

    log_name = "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))

    print("Running with command line: ", args)
    main()

    end_time = datetime.now()
    print('Runtime : {}'.format(end_time - start_time))