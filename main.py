
from args import argument_parser
from data_manager import BlockFrameDataset
from data_split import split_data
from model import CustomModel
from AverageMeter import AverageMeter
from utils.createDataset import createDataset
from utils.loggers import (Logger,RankLogger)
from utils.torchtools import (save_checkpoint,resume_from_checkpoint)
import time
import datetime

from tqdm.notebook import tqdm

import numpy as np
import os.path as osp

import sys

import warnings

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

# global variables
parser = argument_parser()
args = parser.parse_args()



# Main Function Code

def main():

    log_name = "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    # Code For GPU Selection
    has_cuda = True if torch.has_cuda else False
    has_mps = True if torch.has_mps else False

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
    train_data,val_data,test_data = split_data(customDatasetPath)

    val_dataset = BlockFrameDataset(customDatasetPath,val_data,block_size=args.block_size)
    train_dataset = BlockFrameDataset(customDatasetPath,train_data,block_size=args.block_size)
    test_dataset = BlockFrameDataset(customDatasetPath,test_data,block_size=args.block_size)

    val_loader = DataLoader(val_dataset,shuffle=True,batch_size=args.val_batch_size,num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=args.train_batch_size,num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset,shuffle=True,batch_size=args.test_batch_size,num_workers=1, pin_memory=True)

    # Procuring the pretrained model

    model,processor = CustomModel()
    model.to(device)

    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    if args.resume and osp.isdir(args.resume):
        startEpochs = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )
    else:
        startEpochs = args.epochs

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)

    for epoch in range(startEpochs,args.epochs):
        print("Starting Epoch",str(epoch+1),"......")
        avg_train_loss = train(model,processor,train_loader,val_loader,optimizer,device,epoch+1)
        print("Training Epoch",str(epoch+1),"Average Loss :",avg_train_loss)

        

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
            acc_avg = test(model,val_loader,device,is_test=False)
            print("Validation Ended ....................\n")
            
            for name in args.target_names:
                ranklogger.write(name, epoch + 1, acc_avg)


            
    
    print("\nTest Started ....................")
    test(model,test_loader,device)
    print("Test Ended ....................\n")

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Elapsed {elapsed}")
    ranklogger.show_summary()


def train(model, processor, data_loader, val_loader, optimizer, device, epoch):
    loss_meter = AverageMeter()
    model.train()

    for batch_idx, data in enumerate(data_loader):
        frame, label = data[0], data[1]
        frame = torch.squeeze(frame)
        label = label
        frame, label = frame.to(device), label.to(device)
        output = model(frame)
        logits = output.logits

        loss_this = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss_this.backward()
        optimizer.step()
        loss_meter.update(loss_this.item(), label.shape[0])
        
        print("Epoch {:02d} Batch [{:02d}/{:02d}] --> Avg Loss : {:f}".format(epoch,batch_idx+1,len(data_loader),loss_meter.avg))
    
    return loss_meter.avg
    


def test(model, data_loader, device, is_test=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    correct = 0
    model.eval()
    for batch_idx, data in enumerate(data_loader):
        frame, label = data[0], data[1]
        frame = torch.squeeze(frame)
        frame, label = frame.to(device), label.to(device)
        with torch.no_grad():
            output = model(frame)
        logits = output.logits

        loss_this = F.cross_entropy(logits, label)
        pred = logits.argmax(dim=1, keepdim=True)
        correct_this = pred.eq(label.view_as(pred)).sum().item()
        correct += correct_this
        acc_this = correct_this / label.shape[0] * 100.0

        acc_meter.update(acc_this, label.shape[0])
        loss_meter.update(loss_this.item(), label.shape[0])

        print("{:s} Batch [{:02d}/{:02d}] --> Batch Accuracy : {:.2f}%".format("Test" if is_test else "Validation",batch_idx+1,len(data_loader),acc_this))

    if is_test:
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct, len(data_loader.dataset), acc_meter.avg))
    
    else:
        print('Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct, len(data_loader.dataset), acc_meter.avg))

    return acc_meter.avg

if __name__ == "__main__":
    main()