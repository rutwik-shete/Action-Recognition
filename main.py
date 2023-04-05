
from args import argument_parser
from data_manager import BlockFrameDataset
from data_split import split_data
from model import CustomModel
from AverageMeter import AverageMeter

from tqdm.notebook import tqdm

import numpy as np

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

    print("Initializing Dataloader")

    #Split Dataset
    train_data,val_data,test_data = split_data(args.dataset_path)

    val_dataset = BlockFrameDataset(args.dataset_path,val_data,block_size=args.block_size)
    train_dataset = BlockFrameDataset(args.dataset_path,train_data,block_size=args.block_size)
    train_dataset = BlockFrameDataset(args.dataset_path,test_data,block_size=args.block_size)

    val_loader = DataLoader(val_data,shuffle=True,batch_size=args.val_batch_size,num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_data,shuffle=True,batch_size=args.train_batch_size,num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_data,shuffle=True,batch_size=args.test_batch_size,num_workers=1, pin_memory=True)

    # Procuring the pretrained model

    model,processor = CustomModel()

    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    for epoch in range(args.epochs):
        train(model,train_loader,optimizer,device)


def train(model, data_loader, optimizer, device):
    # meter
    loss_meter = AverageMeter()
    # switch to train mode
    model.train()
    tk = tqdm(data_loader, total=int(len(data_loader)), desc='Training', unit='frames', leave=False)
    for batch_idx, data in enumerate(tk):
        # fetch the data
        frame, label = data[0], data[1]
        # after fetching the data, transfer the model to the 
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by 
        frame, label = frame.to(device), label.to(device)
        # compute the forward pass
        output = model(frame)
        # compute the loss function
        loss_this = F.cross_entropy(output, label)
        # initialize the optimizer
        optimizer.zero_grad()
        # compute the backward pass
        loss_this.backward()
        # update the parameters
        optimizer.step()
        # update the loss meter
        loss_meter.update(loss_this.item(), label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg})
    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))

if __name__ == "__main__":
    main()