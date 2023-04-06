
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
    test_dataset = BlockFrameDataset(args.dataset_path,test_data,block_size=args.block_size)

    val_loader = DataLoader(val_dataset,shuffle=True,batch_size=args.val_batch_size,num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=args.train_batch_size,num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset,shuffle=True,batch_size=args.test_batch_size,num_workers=1, pin_memory=True)

    # Procuring the pretrained model

    model,processor = CustomModel()

    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    for epoch in range(args.epochs):
        train(model,processor,train_loader,val_loader,optimizer,device)
    
    test(model,test_loader,device)


def train(model, processor, data_loader, val_loader, optimizer, device):
    loss_meter = AverageMeter()
    model.train()
    model.to(device)
    tk = tqdm(data_loader, total=int(len(data_loader)), desc='Training', unit='frames', leave=False)
    for batch_idx, data in enumerate(tk):
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
        tk.set_postfix({"loss": loss_meter.avg})
    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))
    test(model,val_loader,device,is_test=False)


def test(model, data_loader, device, is_test=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    correct = 0
    model.eval()
    tk = tqdm(data_loader, total=int(len(data_loader)), desc='Test', unit='frames', leave=False)
    for batch_idx, data in enumerate(tk):
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

    if is_test:
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct, len(data_loader.dataset), acc_meter.avg))
    
    else:
        print('Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_meter.avg, correct, len(data_loader.dataset), acc_meter.avg))

if __name__ == "__main__":
    main()