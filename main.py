
from args import argument_parser
from data_manager import BlockFrameDataset
from data_split import split_data

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch

import warnings

import torch

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

    val_dataset = BlockFrameDataset(args.dataset_path,val_data)
    train_dataset = BlockFrameDataset(args.dataset_path,train_data)
    train_dataset = BlockFrameDataset(args.dataset_path,test_data)

    



    

if __name__ == "__main__":
    main()