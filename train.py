import os 
import json
import time
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ConvLSTM import ConvLSTM


def train(args):
    
    gpu = args.gpu_id if torch.cuda.is_available() else None
    print(f'Device - GPU:{args.gpu_id}')
    
    if args.data_source == 'moving_mnist':
        from moving_mnist import MovingMNIST
        train_dataset = MovingMNIST(root=args.data_path,
                                    is_train=True, 
                                    seq_len=args.seq_len, 
                                    horizon=args.horizon)
        
    elif args.data_source == 'sst':
        from sst import SST
        train_dataset = SST(root=args.data_path,
                            is_train=True, 
                            seq_len=args.seq_len, 
                            horizon=args.horizon)
                    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True)
    
    # [TODO: define your model, optimizer, loss function]
    writer = SummaryWriter()  # initialize tensorboard writer
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ConvLSTM(in_channels=xxx, h_channels=xxx, num_layers=xxx, kernel_size=xxx, device=device)
    optimizer = optim.SGD(model.parameters, lr=0.01, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    
    if args.use_teacher_forcing:
        teacher_forcing_rate = 1.0
    else:
        teacher_forcing_rate = None
    
    for epoch in args.num_epochs:
        for i, data in enumerate(train_loader):
            
            # [TODO: train the model with a batch]
            
            # [TODO: use tensorboard to visualize training loss]
            writer.add_scalar()
            
            # [TODO: if using teacher forcing, update teacher forcing rate]

    
    
def main():
    
    parser = argparse.ArgumentParser(description='video_prediction')
    
    # load data from file
    parser.add_argument('--data_path', type=str, default='./data/', help='path to the datasets')
    parser.add_argument('--data_source', type=str, required=True, help='moving_mnist | sst')
    parser.add_argument('--model_name', type=str, required=True, help='name of the saved model')
    parser.add_argument('--seq_len', type=int, required=True, help='input frame length')
    parser.add_argument('--horizon', type=int, required=True, help='output frame length')
    parser.add_argument('--use_teacher_forcing', action='store_true', help='if using teacher forcing, default is False')
    parser.add_argument('--result_path', type=str, default='./results', help='path to the results, containing model and logs')

    # training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)    
    parser.add_argument('--num_epochs', type=int, default=3)   # TODO:
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    
    train(args)
    
    
if __name__ == "__main__":
    main()
