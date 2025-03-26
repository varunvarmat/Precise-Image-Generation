# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import copy
import os
from utils import train_test_loader, train_test_ae, train_test_mlp_ae
# to do:
# 1.copy data(properties) to this machine 
# 2.change properties.json to required format
# 3.build train and test loaders
# 4.design model architecture /data/varun/CrossFlow/libs/model/trans_autoencoder.py

           



        

    

def main():
    print(f"Process ID: {os.getpid()}")
    trainloader,testloader=train_test_loader("properties-mlp-autoencoder",read_from_file=True) #the dataloader has images and labels
    train_test_mlp_ae(trainloader,testloader,"properties-mlp-autoencoder")
    





if __name__=='__main__':
    main()