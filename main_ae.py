# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import copy
import numpy as np
import json
from torch.utils.data import random_split, TensorDataset
from utils import train_test_loader, format_data, train_ae, verify, test_ae
# to do:
# 1.copy data(properties) to this machine 
# 2.change properties.json to required format
# 3.build train and test loaders
# 4.design model architecture /data/varun/CrossFlow/libs/model/trans_autoencoder.py

           



        

    

def main():
    data=format_data()
    trainloader,testloader=train_test_loader(data)
    #print(len(trainloader.dataset))
    #print(len(testloader.dataset))
    #print("reached end")
    autoencoder=train_ae(trainloader)
    test_ae(autoencoder, testloader)
    #verify()
    





if __name__=='__main__':
    main()