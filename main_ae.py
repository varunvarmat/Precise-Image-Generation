# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import copy
import os
from utils import train_test_loader, train_test_ae, get_config
from properties_mlp_autoencoder import MLP_Autoencoder
# to do:
# 1.copy data(properties) to this machine 
# 2.change properties.json to required format
# 3.build train and test loaders
# 4.design model architecture /data/varun/CrossFlow/libs/model/trans_autoencoder.py

           



        

    

def main():
    print(f"Process ID: {os.getpid()}")
    trainloader,testloader=train_test_loader("properties-mlp-autoencoder",read_from_file=True) #the dataloader has images and labels
    batch_size, train_size, test_size, N, schedule, epochs, l_r, lambda_reg, d_model, n_heads, dr_ff, dropout = get_config("properties-mlp-autoencoder")
    model = MLP_Autoencoder(N, schedule)
    train_test_ae(trainloader,testloader,"properties-mlp-autoencoder", model=model, d_model=d_model, dr_ff=dr_ff, no_head=n_heads, N=N, schedule=schedule, dropout=dropout, epochs=epochs, l_r=l_r, lambda_reg=lambda_reg, train_size=train_size, test_size=test_size, batch_size=batch_size)
    





if __name__=='__main__':
    main()