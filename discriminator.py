import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.utils.data import DataLoader, random_split, TensorDataset
import yaml
import numpy as np
import json
import torch
from autoencoder import AutoEncoder
from properties_mlp_autoencoder import MLP_Autoencoder
from image_cnn_autoencoder import CNN_Autoencoder
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import itertools
from image_cnn_autoencoder import CNN_Autoencoder
from utils import train_test_loader, tensor_to_pil, train_test_ae, get_config

class CustomDataset(Dataset):
    def __init__(self, label, img_dir, transform=None, target_transform=None):
        self.img_labels = label
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"image_{idx}.png")
        image = read_image(img_path)
        image = image.float() / 255.0 
        label = self.img_labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Output: 128x128x16
                
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Output: 64x64x32
                
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)  # Output: 32x32x64
            )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 32 * 64, 512),  
            nn.ReLU(),
            nn.Linear(512, 2),  # Binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def create_dataset():
    model=CNN_Autoencoder()
    model.load_state_dict(torch.load("image-cnn-autoencoders/autoencoder_ubbppxo4.pth"))
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    trainloader,testloader=train_test_loader("image-cnn-autoencoder",read_from_file=True)
    model.eval()
    labels=[]
    img_dir="discriminator-images"
    os.makedirs(img_dir, exist_ok=True)
    idx=0
    with torch.no_grad():
        for data in trainloader:
            inputs = data[0]
            inputs = inputs.to(device)
            outputs,_ = model(inputs)
            # inputs = [tensor_to_pil(image) for image in inputs]
            # outputs = [tensor_to_pil(image) for image in outputs]
            for i in range(len(inputs)):
                image = tensor_to_pil(inputs[i])
                labels.append([1,0])
                image.save(f"{img_dir}/image_{idx}.png")
                idx+=1
            for i in range(len(outputs)):
                image = tensor_to_pil(outputs[i])
                labels.append([0,1])
                image.save(f"{img_dir}/image_{idx}.png")
                idx+=1
    print(len(labels))
    print(len(labels[0]))
    return img_dir,labels

def return_train_test_loaders(img_dir, labels, read_from_file=True):
    print("reached1")
    dataset = CustomDataset(labels, img_dir)
    train_indices, test_indices =None, None
    if read_from_file==False:
        # print(type(label))
        train_size=0.8
        test_size=0.2
        train_size = int(train_size * len(dataset))  
        test_size = int(test_size * len(dataset))  
        train_indices, test_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, test_size])
        torch.save(train_indices, f"{img_dir}/train_indices.pt")
        torch.save(test_indices, f"{img_dir}/test_indices.pt")
    
    if train_indices is None and test_indices is None:
        train_indices = torch.load(f"{img_dir}/train_indices.pt",weights_only=False)
        test_indices = torch.load(f"{img_dir}/test_indices.pt",weights_only=False)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    batch_size = 32
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return trainloader,testloader

def main():
    trainloader,testloader=train_test_loader("image-cnn-autoencoder",read_from_file=True) #the dataloader has images and labels
    batch_size, train_size, test_size, N, schedule, epochs, l_r, lambda_reg, d_model, n_heads, dr_ff, dropout = get_config("image-cnn-autoencoder")
    model = CNN_Autoencoder()
    img_dir,labels=train_test_ae(trainloader,testloader,"image-cnn-autoencoder", model=model, d_model=d_model, dr_ff=dr_ff, no_head=n_heads, N=N, schedule=schedule, dropout=dropout, epochs=epochs, l_r=l_r, lambda_reg=lambda_reg, train_size=train_size, test_size=test_size, batch_size=batch_size)
    
    labels=torch.tensor(labels)
    print(labels.shape)
    trainloader, testloader= return_train_test_loaders(img_dir, labels , read_from_file=False)
    model = classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    optimizer = Adam(model.parameters(), lr=1e-6)
    model = model.to(device)
    epochs = 100
    epoch_bar = tqdm(total=epochs, desc="Epoch Progress", colour='blue', position=0)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        bar=tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False,colour='green', position=1)
        for i, data in enumerate(bar):
            inputs, labels=data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs= model(inputs)
            loss=criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        bar.close()
        epoch_bar.update(1)
        epoch_bar.set_postfix(loss=f"{training_loss / len(trainloader):.4f}")  ##!!!!!!!!!!!!
    
    os.makedirs("discriminators", exist_ok=True)
    torch.save(model.state_dict(), "discriminators/discriminator-1.pth")
    model.eval()
    test_loss=0
    with torch.no_grad():
        for data in testloader:
            inputs, labels=data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = criterion(outputs,labels)
            test_loss=test_loss+loss


            #total += inputs.size(0)
    print(f'\nAverage loss: {test_loss/len(testloader)}')
    

if __name__=="__main__":
    main()
