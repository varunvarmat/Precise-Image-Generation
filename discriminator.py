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
    def __init__(self, device,annotations, img_dir, transform=None, target_transform=None):
        self.device = device 
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"image_{idx}.png")
        image = read_image(img_path)
        image = image.float() / 255.0 
        image = image.to(self.device)
        label = self.img_labels[idx]
        label = label.to(self.device)
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
                nn.MaxPool2d(2, 2),  # Output: 32x32x64

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Output: 16x16x128
            )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 16 * 128, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # Binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


def create_dataset(img_dir,device):
    print("reached")
    model = CNN_Autoencoder().to(device)
    model.load_state_dict(torch.load('image-cnn-autoencoders/autoencoder_ufnh641l.pth', map_location=device))
    trainloader,valloader,testloader=train_test_loader(device,"image-cnn-autoencoder",read_from_file=True)
    model.eval()
    labels=[]
    
    os.makedirs(img_dir, exist_ok=True)
    idx=0
    with torch.no_grad():
        loop = tqdm(trainloader, desc="Saving images", leave=False)
        for data in loop:
            inputs = data[0]
            # inputs = inputs.to(device)
            outputs,_ = model(inputs)
            # inputs = [tensor_to_pil(image) for image in inputs]
            # outputs = [tensor_to_pil(image) for image in outputs]
            for input_img, output_img in zip(inputs, outputs):
                input_pil = tensor_to_pil(input_img)
                output_pil = tensor_to_pil(output_img)

                input_pil.save(f"{img_dir}/image_{idx}.png")
                labels.append(1)
                idx += 1

                output_pil.save(f"{img_dir}/image_{idx}.png")
                labels.append(0)
                idx += 1

    print(len(labels))
    # print(len(labels[0]))
    return labels

def return_train_test_loaders(device, img_dir, labels, read_from_file=True):
    print("reached1")
    dataset = CustomDataset(device, labels, img_dir)
    train_indices, test_indices =None, None
    if read_from_file==False:
        # print(type(label))
        train_size=0.8
        test_size=0.1
        val_size=0.1
        train_size = int(train_size * len(dataset))  
        test_size = int(test_size * len(dataset))  
        val_size = len(dataset) - train_size - test_size
        train_indices, val_indices, test_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, val_size, test_size])
        torch.save(train_indices, f"{img_dir}/train_indices.pt")
        torch.save(val_indices, f"{img_dir}/val_indices.pt")
        torch.save(test_indices, f"{img_dir}/test_indices.pt")
    
    if train_indices is None and test_indices is None:
        train_indices = torch.load(f"{img_dir}/train_indices.pt",weights_only=False)
        val_indices = torch.load(f"{img_dir}/val_indices.pt",weights_only=False)
        test_indices = torch.load(f"{img_dir}/test_indices.pt",weights_only=False)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    batch_size = 512
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader

def train(device, trainloader, valloader):
    epochs = 15
    l_r = 1e-4
    run_id = wandb.util.generate_id()
    run = wandb.init(project=f"pig-discriminator", name=run_id, id=run_id, config={
            "epochs": epochs,
            "learning_rate": l_r
        })
    model = classifier()
    
    optimizer = Adam(model.parameters(), lr=l_r)
    model = model.to(device)
    
    epoch_bar = tqdm(total=epochs, desc="Epoch Progress", colour='blue', position=0)
    criterion = nn.BCEWithLogitsLoss()
    training_loss=0
    for epoch in range(epochs):
        correct = 0
        total = 0
        training_loss = 0.0

        bar = tqdm(trainloader, desc=f"Batch progress", leave=False, colour='green', position=1)
        
        for i, data in enumerate(bar):
            inputs, labels = data
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # Accuracy calculation
            preds = (torch.sigmoid(outputs) >= 0.5)
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0
        avg_loss = training_loss / len(trainloader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valloader:
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) >= 0.5)
                val_correct += (preds == labels.bool()).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(valloader)

        
        
        model.train()

        # Logging
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": avg_loss,
            "Training Accuracy": acc,
            "Validation Loss": avg_val_loss,
            "Validation Accuracy": val_acc,
        })
        bar.close()
        epoch_bar.update(1)
        epoch_bar.set_postfix(
            train_loss=f"{avg_loss:.4f}", train_acc=f"{acc*100:.2f}%",
            val_loss=f"{avg_val_loss:.4f}", val_acc=f"{val_acc*100:.2f}%"
        )
        
    
    os.makedirs("discriminators", exist_ok=True)
    torch.save(model.state_dict(), "discriminators/discriminator-1.pth")
    return model

def test(model, device, testloader):
    model = model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Compute predictions
            preds = torch.sigmoid(outputs) >= 0.5  # Bool tensor
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / len(testloader)
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}")
    wandb.log({"Test Loss": avg_loss})
    print(f"Accuracy: {accuracy:.4f}")
    wandb.log({"Test Accuracy": accuracy})


def main():
    read_from_big_file=True
    read_from_file=False
    img_dir="discriminator-images"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # trainloader,testloader=train_test_loader("image-cnn-autoencoder",read_from_file=True) #the dataloader has images and labels
    # batch_size, train_size, test_size, N, schedule, epochs, l_r, lambda_reg, d_model, n_heads, dr_ff, dropout = get_config("image-cnn-autoencoder")
    # model = CNN_Autoencoder()
    # img_dir,labels=train_test_ae(trainloader,testloader,"image-cnn-autoencoder", model=model, d_model=d_model, dr_ff=dr_ff, no_head=n_heads, N=N, schedule=schedule, dropout=dropout, epochs=epochs, l_r=l_r, lambda_reg=lambda_reg, train_size=train_size, test_size=test_size, batch_size=batch_size)
    if read_from_big_file==False:
        labels = create_dataset(img_dir,device)
        labels=torch.tensor(labels)
        print(labels.shape)
        #save tensors
        torch.save(labels, f"{img_dir}/labels.pt")
    else:
        #read from 
        labels= torch.load(f"{img_dir}/labels.pt",weights_only=False).float()

    trainloader, valloader, testloader= return_train_test_loaders(device, img_dir, labels , read_from_file=read_from_file) #returns new trainloaders and testloaders to train the discriminator
    model = train(device, trainloader, valloader)
    # model = classifier()
    # model = model.to(device)
    # model.load_state_dict(torch.load("discriminators/discriminator-1.pth", map_location=device))
    test(model, device, testloader)
    
    

if __name__=="__main__":
    main()
