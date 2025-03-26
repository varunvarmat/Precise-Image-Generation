import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.utils.data import DataLoader, random_split, TensorDataset
import yaml
import numpy as np
import json
import torch
from autoencoder import AutoEncoder
from properties_mlp_autoencoder import MLP_Autoencoder
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset




class CustomDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"image_{idx}.png")
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def train_test_loader(model, read_from_file=True):
    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    label=format_data()
    if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
    dataset = CustomDataset(label, "data/images")
    train_indices,test_indices=None,None
    if read_from_file==False:
        # print(type(label))
        train_size=config[model]['train-size']
        test_size=config[model]['test-size']
        train_size = int(train_size * len(dataset))  
        test_size = int(test_size * len(dataset))  
        # Split dataset
        # train_set, test_set = random_split(dataset, [train_size, test_size])
        # torch.save(train_set, "data/train_data.pt")
        # torch.save(test_set, "data/test_data.pt")
        train_indices, test_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, test_size])
        torch.save(train_indices, "data/train_indices.pt")
        torch.save(test_indices, "data/test_indices.pt")

        
    # else:
        # train_set=torch.load("data/train_data.pt",weights_only=False)
        # test_set=torch.load('data/test_data.pt',weights_only=False)
    if train_indices is None and test_indices is None:
        train_indices = torch.load("data/train_indices.pt",weights_only=False)
        test_indices = torch.load("data/test_indices.pt",weights_only=False)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    batch_size = config[model]['batch-size']
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return trainloader,testloader



def train_test_ae(trainloader,testloader, model_type, model, d_model=None, dr_ff=None, no_head=None, N=None, schedule=None, dropout=None, epochs=None, l_r=None, lambda_reg=None, train_size=None, test_size=None, batch_size=None):

    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    d_model = config[model]['d_model']
    dr_ff = config[model]['dr_ff']
    no_head = config[model]['n_heads']
    N = config[model]['N']
    print(N)
    schedule = config[model]['schedule']
    print(schedule)
    dropout = config[model]['dropout']
    epochs = config[model]['epochs']
    l_r = config[model]['l_r']
    lambda_reg=config[model]['lambda_reg']
    train_size=config[model]['train-size']
    test_size=config[model]['test-size']
    batch_size = config[model]['batch-size']

    run_id = wandb.util.generate_id()
    wandb.init(project=f"pig-{model}-autoencoder", name=run_id, config={
    "train_size": train_size,
    "test_size": test_size,
    "d_model": d_model,
    "dr_ff": dr_ff,
    "no_heads": no_head,
    "number of layers": N,
    "schedule": schedule,
    "dropout": dropout,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": l_r,
    "lambda_reg": lambda_reg,})

    #run_id = wandb.run.id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    autoencoder = AutoEncoder(d_model, dr_ff, N, no_head, schedule, dropout).to(device)
    #print(autoencoder)
    print("reached")
    optimizer = Adam(autoencoder.parameters(), lr=l_r)

    epoch_bar = tqdm(total=epochs, desc="Epoch Progress", colour='blue', position=0)
    for epoch in range(epochs):
        print("reached1")
        training_loss = 0.0
        bar=tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False,colour='green', position=1)
        for i, data in enumerate(bar):
            print("reached2")
            if model=="properties-encoder":
                inputs = data[1]
                # print(inputs.shape)
            elif model=="image-decoder":
                inputs==data[0]
                inputs=inputs.flatten(start_dim=1)
            inputs = inputs.to(device)
            print(inputs.shape)
            
            optimizer.zero_grad()

            reconstructed,latents= autoencoder(inputs)
            loss = compute_loss(inputs, reconstructed, latents, lambda_reg) #per sample

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        bar.close()
        epoch_bar.update(1)
        epoch_bar.set_postfix(loss=f"{training_loss / len(trainloader):.4f}") 
        wandb.log({"Training Loss": training_loss/len(trainloader), "Epoch": epoch + 1})
    if model=="properties-encoder":
        os.makedirs("properties-autoencoders", exist_ok=True)
        model_filename = f"properties-autoencoders/autoencoder_{run_id}.pth"  
    elif model=="image-decoder":
        os.makedirs("image-autoencoders", exist_ok=True)
        model_filename = f"image-autoencoders/autoencoder_{run_id}.pth"

    torch.save(autoencoder.state_dict(), model_filename)
    avg_loss=test_ae(autoencoder, testloader,model)
    wandb.log({"Test Loss": avg_loss})
    wandb.finish()
    return autoencoder

def compute_loss(original, reconstructed, latent=None,lambda_reg=0):
    mse_loss = nn.MSELoss()
    mse = mse_loss(reconstructed, original)  # Reconstruction loss
    if latent is not None and lambda_reg is not None:
        l2_reg = torch.norm(latent, p=2)  # L2 norm (Frobenius norm)
    else:
        l2_reg=0
    return mse + lambda_reg * l2_reg

def test_ae(autoencoder,testloader,model):
    #total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss=0
    autoencoder.eval()
    with torch.no_grad():
        for data in testloader:
            inputs = data[1]
            if model=="properties-encoder" or "properties_mlp_autoencoder":
                inputs = data[1]
            elif model=="image-decoder":
                inputs==data[0]
                inputs=inputs.flatten(start_dim=1)
            inputs = inputs.to(device)

            outputs,_ = autoencoder(inputs)
            loss = compute_loss(inputs, outputs)

            test_loss=test_loss+loss
            #total += inputs.size(0)
        print(f'\nAverage loss: {test_loss/len(testloader)}')
        return test_loss / len(testloader)




def normalize(val,min_val,max_val,property):
    if val<min_val or val>max_val:
        val=np.round(val)
    return (val - min_val) / (max_val - min_val)

def format_data():
    with open("data/properties/properties.json", "r") as f:
        properties = json.load(f)
    n_samples=len(properties)
    data=[]
    for sample in properties:
        azi_cam=sample['azimuth_camera']
        azi_cam/=2*np.pi
        elev_cam=sample['elevation_camera']
        elev_cam=normalize(elev_cam,np.pi/18,np.pi/3,"elevation_camera")
        azi_light=sample['azimuth_light']
        azi_light/=2*np.pi
        elev_light=sample['elevation_light']
        elev_light=normalize(elev_light,np.pi/12,np.pi/2,"elevation_light")
        scene_properties=[azi_cam,elev_cam,azi_light,elev_light]
        if sample['object_properties'][0]['shape']==1:
            #Cube
            #shape_properties=[3752/35330,3750/33152,7500/68480,15000/132352,22.068347666499903/22.068347666499903]
            shape_properties=[1,0,0]
        elif sample['object_properties'][0]['shape']==2:
            #cylinder
            #shape_properties=[35330/35330,33152/33152,68480/68480,132352/132352,18.556787934358/22.068347666499903]
            shape_properties=[0,1,0]
        elif sample['object_properties'][0]['shape']==3:
            #shape_properties=[15362/35330,15360/33152,30720/68480,61440/132352,12.540273923135828/22.068347666499903]
            shape_properties=[0,0,1]
        if sample['object_properties'][0]['material']==1:
            #material_properties=[1.0,0.2]
            material_properties=[1,0]
        elif sample['object_properties'][0]['material']==2:
            #material_properties=[0.0,1.0]
            material_properties=[0,1]
        colour=sample['object_properties'][0]['colour']
        x_pos=sample['object_properties'][0]['x_position']
        y_pos=sample['object_properties'][0]['y_position']
        #z_pos=sample['object_properties'][0]['z_position']
        z_rotation=sample['object_properties'][0]['z_rotation']

        x_pos=normalize(x_pos,-3,3,"x_pos")
        y_pos=normalize(x_pos,-3,3,"y_pos")
        #z_pos=normalize(x_pos,1,3,"z_pos") #need to normalize z
        z_rotation/=2*np.pi

        size=sample['object_properties'][0]['size']
        size=normalize(size,0.5,1.5,"size")

        #object_properties=[x_pos,y_pos,z_pos,z_rotation,size]
        object_properties=[x_pos,y_pos,z_rotation,size]
        property=scene_properties+shape_properties+colour+material_properties+object_properties
        data.append(property)
    data=np.array(data)
    #print(np.shape(data))
    return data



def train_test_mlp_ae(trainloader,testloader,model):

    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    N = config[model]['N']
    print(N)
    schedule = config[model]['schedule']
    print(schedule)
    # dropout = config[model]['dropout']
    epochs = config[model]['epochs']
    l_r = config[model]['l_r']
    lambda_reg=config[model]['lambda_reg']
    train_size=config[model]['train-size']
    test_size=config[model]['test-size']
    batch_size = config[model]['batch-size']

    run_id = wandb.util.generate_id()
    wandb.init(project=f"pig-{model}-autoencoder", name=run_id, config={
    "train_size": train_size,
    "test_size": test_size,
    
    "number of layers": N,
    "schedule": schedule,
    # "dropout": dropout,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": l_r,
    "lambda_reg": lambda_reg,})

    #run_id = wandb.run.id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    autoencoder = MLP_Autoencoder(N,schedule).to(device)
    #print(autoencoder)
    # print("reached")
    optimizer = Adam(autoencoder.parameters(), lr=l_r)

    epoch_bar = tqdm(total=epochs, desc="Epoch Progress", colour='blue', position=0)
    for epoch in range(epochs):
        # print("reached1")
        training_loss = 0.0
        bar=tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False,colour='green', position=1)
        for i, data in enumerate(bar):
            # print("reached2")
            if model=="properties-encoder" or "properties-mlp-autoencoder":
                inputs = data[1]
                # print(inputs.shape)
            elif model=="image-decoder":
                inputs==data[0]
                inputs=inputs.flatten(start_dim=1)
            inputs = inputs.to(device)
            # print(inputs.shape)
            
            optimizer.zero_grad()

            reconstructed,latents= autoencoder(inputs)
            loss = compute_loss(inputs, reconstructed, latents, lambda_reg) #per sample

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        bar.close()
        epoch_bar.update(1)
        epoch_bar.set_postfix(loss=f"{training_loss / len(trainloader):.4f}") 
        wandb.log({"Training Loss": training_loss/len(trainloader), "Epoch": epoch + 1})
    if model=="properties-encoder":
        os.makedirs("properties-autoencoders", exist_ok=True)
        model_filename = f"properties-autoencoders/autoencoder_{run_id}.pth"  
    elif model=="image-decoder":
        os.makedirs("image-autoencoders", exist_ok=True)
        model_filename = f"image-autoencoders/autoencoder_{run_id}.pth"

    torch.save(autoencoder.state_dict(), model_filename)
    avg_loss=test_ae(autoencoder, testloader,model)
    wandb.log({"Test Loss": avg_loss})
    wandb.finish()
    return autoencoder





    
    




    
