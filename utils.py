import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, TensorDataset
import yaml
import numpy as np
import json
import torch
from properties_ae import AutoEncoder
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
def train_test_loader(data):
    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dataset = TensorDataset(data)  # No labels
    train_size=config['autoencoder']['train_size']
    test_size=config['autoencoder']['test_size']
    train_size = int(train_size * len(dataset))  # 80% train
    #val_size = int(0.2 * len(dataset))
    test_size = int(test_size * len(dataset))  # 20% test

    # Split dataset
    train_set, test_set = random_split(dataset, [train_size, test_size])
    batch_size = config['auto-encoder']['batch_size']
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader =DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return trainloader,testloader

def normalize(val,min_val,max_val,property):
    if val<min_val or val>max_val:
        #print(property)
        #print(val)
        val=np.round(val)
        #print(val)
        #print("error")
    return (val - min_val) / (max_val - min_val)

def format_data():
    with open("data/properties/properties.json", "r") as f:
        properties = json.load(f)
    #print(properties[0])
    n_samples=len(properties)
    data=[]
    for sample in properties:
        #property=[]
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
        # print(scene_properties)
        # print(shape_properties)
        # print(colour)
        # print(material_properties)
        # print(object_properties)
        property=scene_properties+shape_properties+colour+material_properties+object_properties
        # print(len(property))  
        # print(property)
        data.append(property)
    data=np.array(data)
    print(np.shape(data))
    return data

def train_ae(trainloader):
    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    d_model = config['autoencoder']['d_model']
    dr_ff = config['autoencoder']['dr_ff']
    no_head = config['autoencoder']['n_classes']
    N = config['autoencoder']['N']
    schedule = config['autoencoder']['schedule']
    dropout = config['autoencoder']['dropout']
    epochs = config['autoencoder']['epochs']
    l_r = config['autoencoder']['l_r']
    lambda_reg=config['autoencoder']['lambda_reg']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    autoencoder = AutoEncoder(d_model, dr_ff, N, no_head, schedule, dropout).to(device)

    optimizer = Adam(autoencoder.parameters(), lr=l_r)
    #loss_curve=[]
    #n_batches=len(train_loader)
    for epoch in range(epochs):

        #loss_per_batch=0
        training_loss = 0.0
        
        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")):
            #print(data)
            inputs = data
            inputs = inputs.to(device)
            
            optimizer.zero_grad()

            reconstructed,latents= autoencoder(inputs)
            loss = compute_loss(inputs, reconstructed, latents, lambda_reg)
            #loss_per_batch+=loss.item()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        #loss_per_batch=loss_per_batch/(i+1)
        #loss_curve.append(loss_per_batch)
        print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(trainloader) :.3f}')
    return autoencoder

def compute_loss(original, reconstructed, latent=None,lambda_reg=None):
    mse_loss = nn.MSELoss()
    mse = mse_loss(reconstructed, original)  # Reconstruction loss
    if latent is not None and lambda_reg is not None:
        l2_reg = torch.norm(latent, p=2)  # L2 norm (Frobenius norm)
    else:
        l2_reg=0
    return mse + lambda_reg * l2_reg

def test_ae(autoencoder,testloader):
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss=0
    with torch.no_grad():
        for data in testloader:
            inputs = data
            inputs = inputs.to(device)

            outputs,_ = autoencoder(inputs)
            loss = compute_loss(inputs, outputs)

            test_loss=test_loss+loss
            total += inputs.size(0)
        print(f'\nAverage loss: {100 * test_loss/total} %')
        return 100 * test_loss / total
