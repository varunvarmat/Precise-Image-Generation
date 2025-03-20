from torch.utils.data import DataLoader, random_split, TensorDataset
import yaml
import numpy as np
import json
import torch
from properties_ae import AutoEncoder
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
def train_test_loader(data,read=True):
    if read==False:
        with open ('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        print(type(data))
        dataset = TensorDataset(data)  # No labels
        train_size=config['auto-encoder']['train-size']
        #print(train_size)
        test_size=config['auto-encoder']['test-size']
        #print(test_size)
        train_size = int(train_size * len(dataset))  # 80% train
        #val_size = int(0.2 * len(dataset))
        test_size = int(test_size * len(dataset))  # 20% test

        # Split dataset
        train_set, test_set = random_split(dataset, [train_size, test_size])
        torch.save(train_set, "data/train_data.pth")
        torch.save(test_set, "data/test_data.pth")
    else:
        train_set=torch.load("train_data.pth")
        test_set=torch.load('test_data.pth')
    batch_size = config['auto-encoder']['batch-size']
    #print(batch_size)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader =DataLoader(test_set, batch_size=batch_size, shuffle=False)
    #print("reached")
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
    #print(np.shape(data))
    return data

def train_ae(trainloader):

    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    d_model = config['auto-encoder']['d_model']
    #print(d_model)
    dr_ff = config['auto-encoder']['dr_ff']
    #print(dr_ff)
    no_head = config['auto-encoder']['n_heads']
    #print(no_head)
    N = config['auto-encoder']['N']
    #print(N)
    schedule = config['auto-encoder']['schedule']
    #print(type(schedule))
    dropout = config['auto-encoder']['dropout']
    #print(dropout)
    epochs = config['auto-encoder']['epochs']
    #print(epochs)
    l_r = config['auto-encoder']['l_r']
    #print(l_r)
    lambda_reg=config['auto-encoder']['lambda_reg']
    train_size=config['auto-encoder']['train-size']
    test_size=config['auto-encoder']['test-size']
    batch_size = config['auto-encoder']['batch-size']
    #print(lambda_reg)

    run_id = wandb.util.generate_id()
    wandb.init(project="pig-properties-autoencoder", name=run_id, config={
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

    optimizer = Adam(autoencoder.parameters(), lr=l_r)

    epoch_bar = tqdm(total=epochs, desc="Epoch Progress", colour='blue', position=0)
    for epoch in range(epochs):

        #loss_per_batch=0
        training_loss = 0.0
        bar=tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False,colour='blue', position=1)
        for i, data in enumerate(bar):
            #print(data)
            # print(type(data))
            # print(len(data))
            #print(data)
            inputs = data[0]
            # print(type(inputs))
            # if not isinstance(data, torch.Tensor):
            #     inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(device)
            
            optimizer.zero_grad()

            reconstructed,latents= autoencoder(inputs)
            loss = compute_loss(inputs, reconstructed, latents, lambda_reg)
            #loss_per_batch+=loss.item()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        bar.close()
        epoch_bar.update(1)
        epoch_bar.set_postfix(loss=f"{training_loss / len(trainloader):.4f}")
        #loss_per_batch=loss_per_batch/(i+1)
        #loss_curve.append(loss_per_batch)
        #print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(trainloader) :.3f}')
    os.makedirs("properties-autoencoders", exist_ok=True)
    model_filename = f"properties-autoencoders/autoencoder_{run_id}.pth"  # Filename with WandB Run ID
    torch.save(autoencoder.state_dict(), model_filename)
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

def test_ae(autoencoder,testloader):
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss=0
    autoencoder.eval()
    with torch.no_grad():
        for data in testloader:
            inputs = data[0]
            inputs = inputs.to(device)

            outputs,_ = autoencoder(inputs)
            loss = compute_loss(inputs, outputs)

            test_loss=test_loss+loss
            total += inputs.size(0)
        print(f'\nAverage loss: {100 * test_loss/total} %')
        return 100 * test_loss / total
    
def verify():
    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    d_model = config['auto-encoder']['d_model']
    #print(d_model)
    dr_ff = config['auto-encoder']['dr_ff']
    #print(dr_ff)
    no_head = config['auto-encoder']['n_heads']
    #print(no_head)
    N = config['auto-encoder']['N']
    #print(N)
    schedule = config['auto-encoder']['schedule']
    #print(type(schedule))
    dropout = config['auto-encoder']['dropout']



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = AutoEncoder(d_model, dr_ff, N, no_head, schedule, dropout).to(device)

    # Create dummy input matching the input shape (batch_size=1, features=10)
    dummy_input = torch.randn(1, 16).to(device)

    # Initialize TensorBoard writer
    writer = SummaryWriter("runs/model_graph")

    # Log the model architecture
    writer.add_graph(model, dummy_input)

    # Close the writer
    writer.close()


