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
from image_autoencoder_ab import ImageAutoencoder
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import itertools
# from accelerate import Accelerator
import signal
import sys
from pytorch_msssim import ssim

from torch.amp import GradScaler, autocast
import lpips
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

def train_test_loader(data_dir,device,model, read_from_file=True):
    image_file=os.path.join(data_dir,"images")
    properties_file=os.path.join(data_dir,"properties/properties.json")
    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    label=format_data(properties_file)
    if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
    
    dataset = CustomDataset(device,label, image_file)
    train_indices,val_indices, test_indices=None,None,None
    if read_from_file==False:
        # print(type(label))
        train_size=config[model]['train-size']
        val_size = config[model]['val-size']
        test_size=config[model]['test-size']
        train_size = int(train_size * len(dataset)) 
        val_size = int(val_size*len(dataset))
        test_size = int(test_size * len(dataset)) 
        print(train_size, val_size, test_size) 
        # Split dataset
        # train_set, test_set = random_split(dataset, [train_size, test_size])
        # torch.save(train_set, "data/train_data.pt")
        # torch.save(test_set, "data/test_data.pt")
        train_indices, val_indices, test_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, val_size,test_size])
        print(len(train_indices),len(test_indices),len(val_indices))
        torch.save(train_indices, f"{data_dir}/train_indices.pt")
        torch.save(val_indices,f"{data_dir}/val_indices.pt")
        torch.save(test_indices, f"{data_dir}/test_indices.pt")

        
    # else:
        # train_set=torch.load("data/train_data.pt",weights_only=False)
        # test_set=torch.load('data/test_data.pt',weights_only=False)
    if train_indices is None and test_indices is None:
        train_indices = torch.load(f"{data_dir}/train_indices.pt",weights_only=False)
        val_indices = torch.load(f"{data_dir}/val_indices.pt",weights_only=False)
        test_indices = torch.load(f"{data_dir}/test_indices.pt",weights_only=False)
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    batch_size = config[model]['batch-size']
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set,batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader



def train_test_ae(res, device,trainloader, valloader, testloader, model_type, model, d_model=None, dr_ff=None, no_head=None, N=None, schedule=None, dropout=None, epochs=None, l_r=None, lambda_reg=None, lambda_lpips=None,train_size=None, val_size = None, test_size=None, batch_size=None,run=None):
    if run is not None:
        run_id = run.id
        wandb.config.update({
            "train_size": train_size,
            "test_size": test_size,
            "val_size": val_size,
            "d_model": d_model,
            "dr_ff": dr_ff,
            "no_heads": no_head,
            "number of layers": N,
            "schedule": schedule,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": l_r,
            "lambda_reg": lambda_reg,
        }, allow_val_change=True)
    else:
        run_id = wandb.util.generate_id()
        run = wandb.init(project=f"pig-{model_type}-{res}", name=run_id, id=run_id, config={
            "train_size": train_size,
            "test_size": test_size,
            "val_size": val_size,
            "d_model": d_model,
            "dr_ff": dr_ff,
            "no_heads": no_head,
            "number of layers": N,
            "schedule": schedule,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": l_r,
            "lambda_reg": lambda_reg,
            "lamda_lpips":lambda_lpips
        })
    wandb.save("*.py")  # Saves all Python files in the directory

    if model_type=="properties-encoder":
        os.makedirs("properties-autoencoders", exist_ok=True)
        model_filename = f"properties-autoencoders/autoencoder_{run_id}.pth"  
    elif model_type=="image-decoder":
        os.makedirs("image-autoencoders", exist_ok=True)
        model_filename = f"image-autoencoders/autoencoder_{run_id}.pth"
    elif model_type=="properties-mlp-autoencoder":
        os.makedirs("properties-mlp-autoencoders", exist_ok=True)
        model_filename = f"properties-mlp-autoencoders/autoencoder_{run_id}.pth"
    elif model_type=="image-cnn-autoencoder":
        os.makedirs("image-cnn-autoencoders", exist_ok=True)
        model_filename = f"image-cnn-autoencoders/autoencoder_{run_id}.pth"
    


    
    # device = accelerator.device
    # trainloader = list(itertools.islice(trainloader, 1))
    optimizer = Adam(model.parameters(), lr=l_r)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-3)

    # model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)
    model = model.to(device)
   
    epoch_bar = tqdm(total=epochs, desc="Epoch Progress", colour='blue', position=0)
    # img_dir="discriminator-images"
    # os.makedirs(img_dir, exist_ok=True)
    # labels=[]
    # idx=0
    val_data=next(iter(valloader))
    val_imgs=val_data[0]
    # def save_model_on_exit(signum, frame):
    #     print("\nSaving model before exit...")
    #     torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #     }, model_filename)
    #     print(f"Model saved to {model_filename}")
    #     sys.exit(0)  # Exit gracefully
    # signal.signal(signal.SIGINT, save_model_on_exit)   # Handles Ctrl+C
    # signal.signal(signal.SIGTERM, save_model_on_exit)  # Handles `kill` command
    scaler = GradScaler()
    criterion2 = nn.MSELoss()
    criterion1 = nn.L1Loss()
    lpips_loss = lpips.LPIPS(net='vgg')  # or net='vgg'
    # Optional: move to GPU
    # lpips_device = torch.device("cuda:0")
    lpips_loss = lpips_loss.to(device)
    lpips_loss.eval()  # Not necessary but often used
    switch_epoch=300
    for epoch in range(epochs):
        training_loss = 0.0
        val_loss=0.0
        bar=tqdm(trainloader, desc=f"Batch progress", leave=False,colour='green', position=1)
        for i, data in enumerate(bar):
            if model_type=="properties-encoder" or model_type=="properties-mlp-autoencoder":
                inputs = data[1]#.to(device)
            elif model_type=="image-decoder" or model_type=="image-cnn-autoencoder":
                inputs=data[0]#.to(device)
            # inputs = inputs.to(device)

            with autocast("cuda"):  # Tensor Cores are utilized here
                reconstructed, latents = model(inputs)
                if epoch<switch_epoch:
                    loss = criterion2(inputs, reconstructed)
                else:
                    loss = criterion1(inputs, reconstructed)
                if lambda_reg!=0:
                    l2_reg = lambda_reg*torch.norm(latents, dim=1).mean()
                    loss=loss+lambda_reg*l2_reg       
                if lambda_lpips!=0 and epoch>50:
                    reconstructed = reconstructed * 2 - 1
                    inputs = inputs * 2 - 1 
                    with torch.no_grad():
                        lpips_mean = lpips_loss(reconstructed, inputs).mean()  
                    loss = loss + lambda_lpips * lpips_mean
                    # reconstructed_lpips = reconstructed.detach().to(lpips_device)
                    # inputs_lpips = inputs.detach().to(lpips_device)
                    # reconstructed_lpips = reconstructed_lpips * 2 - 1
                    # inputs_lpips = inputs_lpips * 2 - 1
                    # lpips_value = lpips_loss(reconstructed_lpips, inputs_lpips)
                    # lpips_mean = lpips_value.mean().to(device)
                    # loss = loss + lambda_lpips * lpips_mean


                


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.item()
            
            # if i==0 and epoch%50==0:
            #     for i in range(len(inputs)):
            #         image = tensor_to_pil(inputs[i])
            #         labels.append([1,0])
            #         image.save(f"{img_dir}/image_{idx}.png")
            #         idx+=1
            #     for i in range(len(reconstructed)):
            #         image = tensor_to_pil(reconstructed[i])
            #         labels.append([0,1])
            #         image.save(f"{img_dir}/image_{idx}.png")
            #         idx+=1
                
            if i==0 and epoch%50==0:
                if lambda_lpips!=0 and epoch>50:
                    reconstructed = (reconstructed + 1) / 2
                    inputs = (inputs + 1) / 2
                input_images = [tensor_to_pil(img) for img in inputs[:5]]  # Log first 5 inputs
                output_images = [tensor_to_pil(img) for img in reconstructed[:5]]  # Log first 5 outputs

                # Log images as input-output pairs
                table = wandb.Table(columns=["Input", "Output"])

                for index in range(5):
                    table.add_data(
                        wandb.Image(input_images[index], caption=f"Input {index}"),
                        wandb.Image(output_images[index], caption=f"Output {index}")
                    )
                    
                # print(f"Table Rows: {len(table.data)}")
                wandb.log({f" Training Input-Output Pairs (Epoch {epoch:03})": table, "Epoch": epoch+1})



                #log validation data
                reconstructed,latents= model(val_imgs)
                # val_loss=compute_loss(inputs, reconstructed, latents, lambda_reg).item() #+ compute_classifier_loss(reconstructed, device)#per sample
                # wandb.log({"Validation Loss": val_loss, "Epoch": epoch + 1})#!!!!!!!!!!!
                input_images = [tensor_to_pil(img) for img in val_imgs[:5]]  # Log first 5 inputs
                output_images = [tensor_to_pil(img) for img in reconstructed[:5]]  # Log first 5 outputs

                # Log images as input-output pairs
                table = wandb.Table(columns=["Input", "Output"])

                for index in range(5):
                    table.add_data(
                        wandb.Image(input_images[index], caption=f"Input {index}"),
                        wandb.Image(output_images[index], caption=f"Output {index}")
                    )
                    
                # print(f"Table Rows: {len(table.data)}")
                wandb.log({f"Validation Input-Output Pairs (Epoch {epoch:03})": table, "Epoch": epoch+1})


                # print("logging done")
            
        for i,data in enumerate(valloader):
            inputs=data[0]
            inputs=inputs.to(device)
            reconstructed,latents= model(inputs)
            if epoch<switch_epoch:
                loss = criterion2(inputs, reconstructed)
            else:
                loss = criterion1(inputs, reconstructed)
                # if lambda_reg!=0:
                #     l2_reg = lambda_reg*torch.norm(latents, dim=1).mean()
                #     loss=loss+lambda_reg*l2_reg       
                # if lambda_lpips!=0 and epoch>50:
                #     reconstructed = reconstructed * 2 - 1
                #     inputs = inputs * 2 - 1 
                #     with torch.no_grad():
                #         lpips_mean = lpips_loss(reconstructed, inputs).mean()  
                #     loss = loss + lambda_lpips * lpips_mean
            val_loss+=loss.item()

        scheduler.step(training_loss/len(trainloader)) 
        bar.close()
        epoch_bar.update(1)
        epoch_bar.set_postfix(loss=f"{training_loss / len(trainloader):.4f}")  ##!!!!!!!!!!!!
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": training_loss / len(trainloader),
            "Validation Loss": val_loss / len(valloader),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    


    torch.save(model.state_dict(), model_filename)
    avg_loss=test_ae(model, testloader,model_type,device, lambda_reg)
    wandb.log({"Test Loss": avg_loss})
    wandb.finish()
    # return img_dir,labels

# def get_mse_loss(original, reconstructed):
#     return nn.MSELoss()(reconstructed, original)  # Reconstruction loss
# def get_mae_loss(original, reconstructed):
#     return   # Reconstruction loss
# def get_d_ssim_loss(original, reconstructed):
#     ssim_val = ssim(original, reconstructed, data_range=1.0, size_average=True)  # average over batch
#     return (1 - ssim_val) / 2
# def get_regularization_loss(latent,lambda_reg):
#     l2_reg = torch.norm(latent, dim=1).mean() if latent is not None else 0
#     return lambda_reg * l2_reg


# def compute_loss(original, reconstructed, latent=None,lambda_reg=0):
#     mse_loss = nn.MSELoss()(reconstructed, original)
#     # mae_loss = nn.L1Loss()(reconstructed, original)
#     # ssim_val = ssim(original, reconstructed, data_range=1.0, size_average=True)
#     # d_ssim_loss = (1 - ssim_val) / 2
#     # regularization_loss = get_regularization_loss(latent, lambda_reg)
#     # return 0.8*mse_loss + 0.2*d_ssim_loss
#     # return mse_loss + regularization_loss
#     return mse_loss

# def compute_classifier_loss(output_imgs, device):#here output images are the result of the image autoencoder
#     from discriminator import classifier
#     model = classifier()
#     model.load_state_dict(torch.load("discriminators/discriminator-1.pth"))
#     model.to(device)
#     model.eval()
#     criterion=nn.BCEWithLogitsLoss()
#     labels = torch.ones(len(output_imgs), dtype=torch.float32, device=device)  
#     pred=model(output_imgs)
#     loss=criterion(pred, labels)
#     return loss

def test_ae(autoencoder,testloader,model_type,device, lambda_reg):
    test_loss=0
    autoencoder.eval()
    logged=0
    criterionL2 = nn.MSELoss()
    # criterionL1 = nn.L1Loss()
    with torch.no_grad():
        for data in testloader:
            if model_type=="properties-encoder" or model_type=="properties-mlp-autoencoder":
                inputs = data[1]
            elif model_type=="image-decoder" or model_type=="image-cnn-autoencoder":
                inputs = data[0]
                # inputs=inputs.flatten(start_dim=1)
            # inputs = inputs.to(device)

            outputs, latent = autoencoder(inputs)
            lossL2 = criterionL2(inputs, outputs)
            # lossL1 = criterionL1(inputs, outputs)
            # loss = 0.7*lossL2 + 0.3*lossL1
            loss=lossL2
            if lambda_reg!=0:
                l2_reg = lambda_reg*torch.norm(latent, dim=1).mean()
                loss=loss+lambda_reg*lambda_reg

            test_loss=test_loss+loss
            if logged==0:
                logged=1
                inputs = [tensor_to_pil(image) for image in inputs[:10]]
                outputs = [tensor_to_pil(image) for image in outputs[:10]]
        
                table = wandb.Table(columns=["Input", "Output"])

                for i in range(len(inputs)):
                    table.add_data(
                        wandb.Image(inputs[i], caption=f"Input {i}"),
                        wandb.Image(outputs[i], caption=f"Output {i}")
                    )

                wandb.log({"Test Input-Output Pairs": table})


            #total += inputs.size(0)
    print(f'\nAverage loss: {test_loss/len(testloader)}')
    # wandb.finish()
    return test_loss / len(testloader)

def tensor_to_pil(image_tensor):
    """Convert a PyTorch tensor to a PIL image"""
    transform = transforms.ToPILImage()
    return transform(image_tensor.cpu().detach().clamp(0, 1))  # Clamp values between 0-1


def normalize(val,min_val,max_val,property):
    if val<min_val or val>max_val:
        val=np.round(val)
    return (val - min_val) / (max_val - min_val)

def format_data(properties_file):
    with open(properties_file, "r") as f:
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
        properties=scene_properties+shape_properties+colour+material_properties+object_properties
        data.append(properties)
    data=np.array(data)
    #print(np.shape(data))
    return data



# def train_test_mlp_ae(trainloader,testloader,model):

#     with open ('config.yaml', 'r') as file:
#         config = yaml.safe_load(file)
#     N = config[model]['N']
#     print(N)
#     schedule = config[model]['schedule']
#     print(schedule)
#     # dropout = config[model]['dropout']
#     epochs = config[model]['epochs']
#     l_r = config[model]['l_r']
#     lambda_reg=config[model]['lambda_reg']
#     train_size=config[model]['train-size']
#     test_size=config[model]['test-size']
#     batch_size = config[model]['batch-size']

#     run_id = wandb.util.generate_id()
#     wandb.init(project=f"pig-{model}-autoencoder", name=run_id, config={
#     "train_size": train_size,
#     "test_size": test_size,
    
#     "number of layers": N,
#     "schedule": schedule,
#     # "dropout": dropout,
#     "epochs": epochs,
#     "batch_size": batch_size,
#     "learning_rate": l_r,
#     "lambda_reg": lambda_reg,})

#     #run_id = wandb.run.id

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
#     autoencoder = MLP_Autoencoder(N,schedule).to(device)
#     #print(autoencoder)
#     # print("reached")
#     optimizer = Adam(autoencoder.parameters(), lr=l_r)

#     epoch_bar = tqdm(total=epochs, desc="Epoch Progress", colour='blue', position=0)
#     for epoch in range(epochs):
#         # print("reached1")
#         training_loss = 0.0
#         bar=tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False,colour='green', position=1)
#         for i, data in enumerate(bar):
#             # print("reached2")
#             if model=="properties-encoder" or "properties-mlp-autoencoder":
#                 inputs = data[1]
#                 # print(inputs.shape)
#             elif model=="image-decoder":
#                 inputs==data[0]
#                 inputs=inputs.flatten(start_dim=1)
#             inputs = inputs.to(device)
#             # print(inputs.shape)
            
#             optimizer.zero_grad()

#             reconstructed,latents= autoencoder(inputs)
#             loss = compute_loss(inputs, reconstructed, latents, lambda_reg) #per sample

#             loss.backward()
#             optimizer.step()

#             training_loss += loss.item()
#         bar.close()
#         epoch_bar.update(1)
#         epoch_bar.set_postfix(loss=f"{training_loss / len(trainloader):.4f}") 
#         wandb.log({"Training Loss": training_loss/len(trainloader), "Epoch": epoch + 1})
#     if model=="properties-encoder":
#         os.makedirs("properties-autoencoders", exist_ok=True)
#         model_filename = f"properties-autoencoders/autoencoder_{run_id}.pth"  
#     elif model=="image-decoder":
#         os.makedirs("image-autoencoders", exist_ok=True)
#         model_filename = f"image-autoencoders/autoencoder_{run_id}.pth"

#     torch.save(autoencoder.state_dict(), model_filename)
#     avg_loss=test_ae(autoencoder, testloader,model)
#     wandb.log({"Test Loss": avg_loss})
#     wandb.finish()
#     return autoencoder

def get_config(model_type):
    with open ('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    batch_size = config[model_type]['batch-size']
    train_size = config[model_type]['train-size']
    test_size = config[model_type]['test-size']
    
    epochs = config[model_type]['epochs']
    l_r = config[model_type]['l_r']
    lambda_reg = config[model_type]['lambda_reg']
    if model_type=="properties-mlp-autoencoder":
        N = config[model_type]['N']
        schedule = config[model_type]['schedule']
        d_model = None
        n_heads = None
        dr_ff = None
        dropout = None
    elif model_type=="image-cnn-autoencoder":
        N = None
        schedule = None
        d_model = None
        n_heads = None
        dr_ff = None
        dropout = None
        lambda_lpips = config[model_type]['lambda_lpips']
        val_size = config[model_type]['val-size']


    return batch_size, train_size, val_size, test_size, N, schedule, epochs, l_r, lambda_reg, lambda_lpips,d_model, n_heads, dr_ff, dropout



def visualize(model_type, model_name,testloader):
    if model_type=="image-cnn-autoencoder":
        model = CNN_Autoencoder()
        wandb.init(project="autoencoder-tracking", name="image-reconstruction")
    elif model_type=="properties-mlp-autoencoder":
        with open ('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        N = config["properties-mlp-autoencoder"]['N']
        schedule = config["properties-mlp-autoencoder"]['schedule']
        model = MLP_Autoencoder(N, schedule)
        wandb.init(project="autoencoder-tracking", name="properties-reconstruction")
    model.load_state_dict(torch.load(model_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logged=0
    with torch.no_grad():
        for data in testloader:
            if model_type=="properties-encoder" or model_type=="properties-mlp-autoencoder":
                inputs = data[1]
            elif model_type=="image-decoder" or model_type=="image-cnn-autoencoder":
                inputs = data[0]
                # inputs=inputs.flatten(start_dim=1)
            inputs = inputs.to(device)
            outputs,_ = model(inputs)
            if logged==0:
                logged=1
                inputs = [tensor_to_pil(image) for image in inputs]
                outputs = [tensor_to_pil(image) for image in outputs]
        
                table = wandb.Table(columns=["Input", "Output"])

                for i in range(len(data[1])):
                    table.add_data(
                        wandb.Image(inputs[i], caption=f"Input {i}"),
                        wandb.Image(outputs[i], caption=f"Output {i}")
                    )

                wandb.log({"Input-Output Pairs": table})
    wandb.finish()

    

        





    
    




    
