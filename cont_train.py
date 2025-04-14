import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# from torch.utils.data import DataLoader, random_split, TensorDataset
# import yaml
import numpy as np
# import json
import torch
from autoencoder import AutoEncoder
# from properties_mlp_autoencoder import MLP_Autoencoder
from image_cnn_autoencoder import CNN_Autoencoder
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb
from PIL import Image
# import torchvision.transforms as transforms
# from torchvision.io import read_image
from torch.utils.data import Dataset
# import itertools
# from accelerate import Accelerator
# import signal
# import sys
# from pytorch_msssim import ssim

from torch.amp import GradScaler, autocast
from utils import train_test_loader, get_config, tensor_to_pil, test_ae
import lpips
def train_test_ae(run_id,device,trainloader, valloader, testloader, model_type, model, d_model=None, dr_ff=None, no_head=None, N=None, schedule=None, dropout=None, epochs=None, l_r=None, lambda_reg=None, lambda_lpips=None,train_size=None, val_size = None, test_size=None, batch_size=None,run=None):
    run = wandb.init(project=f"pig-{model_type}-64", id=run_id, resume="must", config={
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

    
    if model_type=="image-cnn-autoencoder":
        os.makedirs("image-cnn-autoencoders", exist_ok=True)
        model_filename = f"image-cnn-autoencoders/autoencoder_{run_id}V2.pth"
    


    
    # device = accelerator.device
    # trainloader = list(itertools.islice(trainloader, 1))
    optimizer = Adam(model.parameters(), lr=l_r)
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
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    lpips_loss = lpips.LPIPS(net='vgg')  # or net='vgg'
    lpips_loss = lpips_loss.to(device)
    lpips_loss.eval()  # Not necessary but often used
    # lambda_lpips = 0.01
    for epoch in range(epochs):
        training_loss = 0.0
        bar=tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False,colour='green', position=1)
        for i, data in enumerate(bar):
            inputs=data[0]#.to(device)
            # inputs = inputs.to(device)

            with autocast("cuda"):  # Tensor Cores are utilized here
                reconstructed, latents = model(inputs)
                loss = criterion(inputs, reconstructed)
                if lambda_reg!=0:
                    l2_reg = lambda_reg*torch.norm(latents, dim=1).mean()
                    loss=loss+lambda_reg*l2_reg
                if lambda_lpips!=0:
                    reconstructed = reconstructed * 2 - 1
                    inputs = inputs * 2 - 1 
                    with torch.no_grad():
                        lpips_mean = lpips_loss(reconstructed, inputs).mean()  
                    loss = loss + lambda_lpips * lpips_mean

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
                if lambda_lpips!=0:
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
            

            
        bar.close()
        epoch_bar.update(1)
        epoch_bar.set_postfix(loss=f"{training_loss / len(trainloader):.4f}")  ##!!!!!!!!!!!!
        wandb.log({"Training Loss": training_loss/len(trainloader), "Epoch": epoch + 1})#!!!!!!!!!!!
    


    torch.save(model.state_dict(), model_filename)
    avg_loss=test_ae(model, testloader,model_type,device, lambda_reg)
    wandb.log({"Test Loss": avg_loss})
    wandb.finish()
    # return img_dir,labels




def main():
    print(f"Process ID: {os.getpid()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res=64
    data_dir=f"data_res_{res}"
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    trainloader, valloader, testloader=train_test_loader(data_dir, device,"image-cnn-autoencoder",read_from_file=True) #the dataloader has images and labels
    batch_size, train_size, val_size, test_size, N, schedule, epochs, l_r, lambda_reg, lambda_lpips, d_model, n_heads, dr_ff, dropout = get_config("image-cnn-autoencoder")
    model = CNN_Autoencoder().to(device)
    run_id = "pnhakd5c"  # Replace with your actual run ID
    model.load_state_dict(torch.load(f'image-cnn-autoencoders/autoencoder_{run_id}.pth', map_location=device))
    train_test_ae(run_id, device,trainloader,valloader,testloader,"image-cnn-autoencoder", model=model, d_model=d_model, dr_ff=dr_ff, no_head=n_heads, N=N, schedule=schedule, dropout=dropout, epochs=epochs, l_r=l_r, lambda_reg=lambda_reg, lambda_lpips=lambda_lpips,train_size=train_size, val_size = val_size, test_size=test_size, batch_size=batch_size)
if __name__=="__main__":
    main()