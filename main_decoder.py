import os
from utils import train_test_ae,train_test_loader,get_config, visualize, format_data
from image_cnn_autoencoder import CNN_Autoencoder
from image_autoencoder_ab import ImageAutoencoder
import torch

def main():
    print(f"Process ID: {os.getpid()}")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    res=64
    data_dir=f"data_res_{res}"
    trainloader, valloader, testloader=train_test_loader(data_dir, device,"image-cnn-autoencoder",read_from_file=False) #the dataloader has images and labels
    batch_size, train_size, val_size, test_size, N, schedule, epochs, l_r, lambda_reg, lambda_lpips,d_model, n_heads, dr_ff, dropout = get_config("image-cnn-autoencoder")
    model = ImageAutoencoder().to(device)
    train_test_ae(res,device,trainloader,valloader,testloader,"image-cnn-autoencoder", model=model, d_model=d_model, dr_ff=dr_ff, no_head=n_heads, N=N, schedule=schedule, dropout=dropout, epochs=epochs, l_r=l_r, lambda_reg=lambda_reg, lambda_lpips=lambda_lpips,train_size=train_size, val_size = val_size, test_size=test_size, batch_size=batch_size)
    # visualize("image-cnn-autoencoder", "image-cnn-autoencoders/autoencoder_y4ef0nyi.pth", testloader)
if __name__=="__main__":
    main()