import os
from utils import train_test_ae,train_test_loader,get_config, visualize
from image_cnn_autoencoder import CNN_Autoencoder


def main():
    print(f"Process ID: {os.getpid()}")
    trainloader,testloader=train_test_loader("image-cnn-autoencoder",read_from_file=True) #the dataloader has images and labels
    batch_size, train_size, test_size, N, schedule, epochs, l_r, lambda_reg, d_model, n_heads, dr_ff, dropout = get_config("image-cnn-autoencoder")
    model = CNN_Autoencoder()
    train_test_ae(trainloader,testloader,"image-cnn-autoencoder", model=model, d_model=d_model, dr_ff=dr_ff, no_head=n_heads, N=N, schedule=schedule, dropout=dropout, epochs=epochs, l_r=l_r, lambda_reg=lambda_reg, train_size=train_size, test_size=test_size, batch_size=batch_size)
    # visualize("image-cnn-autoencoder", "image-cnn-autoencoders/autoencoder_y4ef0nyi.pth", testloader)
if __name__=="__main__":
    main()