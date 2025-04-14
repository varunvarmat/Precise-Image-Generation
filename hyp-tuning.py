import wandb
import torch
from utils import train_test_loader, train_test_ae, get_config
from image_cnn_autoencoder import CNN_Autoencoder


# Define the sweep configuration
sweep_config = {
    "method": "random",  # or "grid", "bayes"
    "metric": {
        "name": "Validation Loss",
        "goal": "minimize"
    },
    "parameters": {
        "l_r": {
            "min": 1e-6,
            "max": 1e-4
        },
        "lambda_reg": {
            "values": [0.0, 0.001, 0.01, 0.1]
        },
        "lambda_lpips": {
            "values": [0.0, 0.001, 0.01, 0.1]
        }
    }
}
res=64
sweep_id = wandb.sweep(sweep_config, project=f"pig-image-cnn-autoencoder-{res}")





def sweep_train():
    with wandb.init() as run:
        config = run.config


        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model_type = "image-cnn-autoencoder"
        data_dir=f"data_res_{res}"
        trainloader, valloader, testloader=train_test_loader(data_dir, device,"image-cnn-autoencoder",read_from_file=True)
        
        # Use default values for other args, but use wandb.config for tuned ones
        batch_size, train_size, val_size, test_size, N, schedule, epochs, _, _, _, d_model, n_heads, dr_ff, dropout = get_config(model_type)
        model = CNN_Autoencoder().to(device)

        train_test_ae(
            res=64,
            device=device,
            trainloader=trainloader,
            valloader=valloader,
            testloader=testloader,
            model_type=model_type,
            model=model,
            d_model=d_model,
            dr_ff=dr_ff,
            no_head=n_heads,
            N=N,
            schedule=schedule,
            dropout=dropout,
            epochs=epochs,
            l_r=config.l_r,
            lambda_reg=config.lambda_reg,
            lambda_lpips=config.lambda_lpips,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            run = run,)

# Launch the sweep agent
wandb.agent(sweep_id, function=sweep_train, count=8)  # Runs sweep_train() 10 times