import os
from utils import train_test_ae,train_test_loader



def main():
    print(f"Process ID: {os.getpid()}")
    trainloader,testloader=train_test_loader("image-decoder",read_from_file=False)
    train_test_ae(trainloader,testloader,"image-decoder")
if __name__=="__main__":
    main()