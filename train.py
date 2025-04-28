from custom_dataset import MNIST_dataset, Celeb_dataset
import yaml
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import torch.nn as nn
from tqdm import tqdm
from engine import train_step
from pathlib import Path
import numpy as np

def train():
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
     
    # Path / Options
    configPath1 = './MNIST_config.yaml'
    configPath2 = './MNIST_color_config.yaml'
    configPath3 = './Celeb_config.yaml'
    
    dataType1 = 'MNIST'
    dataType2 = 'Celeb'
    
    
    
    # Flag
    showInfo = True
    loadModel = False
    
    # Configurables
    configPath = configPath2
    dataType = dataType1
    saveName = "result_MNIST_color_1"
    loadName = "result_MNIST_color_1"
    
    
    
    
    # SaveFolder
    saveFolder = Path(f"./{saveName}")
    if not saveFolder.is_dir():
        saveFolder.mkdir(parents = True,
                         exist_ok = True)
    
    
    
    
    
    
    
    
    
    # 0. Load the config
    with open(configPath, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            
    if showInfo:
        print("The loaded config is as following: ")
        for key in config.keys():
            print(f"[INFO] {key}: {config[key]}")
        print("\n")
    
    
    
    # 1. Load dataset
    if dataType == 'MNIST':
        trainData = MNIST_dataset(directory = "./MNIST_train",
                                  config = config)
    elif dataType == 'Celeb':
        trainData = Celeb_dataset(directory = "./img_align_celeba",
                                  config = config)
    
    
    # 2. Visualize the dataset
    if showInfo:
        randNum = torch.randint(0, len(trainData)-1, (9,))
        for idx, num in enumerate(randNum):
            trainImg = trainData[num]
            trainImgPlt = (trainImg.permute(1,2,0) + 1) / 2
            plt.subplot(3,3, idx+1)
            plt.imshow(trainImgPlt, cmap="gray")
            plt.axis(False)
        plt.tight_layout()
        plt.show()
        
            
        print(f"The loaded dataset is as following...")
        print(f"[INFO] Number of images in the dataset: {len(trainData)}")
        print(f"[INFO] Size of an image               : {trainImg.shape}")
        print(f"[INFO] Value range in the image       : {trainImg.min()} to {trainImg.max()} ")
        print("\n")
    
    # 3. Load dataloader
    trainDataLoader = DataLoader(dataset = trainData,
                                 batch_size = config['batch_size'],
                                 shuffle = True,
                                 num_workers = config['num_workers'])
    
    # 4. Visualize the dataloader
    if showInfo:
        trainImgBatch  = next(iter(trainDataLoader))
        print("The loaded dataloader is as following: ")
        print(f"[INFO] The number of batches          : {len(trainDataLoader)}")
        print(f"[INFO] The number of images per batch : {len(trainImgBatch)}")
        print(f"[INFO] The size of an image           : {trainImgBatch[0].shape}")
    
    # 5. Instantiate a model
    generator = Generator(config = config).to(device)
    discriminator = Discriminator(config = config).to(device)
    
    # 6. Verify the model
# =============================================================================
#     from torchinfo import summary
#     summary(model = generator,
#             input_size = (1,100,1,1),
#             col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
#             row_settings = ['var_names'])
# =============================================================================

    # 7. Optimizer and loss function
    lossFn = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(params = generator.parameters(),
                                   lr = config['lr'],
                                   betas = (config['betas1'], config['betas2']))
    d_optimizer = torch.optim.Adam(params = discriminator.parameters(),
                                   lr = config['lr'],
                                   betas = (config['betas1'], config['betas2']))
    
    # 8. Training loop
    if loadModel:
        
        loadFile = Path(f"./{loadName}")

        print("[INFO] Loading previously best model...")
        
        trainInfo = torch.load(f = loadFile / "trainInfo.pt",
                               weights_only = False)
        for key in trainInfo.keys():
            if key != 'g_optimizer' and key != 'd_optimizer':
                print(f"{key} : {trainInfo[key]}")
        
        discriminator.load_state_dict(torch.load(f = loadFile / "discriminator.pt",
                                                 weights_only = True))
        generator.load_state_dict(torch.load(f= loadFile / "generator.pt",
                                             weights_only = True))
        g_optimizer.load_state_dict(trainInfo['g_optimizer'])
        d_optimizer.load_state_dict(trainInfo['d_optimizer'])
    
    for epoch in tqdm(range(config['epochs'])):
        
        trainResult = train_step(generator = generator,
                                 discriminator = discriminator,
                                 dataloader = trainDataLoader,
                                 device = device,
                                 g_optimizer = g_optimizer,
                                 d_optimizer = d_optimizer,
                                 loss_fn = lossFn,
                                 config = config,
                                 save_file = saveFolder / "Train Samples",
                                 epoch = epoch)
        
    
        
        print(f"\nThe current epoch: {epoch}")
        for key in trainResult.keys():
            print(f"[INFO] {key} : {trainResult[key]:.4f}")
        print("\n")
        
        torch.save(obj = generator.state_dict(),
                   f = saveFolder / "generator.pt")
        torch.save(obj = discriminator.state_dict(),
                   f = saveFolder / "discriminator.pt")
        
        saveResult = trainResult
        saveResult['d_optimizer'] = d_optimizer.state_dict()
        saveResult['g_optimizer'] = g_optimizer.state_dict()
        
        torch.save(obj = saveResult,
                   f = saveFolder / "trainInfo.pt")



if __name__ == "__main__":
    train()
