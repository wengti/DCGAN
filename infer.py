import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import yaml
from pathlib import Path
from model import Generator
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
import random

def sample(generator, config, save_name, device, noise = None):
    
    """
    Generate an image
    If no noise is None, a random input noise will be created.
    
        - save_name: folder / file_name.png
        - noise: Bxlat_dimx1x1, on cuda device
    """
    
    # When no noise is provided, a set of noise in the shape of num_samples x lat_dim x 1 x 1 is created
    if noise == None:
        lat_dim = config['latent_dim']
        num_samples = 100
        samples_per_row = 10
        noise = torch.randn((num_samples, lat_dim, 1, 1)).to(device)
    else:
        samples_per_row = noise.shape[0]
        
    # Pass the input noise to generator 
    generator.eval()
    with torch.inference_mode():
        
        generated_img = generator(noise)
        generated_img = ((generated_img + 1)/2).cpu() # cpu, range from 0 to 1
        grid = make_grid(generated_img,
                         nrow = samples_per_row)
        output_img = ToPILImage()(grid)
        output_img.save(save_name)


def visualize_latent_space(generator, config, save_name, samples_per_row, device, fixed_inpt = None):
    """
    Visualize the latent space created by 4 input noise at 4 corners of a 2D dimensions
    If fixed_inpt is None, 4 random inpt noise will be created.
    
        - save_name: folder / file_name.png
        - samples_per_row: Number of images to be generated / interpolated for each row
        - fixed_inpt: Bxlat_dimx1x1, where B should be 4, on cuda
    """
    
    lat_dim = config['latent_dim']
    
    # Whe no input noise is provided, 4 random noise is generated as the input
    if fixed_inpt == None:
        tl = torch.randn((1, lat_dim, 1, 1))
        tr = torch.randn((1, lat_dim, 1, 1))
        bl = torch.randn((1, lat_dim, 1, 1))
        br = torch.randn((1, lat_dim, 1, 1))
    else:
        assert fixed_inpt.shape[0] == 4, "[ERROR] Provided input is not 4. Unable to perform a 4 corner interpolation."
        tl = fixed_inpt[0].unsqueeze(0) # Unsqueeze to maintain the shape of 1xlat_dimx1x1
        tr = fixed_inpt[1].unsqueeze(0)
        bl = fixed_inpt[2].unsqueeze(0)
        br = fixed_inpt[3].unsqueeze(0)
    
    factors = torch.linspace(0, 1, samples_per_row) # Number of interpolation to be made between the input images
    
    # Find the value of z for the start and end of each row
    row_start = torch.cat([(1-factor) * tl + factor * bl for factor in factors], dim = 0) # Assume samples_per_row = 10, 10xlat_dimx1x1
    row_end = torch.cat([(1-factor) * tr +  factor * br for factor in factors], dim = 0) # 10xlat_dimx1x1
    
    # Find the interpolated value within each row
    # First row
    inpt = torch.cat([ (row_start[0] * (1-factor) + row_end[0] * factor).unsqueeze(0) 
                        for factor in factors], dim=0) # 10 x lat_dim x 1 x 1
    
    # Second to the last row
    for row_idx in range(1, samples_per_row):
        single_row = torch.cat([ (row_start[row_idx] * (1-factor) + row_end[row_idx] * factor).unsqueeze(0) 
                                for factor in factors], dim=0) # 10 x lat_dim x 1 x 1
        inpt = torch.cat([inpt, single_row], dim = 0) # (10x10) x lat_dim x 1 x 1
    

    # Send the interpolated z values into the generator
    inpt = inpt.to(device)
    generator.eval()
    with torch.inference_mode():
        
        generated_img = generator(inpt)
        generated_img = ((generated_img + 1) / 2).cpu()
        grid = make_grid(generated_img,
                         nrow = samples_per_row)
        output_img = ToPILImage()(grid)
        output_img.save(save_name)








def prep_imgs(recon_dir, data_type, config, color_flag = None):
    
    """
    Read images present in recon_dir and converted into a torch float tensor, range from -1 to 1, in the shape of (BxCxHxW)
    Include operations of resize, center crop, create MNIST images with color
    
        recon_dir: A folder (path object) consists of the images to be prepared
        color_flag: can be either 'from' or 'to', deciding either 'color_from' or 'color_to' are read from the config
                  : limited to be used in the experiment of 'red_to_green' with MNIST_color
    """
    
    im_size = config['im_size']
    im_channels = config['im_channels']
    
    
    recon_dir = Path(recon_dir)
    if not recon_dir.is_dir():
        recon_dir.mkdir(parents = True,
                        exist_ok = True)
    
    # Read all the files in the recon_dir into a list
    path_names = [entry.name for entry in os.scandir(recon_dir)]
    path_list = [recon_dir / name for name in path_names]
    
    # Perform transformation onto each of the list items
    for idx, path in enumerate(path_list):
        img = Image.open(path) 
        if data_type == 'MNIST':
            simple_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((im_size, im_size))])
        elif data_type == 'Celeb':
            simple_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((im_size, im_size)),
                                                   transforms.CenterCrop((im_size, im_size))])
        img = simple_transform(img)
        
        # If its for the MNIST_color, add color
        if data_type == 'MNIST' and im_channels == 3:
            
            # This section is for the experiment of 'red_to_green'
            if color_flag == 'from':
                color = config['color_from']
            elif color_flag == 'to':
                color = config['color_to']
            
            img_color_ch = img * random.uniform(0.5, 1.0) 
            img_color_none = img* random.uniform(0, 0.2)
            if color == 'red':
                img = torch.cat([img_color_ch, img_color_none, img_color_none], dim=0)
            elif color == 'green':
                img = torch.cat([img_color_none, img_color_ch, img_color_none], dim=0)
            elif color == 'blue':
                img = torch.cat([img_color_none, img_color_none, img_color_ch], dim=0)
        
        
        img = (img*2) - 1 # Change the range into -1 to 1
        
        if idx == 0:
            imgs = img.unsqueeze(0)
        else: 
            imgs = torch.cat([imgs, img.unsqueeze(0)], dim = 0) # Concatenate the image tensor, so they are in the form of BxCxHxW
            
# =============================================================================
#         for idx, img in enumerate(imgs):
#             img_plt = ((img+1) / 2).permute(1,2,0)
#             plt.subplot(3, 4, idx+1)
#             plt.imshow(img_plt)
#             plt.axis(False)
#         plt.show()
# =============================================================================
    return imgs # In the shape of BxCxHxW



def inverse_GAN(generator, target_imgs, config, device, save_pic_name, save_x_name, target_loss = None):
    
    """
    Read images (target_imgs) that are prepped using prep_imgs and find the corresponding z input for them
    Return the z input in the shape of Bxlat_dimx1x1
    
        target_imgs: The images where the corresponding z input is to be found, output of prep_imgs
        save_pic_name: folder / file_name.png, the file_name for the generated images of comparison between target_imgs (top row) and reconstructed images (bottom row)
        save_x_name: folder / file_name.pkl, the file name that will be saving the resulted inverse z input for the given images
        target_loss: The minimal loss must be achieved before the process of finding the z input is interrupted
    """
    
    lat_dim = config['latent_dim']
    loss_fn = torch.nn.SmoothL1Loss() # Use SmoothL1Loss
    
    # Freeze the generator
    for parameter in generator.parameters():
        parameter.requires_grad = False
    
    for idx, target in enumerate(target_imgs):
        
        loss = np.inf # Set a high initial loss
        
        # Deciding the target loss
        if target_loss == None:
            recon_loss_target = config['recon_loss_target']
        else:
            recon_loss_target = target_loss
        
        # Plot the target images in console
        plt.subplot(5,5,1)
        plt.imshow(((target+1)/2).permute(1,2,0), cmap="gray" if config['im_channels'] == 1 else None)
        plt.axis(False)
        plt.show()
        
        
        generator.eval()
        # If the loss does not get below the target within 10000, the training process is restarted
        while loss >= recon_loss_target:
            
            # Reiniatialize x
            x = torch.nn.Parameter(torch.zeros((1, lat_dim, 1, 1)).to(device),
                                   requires_grad = True)
            
            # Reinitialize optimizer
            optimizer = torch.optim.Adam(params = [x],
                                         lr = 0.01)
            
            # The training in finding the x is repeated for 10000 steps
            for i in range(10000):
            
                gen_img = generator(x)
                loss = loss_fn(gen_img, target.unsqueeze(0).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 1000 == 0:
                    print(f"\n[INFO] Current step: {i+1}")
                    print(f"[INFO] Loss        : {loss:.4f}")
                
                # Once the loss gets below the target, break the loop
                if loss < recon_loss_target:
                    break
                
                # Add some noise
                if (i+1) != 10000:
                    x.data = x.data + 0.02* torch.randn(x.shape).to(device)
                    
        print(f"\n[INFO] Current step: {i+1}")
        print(f"[INFO] Finel Loss  : {loss:.4f}")
                
        # Show the reconstructed images
        plt.subplot(5,5,1)
        plt.imshow(((gen_img[0]+1)/2).permute(1,2,0).detach().cpu(), cmap="gray" if config['im_channels'] == 1 else None)
        plt.axis(False)
        plt.show()
          
        # Concatenating the found x values, so they are in the shape of Bxlat_dimx1x1
        if idx == 0:
            x_out = x.detach().cpu()
            gen_out = ((gen_img.detach().cpu()) + 1) / 2
        else:
            x_out = torch.cat([x_out, x.detach().cpu()], dim=0)
            gen_out = torch.cat([gen_out, (gen_img.detach().cpu() + 1) / 2], dim=0)
    
    
    # Unfreeze the generator, in case for other uses
    for parameter in generator.parameters():
        parameter.requires_grad = True
        
    # Save results, top row: Target images, bottom_row: Reconstructed images using the found x values
    combined = torch.cat([(target_imgs+1)/2, gen_out], dim = 0)
    grid = make_grid(combined,
                     nrow = target_imgs.shape[0])
    output_img = ToPILImage()(grid)
    output_img.save(save_pic_name)
    
    with open(save_x_name, 'wb') as f:
        pickle.dump(x_out, f)
    
    return x_out # Bxlat_dimx1x1








def interpolate_from_x_to_y(x, y, z, high, generator, device, save_file):
    """
    Perform the estimated transformation from the characteristics of a set of images, x
    to the characteristics of another set of images, y onto a set of input images, z
    
        x: Bxlat_dimx1x1, on cuda, prepared using inverse_GAN
        y: Bxlat_dimx1x1, on cuda, prepared using inverse_GAN
        z: Bxlat_dimx1x1, to be translated from x to y, shares similar characteristics as x, prepared using inverse_GAN
        high: Maximum magnitude of transformation to be applied
        save_file: folder / file_name.png, shows of the gradual change of the images across the estimation
    """
    
    # Get average from the input x and y
    x = torch.mean(x, dim=0) # lat_dim x 1 x 1
    y = torch.mean(y, dim=0) # lat_dim x 1 x 1
    
    batch_size = z.shape[0]
    factors = torch.linspace(0, high, 10).to(device)
    
    generator.eval()
    with torch.inference_mode():
        for idx, factor in enumerate(tqdm(factors, position=0)):
            z_in = z + factor * (y-x) # Perfom the transformation
            z_out = generator(z_in)
            if idx == 0:
                output = z_out
            else:
                output = torch.cat([output, z_out], dim=0)
    
    output = (output+1) / 2
    output_grid = make_grid(output,
                            nrow = batch_size)
    output_img = ToPILImage()(output_grid)
    output_img.save(save_file)
    print('[INFO] Interpolation from x to y has been completed.')
        

            




###############################################################################   
    
    
        
    
    
    
def infer(mode):
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Options
    configPath1 = './MNIST_config.yaml'
    configPath2 = './MNIST_color_config.yaml'
    configPath3 = './Celeb_config.yaml'
    
    dataType1 = 'MNIST'
    dataType2 = 'Celeb'
    
    
    # Configurable Options
    configPath = configPath3
    dataType = dataType2
    loadName = 'result_Celeb_1'
    
    
    # Read config 
    with open(configPath, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
            
    # Load models
    generator = Generator(config = config).to(device)
    loadPath = Path(loadName)
    generator.load_state_dict(torch.load(loadPath / 'generator.pt',
                                         weights_only = True))
    
    
    # Create reconstruction file
    reconDir = loadPath / 'Reconstruction'
    if not reconDir.is_dir():
        reconDir.mkdir(parents = True,
                       exist_ok = True)
    
    reconSaveDir = loadPath / 'Result_Reconstruction'
    if not reconSaveDir.is_dir():
        reconSaveDir.mkdir(parents = True,
                           exist_ok = True)
        
        
    
    # Sample
    # randomly create 100 samples from the latent space using the trained generator
    # The generated samples are saved at {save_name}
    if mode == 'sample':
        sample(generator = generator,
               config = config,
               save_name = loadPath / "Generated Image.png",
               device = device)
    
    # Visualize Latent Space
    # Randomly create 4 samples from the latent space using the trained generator and plot the interpolation between these 4 images
    # The generated interpolation plot are saved in {save_name}
    elif mode == 'visualize_latent_space':
        visualize_latent_space(generator = generator, 
                               config = config, 
                               save_name = loadPath / "latent_space.png", 
                               samples_per_row = 20,
                               device = device)
        
    # Inverse GAN
    # Provide a set of images that are stored in {recon_dir}
    # The set of images are prepared into the form of BxCxHxW, range from -1 to 1, a torch tensor on cuda
    # The corresponding z input in the latent space for these images are found and saved at {save_x_name}
    # The comparison between the input and reconstruction images are saved in {save_pic_name} (top row: Target images, bottom row: Reconstructed images)
    elif mode == 'inverse_GAN':
        
        targetImgs = prep_imgs(recon_dir = reconDir,
                               data_type = dataType,
                               config = config)
        
        targetInput = inverse_GAN(generator = generator,
                                  target_imgs = targetImgs,
                                  config = config,
                                  device = device,
                                  save_pic_name = reconSaveDir / "reconstruction.png",
                                  save_x_name = reconSaveDir / "x.pkl")
        
        sample(generator = generator,
               config = config,
               save_name = reconSaveDir / "reconstruction_with_input_noise.png",
               device = device,
               noise = targetInput.to(device))
    
    
    # Thin to Thick
    # Transform b&w MNIST images that have thin strokes into MNIST images that have thick strokes
    # Note: This mode is only compatible with models that are trained using MNIST
    elif mode == 'thin_to_thick':
        
        thin_to_thick_folder = loadPath / 'thin_to_thick'
        thin_to_thick_recon_folder = thin_to_thick_folder / 'Reconstruction'
        
        thin_folder = thin_to_thick_recon_folder / 'Thin'
        thick_folder = thin_to_thick_recon_folder / 'Thick'
        inpt_folder = thin_to_thick_recon_folder / 'Inpt'
        
        thinFile = thin_to_thick_recon_folder / 'thin.pkl'
        thickFile = thin_to_thick_recon_folder / 'thick.pkl'
        inptFile = thin_to_thick_recon_folder / 'inpt.pkl'
        
        # Prepare the input to reconstruct the images with thin strokes
        if not os.path.exists(thinFile):
            thinTargets = prep_imgs(recon_dir = thin_folder,
                                    data_type = dataType,
                                    config = config)
            
            thinInpt = inverse_GAN(generator = generator,
                                   target_imgs = thinTargets,
                                   config = config,
                                   device = device,
                                   save_pic_name = thin_to_thick_recon_folder / 'thin_reconstruction.png',
                                   save_x_name = thinFile)
        else:
            with open(thinFile, 'rb') as f:
                thinInpt = pickle.load(f)
                print("[INFO] The input to reconstruct the thin images have been loaded successfully.")
        
        
        # Prepare the input to reconstruct the images with thick strokes
        if not os.path.exists(thickFile):
            thickTargets = prep_imgs(recon_dir = thick_folder,
                                     data_type = dataType,
                                     config = config)
            
            thickInpt = inverse_GAN(generator = generator,
                                    target_imgs = thickTargets,
                                    config = config,
                                    device = device,
                                    save_pic_name = thin_to_thick_recon_folder / 'thick_reconstruction.png',
                                    save_x_name = thickFile)
        else:
            with open(thickFile, 'rb') as f:
                thickInpt = pickle.load(f)
                print("[INFO] The input to reconstruct the thick images have been loaded successfully.")
        
            
        # Prepare the input to reconstruct the images to be used as inputs
        if not os.path.exists(inptFile):
            inptTargets = prep_imgs(recon_dir = inpt_folder,
                                    data_type = dataType,
                                    config = config)
            
            inpt = inverse_GAN(generator = generator,
                               target_imgs = inptTargets,
                               config = config,
                               device = device,
                               save_pic_name = thin_to_thick_recon_folder / 'inpt_reconstruction.png',
                               save_x_name = inptFile)
        else:
            with open(inptFile, 'rb') as f:
                inpt = pickle.load(f)
                print("[INFO] The input to reconstruct the input images have been loaded successfully.")
        
        
        
        # The transformation
        print(f"[INFO] Proceed to interpolate the development from thin (x) to thick (y)...")
        interpolate_from_x_to_y(x = thinInpt.to(device),
                                y = thickInpt.to(device),
                                z = inpt.to(device),
                                high = 3,
                                generator = generator,
                                device = device,
                                save_file = thin_to_thick_recon_folder / 'thin_to_thick.png')
    
    
    # Red to Green
    # Transform Colorful MNIST images that are red into Colorful MNIST images that are green
    # Note: This mode is only compatible with models that are trained using MNIST_Color
    elif mode == 'red_to_green':
        red_to_green_folder = loadPath / 'red_to_green'
        red_to_green_recon_folder = red_to_green_folder / 'Reconstruction'
        
        red_folder = red_to_green_recon_folder / 'red'
        green_folder = red_to_green_recon_folder / 'green'
        inpt_folder = red_to_green_recon_folder / 'inpt'
        
        redFile = red_to_green_recon_folder / 'red.pkl'
        greenFile = red_to_green_recon_folder / 'green.pkl'
        inptFile = red_to_green_recon_folder / 'inpt.pkl'
        
        
        # Prepare the input to reconstruct the images that are red
        if not os.path.exists(redFile):
            redTargets = prep_imgs(recon_dir = red_folder,
                                   data_type = dataType,
                                   config = config,
                                   color_flag = 'from')
            
            redInpt = inverse_GAN(generator = generator,
                                  target_imgs = redTargets,
                                  config = config,
                                  device = device,
                                  save_pic_name = red_to_green_recon_folder / 'red_reconstruction.png',
                                  save_x_name = redFile)
        else:
            with open(redFile, 'rb') as f:
                redInpt = pickle.load(f)
                print("[INFO] The input to reconstruct the red images have been loaded successfully.")
        
        
        
        # Prepare the input to reconstruct the images that are green
        if not os.path.exists(greenFile):
            greenTargets = prep_imgs(recon_dir = green_folder,
                                     data_type = dataType,
                                     config = config,
                                     color_flag = 'to')
            
            greenInpt = inverse_GAN(generator = generator,
                                    target_imgs = greenTargets,
                                    config = config,
                                    device = device,
                                    save_pic_name = red_to_green_recon_folder / 'green_reconstruction.png',
                                    save_x_name = greenFile)
        else:
            with open(greenFile, 'rb') as f:
                greenInpt = pickle.load(f)
                print("[INFO] The input to reconstruct the green images have been loaded successfully.")
        
        
        
        # Prepare the input to reconstruct the images to be used as an input
        if not os.path.exists(inptFile):
            inptTargets = prep_imgs(recon_dir = inpt_folder,
                                    data_type = dataType,
                                    config = config,
                                    color_flag = 'from')
            
            inpt = inverse_GAN(generator = generator,
                               target_imgs = inptTargets,
                               config = config,
                               device = device,
                               save_pic_name = red_to_green_recon_folder / 'inpt_reconstruction.png',
                               save_x_name = inptFile)
        else:
            with open(inptFile, 'rb') as f:
                inpt = pickle.load(f)
                print("[INFO] The input to reconstruct the input images have been loaded successfully.")
        
        
        
        # The transformation
        print(f"[INFO] Proceed to interpolate the development from red (x) to green (y)...")
        interpolate_from_x_to_y(x = redInpt.to(device),
                                y = greenInpt.to(device),
                                z = inpt.to(device),
                                high = 1,
                                generator = generator,
                                device = device,
                                save_file = red_to_green_recon_folder / 'red_to_green.png')
    
    
    
    
    
    
    # Visualize Celeve Latent Space
    # Perform interpolation between 4 fixed inputs of celebrity images
    # Note: This mode is only compatible with models that are trained using Celeb
    elif mode == 'visualize_celeb_latent_space':
        visualize_latent_space_with_fixed_input = loadPath / 'visualize_latent_space_with_fixed_input'
        reconDir = visualize_latent_space_with_fixed_input / 'reconDir'
        
        zFile = visualize_latent_space_with_fixed_input / 'recon.pkl'
        
        if not os.path.exists(zFile):
            targImgs = prep_imgs(recon_dir = reconDir,
                                 data_type = dataType,
                                 config = config)
            
            targZ = inverse_GAN(generator = generator,
                                target_imgs = targImgs,
                                config = config,
                                device = device,
                                save_pic_name = visualize_latent_space_with_fixed_input / 'reconstruction.png',
                                save_x_name = zFile)
        
        else:
            with open(zFile, 'rb') as f:
                targZ = pickle.load(f)
                print("[INFO] The input to reconstruct the images have been loaded successfully.")
            
        visualize_latent_space(generator = generator,
                               config = config,
                               save_name = visualize_latent_space_with_fixed_input / 'latent_space.png',
                               samples_per_row = 30,
                               device = device,
                               fixed_inpt = targZ.to(device))
        
    
    # Smiling woman to smiling man
    # Perform the following operation of: smiling_woman - neutral_woman + neutral man
    # Resulting in an image of a smiling man
    elif mode =='smiling_woman_to_smiling_man':
        
        
        neutral_woman_to_smile_man = loadPath / 'neutral_woman_to_smile_man3'
        
        smiling_woman = neutral_woman_to_smile_man / 'smiling_woman'
        neutral_woman = neutral_woman_to_smile_man / 'neutral_woman'
        neutral_man = neutral_woman_to_smile_man / 'neutral_man'   
        
        smiling_woman_file = neutral_woman_to_smile_man / 'smiling_woman.pkl'
        neutral_woman_file = neutral_woman_to_smile_man / 'neutral_woman.pkl'
        neutral_man_file = neutral_woman_to_smile_man / 'neutral_man.pkl'
        
        # Prepare the input to reconstruct the images of a smiling woman
        if not os.path.exists(smiling_woman_file):
            smiling_woman_imgs = prep_imgs(recon_dir = smiling_woman,
                                           data_type = dataType,
                                           config = config)
            
            smiling_woman_z = inverse_GAN(generator = generator,
                                          target_imgs = smiling_woman_imgs,
                                          config = config,
                                          device = device,
                                          save_pic_name = neutral_woman_to_smile_man / 'smiling_woman_recon.png',
                                          save_x_name = smiling_woman_file,
                                          target_loss = 0.012)
        else:
            with open(smiling_woman_file, 'rb') as f:
                smiling_woman_z = pickle.load(f)
                print(f"[INFO] z for smiling woman has been successfully loaded.")
                
                
        
        # Prepare the input to reconstruct the images tof a neutral woman
        if not os.path.exists(neutral_woman_file):
            neutral_woman_imgs = prep_imgs(recon_dir = neutral_woman,
                                           data_type = dataType,
                                           config = config)
            
            neutral_woman_z = inverse_GAN(generator = generator,
                                          target_imgs = neutral_woman_imgs,
                                          config = config,
                                          device = device,
                                          save_pic_name = neutral_woman_to_smile_man / 'neutral_woman_recon.png',
                                          save_x_name = neutral_woman_file,
                                          target_loss = 0.012)
        else:
            with open(neutral_woman_file, 'rb') as f:
                neutral_woman_z = pickle.load(f)
                print(f"[INFO] z for neutral woman has been successfully loaded.")
            
            
        # Prepare the input to reconstruct the images of a neutral man
        if not os.path.exists(neutral_man_file):
            neutral_man_imgs = prep_imgs(recon_dir = neutral_man,
                                           data_type = dataType,
                                           config = config)
            
            neutral_man_z = inverse_GAN(generator = generator,
                                          target_imgs = neutral_man_imgs,
                                          config = config,
                                          device = device,
                                          save_pic_name = neutral_woman_to_smile_man / 'neutral_man_recon.png',
                                          save_x_name = neutral_man_file,
                                          target_loss = 0.012)
        else:
            with open(neutral_man_file, 'rb') as f:
                neutral_man_z = pickle.load(f)
                print(f"[INFO] z for neutral man has been successfully loaded.")
        
        
        # If more than 1 images, find the mean of them
        if smiling_woman_z.shape[0] > 1:
            smiling_woman_z = torch.mean(smiling_woman_z, dim=0).unsqueeze(0)
        if neutral_woman_z.shape[0] > 1:
            neutral_woman_z = torch.mean(neutral_woman_z, dim=0).unsqueeze(0)
        if neutral_man_z.shape[0] > 1:
            neutral_man_z = torch.mean(neutral_man_z, dim = 0).unsqueeze(0)
        
        
        # The transformation
        smiling_man_z = (smiling_woman_z - neutral_woman_z + neutral_man_z)
        inpt_noise = torch.cat([smiling_woman_z, neutral_woman_z, neutral_man_z, smiling_man_z], dim=0) # Concatenate all of them for plotting
        sample(generator = generator,
               config = config,
               save_name = neutral_woman_to_smile_man / "smiling_man.png",
               device = device,
               noise = inpt_noise.to(device))
        
            
        

                
    

# What experiments to be executed?
if __name__ == "__main__":
    infer(mode = 'smiling_woman_to_smiling_man')
    
    