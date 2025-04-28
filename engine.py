import torch
import numpy as np
from tqdm import tqdm
from infer import sample
from pathlib import Path

def train_step(generator, discriminator, dataloader, device, g_optimizer, d_optimizer, loss_fn, config, save_file, epoch):
    
    
    step = 0
    save_file = Path(save_file)
    if not save_file.is_dir():
        save_file.mkdir(parents = True,
                        exist_ok = True)
    
    save_file_epoch = save_file / f"{epoch}"
    if not save_file_epoch.is_dir():
        save_file_epoch.mkdir(parents = True,
                              exist_ok = True)
    
    lat_dim = config['latent_dim']
    
    d_loss_list = []
    d_fake_im_pred_list = []
    d_real_im_pred_list = []
    d_accuracy_list = []
    
    g_loss_list = []
    g_fake_im_pred_list = []
    g_accuracy_list = []
    
    
    for batch, real_im in enumerate(tqdm(dataloader, position = 0)):
        
        discriminator.train()
        generator.train()
        
        batch_size = real_im.shape[0]
        real_im_gt = torch.ones((batch_size)).to(device) * config['mod_label']
        fake_im_gt = torch.zeros((batch_size)).to(device)
        g_real_im_gt = torch.ones((batch_size)).to(device)
        
        # Train discriminator
        ## Freeze generator
        for parameter in discriminator.parameters():
            parameter.requires_grad = True
        for parameter in generator.parameters():
            parameter.requires_grad = False
            
        
        ## Passing in real images
        real_im = real_im.to(device)
        real_im_pred = discriminator(x = real_im)
        d_real_loss = loss_fn(real_im_pred, real_im_gt)
        
        ## Passing in noise -> generate images -> pass in the generated images
        fake_im_noise = torch.randn((batch_size, lat_dim, 1, 1)).to(device)
        fake_im = generator(z = fake_im_noise)
        fake_im_pred = discriminator(x = fake_im.detach())
        d_fake_loss = loss_fn(fake_im_pred, fake_im_gt)
        
        ## Compute total loss and backward propagation
        d_loss = 0.5 * (d_real_loss + d_fake_loss)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        ## Keep track of losses
        d_loss_list.append(d_loss.detach().cpu().item())
        d_real_im_pred_list.append(torch.mean(torch.sigmoid(real_im_pred)).detach().cpu().item())
        d_fake_im_pred_list.append(torch.mean(torch.sigmoid(fake_im_pred)).detach().cpu().item())
        
        ## Keep track of accuracy
        tp = 0
        
        for pred in real_im_pred:
            if torch.sigmoid(pred) > 0.5:
                tp += 1
        
        for pred in fake_im_pred:
            if torch.sigmoid(pred) <= 0.5:
                tp += 1
        
        d_accuracy = tp / (2*batch_size)
        d_accuracy_list.append(d_accuracy)
        
        
        
        # Train generator
        ## Freeze generator
        for parameter in discriminator.parameters():
            parameter.requires_grad = False
        for parameter in generator.parameters():
            parameter.requires_grad = True
        
        ## Passing in noise -> generate images -> pass in the generated images
        g_fake_im_noise = torch.randn((batch_size, lat_dim, 1, 1)).to(device)
        g_fake_im = generator(z = g_fake_im_noise)
        g_fake_im_pred = discriminator(x = g_fake_im)
        
        ## Compute loss and backward propagation
        g_loss = loss_fn(g_fake_im_pred, g_real_im_gt)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        ## Keep track of losses
        g_loss_list.append(g_loss.detach().cpu().item())
        g_fake_im_pred_list.append(torch.mean(torch.sigmoid(g_fake_im_pred)).detach().cpu().item())
        
        
        ## Keep track of accuracy
        g_tp = 0
        
        for pred in g_fake_im_pred:
            if torch.sigmoid(pred) > 0.5:
                g_tp += 1
        
        g_accuracy = g_tp / (batch_size)
        g_accuracy_list.append(g_accuracy)
        
        
        step += 1
        if step % 50 == 0:
            sample(generator = generator,
                   config = config,
                   save_name = save_file_epoch / f"{epoch}_{step}.png",
                   device = device)
            
        
    
    return {'d_loss' : np.mean(d_loss_list),
            'd_fake_im_pred': np.mean(d_fake_im_pred_list),
            'd_real_im_pred': np.mean(d_real_im_pred_list),
            'd_accuracy': np.mean(d_accuracy_list),
            'g_loss' : np.mean(g_loss_list),
            'g_fake_im_pred': np.mean(g_fake_im_pred_list),
            'g_accuracy': np.mean(g_accuracy_list)}
    
    
