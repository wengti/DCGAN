import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.lat_dim = config['latent_dim']
        self.im_channels = config['im_channels']
        self.num_blocks = len(config['generator_kernel_size'])
        
        self.channels = config['generator_channels']
        self.channels = [self.lat_dim] + self.channels + [self.im_channels]
        
        self.generator_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels = self.channels[i],
                                   out_channels = self.channels[i+1],
                                   kernel_size = config['generator_kernel_size'][i],
                                   stride = config['generator_stride'][i],
                                   padding = config['generator_padding'][i],
                                   output_padding = config['generator_output_padding'][i],
                                   bias = False),
                nn.BatchNorm2d(num_features = self.channels[i+1]) if i != (self.num_blocks - 1) else nn.Identity(),
                nn.ReLU() if i != (self.num_blocks - 1) else nn.Tanh()
                ) for i in range(self.num_blocks)])
    
    def forward(self, z):
        """
            z: A float torch tensor, on cuda, (B, lat_dim, 1, 1)
        """
        out = z 
        for block in self.generator_blocks:
            out = block(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.lat_dim = config['latent_dim']
        self.im_channels = config['im_channels']
        self.num_blocks = len(config['discriminator_kernel_size'])
        
        self.channels = config['discriminator_channels']
        self.channels = [self.im_channels] + self.channels + [1]
        
        self.discriminator_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels = self.channels[i],
                          out_channels = self.channels[i+1],
                          kernel_size = config['discriminator_kernel_size'][i],
                          stride = config['discriminator_stride'][i],
                          padding = config['discriminator_padding'][i],
                          bias = False if i != 0 else True),
                nn.BatchNorm2d(num_features = self.channels[i+1]) if i != (self.num_blocks - 1) else nn.Identity(),
                nn.LeakyReLU() if i != (self.num_blocks - 1) else nn.Identity()
                ) for i in range(self.num_blocks)])
    
    def forward(self, x):
        """
            x : a float tensor, on cuda, range from -1 to 1, (B, im_channels, im_size, im_size)
        """
        B  = x.shape[0]
        out = x
        for block in self.discriminator_blocks:
            out = block(out)
        out = out.reshape(B)
        return out
