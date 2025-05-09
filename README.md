# Deep Convolutional Generative Adversarial Network (DCGAN)

## Credits & Acknowledgements
This project is a reimplementaion of [DCGAN-PyTorch] by [explainingai-code] (https://github.com/explainingai-code/DCGAN-Pytorch)
The code has been rewritten from scratch while maintaining the core concepts and functionalities of the original implementation.

## Features
- Build a DCGAN that can be mofified using a config file with a compatible format.
- Various experimentation in the latent space are performed in this code

## Description of Files:
- **extract_mnist.py** - Extracts MNIST data from CSV files.
- **custom_data.py** - Create a custom dataset for both MNIST (b&w and color) and Celeb-HQ dataset.
- **model.py** - Compatible with .yaml config files to create a modifiable discriminator and generator.
- **engine.py** - Defines the train step (for 1 epoch).
- **main.py** - Trains a DCGAN model.
- **infer.py**
  - A) **Sample**
      - Randomly create 100 samples from the latent space using the trained generator
        
      ![Generated MNIST Samples](./result_display/MNIST/Generated%20Image.png)
      ![Generated MNIST_Color_Samples](./result_display/MNIST_color/Generated%20Image.png)
      ![Generated Celeb_Samples](./result_display/Celeb/Generated%20Image.png)
    
  - B) **Visualize Latent Space**
      - Randomly create 4 samples from the latent space using the trained generator and plot the interpolation between these 4 images

      ![Generated MNIST Samples](./result_display/MNIST/latent_space.png)
      ![Generated MNIST_Color_Samples](./result_display/MNIST_color/latent_space.png)
      ![Generated Celeb_Samples](./result_display/Celeb/latent_space.png)

  - C) **Inverse GAN**
      - A set of images are prepared into the form of BxCxHxW, range from -1 to 1, a torch tensor on cuda
      - The corresponding z input in the latent space for these images are then found and saved.
      - The comparison between the input and reconstruction images are saved (top row: Target images, bottom row: Reconstructed images).

      ![Generated MNIST Samples](./result_display/MNIST/reconstruction.png)
      ![Generated MNIST_Color_Samples](./result_display/MNIST_color/reconstruction.png)
      ![Generated Celeb_Samples](./result_display/Celeb/Result_Reconstruction/reconstruction.png)

  - D) **Thin to Thick**
      - Transform b&w MNIST images that have thin strokes into MNIST images that have thick strokes
      - Note: This mode is only compatible with models that are trained using MNIST

      ![Generated MNIST Samples](./result_display/MNIST/thin_to_thick.png)

  - E) **Red to Green**
      - Transform Colorful MNIST images that are red into Colorful MNIST images that are green
      - Note: This mode is only compatible with models that are trained using MNIST_Color

      ![Generated MNIST_Color_Samples](./result_display/MNIST_color/red_to_green.png)

  - F) **Visualize Celeb Latent Space**
      - Perform interpolation between 4 fixed inputs of celebrity images
      - Note: This mode is only compatible with models that are trained using Celeb

      ![Generated Celeb_Samples](./result_display/Celeb/visualize_latent_space_with_fixed_input/latent_space.png)

  - G) **Smiling Woman to Smiling Man**
      - Perform the following operation of: smiling_woman - neutral_woman + neutral man
      - Resulting in an image of a smiling man

      ![Generated Celeb_Samples](./result_display/Celeb/neutral_woman_to_smile_man2/smiling_man.png)
  

  
