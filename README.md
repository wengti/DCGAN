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
