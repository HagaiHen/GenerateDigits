# Generate Handwritten Digits Using GAN

#### The code implements a Generative Adversarial Network (GAN) using PyTorch to generate handwritten digit images from the MNIST dataset. 
### Here's an overview of the steps:

- The MNIST training dataset is loaded using torchvision's datasets.MNIST with the specified root, train flag, download flag, and the defined transformation.

- The generator and discriminator models are initialized by creating instances of the Generator and Discriminator classes.

- The binary cross-entropy loss function (nn.BCELoss()) is defined to measure the difference between the predicted and target labels.

- Adam optimizers are created for both the discriminator and generator models using torch.optim.Adam. The respective model parameters and learning rate are provided.

### The Training Sequence

During each epoch, the trainer function iterates over batches of real samples and labels from the training data. It then generates a set of random latent space samples for the generator. These latent space samples are passed through the generator, which produces fake samples. Both the real and fake samples are assigned corresponding labels.

The real and fake samples, along with their labels, are concatenated to create a combined set of samples. This combined set is then passed through the discriminator, which produces output predictions. The loss function is used to compare these predictions with the corresponding labels, calculating the discriminator loss.

The gradients are backpropagated through the discriminator, and its parameters are updated using the optimizer_discriminator. This step focuses on training the discriminator to distinguish between real and fake samples effectively.

After training the discriminator, the trainer function moves on to training the generator. It generates a new set of random latent space samples and passes them through the generator. The output generated samples are then passed through the discriminator again. The loss function is used to calculate the generator loss by comparing the discriminator's output with the labels indicating the samples as real.

The gradients are backpropagated through the generator, and its parameters are updated using the optimizer_generator. This step aims to train the generator to produce samples that can deceive the discriminator into classifying them as real.

In summary, the code performs the training of a GAN using the MNIST dataset. It trains the generator and discriminator models and generates samples using the trained generator.

### Examples

This is how the real digits looks like:

![real](https://github.com/HagaiHen/GenerateDigits/assets/76903853/ef25b53b-9148-4c0f-95e2-b27d1ce21e18)

This is how the generated digits looks like:

![150_1](https://github.com/HagaiHen/GenerateDigits/assets/76903853/65e49637-c131-470a-a7fd-d401afe863a7)

### Getting Started
- Clone the repository to your local machine.
- Ensure that you have the necessary dependencies installed. You can find the required libraries in the requirements.txt file.
```
pip install -r requirements.txt
```

- Run main.py for start the training

By engaging in this project, you will gain hands-on experience with deep learning techniques, specifically in the field of GANs, and develop skills in generating realistic digit images.
