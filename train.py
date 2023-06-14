import torch
from config import num_epochs, batch_size

def trainer(train_loader, generator, discriminator, loss_function, optimizer_discriminator, optimizer_generator, device):
    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader):
            
            # Move real samples and labels to the specified device
            real_samples = real_samples.to(device=device)
            real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
            
            # Generate random latent space samples
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
            
            # Generate fake samples using the generator
            generated_samples = generator(latent_space_samples)
            
            # Create labels for the generated samples
            generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
            
            # Concatenate real and generated samples and their respective labels
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # ----------- Training the discriminator ----------- 
            
            # Reset gradients of the discriminator
            discriminator.zero_grad()
            
            # Forward pass through the discriminator
            output_discriminator = discriminator(all_samples)
            
            # Calculate the discriminator loss
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            
            # Backpropagate the gradients and update the discriminator's parameters
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # ----------- Training the generator ----------- 
            
            # Generate new random latent space samples
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

            # Reset gradients of the generator
            generator.zero_grad()
            
            # Generate fake samples using the updated generator
            generated_samples = generator(latent_space_samples)
            
            # Forward pass through the discriminator using the generated samples
            output_discriminator_generated = discriminator(generated_samples)
            
            # Calculate the generator loss
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            
            # Backpropagate the gradients and update the generator's parameters
            loss_generator.backward()
            optimizer_generator.step()

            # Print the losses for the current batch
            
            if n == batch_size - 1:
                print(f"---------------- Epoch: {epoch + 1} ----------------")
                print(f"Loss Discriminator: {loss_discriminator}, Loss Generator: {loss_generator}")