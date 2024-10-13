import torch
from tqdm import tqdm  # For progress bar
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from image_loader import image_loader
from image_intial_model import CNN_RNN
from utils import save_checkpoint, load_checkpoint, print_examples
import time
import os
import pickle  # Import pickle for saving models

def trainer():
    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # Load dataset
    try:
        trainloader, dataset = image_loader(
            main_direc='Images', 
            captions='flickrcaptions.txt', 
            transform=transform, 
            batch_s=32,
            workers=8,
            shuffle=True,
            pin_memory=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Save vocabulary to a .pkl file
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(dataset.vocab, f)
    print("Vocabulary saved to vocabulary.pkl")

    # Set device and define flags for model load/save
    torch.backends.cudnn.benchmark = True
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    load_model = False  # Set this to True if you want to load a saved checkpoint
    save_model = True
    train_CNN = False  # Fine-tune CNN or not

    # Hyperparameters
    embeded_size = 256
    hide_size = 256
    vocabulary_size = len(dataset.vocab)
    layers = 1
    learn_rate = 4e-4
    epochs = 100
    checkpoint_interval = 5

    # For tensorboard logging
    writer = SummaryWriter('runs/flickr')
    step = 0
    stop_training = False  # Flag to stop training

    # Initialize model, criterion, optimizer
    model = CNN_RNN(embeded_size, hide_size, vocabulary_size, layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.sti['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Only fine-tune CNN if train_CNN is True
    for name, param in model.encoder.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    # Load from checkpoint if required
    if load_model and os.path.isfile('my_checkpoint.pth.tar'):
        print("Loading checkpoint...")
        step = load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

    # Set model to training mode
    model.train()

    start_time = time.time()  # Track start time for training duration

    for epoch in range(epochs):
        total_loss = 0

        # Progress bar for training loop
        for idx, (images, captions) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images, captions = images.to(device), captions.to(device)

            # Forward pass
            outputs = model(images, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tensorboard logging
            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1

            # Check if 3 hours have passed
            if time.time() - start_time > 48 * 60 * 60:  # Stop after 3 hours
                print(f"Training stopped after {epoch + 1} epochs.")
                stop_training = True  # Set flag to stop training
                break

        if stop_training:
            break  # Break out of the epoch loop if training should stop

        # Average loss per epoch
        avg_loss = total_loss / len(trainloader)

        # Write average loss to TensorBoard
        writer.add_scalar('Training Loss per Epoch', avg_loss, epoch)
        writer.flush()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint after every checkpoint_interval epochs
        if save_model and (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }
            checkpoint_path = os.path.join('/Users/aamoditacharya/Desktop/Projects/Image Captioning Tool/What_Image', f'checkpoint_epoch_{epoch + 1}.pth.tar')
            save_checkpoint(checkpoint, filename=checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # After training (or if stopped), switch to evaluation mode and print examples
    print("Training complete. Starting evaluation on test examples...")
    model.eval()  # Set the model to evaluation mode
    print_examples(model, device, dataset)  # Test the model on the test set

    # Save the encoder and decoder models
    def save_model(model, filename):
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    # Save the models after training is complete
    save_model(model.encoder, 'encoder.pkl')
    save_model(model.decoder, 'decoder.pkl')

if __name__ == '__main__':
    trainer()