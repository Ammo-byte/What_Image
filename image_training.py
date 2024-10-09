import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from image_loader import image_loader
from image_intial_model import CNN_RNN
from utils import save_checkpoint, load_checkpoint, print_examples
import time  # For tracking elapsed time
import os

def trainer():
    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
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
            workers=2
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False  # Set this to True to load a saved checkpoint
    save_model = True

    # Hyperparameters
    embeded_size = 256
    hide_size = 256
    vocabulary_size = len(dataset.vocab)
    layers = 1
    learn_rate = 3e-4
    epochs = 200  # Train for a large number of epochs
    checkpoint_interval = 5  # Save checkpoint every 5 epochs

    # For tensorboard logging
    writer = SummaryWriter('runs/flickr')
    step = 0

    # Initialize model, criterion, optimizer
    model = CNN_RNN(embeded_size, hide_size, vocabulary_size, layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.sti['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Load from checkpoint if required
    if load_model and os.path.isfile('my_checkpoint.pth.tar'):
        print("Loading checkpoint...")
        step = load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

    # Set model to training mode
    model.train()

    start_time = time.time()  # Track start time for training duration

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0

        for indx, (images, captions) in enumerate(trainloader):
            images, captions = images.to(device), captions.to(device)

            # Forward pass
            outputs = model(images, captions[:-1])  # Exclude <EOS> from target
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(trainloader)
        writer.add_scalar('Training Loss per Epoch', avg_loss, epoch)
        writer.flush()  # Make sure logs are written to disk

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Time per epoch: {elapsed_time:.2f}s")

        # Save checkpoint after every checkpoint_interval epochs
        if save_model and (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }
            checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pth.tar'
            save_checkpoint(checkpoint, filename=checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Stop training after a set amount of time (e.g., 3 hours)
        if time.time() - start_time > 3 * 60 * 60:  # Stop after 3 hours
            print(f"Training stopped after {epoch + 1} epochs.")
            break

    # After training, switch to evaluation mode and print examples
    print("Training complete. Starting evaluation on test examples...")
    model.eval()  # Set the model to evaluation mode
    print_examples(model, device, dataset)  # Test the model on the test set

if __name__ == '__main__':
    trainer()
