import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from image_loader import image_loader
from image_intial_model import CNN_RNN

def trainer():
    tranform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainloader, dataset= image_loader(main_direc= 'Images', 
                                       captions = 'flickrcaptions.txt', 
                                       transform = tranform, workers = 2) 
    

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = True

    #Hyperparameters

    embeded_size = 256
    hide_size = 256
    vocabulary_size = len(dataset.vocab)    
    layers = 1
    learn_rate = 3e-4
    epochs = 100

    # For tensorboard

    writer = SummaryWriter('runs/flickr')
    step = 0

    # Initialize model
    model = CNN_RNN(embeded_size, hide_size, vocabulary_size, layers).to(device)