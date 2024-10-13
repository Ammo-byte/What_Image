import pickle
import torch
from image_intial_model import CNN_RNN  # Ensure this imports your model definition
from utils import print_examples  # Assuming you have a method to print examples
from torchvision import transforms
from PIL import Image

with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)
# Load the checkpoint
checkpoint_path = '/Users/aamoditacharya/Desktop/Projects/Image Captioning Tool/What_Image/checkpoint_epoch_20.pth.tar'
checkpoint = torch.load(checkpoint_path, weights_only= False)

# Hyperparameters (must match those used during training)
embeded_size = 256
hide_size = 256
vocabulary_size = len(vocab)  # You need to have access to your vocabulary
layers = 1  # Same number of layers used during training

# Recreate the model
model = CNN_RNN(embeded_size, hide_size, vocabulary_size, layers)

# Load the state dictionary
model.load_state_dict(checkpoint['state_dict'])

# Set the model to evaluation mode
model.eval()

# Prepare the image transformations
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize to the same size used in training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# Load and preprocess your untested images
image_path = '/Users/aamoditacharya/Desktop/Projects/Image Captioning Tool/What_Image/test_examples/boat.png'  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Generate a caption for the image
with torch.no_grad():  # Disable gradient calculations
    caption = model.image_caption(image, vocab)

# Print the generated caption
print("Generated Caption:", " ".join(caption))