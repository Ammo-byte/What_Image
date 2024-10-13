import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import inception_v3, Inception_V3_Weights

# Implementing the CNN encoder and RNN decoder
class CNN_encoder(nn.Module):
    def __init__(self, size, CNN_train=False):
        super(CNN_encoder, self).__init__()
        self.train_CNN = CNN_train
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        output = self.inception(images)
        if isinstance(output, tuple):  # Check if output is a tuple
            features, aux_output = output  # Unpack if it's a tuple
        else:
            features = output  # Otherwise just use the output directly
            aux_output = None  # Set aux_output to None if not available
        return self.dropout(self.relu(features))

class RNN_decoder(nn.Module):
    def __init__(self, size, h_size, vocab_s, layers):
        super(RNN_decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_s, size)
        self.lstm = nn.LSTM(size, h_size, layers)
        self.linear = nn.Linear(h_size, vocab_s)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, caption):
        embedding = self.dropout(self.embedding(caption))
        embedding = torch.cat((features.unsqueeze(0), embedding), dim=0)  # Concatenate features with embeddings
        lstm_out, _ = self.lstm(embedding)
        output = self.linear(lstm_out)
        return output


class CNN_RNN(nn.Module):
    def __init__(self, size, h_size, vocab_s, layers):
        super(CNN_RNN, self).__init__()
        self.encoder = CNN_encoder(size)
        self.decoder = RNN_decoder(size, h_size, vocab_s, layers)

    def forward(self, imgs, caption):
        features = self.encoder(imgs)
        output = self.decoder(features, caption)
        return output

    def image_caption(self, imgs, vocabulary, max_len=50):
        caption = []
        x = self.encoder(imgs).unsqueeze(0)  # Extract features from the image
        batch_size = imgs.size(0)  # Get the batch size from the images
        states = (torch.zeros(self.decoder.lstm.num_layers, batch_size, self.decoder.lstm.hidden_size).to(imgs.device),
                  torch.zeros(self.decoder.lstm.num_layers, batch_size, self.decoder.lstm.hidden_size).to(imgs.device))

        with torch.no_grad():
            for _ in range(max_len):
                hidden, states = self.decoder.lstm(x, states)  # Forward LSTM with initialized state
                output = self.decoder.linear(hidden).squeeze(0)  # Linear transformation to vocab size

                print(output.shape)  # Check the output shape

                if len(output.shape) == 1:  # Ensure output is at least 2D
                    output = output.unsqueeze(0)

                predicted = output.argmax(1)  # Get the predicted word index

                caption.append(predicted.item())  # Append word index to caption
                x = self.decoder.embedding(predicted).unsqueeze(0)  # Embed the predicted word

                if vocabulary.its[predicted.item()] == '<EOS>':  # End the caption generation
                    break

        return [vocabulary.its[indx] for indx in caption]  # Return caption with words
    
    