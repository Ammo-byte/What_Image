import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import inception_v3, Inception_V3_Weights

# Implementing the CNN encoder and RNN decoder
class CNN_encoder(nn.Module):
    def __init__(self, size, CNN_train=False):
        super(CNN_encoder, self).__init__()
        self.size = size
        self.CNN_train = CNN_train
        # Set aux_logits=True
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, size)  # Change output size
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # Set requires_grad for training or freezing the CNN layers
        for name, param in self.inception.named_parameters():
            if 'fc' not in name:
                param.requires_grad = self.CNN_train  # Trainable or frozen CNN layers

    def forward(self, imgs):
        # Get both primary and auxiliary outputs from the Inception model
        features, aux_logits = self.inception(imgs)  # Get primary and auxiliary output

        # Use only the primary output (features) and discard auxiliary logits
        features = self.relu(self.dropout(features))  # Apply dropout and ReLU to the primary output
        return features


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
        states = None

        with torch.no_grad():
            for _ in range(max_len):
                hidden, states = self.decoder.lstm(x, states)  # Forward LSTM with state
                output = self.decoder.linear(hidden).squeeze(0)  # Linear transformation to vocab size
                predicted = output.argmax(1)  # Get the predicted word index

                caption.append(predicted.item())  # Append word index to caption
                x = self.decoder.embedding(predicted).unsqueeze(0)  # Embed the predicted word

                if vocabulary.its[predicted.item()] == '<EOS>':  # End the caption generation
                    break

        return [vocabulary.its[indx] for indx in caption]  # Return caption with words