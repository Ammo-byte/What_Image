import torch
import torch.nn as nn
import torchvision.models as models

#Implementing the CNN encoder and RNN decoder
class CNN_encoder(nn.Module):
    def __init__(self, size, CNN_train = False):
        super(CNN_encoder, self).__init__()
        self.size = size
        self.CNN_train = CNN_train
        self.incep_pretrained = models.inception_v3(pretrained=True, aux_logits=False)
        self.incep_pretrained.fc = nn.Linear(self.incep_pretrained.fc.in_features, self.size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def push_through(self, imgs):
        features = self.incep_pretrained(imgs)
        for names, parameter in self.inception.named_parameters():
            if 'fc.weight' in names or 'fc.bias' in names:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = self.CNN_train
        return self.relu(self.dropout(features))
    
class RNN_decoder (nn.Module):
    def __init__ (self,size, h_size,vocab_s, layers):
        super(RNN_decoder, self).__init__()
        self.embeded = nn.Embedding(vocab_s, size)
        self.lstm = nn.LSTM(size, h_size, layers)
        self.linear = nn.Linear(h_size, vocab_s)
        self.dropout = nn.Dropout(0.5)

    def push_through(self, features, caption):
        embedding = self.dropout(self.embeded(caption))
        embedding = torch.cat((features.unsqueeze(0), embedding), dim = 0)
        lstm_out, _ = self.lstm(embedding)
        output = self.linear(lstm_out)
        return output
    
class CNN_RNN(nn.Module):
    def __init__(self, size, h_size, vocab_s, layers):
        super(CNN_RNN, self).__init__()
        self.encoder = CNN_encoder(size)
        self.decoder = RNN_decoder(size, h_size, vocab_s, layers)
        
    def push_through(self, imgs, caption):
        features = self.encoder.push_through(imgs)
        output = self.decoder.push_through(features, caption)
        return output
   
    def image_caption(self, imgs, vocabulary, max_len = 50):
        caption = []

        with torch.no_grad():

            x = self.encoder.push_through(imgs).unsqueeze(0)
            states = None

            for _ in range(max_len):
                hidden, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hidden).squeeze(0)
                predicted = output.argmax(1)

                caption.append(predicted.item())
                x = self.decoder.embeded(predicted).unsqueeze(0)

                if vocabulary.its[predicted.item()] == '<EOS>':
                    break
        return [vocabulary.its[indx] for indx in caption]
