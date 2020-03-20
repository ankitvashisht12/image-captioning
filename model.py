import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) #downloading pretrained resnet model
        for param in resnet.parameters(): 
            param.requires_grad_(False)           # making every parameter non-trainable
        
        modules = list(resnet.children())[:-1]    
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
 
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.fc(lstm_out[:,:-1,:])
        
        return outputs  
 

    def sample(self, inputs, states=None, max_len=20):
        # (Do not write the sample method yet - you will work with this method when you reach 3_Inference.ipynb.)
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        
        embeddings = inputs
        
        for ii in range(max_len):
            
            
            lstm_out, states = self.lstm(embeddings, states)
                
            out = lstm_out.squeeze(1)
                
            out = self.fc(out)
            _, prediction = out.max(1)
                
                #print(f"ii is {ii} : ",out.size(), torch.squeeze(prediction).item())
            outputs.append(prediction.item())
                
            if prediction == 1:
                break
                
            embeddings = self.embed(prediction).unsqueeze(1)
        return outputs
