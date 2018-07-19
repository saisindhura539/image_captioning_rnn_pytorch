import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
<<<<<<< HEAD
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3, drop_prob=0.2):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_len = 20
        self.n_hidden = hidden_size
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings),1)
        hiddens, _ = self.lstm(embeddings)
        hiddens = self.dropout(hiddens)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        integer_ids = []
        inputs = features
        for i in range(20):
            hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))          # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
        integer_ids = [ int(x) for x in sampled_ids ]
        return integer_ids
                    
=======
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        pass
    
    def forward(self, features, captions):
        pass

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
>>>>>>> 6e797174eddf2078ecbea5729cab4b14cc474326
