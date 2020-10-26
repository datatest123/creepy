#import numpy as np
import torch
import pandas as pd
#import pickle
import re
import torch.nn as nn
import torch.optim as optim

from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import Dataset

# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define save/load paths
#MODEL_PATH = 'model_params.tar'

# BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lang_model = BertModel.from_pretrained('bert-base-uncased')
lang_model.to(device).eval()

# import stories
story_frame = pd.read_pickle('files/BERT_corpus.pkl').sample(frac=1, random_state=14).reset_index(drop=True)

class StoryDataset(Dataset):
    
    def __init__(self, low, high):
        
        self.batch = story_frame.iloc[low:high,:]
        self.embed = [self.embed_story(s) for s in self.batch['story']]
        self.rate = self.batch['rating'].copy().reset_index(drop=True)
        
        
    def __len__(self):
        return len(self.embed)
        
    def __getitem__(self, index):
        return self.embed[index]
    
    
            
    def embed_sentence(self, text):
        tokens = tokenizer.tokenize(text)
        
        # convert tokens to list of int ids
        tokens_with_tags = ['[CLS]'] + tokens + ['[SEP]']
        indices = tokenizer.convert_tokens_to_ids(tokens_with_tags)

        # convert id list to 64 bit int tensor 
        out = lang_model(torch.LongTensor(indices).to(device).unsqueeze(0))

        # Concatenate the last four layers and use that as the embedding
        # source: https://jalammar.github.io/illustrated-bert/
        embeddings_tensor = torch.stack(out[0]).to(device).squeeze(1)[-4:]  # use last 4 layers

        # detach from graph to prevent copying and remove the [CLS] and [SEP] layers
        return embeddings_tensor[:, 1:-1, :].detach()
    
    def embed_story(self, text):
        sens = re.split('[!?.;:]', text)

        return [self.embed_sentence(sen) for sen in sens]

# params
input_size = 3072 #4 x 1 x 768
hidden_size = 128
num_epochs = 2

learning_rate = .001

class NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.predict = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.predict(out)
        
        return out
    
model = NN(input_size, hidden_size).to(torch.float64).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# training loop
bot = 0
top = 10
while top <= 3500:

    data = StoryDataset(bot,top)
    print(data.batch.index)
    
    for i, story in enumerate(data.embed):
        actual = torch.tensor(data.rate[i]).to(device).unsqueeze(0)
     
        predicted = []
        for sentence_tensor in story:
            for j in range(sentence_tensor.shape[1]):
                word_tensor = sentence_tensor[:, j, :].flatten().to(torch.float64)
                predicted.append(model(word_tensor))
                
        predicted = torch.mean(torch.stack(predicted)).to(device).unsqueeze(0)
        
        loss = criterion(predicted, actual)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    bot += 10
    top += 10

        
print(predicted)
print(actual)
# torch.save(model.state_dict(), 'model_params.pt')