# --------------------------
# This script creates a ratings model using BERT and
# a deep feed-foward linear network. Data usage is 
# large, so BERT tensors must be created in batches
# and immediately fed into the NN.
# --------------------------

import torch
import pandas as pd
import re
import torch.nn as nn
import torch.optim as optim

from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import Dataset

# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lang_model = BertModel.from_pretrained('bert-base-uncased')
lang_model.to(device).eval()

# shuffle stories and delete entries incompatable with BERT
story_frame = pd.read_pickle('files/BERT_corpus.pkl').sample(frac=1, random_state=14).reset_index(drop=True).drop([1890,2227,3478]).reset_index(drop=True)

class StoryDataset(Dataset):
    '''
    Create story batches of variable size.
    '''
    
    def __init__(self, low, high):
        '''
        Embed batch using BERT
        '''
        
        self.batch = story_frame.iloc[low:high,:]
        self.embed = [self.embed_story(s) for s in self.batch['story']]
        self.rate = self.batch['rating'].copy().reset_index(drop=True)
        
        
    def __len__(self):
        return len(self.embed)
        
    def __getitem__(self, index):
        return self.embed[index]
    
    
            
    def embed_sentence(self, text):
        '''
        Embed a string using BERT.
        '''
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
        '''
        Apply BERT to full story.
        '''
        sens = re.split('[!?.;:]', text)

        return [self.embed_sentence(sen) for sen in sens]

# params
input_size = 3072 #4 x 1 x 768
hidden_size = 128
num_epochs = 2

learning_rate = .002

class NN(nn.Module):
    '''
    Deep feed-foward linear NN to learn from BERT tensors.
    '''
    def __init__(self, input_size, hidden_size):
        '''
        Use 3 linear layers with 0.2 dropout to reduce overfitting.
        '''
        super(NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.predict = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.fc(x)        
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc2(out)        
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc3(out)        
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.predict(out)
        
        return out
    

# --------------------------------
# Training section
# --------------------------------
model = NN(input_size, hidden_size).to(torch.float64).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# training loop
for epoch in range(num_epochs):
    # use batch size of 8
    bot = 0
    top = 8

    while top <= 2448: # around %70 of data for training

        data = StoryDataset(bot,top)
        print(data.batch.index)

        for i, story in enumerate(data.embed):
            # convert rating to tensor for backprop
            actual = torch.tensor(data.rate[i]).to(device).unsqueeze(0)
        
            predicted = []
            # convert each word in sentence to a tensor
            for sentence_tensor in story:
                for j in range(sentence_tensor.shape[1]):
                    word_tensor = sentence_tensor[:, j, :].flatten().to(torch.float64)
                    predicted.append(model(word_tensor))
                    
            # average model values over all words
            predicted = torch.mean(torch.stack(predicted)).to(device).unsqueeze(0)
            
            loss = criterion(predicted, actual)
            print('loss: %.5f' % loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        bot += 8
        top += 8

# ------------------------------
# Training on the validation set after testing
# ------------------------------

# model = NN(input_size, hidden_size).to(torch.float64).to(device)
# model.load_state_dict(torch.load('files/model_state.pth'))
# model.train()

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     # batch size of 10 
#     bot = 2448
#     top = 2458

#     while top <= 2798:
#         data = StoryDataset(bot,top)
#         print(data.batch.index)

#         for i, story in enumerate(data.embed):
#             actual = torch.tensor(data.rate[i]).to(device).unsqueeze(0)
        
#             predicted = []
#             for sentence_tensor in story:
#                 for j in range(sentence_tensor.shape[1]):
#                     word_tensor = sentence_tensor[:, j, :].flatten().to(torch.float64)
#                     predicted.append(model(word_tensor))
                    
#             predicted = torch.mean(torch.stack(predicted)).to(device).unsqueeze(0)
            
#             loss = criterion(predicted, actual)
#             print('loss: %.5f' % loss)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
            
#         bot += 10
#         top += 10

#     bot = 2448
#     top = 2458 

torch.save(model.state_dict(), 'files/model_state.pth')
#torch.save(model.state_dict(), 'files/model_state2.pth')