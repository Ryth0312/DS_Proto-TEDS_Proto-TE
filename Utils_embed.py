#%%
import torch
from torch.nn.utils.rnn import pad_sequence
# DataLoader-related
from torch.utils.data import Dataset, DataLoader


def Embed2Embed(embeds):
    padded_token_embs = pad_sequence(embeds, batch_first=True)
    return padded_token_embs

#%% Dataloader for evaluation
class MIMICDataset(Dataset):
    def __init__(self, demo_data, EMR_data, lens, labels):
        self.demo_data = demo_data
        self.EMR_data = EMR_data
        self.lens = lens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.demo_data[index], self.EMR_data[index], self.lens[index], self.labels[index]
    
def MIMICLoader(demo_data, EMR_data, lens, labels, batch_size=32, shuffle=False, num_workers=2):

    # Create dataset
    dataset = MIMICDataset(demo_data, EMR_data, lens, labels)
        
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

#%% Dataloader for projection

class MIMICDataset_proj(Dataset):
    def __init__(self, embeddings):
        ''' 
        Load embeddings within a single time step
        Embedding size: N x 1 x D
        
        '''
        self.embeddings = embeddings
        

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return index, self.embeddings[index]
    
def StepLoader(embeddings, batch_size=1024, shuffle=False, num_workers=0):
    '''
    Load embeddings within a single time step
    Embedding size: N x D
    '''
    # Create dataset
    dataset = MIMICDataset_proj(embeddings)
        
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
    
    return dataloader