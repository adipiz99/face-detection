import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import *
from torchvision.transforms import v2

class PairDataset(data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataset = dataframe
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=False),
        ])
        
        self.class_to_indices = {c: dataframe.index[dataframe['label'] == c].tolist() for c in range(2)} # range(x), x = number of different classes 
        
    def generate_pairs(self):
        #create a vector of random couple labels
        self.pair_labels = (np.random.rand(len(self.dataset))>0.5).astype(int)
        
        #paired_idx contains the index of the paired element
        #the i-th element of the dataset is the first element of the i-th pair
        self.paired_idx = []
        #for any element in the paired labels
        for i, l in enumerate(self.pair_labels):
            #get the class of the first element
            c1 = self.dataset.iloc[i]['label']
            if l==0: #the pair is similar
                #choose a random element of the first
                j = self.class_to_indices[c1]
                j = np.random.choice(j)
            else:
                #choose a different class
                diff_class = 1 if c1==0 else 0
                #choose a random element of the different class
                j = self.class_to_indices[diff_class]
                j = np.random.choice(j)
            #save the index of the paired element
            self.paired_idx.append(j)
        
    def __len__(self):
        #as many pairs as elements in the dataset
        return len(self.dataset)
    
    def __getitem__(self, i):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #the first element of the pair is the i-th element of the dataset
        df = self.dataset.drop(columns='label').to_numpy() #.reshape(-1, 12, 1, 1) eventually reshape with the right dimensions
        im1 = df[i]
        im1 = torch.tensor(im1, device=device, dtype=torch.float32)
        l1 = self.dataset.iloc[i]['label']
        im2 = df[self.paired_idx[i]]
        im2 = torch.tensor(im2, device=device, dtype=torch.float32)
        l2 = self.dataset.iloc[self.paired_idx[i]]['label']

        l = self.pair_labels[i] #coup\le label
        
        #return the two elements of the pair, their labels and the couple label
        return im1, im2, l, l1, l2
    
class ResNetEmbedding(nn.Module):
    def __init__(self):
        super(ResNetEmbedding, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1, progress=True)
        #accept x channels
        x = 500
        num_classes = 2 
        self.resnet.conv1 = nn.Conv2d(x, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  #change the first layer to accept x channels
        self.fc1 = nn.Linear(1000, 512) #ResNet18 generates 1000 features, we reduce them to 512
        self.relu1 = nn.ReLU() #activation function
        self.fc2 = nn.Linear(512, 128) #further reduce to 128
        self.relu2 = nn.ReLU() #activation function
        self.fc3 = nn.Linear(128, num_classes) #final layer, num_classes reduction
        self.sftmx = nn.Softmax(dim=0) #softmax function
        
        
    def forward(self, x):
        out = self.resnet(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out) #if retrurns all zeroes, dekete this step
        out = self.fc3(out)
        out = self.sftmx(out)
        return out

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2):
        super(ContrastiveLoss, self).__init__()
        self.m = m

    def forward(self, phi_i, phi_j, l_ij):
        d = F.pairwise_distance(phi_i, phi_j, keepdim=True)

        l = 0.5 * (1 - l_ij.float()) * torch.pow(d,2) + \
            0.5 * l_ij.float() * torch.pow( torch.clamp( self.m - d, min = 0) , 2)
        return l.mean()
    
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
    
class SiameseNetworkTask(pl.LightningModule):
    def __init__(self, 
                 embedding_net, # the embedding network
                 lr=0.0001, # learning rate
                 momentum=0.99, # momentum
                 margin=1, # loss margin
                ):
        super(SiameseNetworkTask, self).__init__()
        self.save_hyperparameters()
        self.embedding_net = embedding_net
        self.embedding_net_state_dict = self.embedding_net.state_dict()
        self.criterion = ContrastiveLoss(margin) #definiamo la loss
        self.criterion_state_dict = self.criterion.state_dict()
        self.best_loss = np.inf
        self.losses = []
        self.val_losses = []

                    
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)
    
    def training_step(self, batch, batch_idx):
        # preleviamo gli elementi I_i e I_j e l'etichetta l_ij
        # scartiamo il resto (le etichette dei singoli elementi)
        I_i, I_j, l_ij, l_i, l_j = batch
        
        #l'implementazione della rete siamese è banale:
        #eseguiamo la embedding net sui due input
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)

        #calcoliamo la loss
        l = self.criterion(phi_i, phi_j, l_ij)
        #Convert tensor to double
        l_arr = l.item()
        #l_arr = l.double()
        self.losses.append(l_arr)
        #self.losses.append(l.detach().to('cpu'))
        

        if l < self.best_loss:
            self.best_loss = l
            self.save_hyperparameters()
            self.embedding_net_state_dict = self.embedding_net.state_dict()
            self.criterion_state_dict = self.criterion.state_dict()
            model_name = 'models/SiameseNet.pth'
            
            torch.save({
            'embedding_state_dict': self.embedding_net.state_dict(),
            'loss': self.criterion.state_dict,
            'best_loss': self.best_loss,
            'hyperparameters': self.hparams,
            }, model_name)
        
        
        self.log('train/loss', l, on_step=True)
        return l
    
    def validation_step(self, batch, batch_idx):
        I_i, I_j, l_ij, l_i, l_j = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        l = self.criterion(phi_i, phi_j, l_ij)
        l_arr = l.item()
        self.val_losses.append(l_arr)

        self.log('val/loss', l, on_step=True)

        #if batch_idx==0:
            #self.logger.experiment.add_embedding(phi_i, batch[3], I_i, global_step=self.global_step)

class Extractor: #Once the model is trained, we can use this class to extract the representations of the data
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    def extract_representations_with_labels(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)
        representations, labels = [], []
        for batch in self.loader:
            x = batch[0].to(device, dtype=torch.float32)
            rep = self.model(x)
            rep = rep.detach().to('cpu').numpy()
            labels.append(batch[1])
            representations.append(rep)
        return np.concatenate(representations), np.concatenate(labels)
    
    def extract_representations(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)
        representations = []
        for batch in self.loader:
            x = batch.clone().detach().to(device, dtype=torch.float32)
            #x = torch.tensor(batch, device=device, dtype=torch.float32)
            #x = batch.to(device, dtype=torch.float32)
            rep = self.model(x)
            rep = rep.detach().to('cpu').numpy()
            representations.append(rep)
        return np.concatenate(representations)