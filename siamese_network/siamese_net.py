import cv2
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
    def __init__(self, dataframe: pd.DataFrame): #Passing a dataframe in which data is stored under the columns 'image' and 'label'. Image contains the path of the images, label contains the class of the image.
        self.dataset = dataframe
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)), #resize the image to 256x256
            transforms.ToTensor()
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
        im1 = self.dataset.iloc[i]['image'] #Image path of the first element
        im1 = cv2.imread(im1) #Read the image
        im1 = self.transforms(im1).to(device) #Use the transforms to convert the image to a reshaped tensor, according to the size of the input layer
        l1 = self.dataset.iloc[i]['label'] #Label of the first element 

        #the second element of the pair is the i-th element in the paired_idx vector
        im2 = self.dataset.iloc[self.paired_idx[i]]['image'] #Image path of the second element
        im2 = cv2.imread(im2) #Read the image
        im2 = self.transforms(im2).to(device) #Use the transforms to convert the image to a reshaped tensor, according to the size of the input layer
        l2 = self.dataset.iloc[self.paired_idx[i]]['label'] #Label of the second element

        l = self.pair_labels[i] #couple label
        
        #return the two elements of the pair, their labels and the couple label
        return im1, im2, l, l1, l2

#### Network architectures ####
class LSTMEmbedding(nn.Module):
    def __init__(self):
        super(LSTMEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=100, num_layers=1, batch_first=True)

        #FC layers
        self.fc1 = nn.Linear(100, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 2)
        self.sftmx = nn.Softmax(dim=0)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sftmx(out)
        return out
    
class LCNNEmbedding(nn.Module):
    def __init__(self):
        super(LCNNEmbedding, self).__init__()

        #LCNN implementation
        self.conv1 = nn.Conv1d(12, 32, 3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3)

        #FC layers
        self.fc1 = nn.Linear(64*2, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 2)
        self.sftmx = nn.Softmax(dim=0)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = out.view(-1, 64*2)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sftmx(out)
        return out
    
class AlexNetEmbedding(nn.Module):
    def __init__(self):
        super(AlexNetEmbedding, self).__init__()
        self.alexnet = models.alexnet(weights= models.AlexNet_Weights.IMAGENET1K_V1, pretrained=True)
        num_classes = 2
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)
        self.sftmx = nn.Softmax(dim=0)
        
    def forward(self, x):
        out = self.alexnet(x)
        out = self.sftmx(out)
        return out
    
class VGGEmbedding(nn.Module):
    def __init__(self):
        super(VGGEmbedding, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1, pretrained=True)
        num_classes = 2
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        self.sftmx = nn.Softmax(dim=0)
        
    def forward(self, x):
        out = self.vgg(x)
        out = self.sftmx(out)
        return out
    
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
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sftmx(out)
        return out

#### Loss function ####
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2):
        super(ContrastiveLoss, self).__init__()
        self.m = m #margin

    def forward(self, phi_i, phi_j, l_ij):
        d = F.pairwise_distance(phi_i, phi_j, keepdim=True) #calculates the euclidean distance between the two embeddings

        l = 0.5 * (1 - l_ij.float()) * torch.pow(d,2) + \
            0.5 * l_ij.float() * torch.pow( torch.clamp( self.m - d, min = 0) , 2) #contrastive loss function with LeCun's formula
        return l.mean() #return the mean of the loss
    
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net #embedding network

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
        self.embedding_net = embedding_net #Initialize the embedding network
        self.embedding_net_state_dict = self.embedding_net.state_dict() #Save the state of the embedding network
        self.criterion = ContrastiveLoss(margin) #Initialize the loss function
        self.criterion_state_dict = self.criterion.state_dict() #Save the state of the loss function
        self.best_loss = np.inf #Initialize the best loss
        self.losses = [] #Initialize the losses array
        self.val_losses = [] #Initialize the validation losses array

                    
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum) #SGD optimizer
    
    def training_step(self, batch, batch_idx):
        # Extract the data from the batch
        I_i, I_j, l_ij, l_i, l_j = batch
        
        #Execute the forward on the two images
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)

        #Compute the loss
        l = self.criterion(phi_i, phi_j, l_ij)
        l_arr = l.item()
        #l_arr = l.double()
        self.losses.append(l_arr)
        #self.losses.append(l.detach().to('cpu'))
        

        if l < self.best_loss: #If the loss is the best loss so far, save the model
            self.best_loss = l
            self.save_hyperparameters()
            self.embedding_net_state_dict = self.embedding_net.state_dict()
            self.criterion_state_dict = self.criterion.state_dict()
            model_name = 'models/SiameseNet.pth' #Save the model in this path
            
            torch.save({
            'embedding_state_dict': self.embedding_net.state_dict(),
            'loss': self.criterion.state_dict,
            'best_loss': self.best_loss,
            'hyperparameters': self.hparams,
            }, model_name) #Save the model
        
        
        self.log('train/loss', l, on_step=True) #Log the loss to tensorboard
        return l
    
    def validation_step(self, batch, batch_idx):
        I_i, I_j, l_ij, l_i, l_j = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        l = self.criterion(phi_i, phi_j, l_ij)
        l_arr = l.item()
        self.val_losses.append(l_arr)

        self.log('val/loss', l, on_step=True)

        if batch_idx==0:
            self.logger.experiment.add_embedding(phi_i, batch[3], I_i, global_step=self.global_step) #Add the embeddings to tensorboard for visualization

class Extractor: #Once the model is trained, we can use this class to extract the representations of the data
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    def extract_representations_with_labels(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval() #Set the model to evaluation mode
        self.model.to(device) #Move the model to the device
        representations, labels = [], [] #Initialize the representations and labels arrays
        for batch in self.loader: #For each batch in the loader
            x = batch[0].to(device, dtype=torch.float32) #Move the batch to the device
            rep = self.model(x) #Get the representation of the batch
            rep = rep.detach().to('cpu').numpy() #Move the representation to the cpu and convert it to a numpy array
            labels.append(batch[1]) #Append the labels of the batch
            representations.append(rep) #Append the representations of the batch
        return np.concatenate(representations), np.concatenate(labels) #Return the representations and labels
    
    def extract_representations(self):
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.model.eval() #Set the model to evaluation mode
        self.model.to(device) #Move the model to the device
        representations = [] #Initialize the representations array
        for batch in self.loader: #For each batch in the loader
            x = batch.clone().detach().to(device, dtype=torch.float32) #Move the batch to the device
            rep = self.model(x) #Get the representation of the batch
            rep = rep.detach().to('cpu').numpy() #Move the representation to the cpu and convert it to a numpy array
            representations.append(rep) #Append the representations of the batch
        return np.concatenate(representations) #Return the representations