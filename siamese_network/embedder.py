import os
import sys
import csv
import torch
import pandas as pd
from torch.utils.data import DataLoader

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import siamese_net as embedding


class DatasetEmbedder():
    '''Class to embed a dataset using a pre-trained model
        Parameters:
            model_path: str
                Path to the model to use for embedding
            dataset_samples: int
                Number of samples to embed from the dataset
            dataset_path: str
                Path to the dataset to embed
            labelled_dataset_path: str (optional)
                Path to the labelled dataset to use for labelling the embedded dataset
            embedded_dataset_path: str (optional)
                Path to save the embedded dataset
            embedded_labelled_dataset_path: str (optional)
                Path to save the embedded labelled dataset
    '''
    def __init__(self, model_path, dataset_samples, dataset_path, labelled_dataset_path=None, embedded_dataset_path=None, embedded_labelled_dataset_path=None):
        self.model_path = model_path
        self.dataset_samples = dataset_samples
        self.dataset_path = dataset_path
        self.labelled_dataset_path = labelled_dataset_path
        self.embedded_dataset_path = embedded_dataset_path
        self.embedded_labelled_dataset_path = embedded_labelled_dataset_path

    def embed_dataset(self):
        if os.path.exists(self.model_path):
            siamese_task = embedding.SiameseNetworkTask(embedding.ResNetEmbedding())
            siamese_task.embedding_net.load_state_dict(torch.load(self.model_path)['embedding_state_dict'])
            siamese_task.criterion.state_dict = torch.load(self.model_path)['loss']
            siamese_task.best_loss = torch.load(self.model_path)['best_loss']           
            hparams = torch.load(self.model_path)['hyperparameters']
            for key in hparams.keys():
                siamese_task.hparams[key]=hparams[key]
        else:
            raise FileNotFoundError("Model not found")
        
        df = pd.read_csv(self.dataset_path)

        df_np = df.to_numpy().reshape(-1, 12, 1, 1) #Reshape the dataset to fit the model
        batch_size = 64
        df_loader = DataLoader(df_np, batch_size=batch_size, num_workers=6, shuffle=False, drop_last=True)

        num_elements = round(self.dataset_samples/batch_size)*batch_size #Number of elements to embed (deletes the last batch if it is not complete)
        extractor = embedding.Extractor(siamese_task.embedding_net, df_loader)
        representations = extractor.extract_representations()

        print("Passed {} files, obtained {} representations".format(num_elements, representations.shape[0]))

        # Building a csv file with the representations and labels
        if self.embedded_dataset_path == None:
            embedded_dataset = 'embedded_dataset.csv'
        else :
            embedded_dataset = self.embedded_dataset_path
        length = len(representations[0])
        headers = {'feature_' + str(x): x for x in range(0, length)}
        with open(embedded_dataset, mode='w') as representations_file:
            representations_writer = csv.writer(representations_file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, lineterminator='\n', escapechar='\\')
            for i in range(len(representations)):
                row = []
                for j in range(len(representations[i])):
                    if i == 0 and j == 0:
                        representations_writer.writerow(headers.keys())
                    row.append(representations[i][j])
                representations_writer.writerow(row)

        # Get an array of the first num_elements labels from the labelled dataset
        if self.embedded_labelled_dataset_path != None and os.path.exists(self.labelled_dataset_path):
            labelled_df = pd.read_csv(self.labelled_dataset_path)
            labels = labelled_df['label'][:num_elements]

            representations_df = pd.read_csv(self.embedded_dataset_path)
            representations_df_label = representations_df.copy()
            representations_df_label['label'] = labels
            representations_df_label.to_csv(self.embedded_labelled_dataset_path, index=False)
        else:
            print("No labelled dataset provided, skipping labelling. Please, label the dataset manually.")
