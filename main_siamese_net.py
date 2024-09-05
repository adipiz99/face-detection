import os
import torch
import numpy as np
import pandas as pd
from deepface import DeepFace
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import siamese_network.siamese_net as embedding
from siamese_network.embedder import DatasetEmbedder
from pytorch_lightning.loggers import TensorBoardLogger
import h5py
import cv2

EPOCHS = 50

def train_siamese_net(train_path, test_path):
    df = pd.read_csv(train_path) #training dataset
    test_df = pd.read_csv(test_path) #validation dataset

    pair_train = embedding.PairDataset(df)
    pair_test = embedding.PairDataset(test_df)

    pair_train_loader = DataLoader(pair_train, batch_size=64, num_workers=0, shuffle=True, drop_last=True)
    pair_test_loader = DataLoader(pair_test, batch_size=64, num_workers=0, shuffle=True, drop_last=True)

    siamese_task = embedding.SiameseNetworkTask(embedding.ResNetEmbedding())
    logger = TensorBoardLogger("metric_logs", name="siamese_embedding")
    trainer = pl.Trainer(logger=logger, max_epochs=EPOCHS, enable_progress_bar=True, accelerator='gpu', check_val_every_n_epoch=1, enable_checkpointing=True)
    trainer.fit(siamese_task, pair_train_loader, pair_test_loader)
    
    return siamese_task

def use_siamese_net(model_name = "SiameseNet.pth", dataset_path = None, num_samples = None, save_path = None):
    siamese_task = embedding.SiameseNetworkTask(embedding.ResNetEmbedding())
    siamese_task.embedding_net.load_state_dict(torch.load("models/" + model_name)['embedding_state_dict'])
    siamese_task.criterion.state_dict = torch.load("models/" + model_name)['loss']
    siamese_task.best_loss = torch.load("models/" + model_name)['best_loss']       
    hparams = torch.load("models/" + model_name)['hyperparameters']
    for key in hparams.keys():
        siamese_task.hparams[key]=hparams[key]
    
    if num_samples is None:
        raise ValueError("Number of samples not provided")
    if dataset_path or save_path is None:
        raise ValueError("Dataset not provided")
    
    dataset_embedder = DatasetEmbedder(model_path=model_name, dataset_samples=num_samples, dataset_path=dataset_path, embedded_dataset_path=save_path)
    dataset_embedder.embed_dataset()
    print("Dataset embedded at ", save_path)


if __name__ == '__main__':
    # Path to the ORL dataset directory
    orl_dir = 'att_faces'

    # Collect all the image paths
    image_paths = []
    for root, dirs, files in os.walk(orl_dir):
        for file in files:
            if file.endswith(".pgm"):
                image_paths.append(os.path.join(root, file))

    # Check the number of images
    print(f"Total number of images: {len(image_paths)}")

    # Initialize a dictionary to store embeddings and corresponding labels
    embeddings_dict = {'VGG-Face': [], 'Facenet': [], 'DeepFace': []}
    labels = []

    # List of models to use
    models = ["VGG-Face", "Facenet", "DeepFace"]

    # Compute embeddings for each image and model
    for image_path in image_paths:
        # Extract the label from the image path (the folder name is the person's ID)
        label = os.path.basename(os.path.dirname(image_path))
        labels.append(label)

        # Compute embeddings using the models
        for model in models:
            embedding = DeepFace.represent(img_path=image_path, model_name=model)
            embeddings_dict[model].append(embedding)

    # Convert lists to numpy arrays for saving
    for model in models:
        embeddings_dict[model] = np.array(embeddings_dict[model])

    labels = np.array(labels)

    import pickle

    # Save the embeddings and labels using pickle
    with open('orl_embeddings.pkl', 'wb') as f:
        pickle.dump((embeddings_dict, labels), f)

    print("Embeddings and labels saved successfully.")

    print(labels)
    print(embeddings_dict)
    print(embeddings_dict['VGG-Face'][0])

    # Face Recognition

    from sklearn.metrics.pairwise import cosine_similarity

    # Load embeddings from the saved pickle file
    with open('orl_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        embeddings_dict = data[0]
        labels = data[1]

    print("Embeddings and labels loaded successfully.")

    # Function to recognize faces
    def recognize_faces(embeddings, model_name):
        numerical_embeddings = np.array([emb[:][0]['embedding'] for emb in embeddings])  # Extract numerical embeddings
        # Compute cosine similarity between embeddings
        similarity_matrix = cosine_similarity(numerical_embeddings)
        np.fill_diagonal(similarity_matrix, 0)  # Set diagonal to zero to ignore self-similarity

        # Find the closest match for each embedding
        matches = []
        for i in range(len(similarity_matrix)):
            closest_match_idx = np.argmax(similarity_matrix[i])
            matches.append((labels[i], labels[closest_match_idx], similarity_matrix[i][closest_match_idx]))

        # Print recognition results
        print(f"Recognition Results using {model_name}:")
        for match in matches:
            print(f"Actual: {match[0]}, Predicted: {match[1]}, Similarity: {match[2]:.4f}")
        print("\n")

    # Perform face recognition for each model
    recognize_faces(embeddings_dict['VGG-Face'], "VGG-Face")
    recognize_faces(embeddings_dict['Facenet'], "Facenet")
    recognize_faces(embeddings_dict['DeepFace'], "DeepFace")

    # --------------------------- Siamese Network ---------------------------
    #Prepare the datasets here and save them to two different csv files
    
    #prepare_train_dataset('train.csv')
    #prepare_test_dataset('test.csv')


    siamese_net = train_siamese_net('train.csv', 'test.csv')
    
    model_path = "models/SiameseNet.pth"
    source_test = "csv_files/source_test.csv"
    embedded_dataset = "csv_files/embedded_dataset.csv"
    samples = 400
    use_siamese_net(model_path, source_test, samples, embedded_dataset)

    #Read the embedded dataset
    classes =[]
    num_classes = 4
    df = pd.read_csv(embedded_dataset)
    for row in df.iterrows():
        argmax = 0
        for i in range(1, num_classes):
            if row[i] > row[argmax]:
                argmax = i
        classes.append(argmax)

    print("Predictions " + str(classes))
