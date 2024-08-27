import os
from deepface import DeepFace
import numpy as np
import h5py
import cv2

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
