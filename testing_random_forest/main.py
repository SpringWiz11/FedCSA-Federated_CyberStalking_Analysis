# main.py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import get_train_test_loaders
from model import SimpleNNModel
from server import CustomServer
from client import CustomClient

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Constants
FILE_PATH = 'dataset.csv'
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
INPUT_SIZE = 5000  # Adjust based on your max_features in TfidfVectorizer
HIDDEN_SIZE = 128
OUTPUT_SIZE = 2  # Assuming binary classification

# Initialize server and clients
server = CustomServer(clients=[CustomClient(id=i, data=None, model=SimpleNNModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)) for i in range(3)])

# Load data
train_loader, test_loader = get_train_test_loaders(FILE_PATH, max_features=INPUT_SIZE)

# Train federated model
for epoch in range(NUM_EPOCHS):
    for client in server.clients:
        client.train(train_loader)
        client.update_model()

    server.aggregate_gradients()
    server.update_global_model()

# Evaluate the global model
server.evaluate_global_model(test_loader)
