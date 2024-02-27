# server.py
import torch
import syft as sy

hook = sy.TorchHook(torch)

class CustomServer:
    def __init__(self, clients):
        self.clients = clients
        self.global_model = SimpleNNModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    def aggregate_gradients(self):
        # Aggregate gradients from clients (implemented using PySyft)
        for param in self.global_model.parameters():
            param.data.set_(sum(client.model.get(param, detach=True) for client in self.clients) / len(self.clients))

    def update_global_model(self):
        # Update the global model with aggregated gradients
        self.global_model = self.global_model.get()

    def evaluate_global_model(self, test_loader):
        # Evaluate the global model
        results = self.clients[0].get_results(test_loader)  # Assume all clients have the same test data
        print(f"Global Model - Accuracy: {results['accuracy']}, Loss: {results['loss']}")
