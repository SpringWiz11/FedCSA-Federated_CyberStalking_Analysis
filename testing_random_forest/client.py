# client.py
import torch
import torch.optim as optim
import syft as sy

hook = sy.TorchHook(torch)

class CustomClient:
    def __init__(self, id, data, model):
        self.id = id
        self.data = data
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, data_loader):
        self.model.train()
        for batch in data_loader:
            data, target = batch['text'], batch['label']
            self.optimizer.zero_grad()
            output = self.model(data.float())
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

    def update_model(self):
        # Send model updates to the server (implemented using PySyft)
        self.model.send(sy.local_worker)

    def get_results(self, test_loader):
        # Evaluate the model on the client's test data
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                data, target = batch['text'], batch['label']
                output = self.model(data.float())
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        return {'accuracy': accuracy, 'loss': test_loss}
