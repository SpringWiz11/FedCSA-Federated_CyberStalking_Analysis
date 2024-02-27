# client.py
import helper
import numpy as np
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.simplefilter('ignore')

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"Client {client_id} received the parameters.")
        return helper.get_params(model)

    def fit(self, parameters, config):
        print("Parameters before setting: ", parameters)
        helper.set_params(model, parameters)
        print("Parameters after setting: ", model.get_params())
    
        model.fit(X_train_tfidf, y_train)
        print(f"Training finished for round {config['server_round']}.")

        trained_params = helper.get_params(model)
        print("Trained Parameters: ", trained_params)

        return trained_params, X_train_tfidf.shape[0], {}

    def evaluate(self, parameters, config):
        helper.set_params(model, parameters)

        y_pred = model.predict(X_test_tfidf)
        loss = log_loss(y_test, y_pred, labels=[0, 1])

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

        line = "-" * 21
        print(line)
        print(f"Accuracy : {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall   : {recall:.8f}")
        print(f"F1 Score : {f1:.8f}")
        print(f"Confusion matrix : {conf_matrix}")
        print(line)

        return loss, X_test_tfidf.shape[0], {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1,}


if __name__ == "__main__":
    client_id = 2
    print(f"Client {client_id}:\n")

    X_train, y_train, X_test, y_test = helper.load_dataset(client_id - 1)
    y_train = y_train.map({'Yes': 1, 'No': 0})
    y_test = y_test.map({'Yes': 1, 'No': 0})

    X_train_series = X_train.squeeze()
    X_test_series = X_test.squeeze()
    vectorizer = TfidfVectorizer(max_features=100)
    X_train_tfidf = vectorizer.fit_transform(X_train_series)
    X_test_tfidf = vectorizer.transform(X_test_series)

    print("The shape of the X_train_tfidf")
    print(X_train_tfidf.shape)
    print("The shape of the X_test_tfidf")
    print(X_test_tfidf.shape)

    model = LogisticRegression(
        solver= 'saga',
        penalty="l2",
        max_iter=50,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    # model = LogisticRegression()
    # helper.set_initial_params(model)
    model.fit(X_train_tfidf, y_train)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
