# helper.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List
from flwr.common import NDArrays

def load_dataset(client_id: int):
    df = pd.read_csv('/home/kishan/Documents/projects/machinelearning_cyberstalking_research/dataset.csv')
    df['text'] = df['text'].str.replace('[^a-zA-Z\s]', '').str.lower()
    df = df.drop(df.columns[2], axis=1)
    X = df['text']
    y = df['label']

    # np.random.seed(42)
    random_choose = np.random.choice(X.index, (len(X) % 3), replace=False)
    X = X.drop(random_choose)
    y = y.drop(random_choose)

    X_split, y_split = np.split(X, 3), np.split(y, 3)
    X1, y1 = X_split[0], y_split[0]
    X2, y2 = X_split[1], y_split[1]
    X3, y3 = X_split[2], y_split[2]

    X_train, y_train, X_test, y_test = [], [], [], []
    train_size = 0.8

    # X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=train_size, random_state=42)
    # X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=train_size, random_state=42)
    # X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, train_size=train_size, random_state=42)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=train_size)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=train_size)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, train_size=train_size)

    X_train.extend([X1_train, X2_train, X3_train])
    y_train.extend([y1_train, y2_train, y3_train])
    X_test.extend([X1_test, X2_test, X3_test])
    y_test.extend([y1_test, y2_test, y3_test])

    return X_train[client_id], y_train[client_id], X_test[client_id], y_test[client_id]


def preprocess_text_data(X_train, X_test, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    return X_train_pad, X_test_pad


def get_params(model) -> NDArrays:
    weights = model.get_weights()
    return [w.copy() for w in weights]


def set_params(model, params: NDArrays):
    model.set_weights(params)
    return model
