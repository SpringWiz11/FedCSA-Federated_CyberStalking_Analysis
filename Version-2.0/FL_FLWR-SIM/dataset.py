import pandas as pd
import numpy as np
from typing import List

def load_dataset(client_id:int):
    df = pd.read_csv('/home/kishan/Documents/projects/machinelearning_cyberstalking_research/dataset.csv')
    df['text'] = df['text'].str.replace('[^a-zA-Z\s]', '').str.lower()
    df = df.drop(df.columns[2], axis=1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    np.random.seed(42)
