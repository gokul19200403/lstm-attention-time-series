import numpy as np

def generate_data(time_steps=1200, features=4):
    np.random.seed(42)
    t = np.arange(time_steps)
    data = np.zeros((time_steps, features))

    for i in range(features):
        trend = 0.01 * t
        seasonality = np.sin(0.02 * t + i)
        noise = np.random.normal(0, 0.2, time_steps)
        data[:, i] = trend + seasonality + noise

    return data

def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

