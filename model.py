import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
#data sets 
np.random.seed(42)
time_steps = 1200
features = 5

t = np.arange(time_steps)
data = np.zeros((time_steps, features))

for i in range(features):
    trend = 0.01 * t
    seasonality = np.sin(0.02 * t + i)
    noise = np.random.normal(0, 0.2, time_steps)
    data[:, i] = trend + seasonality + noise


def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0]) 
    return np.array(X), np.array(y)

SEQ_LEN = 20
X, y = create_sequences(data, SEQ_LEN)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context, weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)
        return out.squeeze(), attn_weights

model = LSTMAttentionModel(features, 64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 15
for epoch in range(EPOCHS):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds, _ = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions, attention_weights = model(X_test)

predictions = predictions.numpy()
y_true = y_test.numpy()

rmse = np.sqrt(mean_squared_error(y_true, predictions))
mae = mean_absolute_error(y_true, predictions)

print("\nEvaluation Metrics:")
print("RMSE:", rmse)
print("MAE:", mae)


avg_attention = attention_weights.mean(dim=0).squeeze().numpy()
print("\nAverage Attention Weights (per time step):")
print(avg_attention)


print("\nSample Predictions:")
print(predictions[:10])
