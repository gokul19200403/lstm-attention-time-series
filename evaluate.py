import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import generate_data, create_sequences
from model import LSTMAttentionModel

data = generate_data()
X, y = create_sequences(data)

split = int(0.8 * len(X))
X_test, y_test = X[split:], y[split:]

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = y_test

model = LSTMAttentionModel(input_dim=X.shape[2])
model.load_state_dict(torch.load("model.pt"))
model.eval()

with torch.no_grad():
    preds, attn = model(X_test)

preds = preds.numpy()

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("RMSE:", rmse)
print("MAE:", mae)

avg_attention = attn.mean(dim=0).squeeze().numpy()
print("Average Attention (last 5 steps):", avg_attention[-5:])

print("Final test predictions:", preds[:5])
