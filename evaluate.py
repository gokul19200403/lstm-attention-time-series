import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import generate_data, create_sequences
from model import LSTMAttentionModel, BaselineLSTM

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

data = generate_data()
X, y = create_sequences(data)

split = int(0.8 * len(X))
X_test, y_test = X[split:], y[split:]
X_test = torch.tensor(X_test, dtype=torch.float32)

models = {
    "attention": LSTMAttentionModel(input_dim=X.shape[2]),
    "baseline": BaselineLSTM(input_dim=X.shape[2])
}

for name, model in models.items():
    model.load_state_dict(torch.load(f"{name}.pt"))
    model.eval()

    with torch.no_grad():
        if name == "attention":
            preds, attn = model(X_test)
        else:
            preds = model(X_test)

    preds = preds.numpy()
    print(f"\n{name.upper()} MODEL")
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("MAPE:", mape(y_test, preds))

    if name == "attention":
        print("Top attention weights:", attn.mean(dim=0).squeeze().numpy()[-5:])

print("Final Predictions:", preds[:5])
