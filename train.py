import torch
from torch.utils.data import DataLoader, TensorDataset
from data import generate_data, create_sequences
from model import LSTMAttentionModel, BaselineLSTM

def train():
    data = generate_data()
    X, y = create_sequences(data)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    models = {
        "attention": LSTMAttentionModel(input_dim=X.shape[2]),
        "baseline": BaselineLSTM(input_dim=X.shape[2])
    }

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        for _ in range(15):
            for xb, yb in loader:
                optimizer.zero_grad()
                if name == "attention":
                    preds, _ = model(xb)
                else:
                    preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), f"{name}.pt")

if __name__ == "__main__":
    train()
