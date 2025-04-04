import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def train_and_plot_Seq2Seq(train_ratio=0.8, n_epochs=1000, window_size=10, lookback=10, batch_size=32):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    df = pd.read_csv("./traffic_density_2_hours_iso.csv", parse_dates=['Timestamp'])

    # Strip any spaces from column names to avoid issues
    df.columns = df.columns.str.strip()

    # Set 'Timestamp' as index
    df.set_index('Timestamp', inplace=True)

    # Sort by time
    df = df.sort_values(by='Timestamp')

    # Select relevant column
    df = df[['Video1 Density']].copy()
    df.rename(columns={"Video1 Density": "value"}, inplace=True)

    # Interpolate and clean
    df['value'] = df['value'].interpolate(method='time', limit=5).fillna(method='ffill').fillna(method='bfill')
    df = df[df['value'] > 0].reset_index(drop=True)

    # Smooth the series
    df['value'] = df['value'].rolling(window=window_size, center=True).mean()
    df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')

    timeseries = df['value'].values.astype('float32')

    # Train-test split
    train_size = int(len(timeseries) * train_ratio)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    def create_dataset(dataset, lookback):
        X, y = [], []
        for i in range(len(dataset) - lookback):
            X.append(dataset[i:i + lookback])
            y.append(dataset[i + lookback])
        return torch.tensor(X).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)

    X_train, y_train = create_dataset(train, lookback)
    X_test, y_test = create_dataset(test, lookback)

    # Move to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    class Seq2SeqModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=300, num_layers=1):
            super().__init__()
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, encoder_input, decoder_input):
            _, (hidden, cell) = self.encoder(encoder_input)
            decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            return output

    model = Seq2SeqModel().to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    best_rmse = float('inf')
    best_model_path = "best_model_Seq2Seq.pt"

    train_losses = []
    test_losses = []

    def create_dataloader(X, y, batch_size):
        dataset = data.TensorDataset(X, y)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dataloader(X_train, y_train, batch_size)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            decoder_input = X_batch[:, -1:, :].repeat(1, lookback, 1)
            y_pred = model(X_batch, decoder_input)
            loss = loss_fn(y_pred[:, -1, :], y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        with torch.no_grad():
            test_decoder_input = X_test[:, -1:, :].repeat(1, lookback, 1)
            y_pred_test = model(X_test, test_decoder_input)[:, -1, :]
            test_rmse = np.sqrt(loss_fn(y_pred_test, y_test).item())
            test_losses.append(test_rmse)

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save(model.state_dict(), best_model_path)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Test RMSE = {test_rmse:.6f}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with torch.no_grad():
        test_decoder_input = X_test[:, -1:, :].repeat(1, lookback, 1)
        y_pred = model(X_test, test_decoder_input)[:, -1].squeeze().cpu().numpy()
        y_true = y_test.squeeze().cpu().numpy()

    test_r2 = r2_score(y_true, y_pred)

    # For plotting
    train_plot = np.ones_like(timeseries) * np.nan
    test_plot = np.ones_like(timeseries) * np.nan

    train_decoder_input = X_train[:, -1:, :].repeat(1, lookback, 1)
    test_decoder_input = X_test[:, -1:, :].repeat(1, lookback, 1)

    with torch.no_grad():
        train_predictions = model(X_train, train_decoder_input)[:, -1, :].squeeze().cpu().numpy()
        test_predictions = model(X_test, test_decoder_input)[:, -1, :].squeeze().cpu().numpy()

    train_plot[lookback:train_size] = train_predictions
    test_plot[train_size + lookback:] = test_predictions

    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, label="Actual Time Series", color="blue", alpha=0.6)
    plt.plot(train_plot, label="Train Predictions", color="red", linestyle="--", linewidth=2)
    plt.plot(test_plot, label="Test Predictions", color="green", linestyle="--", linewidth=2)
    plt.title(f"Time Series Prediction with Seq2Seq (R2: {test_r2:.4f})", fontsize=16)
    plt.xlabel("Time Steps")
    plt.ylabel("Video1 Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss", color="red")
    plt.plot(range(len(test_losses)), test_losses, label="Test RMSE", color="blue")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Test R2 Score: {test_r2:.4f}")
    print(f"Best Test RMSE: {best_rmse:.6f}")


# Run the training
train_and_plot_Seq2Seq(n_epochs=1500)
