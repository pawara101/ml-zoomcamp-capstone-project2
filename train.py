import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("data/Air_Quality_Data.csv")

class AqiCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU()
        )

        # After convs:
        # Input length = 6 (corrected from 7)
        # 6 → 4 → 2
        self.fc = nn.Sequential(
            nn.Linear(32 * 3, 64), # Corrected from 32 * 3 to 32 * 2
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, 6)
        x = x.unsqueeze(1)        # → (batch_size, 1, 6)
        x = self.conv(x)          # → (batch_size, 32, 2)
        x = x.view(x.size(0), -1) # flatten: (batch_size, 32 * 2) = (batch_size, 64)
        return self.fc(x)


def preprocess_train(df):
  df['date'] = pd.to_datetime(df['date'])
  df['year'] = df['date'].dt.year
  df['month'] = df['date'].dt.month
  df['day'] = df['date'].dt.day

  label_encoder = LabelEncoder()
  df['city'] = label_encoder.fit_transform(df['city'])

  feature_cols = ['city','co','o3','no2','so2','pm10','pm25','aqi']
  data_for_aqi_score = data.copy()[feature_cols]

  X = data_for_aqi_score.drop('aqi', axis=1)
  y = data_for_aqi_score['aqi']

  X = X.to_numpy()
  y = y.to_numpy()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  return X_train, X_test, y_train, y_test, label_encoder

def to_tensor(data_array):
    tensor = torch.tensor(data_array, dtype=torch.float32)
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    return tensor


def training_loop(n_epochs, optimiser, model, loss_fn, X_train,  X_val, y_train, y_val):
    for epoch in range(1, n_epochs + 1):
        # Ensure inputs and targets are on the correct device
        X_train_device = X_train.to(device)
        y_train_device = y_train.to(device)
        X_val_device = X_val.to(device)
        y_val_device = y_val.to(device)

        output_train = model(X_train_device) # forwards pass
        loss_train = loss_fn(output_train, y_train_device.unsqueeze(1)) # calculate loss, ensure y_train is 2D
        output_val = model(X_val_device)
        loss_val = loss_fn(output_val, y_val_device.unsqueeze(1)) # ensure y_val is 2D

        optimiser.zero_grad() # set gradients to zero
        loss_train.backward() # backwards pass
        optimiser.step() # update model parameters
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"+
                  f" Validation loss {loss_val.item():.4f}")

X_train, X_test, y_train, y_test, city_label_encoder = preprocess_train(data)

X_train, X_test, y_train, y_test = map(
    to_tensor, (X_train, X_test, y_train, y_test))


model = AqiCNN().to(device)
optimiser = optim.Adam(model.parameters(), lr=0.001)

training_loop(
    n_epochs = 100,
    optimiser = optimiser,
    model = model,
    loss_fn = nn.MSELoss(),
    X_train = X_train[0:1000],
    X_val = X_test[0:200],
    y_train = y_train[0:1000],
    y_val = y_test[0:200])


# torch.save(model.state_dict(), 'model.pth')