from __future__ import annotations
"""
Deep Learning forecasting models (LSTM, GRU) using PyTorch.
Designed for CPU execution with lightweight architectures.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — deep learning models unavailable")

from sklearn.preprocessing import StandardScaler
from app.config import settings

if HAS_TORCH:
    class TimeSeriesDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 30):
            self.X = X
            self.y = y
            self.sequence_length = sequence_length
            
            # Handle case where data is shorter than sequence length
            if len(X) <= sequence_length:
                self.valid_indices = [0]
                # Pad X and y to at least sequence_length
                pad_len = sequence_length - len(X) + 1
                self.X = np.pad(X, ((pad_len, 0), (0, 0)), mode='edge')
                self.y = np.pad(y, (pad_len, 0), mode='edge')
            else:
                self.valid_indices = range(len(X) - sequence_length)

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            start_idx = self.valid_indices[idx]
            end_idx = start_idx + self.sequence_length
            
            x_seq = torch.FloatTensor(self.X[start_idx:end_idx])
            y_target = torch.FloatTensor([self.y[end_idx]])
            return x_seq, y_target

    class LSTMForecaster(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # Disable dropout if num_layers == 1
            lstm_dropout = dropout if num_layers > 1 else 0.0
            
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True,
                dropout=lstm_dropout
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x shape: (batch_size, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Take the output of the last time step
            last_time_step = lstm_out[:, -1, :]
            out = self.dropout(last_time_step)
            return self.fc(out)

    class GRUForecaster(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            gru_dropout = dropout if num_layers > 1 else 0.0
            
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=gru_dropout
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            gru_out, _ = self.gru(x)
            last_time_step = gru_out[:, -1, :]
            out = self.dropout(last_time_step)
            return self.fc(out)


def train_deep_model(
    model_class, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    sequence_length: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 10
) -> dict | None:
    if not HAS_TORCH:
        return None

    try:
        # 1. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        y_train_np = y_train.values
        y_test_np = y_test.values
        
        # 2. Create Datasets and DataLoaders
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_np, sequence_length)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test_np, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 3. Initialize Model
        input_size = X_train.shape[1]
        model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # CPU only
        device = torch.device('cpu')
        model = model.to(device)
        
        # 4. Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 5. Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
                
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Evaluate
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
                    
            avg_val_loss = epoch_val_loss / len(test_loader)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        # 6. Generate final predictions on test set using best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        model.eval()
        predictions = []
        
        # For prediction, we need to handle the whole test set
        # Since TimeSeriesDataset drops the first sequence_length items to form windows,
        # we pad X_test_scaled at the beginning using the end of X_train_scaled
        if len(X_train_scaled) >= sequence_length:
            X_combined = np.vstack([X_train_scaled[-sequence_length:], X_test_scaled])
        else:
            pad_len = sequence_length - len(X_train_scaled)
            X_pad = np.pad(X_train_scaled, ((pad_len, 0), (0, 0)), mode='edge')
            X_combined = np.vstack([X_pad, X_test_scaled])
            
        # Create sliding windows for prediction
        windows = []
        for i in range(len(X_test_scaled)):
            windows.append(X_combined[i:i+sequence_length])
            
        windows_tensor = torch.FloatTensor(np.array(windows)).to(device)
        
        with torch.no_grad():
            preds = model(windows_tensor).cpu().numpy().flatten()
            
        # Clip to >= 0
        preds = np.clip(preds, 0, None)
        
        config = {
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate
        }
        
        return {
            'predictions': preds,
            'model_state': best_model_state,
            'scaler': scaler,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Deep learning training failed: {e}")
        return None


def forecast_future_deep(
    model_state_dict, 
    model_class, 
    scaler, 
    history_features: pd.DataFrame, 
    horizon: int, 
    config: dict
) -> np.ndarray | None:
    if not HAS_TORCH:
        return None
        
    try:
        sequence_length = config.get("sequence_length", 30)
        input_size = history_features.shape[1]
        
        model = model_class(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2)
        )
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # Get the last sequence_length features
        X_hist = history_features.values
        if len(X_hist) < sequence_length:
            pad_len = sequence_length - len(X_hist)
            X_hist = np.pad(X_hist, ((pad_len, 0), (0, 0)), mode='edge')
        
        current_window = X_hist[-sequence_length:].copy()
        future_preds = []
        
        # Import here to avoid circular imports if any
        from app.services.features import create_time_series_features
        from app.services.forecasting import _build_future_feature_row
        
        feature_cols = list(history_features.columns)
        
        # We need a mock history dataframe to build future rows
        # This will be updated iteratively
        dates = pd.date_range(end=pd.Timestamp.today(), periods=len(current_window))
        working_history = pd.DataFrame({
            "Date": dates,
            "Demand": current_window[:, 0]  # Assuming Demand is first or we just need some values for lags
        })
        
        # If Demand is not first, try to find it or just use zeros
        # Since we just need it for building the next row's lags
        
        last_date = pd.Timestamp.today()
        
        for i in range(horizon):
            # Scale current window
            window_scaled = scaler.transform(current_window)
            window_tensor = torch.FloatTensor(window_scaled).unsqueeze(0) # Add batch dim
            
            with torch.no_grad():
                pred = model(window_tensor).item()
                pred = max(0, pred)
                
            future_preds.append(pred)
            
            # Build next row features using the prediction
            next_date = last_date + pd.Timedelta(days=i+1)
            
            # Add prediction to working history
            working_history = pd.concat([
                working_history, 
                pd.DataFrame({"Date": [next_date], "Demand": [pred]})
            ], ignore_index=True)
            
            # Generate next row features
            next_row_df = _build_future_feature_row(working_history, next_date, feature_cols)
            next_row = next_row_df.values[0]
            
            # Shift window and append new row
            current_window = np.vstack([current_window[1:], next_row])
            
        return np.array(future_preds)
        
    except Exception as e:
        logger.error(f"Deep learning future forecast failed: {e}")
        return None
