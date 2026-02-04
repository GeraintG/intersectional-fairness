import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .loss import FairLoss
from ..utils.logger import setup_logger
import os
import numpy as np

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger = setup_logger(config["general"]["output_dir"])
        self.criterion = FairLoss(lambda_df=config["fairness"].get("lambda", 0.5))
        self.best_val_loss = float("inf")
        self.checkpoint_path = os.path.join(config["general"]["output_dir"], "best_model.pt")

    def train(self, X_train, y_train, sensitive_train, X_val, y_val, sensitive_val):
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(self.device)
        sensitive_train_tensor = torch.tensor(sensitive_train.values, dtype=torch.int64).to(self.device)

        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(self.device)
        sensitive_val_tensor = torch.tensor(sensitive_val.values, dtype=torch.int64).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, sensitive_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, sensitive_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["training"]["batch_size"], shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["training"]["learning_rate"])

        log_file = os.path.join(self.config["general"]["output_dir"], "training_log.csv")
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")

        for epoch in range(self.config["training"]["epochs"]):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch, sensitive_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch, sensitive_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch, sensitive_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch, sensitive_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Save best model
            if val_loss < self.best_val_loss:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.best_val_loss = val_loss

            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                             f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f}\n")
