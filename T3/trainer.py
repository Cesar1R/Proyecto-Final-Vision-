# trainer.py
import os
import torch
import torch.nn as nn
from config import Config

class Trainer:
    def __init__(self, model, train_loader, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        # Usamos reducción por media aritmética estable para multietiqueta
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)
        
        for batch_idx, (videos, targets) in enumerate(self.train_loader):
            videos = videos.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(videos)
            loss = self.criterion(logits, targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                print(f"Época [{epoch+1}/{Config.NUM_EPOCHS}] | Batch [{batch_idx+1}/{total_batches}] | Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / total_batches
        return epoch_loss

    def save_checkpoint(self, epoch, filename="best_t3_model.pth"):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"==> Checkpoint guardado con éxito en: {checkpoint_path}")