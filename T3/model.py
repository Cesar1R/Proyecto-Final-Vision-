# model.py
import os
import torch
import torch.nn as nn
import torchvision.models.video as video_models

class MultiLabelAtomicActivityModel(nn.Module):
    def __init__(self, num_classes=64, pretrained=True):
        super(MultiLabelAtomicActivityModel, self).__init__()
        
        #Crear el backbone
        self.backbone = video_models.r3d_18(weights=None)
        
        #Cargar los pesos locales 
        local_weights_path = "./r3d_18-b3b3357e.pth"
        if os.path.exists(local_weights_path):
            print(f"Cargando pesos pre-entrenados locales desde {local_weights_path}...")
            state_dict = torch.load(local_weights_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict)
        else:
            print(f"ADVERTENCIA: No se encontró '{local_weights_path}'. "
                    "El modelo se inicializará desde cero, lo que afectará el rendimiento.")
        
        #Reemplazamos la capa de clasificación completamente
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
        # Cabeza fully-connected dedicada a Multi-label Classification
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits