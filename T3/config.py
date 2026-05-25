# config.py
import os
import torch

class Config:
    # --- Rutas del Dataset ---
    DATA_DIR = "./ap_Town01"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    
    # --- Parámetros del Modelo ---
    NUM_CLASSES = 64
    FRAME_SIZE = (224, 224)
    CLIP_DURATION = 16  # Número de frames por clip de entrada
    SAMPLE_STRIDE = 2   # Muestreo de frames para cubrir más contexto temporal
    
    # --- Parámetros de Entrenamiento ---
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    # --- Umbral de Inferencia ---
    INFERENCE_THRESHOLD = 0.5

    # --- Inicialización de Clases (64 Categorías Atómicas) ---
    # Formato: (region_start -> region_end: agent_type)
    REGIONS = ["lanes", "intersection", "crosswalk", "sidewalk"]
    AGENTS = ["car", "pedestrian", "cyclist", "truck"]
    
    @classmethod
    def get_activity_labels(cls):
        """Genera o retorna la lista estática mapeada de las 64 combinaciones requeridas."""
        labels = []
        for start in cls.REGIONS:
            for end in cls.REGIONS:
                for agent in cls.AGENTS:
                    labels.append(f"{start}->{end}:{agent}")
        # Rellenar con placeholders controlados si la combinación teórica no suma exactamente 64
        while len(labels) < cls.NUM_CLASSES:
            labels.append(f"generic_action_{len(labels)}")
        return labels[:cls.NUM_CLASSES]

# Crear directorios necesarios
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)