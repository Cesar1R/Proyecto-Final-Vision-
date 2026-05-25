# main.py
import argparse
import torch
from config import Config
from model import MultiLabelAtomicActivityModel
from dataloader import get_dataloaders, TACOAtomicActivityDataset
from trainer import Trainer

def run_training():
    print(f"Iniciando Entrenamiento del Reto T3 en el dispositivo: {Config.DEVICE}")
    train_loader = get_dataloaders()
    
    model = MultiLabelAtomicActivityModel(num_classes=Config.NUM_CLASSES, pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    trainer = Trainer(model, train_loader, optimizer, Config.DEVICE)
    
    best_loss = float('inf')
    for epoch in range(Config.NUM_EPOCHS):
        loss = trainer.train_epoch(epoch)
        print(f"--- Resumen Época {epoch+1} | Loss Promedio: {loss:.4f} ---")
        
        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(epoch)

def run_inference(video_path):
    print(f"Ejecutando Inferencia Atómica sobre: {video_path}")
    device = Config.DEVICE
    
    # Inicializar modelo y cargar pesos
    model = MultiLabelAtomicActivityModel(num_classes=Config.NUM_CLASSES, pretrained=False)
    checkpoint_path = f"{Config.CHECKPOINT_DIR}/best_t3_model.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pesos cargados correctamente.")
    except Exception as e:
        print(f"Error cargando pesos en {checkpoint_path}. Asegúrate de entrenar primero. Detalle: {e}")
        return
        
    model = model.to(device)
    model.eval()
    
    # Cargar y preprocesar el clip usando el dataset helper directamente
    dataset_helper = TACOAtomicActivityDataset(Config.DATA_DIR)
    video_tensor = dataset_helper._load_video_frames(video_path).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(video_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
    labels_map = Config.get_activity_labels()
    
    print("\n--- Actividades Atómicas Detectadas (ROAD T3) ---")
    detected = False
    for idx, prob in enumerate(probabilities):
        if prob >= Config.INFERENCE_THRESHOLD:
            print(f" Actividad: [{labels_map[idx]}] -> Confianza: {prob:.2%}")
            detected = True
            
    if not detected:
        # Fallback de máxima verosimilitud en caso de confianza baja generalizada
        max_idx = probabilities.argmax()
        print(f"Ninguna superó el umbral. Predicción más probable: [{labels_map[max_idx]}] con {probabilities[max_idx]:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ROAD-ECCV2024 Reto T3")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"], help="Modo de ejecución")
    parser.add_argument("--video_path", type=str, default="", help="Ruta al video .mp4 para inferencia")
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training()
    elif args.mode == "infer":
        if not args.video_path:
            print("Error: Se requiere definir --video_path para el modo inferencia.")
        else:
            run_inference(args.video_path)