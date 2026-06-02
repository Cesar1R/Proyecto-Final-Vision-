import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from models.action_slot import ACTION_SLOT

class Args:
    backbone       = 'x3d'
    dataset        = 'taco'
    pretrain       = 'taco'
    channel        = 256  
    seq_len        = 16   
    allocated_slot = True
    bg_slot        = True

def load_and_sample_video(video_path, target_seq_len=16):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"el archivo de video no existe en: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0 or orig_fps == 0:
        raise ValueError(f"no se puede leer el arcchivo {video_path}.")

    # tomar los FPS
    # T_salida / (N_total / FPS_original) = FPS_ajustado
    duration = total_frames / orig_fps
    output_fps = target_seq_len / duration

    frame_indices = np.linspace(0, total_frames - 1, num=target_seq_len, dtype=int)
    
    frames = []
    current_idx = 0
    
    for idx in frame_indices:
        if idx != current_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            current_idx = idx
            
        ret, frame = cap.read()
        if not ret:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((256, 768, 3), dtype=np.uint8))
        else:
            frames.append(frame)
            current_idx += 1
            
    cap.release()
    return frames, output_fps

def preprocess_frames(frames, device, target_height=256, target_width=768):
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor_list = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame_tensor = transform_pipeline(frame_resized).unsqueeze(0).to(device)
        tensor_list.append(frame_tensor)
        
    return tensor_list

def main(NAME):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Dispositivo activo: {device}")
    
    args = Args()
    num_ego_class = 4
    num_actor_class = 64
    
    model = ACTION_SLOT(
        args=args,
        num_ego_class=num_ego_class,
        num_actor_class=num_actor_class,
    ).to(device)
    
    checkpoint_path = 'weights/taco_action_slot_best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("[INFO] Estado del modelo configurado en EVAL y pesos cargados.")
    
    #####
    # NAME = "pruebaTaco4"
    video_input_path = f"PruebasPersonales/{NAME}.mp4"
    print(f"[INFO] Procesando video: {video_input_path}...")
    
    raw_frames, output_fps = load_and_sample_video(video_input_path, target_seq_len=args.seq_len)
    dummy_video = preprocess_frames(raw_frames, device, target_height=256, target_width=768)
    
    with torch.no_grad():
        ego_out, actor_out, attn_masks = model(dummy_video)
        
        ego_probs = torch.sigmoid(ego_out).cpu().numpy()[0]
        actor_probs = torch.sigmoid(actor_out).cpu().numpy()[0]
        
    print("\n" + "="*30 + " RESULTADOS DE INFERENCIA " + "="*30)
    active_actors = np.where(actor_probs > 0.5)[0]
    active_ego = np.where(ego_probs > 0.5)[0]
    print(f"Clases Ego activas: {active_ego} con probs {np.round(ego_probs, 4)}")
    print(f"Clases Actor activas: {active_actors}")

    # -----------------------------------------------------------------------
    # VISUALIZACIÓN Y GENERACION DE VIDEO DE OUTPUT
    # -----------------------------------------------------------------------
    output_video_path = 'PruebasPersonalesResult/' + NAME + '_result.mp4'
    orig_h, orig_w, _ = raw_frames[0].shape
    
    # el contenedor de video con el códec estandarizado mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (orig_w, orig_h))
    
    # tomamos las mascaras: Forma final [16, 65, 8, 24]
    masks_numpy = attn_masks[0].cpu().numpy()
    
    print(f"[INFO] Renderizando video de salida en: {output_video_path}...")
    
    for t in range(args.seq_len):
        frame_visual = raw_frames[t].copy()
        
        # si se detectaron actores activos, superponer la mascara del primero de ellos
        if len(active_actors) > 0:
            target_slot = active_actors[0] # Tomamos la clase con confianza > 0.5
            
            # extraemos el  mapa de atencion espacial de resolución estática (8, 24)
            attn_map = masks_numpy[t, target_slot, :, :]
            
            # normalizacion Min-Max para escalar los valores de activación al rango dinámico [0, 255]
            attn_map_normalized = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_byte = (attn_map_normalized * 255).astype(np.uint8)
            
            # interpolacion bilineal para proyectar la resolución latente a la geometría nativa del video
            attn_resized = cv2.resize(attn_byte, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Mapeo de densidad a espacio de color pseudo-RGB (Jet Color Map)
            heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
            
            # combinacion lineal ponderada (Alpha Blending) para transparencia
            # I_out = alpha * I_original + beta * I_heatmap + gamma
            frame_visual = cv2.addWeighted(frame_visual, 0.65, heatmap, 0.35, 0)
            
            # dibujamos la etiqueta de la accion del Actor
            cv2.putText(frame_visual, f"Actor Slot {target_slot} Attn", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # superpocion de textos analíticos en la región superior izquierda
        cv2.putText(frame_visual, f"Ego Activa: {active_ego} (Prob: {ego_probs[active_ego[0]]:.2f})", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        

        # escribimos  el frame en el archivo
        video_writer.write(frame_visual)
        
    video_writer.release()
    print(f"[ÉXITO] Video guardado correctamente. Ejecuta 'vlc {output_video_path}' o ábrelo para inspección visual.")

if __name__ == '__main__':
    main("pruebaGuanajuato7")