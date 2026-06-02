import torch
from models.action_slot import ACTION_SLOT

## Este archivo lo use para debuggear


# USAR RTX
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Instanciar el modelo con los mismos argumentos del entrenamiento
class Args:
    # Backbone opciones 'r50' | 'i3d' | 'x3d' | 'slowfast'
    backbone      = 'x3d'

    # Dataset 'taco' | 'oats' | 'nuscenes'
    dataset       = 'taco'

    # preentrenamiento base (esto afecta a los num_slots en el datasets)
    pretrain      = 'taco'

    # dim de los slots (el canal oculto del SlotAttention y convs)
    channel       = 256 # <---- IMPORTANTE esa cantidad justa

    # tamano de la secuencia de frames que procesa el backbone
    seq_len       = 16

    # allocated_slot=True -> usa Allocated_Head; False -> usa Head normal
    allocated_slot = True

    # bg_slot=True -> esto agrega 1 slot extra de background
    bg_slot        = True


def main():
    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] corriendo en: {device}")
    if device.type == 'cuda':
        print(f"[INFO] gpu activa: {torch.cuda.get_device_name(0)}")

    args = Args()

    # num_actor_class:         Para TACO el modelo usa 64 clases de actor
    # num_ego_class  = 0   # 0 = sin rama ego; cambia a p.ej. 4 si usas OATS/nuScenes
    num_ego_class = 4
    num_actor_class = 64  # clases de actor del dataset TACO

    print("[INFO] levantando ACTION_SLOT...")


    # -----------------------------------------------------------------------
    # INICIALIZACION

    model = ACTION_SLOT(
        args=args,
        num_ego_class=num_ego_class,
        num_actor_class=num_actor_class,
    )
    model = model.to(device)
    model.eval()
    print("[INFO] modelo en modo eval (BN y Dropout fijos)")


    
    # -----------------------------------------------------------------------
    # cARGAR PESOS
    checkpoint = torch.load('weights/taco_action_slot_best_model.pth', map_location=device)
    model.load_state_dict(checkpoint, strict=True) # Si da error aquí, es que faltan argumentos en __init__
    print("[INFO] Pesos cargados con éxito.")

    # el input correcto para deberia ser I3D
    # [Batch, Channels, Time, Height, Width]
    seq_len = 16

    # dummy_video = torch.randn(1, 3, 16, 224, 224).to(device)
    # dummy_video = [torch.randn(1, 3, 224, 224).to(device) for _ in range(seq_len)]
    dummy_video = [torch.randn(1, 3, 256, 768).to(device) for _ in range(seq_len)]

    model.eval()
    with torch.no_grad():
        # output = model(dummy_video)
        # print("Inferencia exitosa:", output.shape)

        ego_out, actor_out, attn_masks = model(dummy_video)
        
        print("\n=== Inferencia Exitosa ===")
        print("Ego output shape (Acciones ego):   ", ego_out.shape)   # deber ser [1, 4]
        print("Actor output shape (Acciones actor):", actor_out.shape) # deber ser  [1, 64, 64] 
        print("Attention masks shape:             ", attn_masks.shape) # deber ser  [1, 16, 65, 8, 24]

if __name__ == '__main__':
    main()