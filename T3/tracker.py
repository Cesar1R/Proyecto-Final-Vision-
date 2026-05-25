# tracker.py
import json
import os
import numpy as np

class ActivityTracker:
    def __init__(self):
        self.active_tracks = {}

    def load_scenario_bboxes(self, bbox_json_path):
        """Carga las anotaciones crudas de bounding boxes del escenario."""
        if not os.path.exists(bbox_json_path):
            return {}
        with open(bbox_json_path, 'r') as f:
            return json.load(f)

    def extract_agent_trajectories(self, bbox_data):
        """
        Procesa el archivo bbox.json para extraer trayectorias continuas por Agent ID.
        Retorna un diccionario indexado por track_id con sus respectivas posiciones viales.
        """
        trajectories = {}
        # Estructura de TACO bbox.json
        for frame_idx, frame_data in bbox_data.get("frames", {}).items():
            for obj in frame_data.get("objects", []):
                obj_id = obj.get("id")
                obj_type = obj.get("type", "car")
                # Extraemos la posición relativa en base al box o datos 3D si están disponibles
                box = obj.get("box_2d", [0, 0, 0, 0])
                center_x = (box[0] + box[2]) / 2.0
                center_y = (box[1] + box[3]) / 2.0
                
                if obj_id not in trajectories:
                    trajectories[obj_id] = {"type": obj_type, "coords": []}
                
                trajectories[obj_id]["coords"].append((center_x, center_y))
                
        return trajectories