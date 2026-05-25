# topology_utils.py
import numpy as np

class TopologyMapper:
    def __init__(self):
        # Definición geométrica simplificada de zonas viales relativas a la intersección
        self.intersection_radius = 25.0  # metros
        self.crosswalk_min = 25.0
        self.crosswalk_max = 30.0

    def get_region_by_coordinates(self, x, y, ego_speed=0.0):
        """
        Calcula la región vial basada en las coordenadas 2D/3D relativas extraídas 
        del JSON del simulador o estimadas del tracking.
        """
        distance = np.sqrt(x**2 + y**2)
        
        if distance <= self.intersection_radius:
            return "intersection"
        elif self.crosswalk_min < distance <= self.crosswalk_max:
            return "crosswalk"
        elif abs(x) > 12.0 and distance > self.crosswalk_max:
            return "sidewalk"
        else:
            return "lanes"

    def match_trajectory_to_activity(self, track_history, agent_type):
        """
        Analiza el histórico de posiciones de un agente para determinar la transición 
        vial (region_start -> region_end).
        """
        if len(track_history) < 2:
            return "lanes->lanes"
            
        start_pt = track_history[0]  # (x, y)
        end_pt = track_history[-1]
        
        start_region = self.get_region_by_coordinates(start_pt[0], start_pt[1])
        end_region = self.get_region_by_coordinates(end_pt[0], end_pt[1])
        
        return f"{start_region}->{end_region}:{agent_type}"