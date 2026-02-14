# modules/geometry.py
import trimesh
import numpy as np

class WorldModel:
    def __init__(self):
        self.scene = trimesh.Scene()

    def add_pipe(self, start_point, end_point, radius):
        """
        Создает 3D цилиндр (трубу) в мире
        """
        # Математика создания цилиндра между двумя точками
        # (Упрощенно используем встроенный примитив)
        height = np.linalg.norm(np.array(end_point) - np.array(start_point))
        pipe = trimesh.creation.cylinder(radius=radius, height=height)
        
        # Здесь должна быть матрица поворота и смещения (transformation matrix)
        # чтобы поставить трубу на место. Пока упростим.
        pipe.visual.face_colors = [100, 100, 100, 255] # Серый цвет
        self.scene.add_geometry(pipe)
        return "pipe_added"

    def check_collision(self, scaffold_beam_coords):
        """
        Проверяет, не врезается ли балка лесов в трубу
        """
        # Логика Ray Casting (луч)
        # Если луч пересекает геометрию сцены — значит коллизия
        return False # Пока заглушка