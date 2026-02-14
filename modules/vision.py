# modules/vision.py
from ultralytics import YOLO
import numpy as np
import cv2

class Eyes:
    def __init__(self):
        # Загружаем модель. В будущем заменим 'yolov8n.pt' на твою обученную 'scaffold-pipes.pt'
        self.model = YOLO('yolov8n.pt') 

    def detect_obstacles(self, image_bytes: bytes):
        """
        Принимает фото, находит на нем трубы, стены, людей.
        Возвращает список объектов с их координатами (bounding box).
        """
        # Конвертация байтов в картинку для OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Запуск AI
        results = self.model(img)
        
        obstacles = []
        for result in results:
            for box in result.boxes:
                # Получаем координаты и класс объекта
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = self.model.names[cls]

                # Нас интересуют только определенные объекты (пример)
                if name in ['person', 'chair', 'bottle']: # Пока стандартные, потом заменим на 'pipe', 'beam'
                    obstacles.append({
                        "type": name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                        # В будущем здесь будет привязка к 3D координатам через ARCore данные
                    })
        
        return obstacles