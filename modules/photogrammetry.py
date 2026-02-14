# modules/photogrammetry.py
"""
Модуль многоракурсной фотограмметрии для объединения данных из нескольких фото.
Компенсирует погрешности AR-датчиков через триангуляцию.
"""
import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
from typing import List, Dict, Tuple
try:
    from scipy.spatial import distance
except Exception:  # pragma: no cover
    distance = None

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    DBSCAN = None

class MultiViewFusion:
    """
    Объединяет данные из нескольких ракурсов в единую точную 3D-модель.
    """
    
    def __init__(self):
        self.point_clouds = []  # Список облаков точек из разных ракурсов
        self.camera_poses = []  # Позиции камеры для каждого снимка
    
    def add_view(self, ar_points: List[Dict], camera_pose: Dict):
        """
        Добавляет новый ракурс.
        
        Args:
            ar_points: облако точек от ARCore [{x, y, z}, ...]
            camera_pose: позиция и ориентация камеры {
                "position": [x, y, z],
                "rotation": [qx, qy, qz, qw]  # кватернион
            }
        """
        self.point_clouds.append(ar_points)
        self.camera_poses.append(camera_pose)
    
    def fuse_point_clouds(self, merge_threshold: float = 0.05) -> List[Dict]:
        """
        Объединяет облака точек в единую модель, убирая дубликаты.
        
        Args:
            merge_threshold: порог расстояния для слияния точек (метры)
        
        Returns:
            Объединенное облако точек с усредненными координатами
        """
        if len(self.point_clouds) < 2:
            # Недостаточно данных для фьюжна
            return self.point_clouds[0] if self.point_clouds else []
        
        # 1. Собираем все точки в один массив
        all_points = []
        for cloud in self.point_clouds:
            for point in cloud:
                all_points.append([point['x'], point['y'], point['z']])
        
        if not all_points:
            return []
        
        all_points = np.array(all_points)
        
        # 2. Кластеризация через DBSCAN (группируем близкие точки)
        if DBSCAN is not None:
            clustering = DBSCAN(eps=merge_threshold, min_samples=1).fit(all_points)
            labels = clustering.labels_
        else:
            # Fallback: каждая точка отдельным кластером
            labels = np.arange(len(all_points))
        
        # 3. Усредняем точки в каждом кластере
        unique_labels = set(labels)
        fused_points = []
        
        for label in unique_labels:
            if label == -1:  # Шумовые точки
                continue
            
            cluster_mask = labels == label
            cluster_points = all_points[cluster_mask]
            
            # Усредняем координаты
            avg_point = np.mean(cluster_points, axis=0)
            
            # Оценка качества (чем больше точек в кластере, тем надежнее)
            confidence = min(1.0, len(cluster_points) / len(self.point_clouds))
            
            fused_points.append({
                "x": float(avg_point[0]),
                "y": float(avg_point[1]),
                "z": float(avg_point[2]),
                "confidence": round(confidence, 3),
                "source_count": len(cluster_points)
            })
        
        return fused_points
    
    def detect_ar_drift(self) -> Dict:
        """
        Обнаруживает дрейф AR-координат между ракурсами.
        
        Returns:
            {
                "drift_detected": bool,
                "max_drift_m": float,
                "quality_score": float (0-1)
            }
        """
        if len(self.point_clouds) < 2:
            return {"drift_detected": False, "max_drift_m": 0, "quality_score": 1.0}
        
        # Находим общие точки между ракурсами (близкие по координатам)
        drifts = []
        
        for i in range(len(self.point_clouds) - 1):
            cloud1 = np.array([[p['x'], p['y'], p['z']] for p in self.point_clouds[i]])
            cloud2 = np.array([[p['x'], p['y'], p['z']] for p in self.point_clouds[i + 1]])
            
            if len(cloud1) == 0 or len(cloud2) == 0:
                continue
            
            # Для каждой точки в cloud1 находим ближайшую в cloud2
            if distance is not None:
                distances = distance.cdist(cloud1, cloud2, 'euclidean')
            else:
                distances = np.linalg.norm(cloud1[:, None, :] - cloud2[None, :, :], axis=2)
            min_distances = np.min(distances, axis=1)
            
            # Медианное расстояние = мера дрейфа
            median_drift = np.median(min_distances)
            drifts.append(median_drift)
        
        max_drift = max(drifts) if drifts else 0
        
        # Quality score: чем меньше дрейф, тем лучше
        quality_score = max(0, 1.0 - (max_drift / 0.5))  # 0.5м - критичный дрейф
        
        return {
            "drift_detected": max_drift > 0.1,  # 10 см порог
            "max_drift_m": round(max_drift, 3),
            "quality_score": round(quality_score, 3)
        }
    
    def suggest_next_view(self, current_coverage: List[Dict]) -> Dict:
        """
        Предлагает оптимальную позицию для следующего снимка.
        
        Args:
            current_coverage: текущее облако точек
        
        Returns:
            Рекомендация где сфотографировать {
                "direction": str,
                "distance": float,
                "reason": str
            }
        """
        if len(current_coverage) < 10:
            return {
                "direction": "Любое",
                "distance": 3.0,
                "reason": "Недостаточно данных. Сделайте несколько общих снимков."
            }
        
        # Анализируем распределение точек
        points = np.array([[p['x'], p['y'], p['z']] for p in current_coverage])
        
        # Находим центр масс
        centroid = np.mean(points, axis=0)
        
        # Анализируем покрытие по осям
        x_range = np.ptp(points[:, 0])  # размах по X
        y_range = np.ptp(points[:, 1])  # размах по Y
        z_range = np.ptp(points[:, 2])  # размах по Z
        
        # Определяем слабо покрытую область
        if y_range < x_range * 0.5:
            return {
                "direction": "Справа/Слева (по оси Y)",
                "distance": 3.0,
                "reason": "Недостаточно данных по глубине. Переместитесь вбок."
            }
        elif z_range < x_range * 0.3:
            return {
                "direction": "Снизу или Сверху",
                "distance": 3.0,
                "reason": "Мало данных по высоте. Сфотографируйте с другого уровня."
            }
        else:
            return {
                "direction": "Сзади (противоположная сторона)",
                "distance": 3.0,
                "reason": "Получите задний ракурс для полноты модели."
            }


class FeatureMatching:
    """
    Сопоставление характерных точек между фотографиями для калибровки.
    """
    
    def __init__(self):
        # Используем SIFT или ORB для поиска особых точек
        self.detector = None
        if cv2 is not None:
            try:
                self.detector = cv2.SIFT_create()
            except Exception:
                self.detector = cv2.ORB_create()
    
    def match_images(self, img1_bytes: bytes, img2_bytes: bytes) -> Dict:
        """
        Находит соответствие между двумя изображениями.
        
        Returns:
            {
                "matches_count": int,
                "confidence": float,
                "rotation": float,  # угол поворота между снимками
                "translation": [dx, dy]  # смещение
            }
        """
        if cv2 is None or self.detector is None:
            return {"matches_count": 0, "confidence": 0, "rotation_deg": 0, "translation_px": [0, 0]}

        # Декодируем изображения
        nparr1 = np.frombuffer(img1_bytes, np.uint8)
        nparr2 = np.frombuffer(img2_bytes, np.uint8)
        
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return {"matches_count": 0, "confidence": 0, "rotation": 0, "translation": [0, 0]}
        
        # Находим ключевые точки и дескрипторы
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return {"matches_count": 0, "confidence": 0, "rotation": 0, "translation": [0, 0]}
        
        # Сопоставление через FLANN или BruteForce
        if hasattr(cv2, "SIFT") and isinstance(self.detector, cv2.SIFT):
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Берем топ-N лучших совпадений
        good_matches = matches[:min(50, len(matches))]
        
        # Confidence на основе качества совпадений
        if len(good_matches) > 0:
            avg_distance = np.mean([m.distance for m in good_matches])
            confidence = max(0, 1.0 - (avg_distance / 100))  # Нормализация
        else:
            confidence = 0
        
        # Оценка поворота и смещения (упрощенно через центры масс ключевых точек)
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Извлекаем угол поворота
                rotation = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
                translation = [float(M[0, 2]), float(M[1, 2])]
            else:
                rotation = 0
                translation = [0, 0]
        else:
            rotation = 0
            translation = [0, 0]
        
        return {
            "matches_count": len(good_matches),
            "confidence": round(confidence, 3),
            "rotation_deg": round(rotation, 2),
            "translation_px": [round(translation[0], 1), round(translation[1], 1)]
        }


class PhotogrammetrySystem:
    """
    Полная система многоракурсной фотограмметрии.
    Объединяет fusion, matching, и валидацию.
    """
    
    def __init__(self):
        self.fusion = MultiViewFusion()
        self.matcher = FeatureMatching()
        self.views = []  # История снимков
    
    def add_photo_view(self, image_bytes: bytes, ar_points: List[Dict], 
                      camera_pose: Dict) -> Dict:
        """
        Добавляет новое фото с AR-данными.
        
        Returns:
            {
                "view_id": int,
                "quality": float,
                "matches_with_previous": int,
                "suggestions": [...]
            }
        """
        view_id = len(self.views)
        
        # Добавляем в fusion
        self.fusion.add_view(ar_points, camera_pose)
        
        # Если есть предыдущее фото, сопоставляем
        matches_count = 0
        if view_id > 0:
            prev_image = self.views[-1]['image_bytes']
            match_result = self.matcher.match_images(prev_image, image_bytes)
            matches_count = match_result['matches_count']
        
        # Сохраняем view
        self.views.append({
            "view_id": view_id,
            "image_bytes": image_bytes,
            "ar_points": ar_points,
            "camera_pose": camera_pose
        })
        
        # Оценка качества
        quality = self._assess_view_quality(ar_points, matches_count)
        
        # Предложения для следующего снимка
        current_cloud = self.fusion.fuse_point_clouds()
        next_view = self.fusion.suggest_next_view(current_cloud)
        
        return {
            "view_id": view_id,
            "quality_score": quality,
            "matches_with_previous": matches_count,
            "total_views": len(self.views),
            "next_view_suggestion": next_view
        }
    
    def get_final_model(self) -> Dict:
        """
        Возвращает финальную объединенную модель.
        
        Returns:
            {
                "points": [...],
                "quality_metrics": {...},
                "is_ready": bool
            }
        """
        fused_points = self.fusion.fuse_point_clouds()
        drift_analysis = self.fusion.detect_ar_drift()
        
        is_ready = (
            len(fused_points) >= 20 and 
            len(self.views) >= 2 and
            drift_analysis['quality_score'] > 0.7
        )
        
        return {
            "points": fused_points,
            "quality_metrics": {
                "total_points": len(fused_points),
                "total_views": len(self.views),
                "drift_quality": drift_analysis['quality_score'],
                "max_drift_m": drift_analysis['max_drift_m']
            },
            "is_ready_for_design": is_ready,
            "warnings": self._generate_warnings(fused_points, drift_analysis)
        }
    
    def _assess_view_quality(self, ar_points: List[Dict], matches_count: int) -> float:
        """Оценка качества отдельного снимка"""
        # Факторы: количество точек, количество совпадений
        points_score = min(1.0, len(ar_points) / 30)
        matches_score = min(1.0, matches_count / 50) if matches_count > 0 else 0.5
        
        return (points_score + matches_score) / 2
    
    def _generate_warnings(self, points: List[Dict], drift_analysis: Dict) -> List[str]:
        """Генерирует предупреждения"""
        warnings = []
        
        if len(points) < 20:
            warnings.append("Мало опорных точек. Добавьте еще 2-3 снимка.")
        
        if drift_analysis['drift_detected']:
            warnings.append(
                f"Обнаружен дрейф AR ({drift_analysis['max_drift_m']*100:.0f} см). "
                f"Перезапустите AR-сессию."
            )
        
        if drift_analysis['quality_score'] < 0.5:
            warnings.append("Низкое качество данных. Улучшите освещение и стабильность камеры.")
        
        return warnings