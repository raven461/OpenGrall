#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ sensors/map_storage.py — ХРАНИЛИЩЕ КАРТ И МЕТАДАННЫХ                        ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Хранит карты комнат (PNG) и двери (JSON)                                ║
║   • Хранит глобальный индекс помещений с GPS/высотой                         ║
║   • Определяет помещение по GPS (примерно)                                   ║
║   • Подтверждает помещение сверкой скана лидара с картами                    ║
║                                                                              ║
║ ЛОГИКА ОПРЕДЕЛЕНИЯ ПОМЕЩЕНИЯ:                                                ║
║   1. GPS + высота → список кандидатов (может быть несколько)                 ║
║   2. Для каждого кандидата: scan matching по всем комнатам                   ║
║   3. Если хотя бы одна комната совпала → помещение подтверждено              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import time
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MapStorage:
    """Хранилище карт, дверей и метаданных помещений."""
    
    def __init__(self, maps_dir: str = None):
        if maps_dir is None:
            maps_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "maps")
        self.maps_dir = os.path.abspath(maps_dir)
        self.metadata_path = os.path.join(self.maps_dir, "metadata.json")
        os.makedirs(self.maps_dir, exist_ok=True)
        self._metadata = None
    
    # ==================== МЕТАДАННЫЕ ====================
    
    def _load_metadata(self) -> Dict:
        """Загружает metadata.json (кэширует в памяти)"""
        if self._metadata is None:
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._metadata = {"locations": {}}
        return self._metadata
    
    def _save_metadata(self):
        """Сохраняет metadata.json"""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        logger.debug("💾 metadata.json сохранён")
    
    # ==================== ПОМЕЩЕНИЯ (LOCATIONS) ====================
    
    def add_location(self, location_id: str, gps: Tuple[float, float], 
                     altitude: float, map_dir: str = None):
        """
        Регистрирует новое помещение.
        
        Args:
            location_id: уникальный ID («home», «office»)
            gps: (latitude, longitude)
            altitude: высота в метрах
            map_dir: имя директории для карт (по умолчанию = location_id)
        """
        meta = self._load_metadata()
        
        if map_dir is None:
            map_dir = location_id.replace(" ", "_").lower()
        
        meta["locations"][location_id] = {
            "gps": {"lat": gps[0], "lon": gps[1]},
            "altitude": altitude,
            "map_dir": map_dir,
            "rooms": [],
            "default_room": None,
            "created_at": time.time()
        }
        
        # Создаём директорию для карт
        room_dir = os.path.join(self.maps_dir, map_dir)
        os.makedirs(room_dir, exist_ok=True)
        
        self._save_metadata()
        logger.info(f"📍 Добавлено помещение: {location_id} ({gps[0]}, {gps[1]}, {altitude}м)")
    
    def get_location_by_gps(self, lat: float, lon: float, altitude: float = None,
                           max_distance_km: float = 0.1) -> List[str]:
        """
        Находит помещения-кандидаты по GPS.
        Возвращает список ID (может быть несколько при одинаковых координатах).
        
        Args:
            lat, lon: текущие координаты
            altitude: текущая высота (опционально, для фильтрации по этажу)
            max_distance_km: максимальное расстояние (по умолчанию 100 м)
        
        Returns:
            Список location_id кандидатов, отсортированный по релевантности
        """
        meta = self._load_metadata()
        candidates = []
        
        for loc_id, loc_data in meta.get("locations", {}).items():
            loc_gps = loc_data.get("gps")
            if not loc_gps:
                continue
            
            # Расстояние по GPS
            distance = self._haversine(lat, lon, loc_gps["lat"], loc_gps["lon"])
            
            if distance > max_distance_km:
                continue
            
            # Фильтр по высоте (если указана)
            score = 1.0 / (distance + 0.001)  # чем ближе, тем выше скор
            
            if altitude is not None and loc_data.get("altitude") is not None:
                alt_diff = abs(altitude - loc_data["altitude"])
                if alt_diff > 5.0:  # больше 5 метров — разные этажи
                    continue
                # Штраф за разницу в высоте
                score *= 1.0 / (alt_diff + 1.0)
            
            candidates.append((loc_id, score))
        
        # Сортируем по скору (выше = лучше совпадение)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]
    
    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Расстояние между двумя GPS-точками в километрах (формула гаверсинусов)"""
        R = 6371.0  # радиус Земли в км
        
        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    # ==================== КОМНАТЫ (ROOMS) ====================
    
    def save_room(self, location_id: str, room_name: str, 
                  map_image: np.ndarray, doors: List[Dict]):
        """
        Сохраняет карту комнаты и её двери.
        
        Args:
            location_id: ID помещения
            room_name: название комнаты
            map_image: occupancy grid (numpy array)
            doors: список дверей [{"x": ..., "y": ..., "width": ..., "direction": ...}, ...]
        """
        import cv2
        
        meta = self._load_metadata()
        loc_data = meta["locations"].get(location_id)
        if not loc_data:
            logger.error(f"Помещение {location_id} не найдено")
            return
        
        map_dir = loc_data["map_dir"]
        room_dir = os.path.join(self.maps_dir, map_dir)
        os.makedirs(room_dir, exist_ok=True)
        
        # Сохраняем PNG
        png_path = os.path.join(room_dir, f"{room_name}.png")
        
        # Конвертируем occupancy grid в изображение
        image = np.zeros((map_image.shape[0], map_image.shape[1]), dtype=np.uint8)
        image[map_image == -1] = 0      # free space — чёрный
        image[map_image == 0] = 128     # unknown — серый
        image[map_image == 100] = 255   # occupied — белый
        cv2.imwrite(png_path, image)
        
        # Обновляем двери
        doors_path = os.path.join(room_dir, "doors.json")
        all_doors = {}
        if os.path.exists(doors_path):
            with open(doors_path, 'r') as f:
                all_doors = json.load(f)
        
        # Добавляем двери для этой комнаты (с указанием соседа, если ещё нет)
        if room_name not in all_doors:
            all_doors[room_name] = {"doors": []}
        
        for door in doors:
            all_doors[room_name]["doors"].append({
                "x": door["x"],
                "y": door["y"],
                "width": door["width"],
                "direction": door.get("direction", "unknown"),
                "to": door.get("to", "unknown")
            })
        
        with open(doors_path, 'w') as f:
            json.dump(all_doors, f, indent=2)
        
        # Обновляем список комнат
        if room_name not in loc_data["rooms"]:
            loc_data["rooms"].append(room_name)
        if loc_data["default_room"] is None:
            loc_data["default_room"] = room_name
        
        self._save_metadata()
        logger.info(f"💾 Комната '{room_name}' сохранена в {location_id} "
                   f"(карта: {png_path}, дверей: {len(doors)})")
    
    def load_map(self, location_id: str, room_name: str) -> Optional[np.ndarray]:
        """Загружает карту комнаты как numpy array"""
        import cv2
        
        meta = self._load_metadata()
        loc_data = meta["locations"].get(location_id)
        if not loc_data:
            return None
        
        png_path = os.path.join(self.maps_dir, loc_data["map_dir"], f"{room_name}.png")
        
        if not os.path.exists(png_path):
            return None
        
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        
        # Конвертируем обратно в occupancy grid
        grid = np.zeros_like(image, dtype=np.int8)
        grid[image == 0] = -1     # free space
        grid[image == 128] = 0    # unknown
        grid[image == 255] = 100  # occupied
        
        return grid
    
    def get_doors(self, location_id: str) -> Dict:
        """Загружает двери для всех комнат помещения"""
        meta = self._load_metadata()
        loc_data = meta["locations"].get(location_id)
        if not loc_data:
            return {}
        
        doors_path = os.path.join(self.maps_dir, loc_data["map_dir"], "doors.json")
        try:
            with open(doors_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def list_rooms(self, location_id: str) -> List[str]:
        """Список комнат в помещении"""
        meta = self._load_metadata()
        loc_data = meta["locations"].get(location_id)
        if not loc_data:
            return []
        return loc_data.get("rooms", [])
    
    def list_locations(self) -> List[str]:
        """Список всех помещений"""
        meta = self._load_metadata()
        return list(meta.get("locations", {}).keys())
    
    # ==================== ПОДТВЕРЖДЕНИЕ ПОМЕЩЕНИЯ ====================
    
    def confirm_location(self, lat: float, lon: float, altitude: float = None,
                        current_scan_points: np.ndarray = None,
                        scan_matcher=None) -> Optional[str]:
        """
        Определяет помещение по GPS + сверка с картами лидара.
        
        Алгоритм:
        1. GPS → список кандидатов
        2. Для каждого кандидата пробуем scan matching по комнатам
        3. Если хотя бы одна комната совпала → возвращаем location_id
        4. Если совпадений нет → возвращаем лучшего кандидата по GPS
        
        Args:
            lat, lon: GPS координаты
            altitude: высота (опционально)
            current_scan_points: точки текущего скана лидара
            scan_matcher: функция scan_matching(scan_points, map_image) -> (x, y, theta) или None
        
        Returns:
            location_id или None
        """
        candidates = self.get_location_by_gps(lat, lon, altitude)
        
        if not candidates:
            logger.warning("GPS не совпал ни с одним помещением")
            return None
        
        # Если scan matching доступен — подтверждаем лидаром
        if current_scan_points is not None and scan_matcher is not None:
            for loc_id in candidates:
                rooms = self.list_rooms(loc_id)
                
                for room in rooms:
                    map_image = self.load_map(loc_id, room)
                    if map_image is None:
                        continue
                    
                    # Пробуем scan matching
                    pose = scan_matcher(current_scan_points, map_image)
                    if pose is not None:
                        logger.info(f"✅ Помещение подтверждено лидаром: {loc_id} (комната: {room})")
                        return loc_id
            
            logger.warning("Scan matching не подтвердил ни одно помещение")
        
        # Fallback: возвращаем лучшего кандидата по GPS
        best = candidates[0]
        logger.info(f"📍 Помещение определено по GPS: {best} (без подтверждения лидаром)")
        return best
    
    # ==================== СТАТИСТИКА ====================
    
    def get_stats(self) -> Dict:
        """Статистика хранилища"""
        meta = self._load_metadata()
        total_rooms = sum(len(loc.get("rooms", [])) for loc in meta.get("locations", {}).values())
        
        return {
            "locations_count": len(meta.get("locations", {})),
            "total_rooms": total_rooms,
            "maps_dir": self.maps_dir
        }