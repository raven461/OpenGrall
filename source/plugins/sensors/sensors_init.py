#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/sensors/__init__.py — ПЛАГИН СЕНСОРОВ                                ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Обрабатывает сырые данные лидара в структурированный формат              ║
║   • 8 секторов, кластеры объектов, скорость и направление движения          ║
║   • Публикация обработанных данных в SensorMemory                           ║
║   • Предоставляет LidarProcessor для SLAM-плагина (сырые точки)             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

from plugins.plugin import Plugin

# Импортируем правильный LidarProcessor (с поддержкой сырых точек)
from sensors.lidar_processor import LidarProcessor

logger = logging.getLogger(__name__)


class SensorsPlugin(Plugin):
    """
    Плагин сенсоров — обрабатывает лидар и публикует данные в SensorMemory.
    Также предоставляет LidarProcessor для SLAM-плагина.
    """
    
    name = "sensors"
    version = "1.0"
    dependencies = []
    
    def __init__(self, agent):
        super().__init__(agent)
        
        self.lidar_processor: Optional[LidarProcessor] = None
        
        # Фоновая задача опроса лидара
        self._lidar_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def on_load(self):
        """Инициализация процессора лидара"""
        
        # Параметры из конфига
        cluster_threshold = self.agent.get_config('LIDAR_CLUSTER_THRESHOLD', 0.3)
        min_cluster_points = self.agent.get_config('LIDAR_MIN_CLUSTER_POINTS', 5)
        max_tracking_age = self.agent.get_config('LIDAR_MAX_TRACKING_AGE', 2.0)
        
        self.lidar_processor = LidarProcessor(
            source_id="lidar",
            cluster_threshold=cluster_threshold,
            min_cluster_points=min_cluster_points,
            max_tracking_age=max_tracking_age
        )
        
        logger.info("✅ SensorsPlugin loaded")
        await super().on_load()
    
    async def on_unload(self):
        """Остановка"""
        self._running = False
        if self._lidar_task:
            self._lidar_task.cancel()
        
        await super().on_unload()
    
    # ==================== API ДЛЯ SLAM-ПЛАГИНА ====================
    
    def get_lidar_processor(self) -> Optional[LidarProcessor]:
        """
        Возвращает LidarProcessor для SLAM-плагина.
        SLAM-плагин использует этот метод для получения сырых точек лидара.
        """
        return self.lidar_processor
    
    # ==================== ОСТАЛЬНЫЕ МЕТОДЫ ====================
    
    def start_lidar_polling(self, interval: float = 0.05):
        """Запускает фоновый опрос лидара."""
        if self._lidar_task and not self._lidar_task.done():
            return
        
        self._running = True
        self._lidar_task = asyncio.create_task(self._poll_lidar_loop(interval))
        logger.info(f"🔄 LiDAR polling started (interval={interval}s)")
    
    async def _poll_lidar_loop(self, interval: float):
        """Фоновый цикл опроса лидара."""
        while self._running and self.agent.is_running():
            try:
                network = self.agent.get_plugin("network")
                if network and network.is_connected():
                    await network.send({"type": "get_lidar"})
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LiDAR polling error: {e}")
                await asyncio.sleep(interval)
    
    async def process_lidar_data(self, ranges: np.ndarray, 
                                  angle_min: float = -np.pi,
                                  angle_max: float = np.pi,
                                  angle_increment: float = np.pi/360,
                                  timestamp: float = None):
        """Обрабатывает сырые данные лидара (вызывается из on_message)."""
        if self.lidar_processor is None:
            return
        
        if ranges is None or len(ranges) == 0:
            return
        
        # Получаем сырые точки через LidarProcessor
        raw_scan = self.lidar_processor.get_raw_scan(
            ranges=ranges,
            angle_min=angle_min,
            angle_max=angle_max,
            angle_increment=angle_increment,
            timestamp=timestamp
        )
        
        if raw_scan and "points" in raw_scan:
            points = raw_scan["points"]
            # Конвертируем в numpy для обработки
            if isinstance(points, list) and len(points) > 0:
                if isinstance(points[0], dict):
                    scan_points = np.array([[p.get("x", 0), p.get("y", 0)] for p in points])
                else:
                    scan_points = np.array(points)
                
                # Обрабатываем скан (8 секторов + кластеры)
                processed = self.lidar_processor.process_raw_points(scan_points, timestamp=timestamp)
                
                if processed:
                    # Публикуем в SensorMemory (только 8 секторов и кластеры, НЕ сырые точки)
                    sensor_memory = self.agent.get_sensor_memory()
                    if sensor_memory:
                        sensor_memory.update(
                            source="lidar",
                            data={
                                "distances": {
                                    "front": processed.front_distance,
                                    "left": processed.left_distance,
                                    "right": processed.right_distance,
                                    "back": processed.back_distance
                                },
                                "sectors": processed.sectors,
                                "obstacles": {
                                    "front": processed.obstacle_front,
                                    "left": processed.obstacle_left,
                                    "right": processed.obstacle_right,
                                    "back": processed.obstacle_back
                                },
                                "clusters": [c.to_dict() for c in processed.clusters],
                                "moving_objects": processed.moving_objects,
                                "quality": processed.data_quality,
                                "points_count": processed.points_count
                            },
                            weight=processed.weight,
                            meta={"capability": "sensor.lidar.scan"}
                        )
    
    async def on_message(self, msg_type: str, data: Dict[str, Any]):
        """Обработка входящих сообщений от сервера."""
        
        # Данные лидара
        if msg_type == 'lidar_data' or data.get('capability') == 'sensor.lidar.scan':
            lidar_data = data.get('data', {})
            ranges = lidar_data.get('ranges')
            angle_min = lidar_data.get('angle_min', -np.pi)
            angle_max = lidar_data.get('angle_max', np.pi)
            angle_increment = lidar_data.get('angle_increment', np.pi/360)
            timestamp = data.get('timestamp')
            
            if ranges is not None:
                await self.process_lidar_data(
                    ranges=np.array(ranges),
                    angle_min=angle_min,
                    angle_max=angle_max,
                    angle_increment=angle_increment,
                    timestamp=timestamp
                )
        
        # Одометрия
        elif msg_type == 'odometry' or data.get('capability') == 'sensor.odometry':
            odom_data = data.get('data', {})
            sensor_memory = self.agent.get_sensor_memory()
            if sensor_memory:
                sensor_memory.update(
                    source="odometry",
                    data=odom_data,
                    weight=0.65,
                    meta={"capability": "sensor.odometry"}
                )
    
    def get_lidar_summary(self) -> Optional[str]:
        """Возвращает компактную сводку с лидара для LLM."""
        if not self.lidar_processor:
            return None
        
        sensor_memory = self.agent.get_sensor_memory()
        if sensor_memory:
            lidar_data = sensor_memory.get("lidar")
            if lidar_data:
                sectors = lidar_data.data.get("sectors", {})
                if sectors:
                    return ", ".join([f"{s}={d:.1f}м" for s, d in sectors.items()])
        
        return None
    
    def get_odometry_summary(self) -> Optional[str]:
        """Возвращает сводку одометрии."""
        sensor_memory = self.agent.get_sensor_memory()
        if sensor_memory:
            odom = sensor_memory.get("odometry")
            if odom:
                speed_left = odom.data.get("speed_left", 0)
                speed_right = odom.data.get("speed_right", 0)
                heading = odom.data.get("heading", 0)
                return f"vл={speed_left:.1f}, vп={speed_right:.1f}, курс={heading:.0f}°"
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика плагина."""
        stats = {
            "name": self.name,
            "version": self.version,
        }
        
        if self.lidar_processor:
            stats["lidar"] = self.lidar_processor.get_stats()
        
        return stats