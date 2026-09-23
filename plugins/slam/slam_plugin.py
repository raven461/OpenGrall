#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/slam/__init__.py — SLAM-ПЛАГИН                                        ║
║                                                                              ║
║ ИНСТРУМЕНТЫ:                                                                 ║
║   • move_to(target) — переместиться к объекту в комнате (диван, стол)        ║
║   • move_to_room(target) — переместиться в другую комнату                    ║
║   • get_room_map() — получить карту текущей комнаты (список объектов)        ║
║   • explore_room(room_name) — автономное исследование комнаты                ║
║   • read_map_metadata(room_name) — чтение метаданных карты                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from plugins.plugin import Plugin
from orchestration.tool import Tool

# Импорты SLAM-модулей
from sensors.lidar_navigator import LidarNavigator
from sensors.lidar_mapper import LidarMapper
from sensors.map_storage import MapStorage

logger = logging.getLogger(__name__)


# ================================================================
# ИНСТРУМЕНТЫ SLAM
# ================================================================

class MoveToTool(Tool):
    """
    Инструмент: move_to — перемещение к объекту внутри комнаты.
    Использует A* для обхода препятствий.
    """
    name = "move_to"
    description = "Переместиться к объекту в текущей комнате. Пример: move_to(target='диван')"
    latency = 5.0

    def __init__(self, agent, slam_plugin):
        self.agent = agent
        self.slam = slam_plugin

    async def forward(self, **kwargs) -> str:
        target = kwargs.get('target')
        if not target:
            return "Error: missing 'target' parameter. Example: move_to(target='диван')"

        if not self.slam.navigator or not self.slam.navigator.current_room:
            return "No map loaded. Cannot navigate."

        # Получаем объекты из текущей комнаты
        objects = []
        if self.slam.map_storage and self.slam.navigator.current_location:
            metadata = self.slam.map_storage.get_room_metadata(
                self.slam.navigator.current_location,
                self.slam.navigator.current_room
            )
            objects = metadata.get("objects", [])

        if not objects and self.slam.mapper:
            objects = self.slam.mapper.identified_objects

        # Ищем объект
        target_obj = None
        for obj in objects:
            label = obj.get("label", obj.get("name", ""))
            if label.lower() == target.lower():
                target_obj = obj
                break

        if target_obj is None:
            return f"Object '{target}' not found in current room '{self.slam.navigator.current_room}'"

        obj_x = target_obj.get("x", 0)
        obj_y = target_obj.get("y", 0)

        if obj_x == 0 and obj_y == 0:
            return f"Object '{target}' has no coordinates"

        # Выполняем навигацию через A*
        success, msg = await self.slam.navigator.execute_move_to(obj_x, obj_y)

        if success:
            self.slam.navigator.update_pose(obj_x, obj_y, self.slam.navigator.current_pose[2])
            return f"Arrived at '{target}'. {msg}"
        else:
            return f"Failed to reach '{target}': {msg}"


class MoveToRoomTool(Tool):
    """
    Инструмент: move_to_room — перемещение между комнатами.
    """
    name = "move_to_room"
    description = "Переместиться в другую комнату. Пример: move_to_room(target='кухня')"
    latency = 5.0

    def __init__(self, agent, slam_plugin):
        self.agent = agent
        self.slam = slam_plugin

    async def forward(self, **kwargs) -> str:
        target = kwargs.get('target')
        if not target:
            return "Error: missing 'target' parameter. Example: move_to_room(target='кухня')"

        if not self.slam.navigator or not self.slam.navigator.current_room:
            return "No map loaded. Cannot navigate."

        # Пробуем как комнату
        rooms = []
        if self.slam.navigator.current_location:
            rooms = self.slam.map_storage.list_rooms(self.slam.navigator.current_location) if self.slam.map_storage else []

        if target in rooms:
            success, msg = await self.slam.navigator.go_to_room_bfs(target, self.slam.mapper.room_graph)
            return msg
        else:
            return f"Room '{target}' not found. Available rooms: {', '.join(rooms)}"


class GetRoomMapTool(Tool):
    """
    Инструмент: get_room_map — возвращает JSON с объектами в текущей комнате.
    """
    name = "get_room_map"
    description = "Получить карту текущей комнаты: список объектов"
    latency = 0.1

    def __init__(self, agent, slam_plugin):
        self.agent = agent
        self.slam = slam_plugin

    async def forward(self, **kwargs) -> str:
        nav = self.slam.navigator

        if not nav.current_room:
            return json.dumps({"error": "Неизвестна текущая комната"}, ensure_ascii=False)

        # Получаем объекты
        objects = []
        if self.slam.map_storage and nav.current_location:
            metadata = self.slam.map_storage.get_room_metadata(nav.current_location, nav.current_room)
            objects = metadata.get("objects", [])
        elif self.slam.mapper and self.slam.mapper.current_room == nav.current_room:
            objects = self.slam.mapper.identified_objects

        # Формируем JSON только с именами (LLM не нужны координаты)
        result_objects = []
        for obj in objects:
            label = obj.get("label", obj.get("name", "неизвестно"))
            if label != "room_graph":
                result_objects.append({"label": label})

        return json.dumps({
            "room": nav.current_room,
            "objects": result_objects
        }, ensure_ascii=False, indent=2)


class ExploreRoomTool(Tool):
    """
    Инструмент: explore_room — автономное исследование комнаты.
    """
    name = "explore_room"
    description = "Автономное исследование комнаты: построение карты, поиск дверей, идентификация объектов"
    latency = 60.0

    def __init__(self, agent, slam_plugin):
        self.agent = agent
        self.plugin = slam_plugin

    async def forward(self, **kwargs) -> str:
        room_name = kwargs.get('room_name', 'unknown_room')
        max_scans = kwargs.get('max_scans', 200)
        min_coverage = kwargs.get('min_coverage', 0.85)
        enable_vlm = kwargs.get('enable_vlm_verification', True)

        if not self.plugin.mapper:
            return "Error: LidarMapper not initialized"

        self.plugin.mapper.agent = self.agent

        logger.info(f"🗺️ Starting exploration of room: {room_name}")

        result = await self.plugin.mapper.explore_room(
            room_name=room_name,
            max_scans=max_scans,
            min_coverage=min_coverage,
            max_idle_scans=kwargs.get('max_idle_scans', 20),
            enable_vlm_verification=enable_vlm
        )

        if result.get("success"):
            return (f"Room '{room_name}' explored successfully. "
                    f"Coverage: {result['coverage']:.1%}, "
                    f"Doors found: {result['doors_confirmed']}, "
                    f"Objects identified: {result['objects_identified']}")
        else:
            return f"Room '{room_name}' exploration incomplete. Coverage: {result['coverage']:.1%}"


class ReadMapMetadataTool(Tool):
    """
    Инструмент: read_map_metadata — чтение метаданных карты.
    """
    name = "read_map_metadata"
    description = "Прочитать метаданные карты: двери, объекты, граф помещений"
    latency = 0.1

    def __init__(self, agent, slam_plugin):
        self.agent = agent
        self.plugin = slam_plugin

    async def forward(self, **kwargs) -> str:
        room_name = kwargs.get('room_name')

        if not room_name and self.plugin.navigator:
            room_name = self.plugin.navigator.current_room

        if not room_name:
            return "Error: missing 'room_name' parameter or no current room"

        if self.plugin.mapper and self.plugin.mapper.current_room == room_name:
            metadata = {
                "room": room_name,
                "coverage": self.plugin.mapper.get_coverage(),
                "doors": self.plugin.mapper.detected_doors,
                "objects": self.plugin.mapper.identified_objects,
                "room_graph": self.plugin.mapper.build_room_graph()
            }
            return json.dumps(metadata, indent=2, ensure_ascii=False)

        if self.plugin.map_storage:
            for loc_id in self.plugin.map_storage.list_locations():
                if room_name in self.plugin.map_storage.list_rooms(loc_id):
                    doors_data = self.plugin.map_storage.get_doors(loc_id)
                    if room_name in doors_data:
                        metadata = {
                            "room": room_name,
                            "location": loc_id,
                            "doors": doors_data[room_name].get("doors", [])
                        }
                        return json.dumps(metadata, indent=2, ensure_ascii=False)

        return f"No metadata found for room '{room_name}'"


# ================================================================
# ПЛАГИН SLAM
# ================================================================

class SLAMPlugin(Plugin):
    name = "slam"
    version = "1.0"
    dependencies = ["sensors"]

    def __init__(self, agent):
        super().__init__(agent)

        self.mapper: Optional[LidarMapper] = None
        self.navigator: Optional[LidarNavigator] = None
        self.map_storage: Optional[MapStorage] = None
        self._lidar_processor = None

        self._publish_task: asyncio.Task = None
        self._scan_task: asyncio.Task = None

    async def on_load(self):
        """Инициализация SLAM-модулей"""
        # Получаем LidarProcessor из сенсорного плагина
        sensors_plugin = self.agent.get_plugin("sensors")
        if sensors_plugin and hasattr(sensors_plugin, 'get_lidar_processor'):
            self._lidar_processor = sensors_plugin.get_lidar_processor()
            logger.info("🔗 LidarProcessor получен из SensorsPlugin")

        # Конфигурация
        config = {
            'robot_length': self.agent.get_config('ROBOT_LENGTH', 0.51),
            'robot_width': self.agent.get_config('ROBOT_WIDTH', 0.32),
            'robot_height': self.agent.get_config('ROBOT_HEIGHT', 0.37),
            'lidar_offset_x': self.agent.get_config('LIDAR_OFFSET_X', 0.0),
            'lidar_offset_y': self.agent.get_config('LIDAR_OFFSET_Y', 0.0),
            'lidar_height': self.agent.get_config('LIDAR_HEIGHT', 0.37),
            'map_resolution': self.agent.get_config('MAP_RESOLUTION', 0.05),
            'map_size': self.agent.get_config('MAP_SIZE', 20.0),
        }

        self.map_storage = MapStorage()
        self.mapper = LidarMapper(config)
        self.navigator = LidarNavigator(config)

        # Устанавливаем ссылку на агента
        self.navigator.set_agent(self.agent)

        logger.info("🗺️ SLAM-модули инициализированы")

        await self._autoload_map()
        self._publish_task = asyncio.create_task(self._publish_slam_loop())
        self._scan_task = asyncio.create_task(self._process_scan_loop())

        await super().on_load()

    async def on_unload(self):
        if self._publish_task:
            self._publish_task.cancel()
        if self._scan_task:
            self._scan_task.cancel()

        try:
            if self._publish_task:
                await self._publish_task
            if self._scan_task:
                await self._scan_task
        except asyncio.CancelledError:
            pass

        await super().on_unload()

    def get_tools(self) -> List[Tool]:
        """Возвращает SLAM-инструменты"""
        tools = [
            ExploreRoomTool(self.agent, self),
            ReadMapMetadataTool(self.agent, self),
            GetRoomMapTool(self.agent, self),
        ]

        if self.navigator and self.navigator.current_room:
            tools.append(MoveToTool(self.agent, self))
            tools.append(MoveToRoomTool(self.agent, self))
            logger.info(f"🗺️ Навигационные инструменты активированы (комната: {self.navigator.current_room})")

        return tools

    async def on_message(self, msg_type: str, data: Dict[str, Any]):
        pass

    # ==================== ПОЛУЧЕНИЕ СЫРЫХ ТОЧЕК ====================

    async def _get_raw_lidar_scan(self) -> Optional[List[Tuple[float, float]]]:
        if self._lidar_processor is None:
            return None

        try:
            raw_data = self._lidar_processor.get_raw_scan()
            if raw_data and "points" in raw_data:
                points = raw_data["points"]
                if isinstance(points, list) and len(points) > 0:
                    if isinstance(points[0], dict):
                        return [(p.get("x", 0), p.get("y", 0)) for p in points]
                    elif isinstance(points[0], (list, tuple)):
                        return [(p[0], p[1]) for p in points]
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка получения сырого скана: {e}")
            return None

    # ==================== ФОНОВЫЕ ЗАДАЧИ ====================

    async def _process_scan_loop(self):
        """Фоновый цикл обработки сканов (10 Гц)"""
        while self._loaded and self.agent.is_running():
            if self.navigator and self.navigator.current_room:
                raw_points = await self._get_raw_lidar_scan()
                if raw_points and len(raw_points) > 10:
                    scan_points = np.array(raw_points)
                    pose = await self._get_robot_pose()

                    if self.mapper:
                        self.mapper.add_scan(scan_points, pose)
                    self.navigator.localize_on_map(scan_points)

            await asyncio.sleep(0.1)

    async def _get_robot_pose(self) -> Tuple[float, float, float]:
        sensor_memory = self.agent.get_sensor_memory()
        if sensor_memory:
            odom = sensor_memory.get("odometry")
            if odom and hasattr(odom, 'data'):
                return (
                    odom.data.get("x", 0.0),
                    odom.data.get("y", 0.0),
                    odom.data.get("theta", 0.0)
                )
        return (0.0, 0.0, 0.0)

    async def _publish_slam_loop(self):
        """Публикует SLAM-строку в SensorMemory (2 Гц)"""
        while self._loaded and self.agent.is_running():
            if self.navigator and self.navigator.current_room:
                slam_summary = self.navigator.get_slam_summary()

                sensor_memory = self.agent.get_sensor_memory()
                if sensor_memory:
                    sensor_memory.update(
                        source="slam",
                        data={
                            "summary": slam_summary,
                            "room": self.navigator.current_room,
                            "pose": self.navigator.current_pose
                        },
                        weight=0.90,
                        meta={"capability": "slam.localization"}
                    )

            await asyncio.sleep(0.5)

    async def _autoload_map(self):
        """Автоматическое определение помещения по скану"""
        if not self.navigator or not self.map_storage:
            return

        raw_points = await self._get_raw_lidar_scan()
        if raw_points is None or len(raw_points) < 10:
            logger.info("📍 Не удалось получить скан для автозагрузки")
            return

        scan_points = np.array(raw_points)
        logger.info(f"📍 Поиск помещения по скану ({len(scan_points)} точек)...")

        best_location = None
        best_room = None
        best_pose = None
        best_score = float('inf')

        for location_id in self.map_storage.list_locations():
            for room in self.map_storage.list_rooms(location_id):
                map_image = self.map_storage.load_map(location_id, room)
                if map_image is None:
                    continue

                self.navigator.room_map = map_image
                self.navigator.current_location = location_id
                self.navigator.current_room = room

                pose = self.navigator.localize_on_map(scan_points)

                if pose is not None:
                    score = self._evaluate_scan_match(scan_points, pose, map_image)
                    if score < best_score:
                        best_score = score
                        best_location = location_id
                        best_room = room
                        best_pose = pose

        self.navigator.room_map = None
        self.navigator.current_room = None
        self.navigator.current_location = None

        if best_location and best_room and best_pose:
            map_image = self.map_storage.load_map(best_location, best_room)
            doors_data = self.map_storage.get_doors(best_location)
            room_doors = doors_data.get(best_room, {}).get('doors', [])

            self.navigator.set_room(best_location, best_room, map_image, room_doors)
            self.navigator.update_pose(*best_pose)

            logger.info(f"📍 Автоопределение: {best_location}/{best_room}")
        else:
            logger.info("📍 Помещение не опознано")

    def _evaluate_scan_match(self, scan_points, pose, map_image, max_search=10):
        if map_image is None or len(scan_points) < 10:
            return float('inf')

        x, y, theta = pose
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        resolution = 0.05
        map_size = map_image.shape[0] * resolution
        origin = -map_size / 2

        total_error = 0
        count = 0

        for lx, ly in scan_points[:50]:
            wx = x + lx * cos_t - ly * sin_t
            wy = y + lx * sin_t + ly * cos_t

            px = int((wx - origin) / resolution)
            py = int((wy - origin) / resolution)

            if 0 <= px < map_image.shape[1] and 0 <= py < map_image.shape[0]:
                if map_image[py, px] == 100:
                    total_error += 0
                    count += 1
                elif map_image[py, px] == -1 or map_image[py, px] == 0:
                    min_dist = self._min_dist_to_wall(px, py, map_image, max_search)
                    total_error += min_dist
                    count += 1

        return total_error / count if count > 0 else float('inf')

    def _min_dist_to_wall(self, px, py, map_image, max_search=10):
        for radius in range(1, max_search + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < map_image.shape[1] and 0 <= ny < map_image.shape[0]:
                        if map_image[ny, nx] == 100:
                            return radius
        return max_search

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "name": self.name,
            "version": self.version,
            "lidar_processor_linked": self._lidar_processor is not None
        }

        if self.navigator:
            stats["navigator"] = self.navigator.get_stats()
        if self.mapper:
            stats["mapper"] = self.mapper.get_stats()
        if self.map_storage:
            stats["storage"] = self.map_storage.get_stats()

        return stats