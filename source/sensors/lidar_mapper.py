#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ sensors/lidar_mapper.py — КАРТОГРАФ ПОМЕЩЕНИЙ ПО ЛИДАРУ                       ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Строит occupancy grid карту комнаты (PNG)                                ║
║   • Ищет дверные проёмы по лидарным данным (gap detection)                   ║
║   • Автономное исследование комнаты (frontier-based)                         ║
║   • Фильтрация динамических объектов (временная)                             ║
║   • VLM-верификация дверей (через agent.tools: ask_vlm)                      ║
║   • Опрос человека о назначении дверей (через agent.tools: ask_human)        ║
║   • VLM-идентификация крупных объектов — мебель (через ask_vlm)              ║
║   • Построение графа помещений (room_graph)                                  ║
║   • Сохраняет карты (PNG) и метаданные (JSON)                                ║
║                                                                              ║
║ АРХИТЕКТУРНОЕ ПРАВИЛО:                                                       ║
║   • Сырые точки лидара → LidarProcessor.get_raw_scan() → LidarMapper        ║
║   • 8 секторов + кластеры → SensorMemory → ContextBuilder → LLM             ║
║   • Одометрия (x, y, θ) → SensorMemory → могут использовать все             ║
║   • Сырые точки НИКОГДА не попадают в SensorMemory (забьют промпт LLM)      ║
║                                                                              ║
║ ЗАВИСИМОСТИ:                                                                 ║
║   • numpy, cv2, scipy                                                        ║
║   • config.py — габариты робота, смещение лидара, высота                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import numpy as np
import cv2
import json
import os
import time
import logging
from typing import Optional, List, Dict, Tuple, Callable

logger = logging.getLogger(__name__)


class LidarMapper:
    """
    Картограф. Строит occupancy grid, находит двери,
    фильтрует динамику и верифицирует объекты через VLM.
    
    Важно: сырые точки лидара получает напрямую от LidarProcessor,
    а не из SensorMemory (где только 8 секторов для LLM).
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: словарь с параметрами (из config.py)
                - robot_length, robot_width, robot_height (метры)
                - lidar_offset_x, lidar_offset_y (смещение от центра, метры)
                - lidar_height (высота установки, метры)
                - map_resolution (метры/пиксель, по умолчанию 0.05)
                - map_size (метры, по умолчанию 20.0)
                - temporal_persist_threshold (сколько сканов для уверенности, по умолчанию 3)
                - temporal_decay_interval (интервал затухания в сканах, по умолчанию 5)
        """
        self.config = config or {}

        # Габариты робота
        self.robot_length = self.config.get('robot_length', 0.51)
        self.robot_width = self.config.get('robot_width', 0.32)
        self.robot_height = self.config.get('robot_height', 0.37)

        # Смещение лидара
        self.lidar_offset_x = self.config.get('lidar_offset_x', 0.0)
        self.lidar_offset_y = self.config.get('lidar_offset_y', 0.0)
        self.lidar_height = self.config.get('lidar_height', 0.37)

        # Параметры карты
        self.resolution = self.config.get('map_resolution', 0.05)
        self.map_size = self.config.get('map_size', 20.0)
        self.grid_size = int(self.map_size / self.resolution)
        self.origin = (-self.map_size / 2, -self.map_size / 2)

        # Основная карта (occupancy grid)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Временной слой для фильтрации динамики
        self.temporal_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.persist_threshold = self.config.get('temporal_persist_threshold', 3)
        self.decay_interval = self.config.get('temporal_decay_interval', 5)
        self.scan_counter = 0

        # Найденные двери
        self.detected_doors: List[Dict] = []

        # Идентифицированные объекты (мебель)
        self.identified_objects: List[Dict] = []

        # ГРАФ ПОМЕЩЕНИЙ
        self.room_graph: Dict[str, Dict] = {}
        self.current_room: str = "unknown"

        # Ссылка на агента (устанавливается при создании ExploreRoomTool)
        self.agent = None

        logger.info(f"✅ LidarMapper инициализирован "
                    f"(resolution={self.resolution}м/px, "
                    f"temporal_threshold={self.persist_threshold})")

    # ==================== БАЗОВЫЕ МЕТОДЫ ====================

    def reset(self):
        """Сбрасывает карту для новой комнаты"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.temporal_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.scan_counter = 0
        self.detected_doors = []
        self.identified_objects = []

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Переводит мировые координаты в индексы сетки"""
        px = int((wx - self.origin[0]) / self.resolution)
        py = int((wy - self.origin[1]) / self.resolution)
        return (px, py)

    def grid_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Переводит индексы сетки в мировые координаты"""
        wx = px * self.resolution + self.origin[0]
        wy = py * self.resolution + self.origin[1]
        return (wx, wy)

    def in_bounds(self, px: int, py: int) -> bool:
        """Проверяет, в пределах ли сетки точка"""
        return 0 <= px < self.grid_size and 0 <= py < self.grid_size

    async def _execute_tool(self, tool_name: str, params: dict = None):
        """
        Вызывает инструмент агента.
        Все вызовы идут через agent.tools (move_forward, ask_vlm, ask_human, ...)
        """
        if self.agent is None:
            logger.error(f"❌ Агент не привязан к LidarMapper. Вызов {tool_name} невозможен.")
            return None

        for tool in self.agent.tools:
            if tool.name == tool_name:
                try:
                    result = await tool.forward(**(params or {}))
                    return result
                except Exception as e:
                    logger.error(f"❌ Ошибка вызова {tool_name}: {e}")
                    return None

        logger.warning(f"⚠️ Инструмент {tool_name} не найден среди {[t.name for t in self.agent.tools]}")
        return None

    # ==================== ПОЛУЧЕНИЕ ДАННЫХ С СЕНСОРОВ ====================

    async def _get_raw_lidar_scan(self) -> Optional[List[Tuple[float, float]]]:
        """
        Получает сырые точки лидара НАПРЯМУЮ от LidarProcessor.
        
        НЕ использует SensorMemory — сырые точки не должны попадать в промпт LLM.
        LidarProcessor отдаёт 8 секторов + кластеры в SensorMemory для LLM,
        а сырые точки — только через get_raw_scan() для картографа.
        """
        if self.agent is None:
            logger.error("❌ Агент не привязан к LidarMapper")
            return None

        try:
            # Получаем сырые точки от LidarProcessor
            if hasattr(self.agent.sensor_collector, 'lidar'):
                raw_scan = self.agent.sensor_collector.lidar.get_raw_scan()
                if raw_scan and "points" in raw_scan:
                    return raw_scan["points"]

            # Fallback: если структура другая — попробуем через get_latest
            logger.warning("⚠️ LidarProcessor.get_raw_scan() недоступен, пробую альтернативу")
            if hasattr(self.agent.sensor_collector, 'get_latest_raw_scan'):
                return self.agent.sensor_collector.get_latest_raw_scan()

            return None

        except Exception as e:
            logger.error(f"❌ Ошибка получения сырого скана: {e}")
            return None

    async def _get_robot_pose(self) -> Tuple[float, float, float]:
        """
        Получает позу робота из одометрии.
        
        Можно брать из SensorMemory — это всего 3 числа (x, y, theta),
        которые не забивают контекст LLM.
        """
        if self.agent and self.agent.sensor_memory:
            odom_data = self.agent.sensor_memory.get("odometry")
            if odom_data and hasattr(odom_data, 'data'):
                return (
                    odom_data.data.get("x", 0.0),
                    odom_data.data.get("y", 0.0),
                    odom_data.data.get("theta", 0.0)
                )

        # Fallback: нулевая поза
        return (0.0, 0.0, 0.0)

    async def _get_vlm_answer(self) -> Optional[str]:
        """
        Получает ответ VLM из SensorMemory после ask_vlm.
        
        Ответ на pending_question приходит в следующем кадре VLMScanner
        и сохраняется в поле answer_llm.
        """
        if self.agent is None or self.agent.sensor_memory is None:
            return None

        # Ждём до 3 циклов VLM (~1.5 сек)
        for attempt in range(3):
            await asyncio.sleep(0.5)

            vlm_data = self.agent.sensor_memory.get("vlm")
            if vlm_data and hasattr(vlm_data, 'data'):
                answer = vlm_data.data.get("answer_llm")
                if answer and answer.strip():
                    return answer.strip()

        return None

    # ==================== КАРТОГРАФИРОВАНИЕ С ВРЕМЕННОЙ ФИЛЬТРАЦИЕЙ ====================

    def add_scan(
        self,
        scan_points: np.ndarray,
        robot_pose: Tuple[float, float, float]
    ):
        """
        Добавляет скан лидара в карту с временной фильтрацией динамических объектов.
        
        Клетка помечается как occupied только если наблюдается persist_threshold раз.
        Клетки, которые перестали наблюдаться, со временем теряют уверенность (decay).
        """
        rx, ry, rtheta = robot_pose
        cos_t = np.cos(rtheta)
        sin_t = np.sin(rtheta)

        # Позиция лидара с учётом смещения
        lidar_wx = rx + self.lidar_offset_x * cos_t - self.lidar_offset_y * sin_t
        lidar_wy = ry + self.lidar_offset_x * sin_t + self.lidar_offset_y * cos_t

        lidar_px, lidar_py = self.world_to_grid(lidar_wx, lidar_wy)

        # Множество клеток, которые «видны» в этом скане (для decay)
        observed_cells = set()

        for lx, ly in scan_points:
            # В мировые координаты
            wx = rx + lx * cos_t - ly * sin_t
            wy = ry + lx * sin_t + ly * cos_t

            px, py = self.world_to_grid(wx, wy)

            if not self.in_bounds(px, py) or not self.in_bounds(lidar_px, lidar_py):
                continue

            # Bresenham: отмечаем free space
            cells_on_line = self._bresenham_line(lidar_px, lidar_py, px, py)
            for cell_px, cell_py in cells_on_line:
                if self.in_bounds(cell_px, cell_py):
                    observed_cells.add((cell_px, cell_py))
                    # Free space — сразу сбрасываем счётчик
                    if self.grid[cell_py, cell_px] == 0:
                        self.grid[cell_py, cell_px] = -1
                    self.temporal_grid[cell_py, cell_px] = 0

            # Конечная точка: увеличиваем счётчик наблюдений
            observed_cells.add((px, py))
            self.temporal_grid[py, px] = min(self.temporal_grid[py, px] + 1, 255)

            # Если видели достаточно раз — помечаем как занятое
            if self.temporal_grid[py, px] >= self.persist_threshold:
                self.grid[py, px] = 100

        # Затухание (decay): уменьшаем счётчики клеток, которые не видели в этом скане
        self.scan_counter += 1
        if self.scan_counter % self.decay_interval == 0:
            self._decay_unobserved_cells(observed_cells)

    def _bresenham_line(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> List[Tuple[int, int]]:
        """
        Алгоритм Брезенхема. Возвращает список клеток на линии.
        Конечная точка (x1, y1) НЕ включается — она обрабатывается отдельно.
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            cells.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells

    def _decay_unobserved_cells(self, observed_cells: set):
        """
        Уменьшает счётчик temporal_grid для клеток,
        которые не были в observed_cells в этом раунде.
        Если счётчик падает ниже порога — клетка становится unknown.
        """
        # Только для клеток со счётчиком > 0, которых нет в observed
        active_mask = self.temporal_grid > 0

        # Создаём маску observed
        observed_mask = np.zeros_like(self.temporal_grid, dtype=bool)
        for px, py in observed_cells:
            if self.in_bounds(px, py):
                observed_mask[py, px] = True

        # Клетки, которые активны, но не наблюдались
        decay_mask = active_mask & ~observed_mask
        self.temporal_grid[decay_mask] -= 1

        # Там где счётчик упал ниже порога — сбрасываем occupied
        reset_mask = (self.temporal_grid < self.persist_threshold) & (self.grid == 100)
        self.grid[reset_mask] = 0
        self.temporal_grid[reset_mask] = 0

        decayed = np.sum(decay_mask)
        if decayed > 0:
            logger.debug(f"⏳ Decay: {decayed} клеток теряют уверенность")

    # ==================== ДЕТЕКЦИЯ ДВЕРЕЙ ====================

    def detect_doors(
        self,
        scan_points: np.ndarray,
        robot_pose: Tuple[float, float, float]
    ) -> List[Dict]:
        """
        Ищет дверные проёмы по лидарным данным (gap detection).
        
        Алгоритм:
        1. Gap detection: ищем разрывы > 0.3 м между соседними точками
        2. Geometry classification: отличаем дверь от пустого пространства
           - ширина проёма 0.5-2.0 м
           - за проёмом есть продолжение (дальняя стена)
           - обрамлён косяками (резкое приближение → пустота → приближение)
        """
        doors = []

        if len(scan_points) < 10:
            return doors

        # Вычисляем расстояния и углы для каждой точки
        distances = np.sqrt(scan_points[:, 0]**2 + scan_points[:, 1]**2)
        angles = np.degrees(np.arctan2(scan_points[:, 1], scan_points[:, 0]))

        # Сортируем по углу
        sort_idx = np.argsort(angles)
        distances = distances[sort_idx]
        angles = angles[sort_idx]
        scan_points_sorted = scan_points[sort_idx]

        # Gap detection: ищем разрывы > 0.3 м между соседними точками
        gaps = []
        for i in range(1, len(distances)):
            diff = abs(distances[i] - distances[i-1])
            if diff > 0.3:
                gaps.append(i)

        # Анализируем каждый разрыв
        for gap_idx in gaps:
            left_start = max(0, gap_idx - 10)
            right_end = min(len(distances), gap_idx + 10)

            left_points = scan_points_sorted[left_start:gap_idx]
            right_points = scan_points_sorted[gap_idx:right_end]

            if len(left_points) < 3 or len(right_points) < 3:
                continue

            left_dist = np.mean(distances[left_start:gap_idx])
            right_dist = np.mean(distances[gap_idx:right_end])

            if abs(left_dist - right_dist) < 0.2:
                continue

            if left_dist < right_dist:
                near_points = left_points
                far_points = right_points
                near_angle = np.mean(angles[left_start:gap_idx])
            else:
                near_points = right_points
                far_points = left_points
                near_angle = np.mean(angles[gap_idx:right_end])

            if len(far_points) < 3:
                continue

            far_angles = np.degrees(np.arctan2(far_points[:, 1], far_points[:, 0]))
            angular_width = np.max(far_angles) - np.min(far_angles)
            far_distance = np.mean(np.sqrt(far_points[:, 0]**2 + far_points[:, 1]**2))
            door_width_m = 2 * far_distance * np.tan(np.radians(angular_width / 2))

            if 0.5 < door_width_m < 2.0:
                door_angle = near_angle
                door_distance = np.mean(np.sqrt(near_points[:, 0]**2 + near_points[:, 1]**2))

                rx, ry, rtheta = robot_pose
                door_wx = rx + door_distance * np.cos(np.radians(door_angle + np.degrees(rtheta)))
                door_wy = ry + door_distance * np.sin(np.radians(door_angle + np.degrees(rtheta)))

                direction = self._door_direction(door_angle, rtheta)

                doors.append({
                    "x": round(door_wx, 2),
                    "y": round(door_wy, 2),
                    "width": round(door_width_m, 2),
                    "angle": round(door_angle, 1),
                    "distance": round(door_distance, 2),
                    "direction": direction
                })

        doors = self._merge_duplicate_doors(doors)
        self.detected_doors = doors
        logger.info(f"🚪 Найдено {len(doors)} кандидатов в двери (по лидару)")

        return doors

    def _door_direction(self, angle: float, robot_theta: float) -> str:
        """Определяет направление двери относительно севера (0° = north)"""
        global_angle = (angle + np.degrees(robot_theta) + 360) % 360

        if 45 <= global_angle < 135:
            return "east"
        elif 135 <= global_angle < 225:
            return "south"
        elif 225 <= global_angle < 315:
            return "west"
        else:
            return "north"

    def _merge_duplicate_doors(
        self, doors: List[Dict], dist_threshold: float = 0.5
    ) -> List[Dict]:
        """Объединяет дубликаты дверей, найденные с разных позиций"""
        if len(doors) < 2:
            return doors

        merged = []
        used = set()

        for i in range(len(doors)):
            if i in used:
                continue

            group = [doors[i]]
            used.add(i)

            for j in range(i + 1, len(doors)):
                if j in used:
                    continue

                dx = doors[i]['x'] - doors[j]['x']
                dy = doors[i]['y'] - doors[j]['y']
                dist = np.sqrt(dx**2 + dy**2)

                if dist < dist_threshold:
                    group.append(doors[j])
                    used.add(j)

            avg_x = np.mean([d['x'] for d in group])
            avg_y = np.mean([d['y'] for d in group])
            avg_width = np.mean([d['width'] for d in group])
            direction = group[0]['direction']

            merged.append({
                "x": round(avg_x, 2),
                "y": round(avg_y, 2),
                "width": round(avg_width, 2),
                "direction": direction
            })

        return merged

    # ==================== VLM-ВЕРИФИКАЦИЯ ДВЕРЕЙ ====================

    async def verify_door_with_vlm(
        self,
        door_candidate: Dict,
        robot_pose: Tuple[float, float, float]
    ) -> bool:
        """
        Подъезжает носом к кандидату в дверь и спрашивает VLM через ask_vlm.
        
        Использует agent.execute_tool("turn_left"/"turn_right"/"move_forward") 
        для движения и agent.execute_tool("ask_vlm") для вопроса к VLM.
        Ответ получает из SensorMemory (поле answer_llm в следующем кадре).
        """
        rx, ry, rtheta = robot_pose
        door_x = door_candidate["x"]
        door_y = door_candidate["y"]

        # Точка остановки: 1.5м перед дверью
        dx = door_x - rx
        dy = door_y - ry
        dist_to_door = np.sqrt(dx**2 + dy**2)

        if dist_to_door > 1.5:
            # Поворачиваемся к точке остановки
            target_angle = np.degrees(np.arctan2(dy, dx))
            turn_angle = target_angle - np.degrees(rtheta)
            turn_angle = (turn_angle + 180) % 360 - 180

            if abs(turn_angle) > 5:
                if turn_angle > 0:
                    await self._execute_tool("turn_right", {"angle": abs(turn_angle)})
                else:
                    await self._execute_tool("turn_left", {"angle": abs(turn_angle)})

            # Подъезжаем
            await self._execute_tool("move_forward", {"distance": dist_to_door - 1.5})

        # Поворачиваемся носом к двери
        door_angle = np.degrees(np.arctan2(door_y - ry, door_x - rx))
        turn_to_door = door_angle - np.degrees(rtheta)
        turn_to_door = (turn_to_door + 180) % 360 - 180

        if abs(turn_to_door) > 5:
            if turn_to_door > 0:
                await self._execute_tool("turn_right", {"angle": abs(turn_to_door)})
            else:
                await self._execute_tool("turn_left", {"angle": abs(turn_to_door)})

        # Спрашиваем VLM через ask_vlm (вопрос на английском)
        logger.info(f"👃 Спрашиваю VLM про дверь на ({door_x:.1f}, {door_y:.1f})")

        await self._execute_tool("ask_vlm", {
            "prompt": "Look straight ahead. Is this a doorway or an open passage "
                      "to another room? Answer ONLY YES or NO."
        })

        # Ответ придёт в следующем кадре VLMScanner
        answer = await self._get_vlm_answer()

        if answer:
            logger.info(f"🤖 VLM ответ: '{answer}'")
            is_door = answer.strip().upper().startswith("YES")
            if is_door:
                logger.info(f"✅ Дверь подтверждена VLM: ({door_x:.1f}, {door_y:.1f})")
            else:
                logger.info(f"❌ VLM отвергла кандидата: ({door_x:.1f}, {door_y:.1f})")
            return is_door

        logger.warning("⚠️ VLM не дала ответа")
        return False

    # ==================== ОПРОС ЧЕЛОВЕКА О ДВЕРЯХ ====================

    async def ask_human_about_door(self, door: Dict) -> Optional[str]:
        """
        Спрашивает человека, куда ведёт дверь.
        
        Использует agent.execute_tool("ask_human") — AskHumanTool
        ждёт ответа через _wait_for_speech() с таймаутом 10 секунд.
        """
        direction = door.get("direction", "неизвестно")
        room_name = self.current_room

        question = (
            f"Я нашёл дверь на {self._direction_to_russian(direction)} "
            f"в комнате «{room_name}». Куда она ведёт?"
        )

        logger.info(f"❓ Спрашиваю человека: {question}")

        # AskHumanTool возвращает "Человек ответил: ..." или "Человек не ответил"
        response = await self._execute_tool("ask_human", {"question": question})

        if response and "Человек ответил:" in str(response):
            answer = str(response).split("Человек ответил:", 1)[1].strip()
            destination = answer.lower()
            logger.info(f"✅ Человек ответил: дверь на {direction} → {destination}")
            self.update_room_graph_after_door(door, destination)
            return destination

        logger.info(f"⏰ Нет ответа, дверь на {direction} → ?")
        self.update_room_graph_after_door(door, "?")
        return "?"

    def _direction_to_russian(self, direction: str) -> str:
        """Переводит направление на русский"""
        translations = {
            "north": "север",
            "south": "юг",
            "east": "восток",
            "west": "запад",
            "unknown": "неизвестное направление"
        }
        return translations.get(direction, direction)

    def _direction_to_opposite(self, direction: str) -> str:
        """Возвращает противоположное направление"""
        opposites = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east"
        }
        return opposites.get(direction, "unknown")

    def update_room_graph_after_door(self, door: Dict, destination: str):
        """
        Обновляет граф помещений после подтверждения двери.
        
        Создаёт обратную связь: если дверь ведёт из А в Б,
        то в графе комнаты Б появляется обратная дверь в А.
        """
        direction = door.get("direction", "unknown")

        # Инициализируем текущую комнату в графе если нужно
        if self.current_room not in self.room_graph:
            self.room_graph[self.current_room] = {
                "doors": {},
                "objects": [obj["label"] for obj in self.identified_objects]
            }

        # Добавляем дверь в текущую комнату
        self.room_graph[self.current_room]["doors"][direction] = destination

        # Если destination известна — создаём обратную связь
        if destination != "?":
            opposite = self._direction_to_opposite(direction)

            if destination not in self.room_graph:
                self.room_graph[destination] = {
                    "doors": {},
                    "objects": []
                }

            # Обратная дверь ведёт в текущую комнату
            self.room_graph[destination]["doors"][opposite] = self.current_room

        logger.info(f"🗺️ Граф обновлён: {self.current_room} → {direction} → {destination}")

    def build_room_graph(self) -> Dict:
        """
        Строит граф помещений на основе подтверждённых дверей.
        
        Returns:
            {
                "current_room": "гостиная",
                "rooms": {
                    "гостиная": {
                        "doors": {"north": "кухня", "south": "?"},
                        "objects": ["диван", "стол"]
                    },
                    "кухня": {
                        "doors": {"south": "гостиная"},
                        "objects": []
                    }
                },
                "unexplored_doors": [
                    {"from_room": "гостиная", "direction": "south", "destination": "?"}
                ]
            }
        """
        graph = {
            "current_room": self.current_room,
            "rooms": {},
            "unexplored_doors": []
        }

        # Добавляем текущую комнату если её ещё нет
        if self.current_room not in self.room_graph:
            self.room_graph[self.current_room] = {
                "doors": {},
                "objects": [obj["label"] for obj in self.identified_objects]
            }

        # Собираем все комнаты из графа
        for room_name, room_data in self.room_graph.items():
            graph["rooms"][room_name] = {
                "doors": room_data.get("doors", {}),
                "objects": room_data.get("objects", [])
            }

            # Ищем неисследованные двери
            for direction, destination in room_data.get("doors", {}).items():
                if destination == "?" or destination not in self.room_graph:
                    graph["unexplored_doors"].append({
                        "from_room": room_name,
                        "direction": direction,
                        "destination": destination
                    })

        return graph

    # ==================== VLM-ИДЕНТИФИКАЦИЯ ОБЪЕКТОВ ====================

    async def identify_objects_with_vlm(
        self,
        robot_pose: Tuple[float, float, float],
        min_cluster_size: int = 10,
        max_aspect_ratio: float = 5.0,
        min_wall_size_m: float = 1.5
    ) -> List[Dict]:
        """
        Находит крупные occupied-кластеры (не стены),
        подъезжает к каждому и спрашивает VLM: «Что ты видишь?»
        
        Фильтрует стены по соотношению сторон (aspect ratio).
        """
        from scipy.ndimage import label as nd_label

        occupied_mask = self.grid == 100
        labeled, num_features = nd_label(occupied_mask)

        if num_features == 0:
            logger.info("📦 Нет кластеров для идентификации")
            return []

        identified_objects = []
        rx, ry, rtheta = robot_pose

        for cluster_id in range(1, num_features + 1):
            cluster_py, cluster_px = np.where(labeled == cluster_id)

            if len(cluster_px) < min_cluster_size:
                continue

            # Центр и размеры кластера в мировых координатах
            avg_px = np.mean(cluster_px)
            avg_py = np.mean(cluster_py)
            wx, wy = self.grid_to_world(avg_px, avg_py)

            width_m = (np.max(cluster_px) - np.min(cluster_px)) * self.resolution
            height_m = (np.max(cluster_py) - np.min(cluster_py)) * self.resolution

            # Фильтруем стены (длинные и тонкие)
            min_dim = max(min(width_m, height_m), 0.01)
            aspect_ratio = max(width_m, height_m) / min_dim
            max_dim = max(width_m, height_m)

            if aspect_ratio > max_aspect_ratio and max_dim > min_wall_size_m:
                logger.debug(f"⏭️ Кластер #{cluster_id} похож на стену "
                            f"(aspect={aspect_ratio:.1f}, size={width_m:.1f}x{height_m:.1f})")
                continue

            # Подъезжаем к объекту на 1.0м
            dx = wx - rx
            dy = wy - ry
            dist = np.sqrt(dx**2 + dy**2)

            if dist > 1.0:
                target_angle = np.degrees(np.arctan2(dy, dx))
                turn_angle = target_angle - np.degrees(rtheta)
                turn_angle = (turn_angle + 180) % 360 - 180

                if abs(turn_angle) > 5:
                    if turn_angle > 0:
                        await self._execute_tool("turn_right", {"angle": abs(turn_angle)})
                    else:
                        await self._execute_tool("turn_left", {"angle": abs(turn_angle)})

                await self._execute_tool("move_forward", {"distance": dist - 1.0})

            # Поворачиваемся носом к объекту
            target_angle = np.degrees(np.arctan2(dy, dx))
            turn_to_obj = target_angle - np.degrees(rtheta)
            turn_to_obj = (turn_to_obj + 180) % 360 - 180

            if abs(turn_to_obj) > 5:
                if turn_to_obj > 0:
                    await self._execute_tool("turn_right", {"angle": abs(turn_to_obj)})
                else:
                    await self._execute_tool("turn_left", {"angle": abs(turn_to_obj)})

            # Спрашиваем VLM
            await self._execute_tool("ask_vlm", {
                "prompt": "What is the large object directly in front of you? "
                          "Answer with one word or short phrase in English."
            })

            answer = await self._get_vlm_answer()
            label = answer if answer else "неизвестно"

            logger.info(f"🏷️ VLM: '{label}' на ({wx:.1f}, {wy:.1f})")

            identified_objects.append({
                "label": label,
                "x": round(wx, 2),
                "y": round(wy, 2),
                "width_m": round(width_m, 2),
                "height_m": round(height_m, 2),
                "verified_by_vlm": True
            })

        self.identified_objects = identified_objects
        logger.info(f"📦 Идентифицировано {len(identified_objects)} объектов через VLM")

        return identified_objects

    # ==================== ИССЛЕДОВАНИЕ ПОМЕЩЕНИЯ ====================

    def get_coverage(self) -> float:
        """Оценивает процент покрытия комнаты (доля известных клеток)."""
        total_cells = self.grid_size * self.grid_size
        if total_cells == 0:
            return 0.0
        known_cells = np.sum(self.grid != 0)  # -1 (free) или 100 (occupied)
        return known_cells / total_cells

    def is_room_covered(self, min_coverage: float = 0.85) -> bool:
        """Проверяет, достаточно ли покрыта комната для завершения исследования."""
        return self.get_coverage() >= min_coverage

    def find_unexplored_direction(
        self, robot_pose: Tuple[float, float, float]
    ) -> Optional[float]:
        """
        Находит направление к ближайшей неизведанной области.
        Использует frontier-based подход: ищем границу между известным и неизвестным.
        """
        rx, ry, _ = robot_pose
        rpx, rpy = self.world_to_grid(rx, ry)

        # Ищем frontier cells: известные клетки на границе с неизвестными
        frontier_cells = []

        for py in range(1, self.grid_size - 1):
            for px in range(1, self.grid_size - 1):
                if self.grid[py, px] == 0:
                    continue  # неизвестная клетка — не подходит

                # Проверяем соседей: есть ли неизвестные рядом
                has_unknown_neighbor = (
                    self.grid[py-1, px] == 0 or
                    self.grid[py+1, px] == 0 or
                    self.grid[py, px-1] == 0 or
                    self.grid[py, px+1] == 0
                )

                if has_unknown_neighbor:
                    frontier_cells.append((px, py))

        if not frontier_cells:
            return None

        # Находим ближайшую frontier cell
        min_dist = float('inf')
        best_px, best_py = frontier_cells[0]

        for px, py in frontier_cells:
            dist = np.sqrt((px - rpx)**2 + (py - rpy)**2)
            if dist < min_dist:
                min_dist = dist
                best_px, best_py = px, py

        # Конвертируем в мировые координаты и вычисляем угол
        wx, wy = self.grid_to_world(best_px, best_py)
        dx = wx - rx
        dy = wy - ry
        angle = np.degrees(np.arctan2(dy, dx))
        return (angle + 360) % 360

    async def explore_room(
        self,
        room_name: str,
        max_scans: int = 200,
        min_coverage: float = 0.85,
        max_idle_scans: int = 20,
        enable_vlm_verification: bool = True
    ) -> Dict:
        """
        Основной цикл исследования комнаты.
        
        Получает сырые точки лидара НАПРЯМУЮ от LidarProcessor
        (НЕ из SensorMemory — там только 8 секторов для LLM).
        Двигается через agent.tools (turn_left, turn_right, move_forward).
        
        Args:
            room_name: название комнаты
            max_scans: максимальное количество сканов (защита от бесконечного цикла)
            min_coverage: минимальный процент покрытия для завершения
            max_idle_scans: сколько сканов без прогресса до остановки
            enable_vlm_verification: запускать ли VLM-верификацию после сканирования
        
        Returns:
            Dict: статистика исследования + room_graph
        """
        logger.info(f"🔍 Начинаю исследование комнаты: {room_name}")
        self.reset()
        self.current_room = room_name

        idle_scans = 0
        last_coverage = 0.0
        scan_count = 0
        robot_pose = (0.0, 0.0, 0.0)  # будет обновляться из одометрии

        autosave_interval = 20

        for scan_count in range(max_scans):
            # 1. Получаем СЫРОЙ скан лидара НАПРЯМУЮ от LidarProcessor
            #    (НЕ из SensorMemory — там только 8 секторов для LLM!)
            raw_scan = await self._get_raw_lidar_scan()
            if raw_scan is None:
                logger.warning(f"⚠️ Скан #{scan_count}: нет сырых данных лидара")
                idle_scans += 1
                if idle_scans >= max_idle_scans:
                    logger.warning("🛑 Слишком много пропущенных сканов")
                    break
                await asyncio.sleep(0.1)
                continue

            # Конвертируем в numpy массив [(x, y), ...]
            if isinstance(raw_scan, list) and len(raw_scan) > 0:
                if isinstance(raw_scan[0], dict):
                    scan_points = np.array([[p.get("x", 0), p.get("y", 0)] for p in raw_scan])
                elif isinstance(raw_scan[0], (list, tuple)):
                    scan_points = np.array(raw_scan)
                else:
                    logger.warning(f"⚠️ Неизвестный формат сырых точек: {type(raw_scan[0])}")
                    continue
            else:
                logger.warning(f"⚠️ Пустой скан")
                continue

            # 2. Поза из одометрии (можно из SensorMemory — это всего 3 числа)
            robot_pose = await self._get_robot_pose()

            # 3. Добавляем скан в карту (с временной фильтрацией динамики)
            self.add_scan(scan_points, robot_pose)

            # 4. Ищем двери
            doors = self.detect_doors(scan_points, robot_pose)

            # 5. Проверяем покрытие
            coverage = self.get_coverage()
            logger.info(f"📊 Скан #{scan_count}: покрытие={coverage:.1%}, "
                        f"дверей={len(doors)}, "
                        f"поза=({robot_pose[0]:.1f}, {robot_pose[1]:.1f})")

            # 6. Проверяем завершение
            if self.is_room_covered(min_coverage):
                logger.info(f"✅ Комната {room_name} исследована (покрытие: {coverage:.1%})")
                break

            # 7. Проверяем прогресс
            if coverage > last_coverage + 0.01:
                last_coverage = coverage
                idle_scans = 0
            else:
                idle_scans += 1
                if idle_scans >= max_idle_scans:
                    logger.info(f"✅ Покрытие не улучшается ({idle_scans} сканов), "
                              f"комната {room_name} исследована (покрытие: {coverage:.1%})")
                    break

            # 8. Находим направление движения
            direction = self.find_unexplored_direction(robot_pose)
            if direction is None:
                logger.info(f"✅ Неизведанных областей не осталось, комната {room_name} исследована")
                break

            # 9. Двигаемся в найденном направлении
            current_theta = np.degrees(robot_pose[2])
            turn_angle = direction - current_theta
            # Нормализуем угол в [-180, 180]
            turn_angle = (turn_angle + 180) % 360 - 180

            if abs(turn_angle) > 5:
                if turn_angle > 0:
                    await self._execute_tool("turn_right", {"angle": abs(turn_angle)})
                else:
                    await self._execute_tool("turn_left", {"angle": abs(turn_angle)})

            await self._execute_tool("move_forward", {"distance": 0.3})
            idle_scans = 0

            # 10. Автосохранение
            if scan_count > 0 and scan_count % autosave_interval == 0:
                self.save_map(f"maps/{room_name}_autosave_{scan_count}.png")
                self.save_metadata(f"maps/{room_name}_autosave_{scan_count}.json")

            await asyncio.sleep(0.1)

        logger.info(f"🏁 Сканирование завершено: {scan_count + 1} сканов, "
                    f"покрытие={self.get_coverage():.1%}")

        # ==================== VLM-ВЕРИФИКАЦИЯ ПОСЛЕ СКАНИРОВАНИЯ ====================
        if enable_vlm_verification and self.agent:
            logger.info("🤖 Начинаю VLM-верификацию дверей")

            confirmed_doors = []
            for candidate in self.detected_doors:
                # 1. VLM-верификация
                is_confirmed = await self.verify_door_with_vlm(candidate, robot_pose)

                if is_confirmed:
                    # 2. Спрашиваем человека куда ведёт дверь
                    await self.ask_human_about_door(candidate)

                    # 3. Получаем destination из графа (только что обновлён)
                    destination = self.room_graph.get(
                        self.current_room, {}
                    ).get("doors", {}).get(
                        candidate.get("direction", "unknown"), "?"
                    )

                    confirmed_doors.append({
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "width": candidate.get("width", 0.0),
                        "direction": candidate.get("direction", "unknown"),
                        "destination": destination,
                        "verified_by_vlm": True,
                        "confirmed_by_human": destination != "?"
                    })

            self.detected_doors = confirmed_doors

            # Идентифицируем крупные объекты (мебель)
            await self.identify_objects_with_vlm(robot_pose)

        # Строим граф помещений
        room_graph = self.build_room_graph()

        # Финальное сохранение
        self.save_map(f"maps/{room_name}_final.png")
        self.save_metadata(f"maps/{room_name}_metadata.json",
                          extra_objects=[{"room_graph": room_graph}])

        return {
            "room_name": room_name,
            "scans": scan_count + 1,
            "coverage": self.get_coverage(),
            "doors_confirmed": len(self.detected_doors),
            "objects_identified": len(self.identified_objects),
            "room_graph": room_graph,
            "success": self.is_room_covered(min_coverage)
        }

    # ==================== СОХРАНЕНИЕ ====================

    def save_map(self, path: str):
        """Сохраняет карту как PNG"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Конвертируем: -1 (free) → 0, 0 (unknown) → 128, 100 (occupied) → 255
        image = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        image[self.grid == -1] = 0      # free space — чёрный
        image[self.grid == 0] = 128     # unknown — серый
        image[self.grid == 100] = 255   # occupied — белый

        cv2.imwrite(path, image)
        logger.info(f"💾 Карта сохранена: {path}")

    def save_metadata(self, path: str, extra_objects: List[Dict] = None):
        """
        Сохраняет метаданные комнаты в JSON для бортовой LLM.
        
        JSON содержит: имя комнаты, статистику, двери (с destination),
        идентифицированные объекты, граф помещений.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        all_objects = list(self.identified_objects)
        if extra_objects:
            all_objects.extend(extra_objects)

        metadata = {
            "room_name": self.current_room,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stats": self.get_stats(),
            "doors": self.detected_doors,
            "objects": all_objects,
            "room_graph": self.build_room_graph(),
            "map_info": {
                "resolution_m_per_px": self.resolution,
                "map_size_meters": self.map_size,
                "grid_size_px": self.grid_size,
                "origin": self.origin
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"📋 Метаданные сохранены: {path} "
                    f"(дверей: {len(self.detected_doors)}, "
                    f"объектов: {len(all_objects)})")

    def get_map_image(self) -> np.ndarray:
        """Возвращает карту как изображение"""
        return self.grid.copy()

    def get_stats(self) -> Dict:
        """Статистика картографа"""
        occupied = np.sum(self.grid == 100)
        free = np.sum(self.grid == -1)
        unknown = np.sum(self.grid == 0)

        return {
            "total_cells": self.grid_size ** 2,
            "occupied_cells": int(occupied),
            "free_cells": int(free),
            "unknown_cells": int(unknown),
            "coverage": round(self.get_coverage(), 4),
            "doors_detected": len(self.detected_doors),
            "objects_identified": len(self.identified_objects),
            "temporal_active_cells": int(np.sum(self.temporal_grid > 0)),
            "resolution": self.resolution,
            "map_size_meters": self.map_size
        }