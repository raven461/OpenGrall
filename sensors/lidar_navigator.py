#!/usr/bin/env python3
"""
sensors/lidar_navigator.py — НАВИГАЦИЯ ПО КАРТЕ (A*, BFS, ICP)
"""

import numpy as np
import heapq
import logging
from typing import Optional, Tuple, Dict, List, Any

logger = logging.getLogger(__name__)


class LidarNavigator:
    """
    Навигатор. Локализация (ICP), планирование пути (A*),
    навигация между комнатами (BFS), исполнение команд через агента.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Текущая поза
        self.current_pose = (0.0, 0.0, 0.0)  # x, y, theta

        # Текущая комната и карта
        self.current_room: Optional[str] = None
        self.current_location: Optional[str] = None
        self.room_map: Optional[np.ndarray] = None
        self.room_doors: List[Dict] = []

        # Параметры из конфига
        self.robot_radius = max(
            self.config.get('robot_length', 0.51),
            self.config.get('robot_width', 0.32)
        ) / 2 + 0.1  # + запас 10 см

        self.map_resolution = self.config.get('map_resolution', 0.05)

        # Ссылка на агента (устанавливается извне)
        self.agent = None

        # Кэш для построения путей
        self._room_graph_cache = None

        logger.info("✅ LidarNavigator инициализирован")

    # ==================== СВЯЗЬ С АГЕНТОМ ====================

    def set_agent(self, agent):
        """Устанавливает ссылку на агента для вызова инструментов движения."""
        self.agent = agent

    async def _call_agent_tool(self, tool_name: str, params: dict) -> Any:
        """Вызывает инструмент агента по имени."""
        if self.agent is None:
            return None
        return await self.agent.execute_tool(tool_name, params)

    # ==================== УПРАВЛЕНИЕ КАРТОЙ И КОМНАТОЙ ====================

    def set_room(self, location: str, room: str, map_image: np.ndarray, doors: List[Dict]):
        """Загружает карту комнаты."""
        self.current_location = location
        self.current_room = room
        self.room_map = map_image
        self.room_doors = doors
        logger.info(f"📍 Загружена комната: {room} ({location})")

    def update_pose(self, x: float, y: float, theta: float):
        """Обновляет текущую позу."""
        self.current_pose = (x, y, theta)

    # ==================== A* ПЛАНИРОВАНИЕ ПУТИ ====================

    def plan_path(self, target_x: float, target_y: float) -> Optional[List[Tuple[float, float]]]:
        """
        Прокладывает путь от текущей позиции до цели по карте (A*).

        Returns:
            Список точек [(x, y), ...] или None, если пути нет.
        """
        if self.room_map is None:
            logger.warning("Нет карты для планирования пути")
            return None

        resolution = self.map_resolution
        map_size = self.room_map.shape[0] * resolution
        origin = -map_size / 2

        # Начальная и конечная точки в сетке
        start_px = int((self.current_pose[0] - origin) / resolution)
        start_py = int((self.current_pose[1] - origin) / resolution)

        goal_px = int((target_x - origin) / resolution)
        goal_py = int((target_y - origin) / resolution)

        # Проверяем границы и стены
        if not (0 <= goal_px < self.room_map.shape[1] and 0 <= goal_py < self.room_map.shape[0]):
            logger.warning(f"Цель ({target_x}, {target_y}) за пределами карты")
            return None

        robot_radius_px = int(self.robot_radius / resolution)

        # A*
        open_set = []
        heapq.heappush(open_set, (0, start_px, start_py))

        came_from = {}
        g_score = {(start_px, start_py): 0}

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (goal_px, goal_py):
                # Восстанавливаем путь
                path = []
                while (cx, cy) in came_from:
                    wx = cx * resolution + origin
                    wy = cy * resolution + origin
                    path.append((wx, wy))
                    cx, cy = came_from[(cx, cy)]
                path.reverse()
                logger.info(f"📏 Путь построен: {len(path)} сегментов")
                return path

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < self.room_map.shape[1] and 0 <= ny < self.room_map.shape[0]):
                    continue

                if self._is_cell_blocked(nx, ny, robot_radius_px):
                    continue

                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                new_g = g_score.get((cx, cy), float('inf')) + cost

                if new_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = new_g

                    h = ((nx - goal_px)**2 + (ny - goal_py)**2)**0.5
                    heapq.heappush(open_set, (new_g + h, nx, ny))

        logger.warning("Путь не найден!")
        return None

    def _is_cell_blocked(self, px: int, py: int, radius: int) -> bool:
        """Проверяет, заблокирована ли клетка (стена или рядом со стеной)."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.room_map.shape[1] and 0 <= ny < self.room_map.shape[0]:
                    if self.room_map[ny, nx] == 100:  # occupied
                        return True
        return False

    # ==================== ИСПОЛНЕНИЕ ПУТИ ====================

    async def execute_move_to(self, target_x: float, target_y: float) -> Tuple[bool, str]:
        """
        Прокладывает и выполняет путь до цели (по точкам A*).

        Returns:
            (success, message)
        """
        if self.agent is None:
            return (False, "Agent not set in LidarNavigator")

        # 1. Планируем путь
        path = self.plan_path(target_x, target_y)
        if path is None:
            # Fallback: прямой путь, если A* не справился
            path = [(target_x, target_y)]
            logger.warning("A* не нашёл путь, пробую прямой")

        logger.info(f"🚶 Выполнение пути из {len(path)} сегментов")

        # 2. Исполняем путь по сегментам
        for i, (wx, wy) in enumerate(path):
            dx = wx - self.current_pose[0]
            dy = wy - self.current_pose[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 0.05:
                continue

            # Угол до следующей точки
            global_angle = np.degrees(np.arctan2(dy, dx)) % 360
            current_deg = np.degrees(self.current_pose[2]) % 360

            turn_angle = (global_angle - current_deg + 360) % 360
            if turn_angle > 180:
                turn_angle -= 360

            # Поворачиваемся
            if abs(turn_angle) > 5:
                tool = "turn_right" if turn_angle > 0 else "turn_left"
                result = await self._call_agent_tool(tool, {"angle": abs(turn_angle)})
                if result and "error" in str(result).lower():
                    return (False, f"Turn failed: {result}")

            # Едем (не доезжаем до конечной точки, чтобы не упереться)
            move_dist = dist - 0.05 if i == len(path) - 1 else dist
            if move_dist > 0.05:
                result = await self._call_agent_tool("move_forward", {"distance": move_dist})
                if result and "error" in str(result).lower():
                    return (False, f"Move failed: {result}")

            # Обновляем позу (одометрия обновится в следующем цикле)
            self.current_pose = (wx, wy, self.current_pose[2] + np.radians(turn_angle))

        await self._call_agent_tool("stop")
        return (True, f"Arrived at ({target_x:.1f}, {target_y:.1f})")

    # ==================== НАВИГАЦИЯ К ОБЪЕКТУ ====================

    async def go_to_object(self, object_name: str, objects_list: List[Dict]) -> Tuple[bool, str]:
        """
        Навигация к объекту (диван, стол и т.д.) по его координатам.

        Args:
            object_name: название объекта
            objects_list: список объектов из метаданных комнаты

        Returns:
            (success, message)
        """
        # Ищем объект
        target_obj = None
        for obj in objects_list:
            label = obj.get("label", obj.get("name", ""))
            if label.lower() == object_name.lower():
                target_obj = obj
                break

        if target_obj is None:
            return (False, f"Object '{object_name}' not found in current room")

        obj_x = target_obj.get("x", 0)
        obj_y = target_obj.get("y", 0)

        if obj_x == 0 and obj_y == 0:
            return (False, f"Object '{object_name}' has no coordinates")

        return await self.execute_move_to(obj_x, obj_y)

    # ==================== НАВИГАЦИЯ МЕЖДУ КОМНАТАМИ (BFS) ====================

    async def go_to_room_bfs(self, target_room: str, room_graph: Dict) -> Tuple[bool, str]:
        """
        Перемещается в другую комнату, используя BFS по графу помещений.

        Args:
            target_room: название целевой комнаты
            room_graph: граф помещений (из LidarMapper)

        Returns:
            (success, message)
        """
        if self.agent is None:
            return (False, "Agent not set in LidarNavigator")

        current = self.current_room
        if current == target_room:
            return (True, f"Already in room '{target_room}'")

        # BFS поиск пути
        visited = {current}
        queue = [(current, [current])]
        path = None

        while queue:
            node, node_path = queue.pop(0)
            if node == target_room:
                path = node_path
                break

            for door in room_graph.get(node, {}).get("doors", {}).values():
                if door not in visited and door != "?":
                    visited.add(door)
                    queue.append((door, node_path + [door]))

        if not path:
            return (False, f"No path found from '{current}' to '{target_room}'")

        # Проходим по дверям
        for i in range(len(path) - 1):
            from_room = path[i]
            to_room = path[i + 1]

            door = self._find_door_between(from_room, to_room, room_graph)
            if not door:
                return (False, f"Door from '{from_room}' to '{to_room}' not found")

            success, msg = await self._move_through_door(door)
            if not success:
                return (False, msg)

            # Обновляем текущую комнату
            self.current_room = to_room

            # Загружаем карту новой комнаты
            if self.agent and hasattr(self.agent, 'slam'):
                # Предполагается, что SLAMPlugin предоставляет map_storage
                map_image = self.agent.slam.map_storage.load_map(self.current_location, to_room)
                if map_image is not None:
                    self.room_map = map_image

        return (True, f"Arrived at room '{target_room}'")

    def _find_door_between(self, from_room: str, to_room: str, room_graph: Dict) -> Optional[Dict]:
        """Находит дверь между двумя комнатами в графе."""
        for direction, dest in room_graph.get(from_room, {}).get("doors", {}).items():
            if dest == to_room:
                # Ищем координаты двери в self.room_doors
                for door in self.room_doors:
                    if door.get("to") == to_room or door.get("direction") == direction:
                        return door
                return {"direction": direction, "to": to_room}
        return None

    async def _move_through_door(self, door: Dict) -> Tuple[bool, str]:
        """Подъезжает к двери и проезжает через неё."""
        if self.agent is None:
            return (False, "Agent not set")

        # Получаем координаты двери (если есть)
        door_x = door.get("x", 0)
        door_y = door.get("y", 0)

        if door_x == 0 and door_y == 0:
            # Если координат нет — просто едем вперёд
            await self._call_agent_tool("move_forward", {"distance": 0.5})
            return (True, "Moved through door (no coordinates)")

        # Подъезжаем к двери на 0.5 м
        success, msg = await self.execute_move_to(door_x, door_y)
        if not success:
            return (False, f"Cannot approach door: {msg}")

        # Проезжаем через дверь
        await self._call_agent_tool("move_forward", {"distance": 0.5})
        return (True, f"Moved through door to {door.get('to', 'unknown')}")

    # ==================== SCAN MATCHING (ICP) ====================

    def scan_matching(self, scan_points: np.ndarray,
                      max_iterations: int = 20,
                      max_correspondence_dist: float = 0.3,
                      epsilon_xy: float = 0.001,
                      epsilon_theta: float = 0.001) -> Optional[Tuple[float, float, float]]:
        """ICP — определяет позу робота на карте."""
        if self.room_map is None or len(scan_points) < 10:
            return None

        x, y, theta = self.current_pose
        map_points = self._map_to_points()
        if len(map_points) < 10:
            return None

        for _ in range(max_iterations):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            transformed = np.zeros_like(scan_points)
            transformed[:, 0] = x + scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t
            transformed[:, 1] = y + scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t

            correspondences = []
            for tp in transformed:
                dists = np.sqrt((map_points[:, 0] - tp[0])**2 + (map_points[:, 1] - tp[1])**2)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                if min_dist < max_correspondence_dist:
                    correspondences.append((tp, map_points[min_idx]))

            if len(correspondences) < 5:
                return None

            src_points = np.array([c[0] for c in correspondences])
            dst_points = np.array([c[1] for c in correspondences])

            src_centroid = np.mean(src_points, axis=0)
            dst_centroid = np.mean(dst_points, axis=0)

            src_centered = src_points - src_centroid
            dst_centered = dst_points - dst_centroid

            H = src_centered.T @ dst_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            dx = dst_centroid[0] - src_centroid[0]
            dy = dst_centroid[1] - src_centroid[1]
            dtheta = np.arctan2(R[1, 0], R[0, 0])

            x += dx
            y += dy
            theta += dtheta

            if abs(dx) < epsilon_xy and abs(dy) < epsilon_xy and abs(dtheta) < epsilon_theta:
                break

        return (x, y, theta % (2 * np.pi))

    def _map_to_points(self) -> np.ndarray:
        """Преобразует occupancy grid в облако точек (только occupied)."""
        if self.room_map is None:
            return np.array([])

        points = []
        map_size = self.room_map.shape[0] * self.map_resolution
        origin = -map_size / 2

        for py in range(self.room_map.shape[0]):
            for px in range(self.room_map.shape[1]):
                if self.room_map[py, px] == 100:
                    wx = px * self.map_resolution + origin
                    wy = py * self.map_resolution + origin
                    points.append([wx, wy])
        return np.array(points)

    def localize_on_map(self, scan_points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Определяет позу робота на текущей карте через ICP."""
        pose = self.scan_matching(scan_points)
        if pose is not None:
            self.current_pose = pose
            logger.debug(f"📍 Локализация: ({pose[0]:.2f}, {pose[1]:.2f}, {np.degrees(pose[2]):.0f}°)")
        return pose

    # ==================== ВАЛИДАЦИЯ ТОЧЕК ====================

    def validate_move_to(self, target_x: float, target_y: float) -> Tuple[bool, str]:
        """Проверяет, безопасно ли ехать в точку (x, y)."""
        if self.room_map is None:
            return (False, "нет карты комнаты")

        map_size = self.room_map.shape[0] * self.map_resolution
        origin = -map_size / 2

        px = int((target_x - origin) / self.map_resolution)
        py = int((target_y - origin) / self.map_resolution)

        if not (0 <= px < self.room_map.shape[1] and 0 <= py < self.room_map.shape[0]):
            return (False, f"точка ({target_x:.1f}, {target_y:.1f}) за пределами комнаты")

        robot_radius_px = int(self.robot_radius / self.map_resolution)
        for dy in range(-robot_radius_px, robot_radius_px + 1):
            for dx in range(-robot_radius_px, robot_radius_px + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.room_map.shape[1] and 0 <= ny < self.room_map.shape[0]:
                    if self.room_map[ny, nx] == 100:
                        return (False, f"точка ({target_x:.1f}, {target_y:.1f}) в стене")

        # Упрощённая проверка прямого пути (через линию)
        rx, ry, _ = self.current_pose
        rpx = int((rx - origin) / self.map_resolution)
        rpy = int((ry - origin) / self.map_resolution)

        if 0 <= rpx < self.room_map.shape[1] and 0 <= rpy < self.room_map.shape[0]:
            for (cx, cy) in self._line_iter(rpx, rpy, px, py):
                if self.room_map[cy, cx] == 100:
                    return (False, "прямой путь заблокирован")

        return (True, "OK")

    def _line_iter(self, x0: int, y0: int, x1: int, y1: int):
        """Итератор по точкам линии (Bresenham)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            yield (x0, y0)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # ==================== SLAM-СТРОКА ДЛЯ LLM ====================

    def get_slam_summary(self) -> str:
        """Генерирует компактную строку SLAM для промпта LLM."""
        if not self.current_room:
            return "неизвестно"

        x, y, theta = self.current_pose
        theta_deg = np.degrees(theta) % 360

        door_strs = []
        for door in self.room_doors:
            door_name = door.get('to', '?')
            door_x = door.get('x', 0)
            door_y = door.get('y', 0)

            dx = door_x - x
            dy = door_y - y
            distance = np.sqrt(dx**2 + dy**2)
            if distance < 0.01:
                continue

            global_angle = np.degrees(np.arctan2(dy, dx)) % 360
            relative_angle = (global_angle - theta_deg + 360) % 360
            arrow = self._angle_to_arrow(relative_angle)

            door_strs.append(f"{door_name} {arrow}{distance:.1f}м")

        return (f"{self.current_room}, ({x:.1f}, {y:.1f}), "
                f"двери: {', '.join(door_strs)}")

    def _angle_to_arrow(self, relative_angle: float) -> str:
        sectors = [
            (0, 22.5, "↑"), (22.5, 67.5, "↗"), (67.5, 112.5, "→"),
            (112.5, 157.5, "↘"), (157.5, 202.5, "↓"), (202.5, 247.5, "↙"),
            (247.5, 292.5, "←"), (292.5, 337.5, "↖"), (337.5, 360, "↑"),
        ]
        for start, end, arrow in sectors:
            if start <= relative_angle < end:
                return arrow
        return "↑"

    # ==================== СТАТИСТИКА ====================

    def get_stats(self) -> Dict:
        return {
            "current_room": self.current_room,
            "current_location": self.current_location,
            "pose": self.current_pose,
            "doors_count": len(self.room_doors),
            "has_map": self.room_map is not None
        }