#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/motion/__init__.py — ПЛАГИН ДВИЖЕНИЯ                                 ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Предоставляет инструменты движения (move_forward, turn_left и т.д.)     ║
║   • Предоставляет инструменты света и ожидания                               ║
║   • Управляет записью и выполнением маршрутов                                ║
║   • Обрабатывает рефлексы TinyML (экстренные остановки)                      ║
║                                                                              ║
║ ИНСТРУМЕНТЫ:                                                                 ║
║   • move_forward(speed, duration, distance) — движение вперёд               ║
║   • move_backward(speed, duration, distance) — движение назад               ║
║   • turn_left(speed, duration, angle) — поворот налево                      ║
║   • turn_right(speed, duration, angle) — поворот направо                    ║
║   • stop() — экстренная остановка                                           ║
║   • wait(seconds) — ожидание                                                ║
║   • set_light(state) — управление подсветкой                                ║
║   • record_route(action, name) — запись маршрута                            ║
║   • execute_route(name) — выполнение маршрута                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List

from plugins.plugin import Plugin
from orchestration.tool import Tool

logger = logging.getLogger(__name__)


# ================================================================
# ИНСТРУМЕНТЫ ДВИЖЕНИЯ (исправлены на **kwargs)
# ================================================================

class MoveForwardTool(Tool):
    name = "move_forward"
    description = "Двигаться вперёд. speed: 0-1023 (512=половина), duration: секунды, distance: метры"
    latency = 0.5

    def __init__(self, agent, motion_plugin):
        self.agent = agent
        self.plugin = motion_plugin

    async def forward(self, **kwargs) -> str:
        speed = kwargs.get('speed', 512)
        duration = kwargs.get('duration')
        distance = kwargs.get('distance')
        
        # Уведомляем VLM о движении (если есть)
        vision_plugin = self.agent.get_plugin("vision")
        if vision_plugin:
            vision_plugin.on_movement()

        # Отправляем команду
        from core.protocol_v5 import RobotProtocolV5
        msg = RobotProtocolV5.create_motor_command(
            left=speed, right=speed,
            duration=duration,
            distance=distance,
            source="agent"
        )
        await self.agent.send_command(msg)
        
        # Если идёт запись маршрута — добавляем команду
        if self.plugin.is_recording:
            self.plugin.record_command("move_forward", {
                "speed": speed,
                "duration": duration,
                "distance": distance
            })
        
        return f"Движение вперёд (speed={speed})"


class MoveBackwardTool(Tool):
    name = "move_backward"
    description = "Двигаться назад. speed: 0-1023 (512=половина), duration: секунды, distance: метры"
    latency = 0.5

    def __init__(self, agent, motion_plugin):
        self.agent = agent
        self.plugin = motion_plugin

    async def forward(self, **kwargs) -> str:
        speed = kwargs.get('speed', 512)
        duration = kwargs.get('duration')
        distance = kwargs.get('distance')
        
        vision_plugin = self.agent.get_plugin("vision")
        if vision_plugin:
            vision_plugin.on_movement()

        from core.protocol_v5 import RobotProtocolV5
        msg = RobotProtocolV5.create_motor_command(
            left=-speed, right=-speed,
            duration=duration,
            distance=distance,
            source="agent"
        )
        await self.agent.send_command(msg)
        
        if self.plugin.is_recording:
            self.plugin.record_command("move_backward", {
                "speed": speed,
                "duration": duration,
                "distance": distance
            })
        
        return f"Движение назад (speed={speed})"


class TurnLeftTool(Tool):
    name = "turn_left"
    description = "Повернуть налево. speed: 0-1023 (512=половина), duration: секунды, angle: градусы"
    latency = 0.4

    def __init__(self, agent, motion_plugin):
        self.agent = agent
        self.plugin = motion_plugin

    async def forward(self, **kwargs) -> str:
        speed = kwargs.get('speed', 512)
        duration = kwargs.get('duration')
        angle = kwargs.get('angle')
        
        vision_plugin = self.agent.get_plugin("vision")
        if vision_plugin:
            vision_plugin.on_movement()

        from core.protocol_v5 import RobotProtocolV5
        msg = RobotProtocolV5.create_motor_command(
            left=-speed, right=speed,
            duration=duration,
            angle=angle,
            source="agent"
        )
        await self.agent.send_command(msg)
        
        if self.plugin.is_recording:
            self.plugin.record_command("turn_left", {
                "speed": speed,
                "duration": duration,
                "angle": angle
            })
        
        return f"Поворот налево (speed={speed})"


class TurnRightTool(Tool):
    name = "turn_right"
    description = "Повернуть направо. speed: 0-1023 (512=половина), duration: секунды, angle: градусы"
    latency = 0.4

    def __init__(self, agent, motion_plugin):
        self.agent = agent
        self.plugin = motion_plugin

    async def forward(self, **kwargs) -> str:
        speed = kwargs.get('speed', 512)
        duration = kwargs.get('duration')
        angle = kwargs.get('angle')
        
        vision_plugin = self.agent.get_plugin("vision")
        if vision_plugin:
            vision_plugin.on_movement()

        from core.protocol_v5 import RobotProtocolV5
        msg = RobotProtocolV5.create_motor_command(
            left=speed, right=-speed,
            duration=duration,
            angle=angle,
            source="agent"
        )
        await self.agent.send_command(msg)
        
        if self.plugin.is_recording:
            self.plugin.record_command("turn_right", {
                "speed": speed,
                "duration": duration,
                "angle": angle
            })
        
        return f"Поворот направо (speed={speed})"


class StopTool(Tool):
    name = "stop"
    description = "Экстренная остановка всех двигателей"
    latency = 0.1

    def __init__(self, agent):
        self.agent = agent

    async def forward(self, **kwargs) -> str:
        from core.protocol_v5 import RobotProtocolV5
        msg = RobotProtocolV5.create_message(
            source="agent",
            capability="motor.stop",
            source_type="agent"
        )
        await self.agent.send_command(msg)
        
        # Если есть активная задача — прерываем
        tasks_plugin = self.agent.get_plugin("tasks")
        if tasks_plugin and tasks_plugin.has_active_task():
            tasks_plugin.cancel_task()
        
        return "Двигатели остановлены"


class WaitTool(Tool):
    name = "wait"
    description = "Подождать указанное количество секунд"
    latency = 0.0

    async def forward(self, **kwargs) -> str:
        seconds = kwargs.get('seconds')
        if seconds is None:
            return "Error: missing 'seconds' parameter. Example: wait(seconds=5)"
        
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            return "Error: 'seconds' must be a positive number"
        
        self.latency = seconds
        await asyncio.sleep(seconds)
        return f"Ожидание {seconds}с завершено"


class SetLightTool(Tool):
    name = "set_light"
    description = "Включить/выключить подсветку. state: true или false"
    latency = 0.1

    def __init__(self, agent):
        self.agent = agent

    async def forward(self, **kwargs) -> str:
        state = kwargs.get('state')
        if state is None:
            return "Error: missing 'state' parameter. Example: set_light(state=True)"
        
        from core.protocol_v5 import RobotProtocolV5
        msg = RobotProtocolV5.create_message(
            source="agent",
            capability="light.set",
            data={"state": bool(state)},
            source_type="agent"
        )
        await self.agent.send_command(msg)
        return f"Свет {'включён' if state else 'выключен'}"


# ================================================================
# ИНСТРУМЕНТЫ МАРШРУТОВ (исправлены на **kwargs)
# ================================================================

class RecordRouteTool(Tool):
    name = "record_route"
    description = "Управление записью маршрута. action='start' или 'stop', name — имя маршрута"
    latency = 0.05

    def __init__(self, agent, motion_plugin):
        self.agent = agent
        self.plugin = motion_plugin

    async def forward(self, **kwargs) -> str:
        action = kwargs.get('action')
        if not action:
            return "Error: missing 'action' parameter. Use 'start' or 'stop'"
        
        name = kwargs.get('name')
        
        if action == "start":
            if not name:
                return "Error: missing 'name' parameter for start action"
            self.plugin.start_recording(name)
            return f"Запись маршрута '{name}' начата"

        elif action == "stop":
            if not self.plugin.is_recording:
                return "Нет активной записи"
            route_name = self.plugin.stop_recording()
            return f"Маршрут '{route_name}' сохранён ({self.plugin.get_route_length(route_name)} команд)"

        else:
            return f"Неизвестное действие: {action}. Используйте 'start' или 'stop'"


class ExecuteRouteTool(Tool):
    name = "execute_route"
    description = "Выполнить сохранённый маршрут"
    latency = 0.1

    def __init__(self, agent, motion_plugin):
        self.agent = agent
        self.plugin = motion_plugin

    async def forward(self, **kwargs) -> str:
        name = kwargs.get('name')
        if not name:
            return "Error: missing 'name' parameter. Example: execute_route(name='my_route')"
        
        if not self.plugin.route_memory:
            return "Память маршрутов недоступна"
        
        commands = self.plugin.route_memory.get_route(name)
        if not commands:
            return f"Маршрут '{name}' не найден"
        
        asyncio.create_task(self.plugin.execute_route(commands))
        return f"Начинаю выполнение маршрута '{name}' ({len(commands)} команд)"


# ================================================================
# ПЛАГИН ДВИЖЕНИЯ
# ================================================================

class MotionPlugin(Plugin):
    """
    Плагин управления движением.
    
    Предоставляет инструменты для движения, записи и выполнения маршрутов.
    Обрабатывает рефлексы от TinyML.
    """
    
    name = "motion"
    version = "1.0"
    dependencies = []  # независим от других плагинов
    
    def __init__(self, agent):
        super().__init__(agent)
        
        # Состояние записи маршрута
        self.is_recording = False
        self.recording_name: Optional[str] = None
        self.recording_commands: List[Dict] = []
        
        # Память маршрутов (будет получена из Memory-плагина)
        self.route_memory = None
        
        # Фоновые задачи
        self._route_task: Optional[asyncio.Task] = None
    
    async def on_load(self):
        """Инициализация плагина"""
        
        # Получаем память маршрутов из Memory-плагина
        memory_plugin = self.agent.get_plugin("memory")
        if memory_plugin:
            self.route_memory = memory_plugin.get_route_memory()
            if self.route_memory:
                logger.info(f"✅ Память маршрутов загружена ({len(self.route_memory.list_routes())} маршрутов)")
        else:
            logger.warning("⚠️ Memory-плагин не загружен, маршруты не будут сохраняться")
        
        await super().on_load()
    
    # ==================== УПРАВЛЕНИЕ МАРШРУТАМИ ====================
    
    def start_recording(self, name: str):
        """Начинает запись маршрута"""
        self.is_recording = True
        self.recording_name = name
        self.recording_commands = []
        logger.info(f"🎬 Начата запись маршрута: {name}")
    
    def record_command(self, action: str, params: Dict):
        """Записывает команду в текущий маршрут"""
        if self.is_recording:
            self.recording_commands.append({
                "action": action,
                "params": params,
                "timestamp": time.time()
            })
            logger.debug(f"📝 Записана команда: {action}")
    
    def stop_recording(self) -> Optional[str]:
        """Завершает запись маршрута и сохраняет его"""
        if not self.is_recording:
            return None
        
        name = self.recording_name
        if self.route_memory and self.recording_commands:
            # Убираем временные метки перед сохранением
            clean_commands = [
                {"action": cmd["action"], "params": cmd["params"]}
                for cmd in self.recording_commands
            ]
            self.route_memory.save_route(name, clean_commands)
            logger.info(f"💾 Маршрут '{name}' сохранён ({len(clean_commands)} команд)")
        
        self.is_recording = False
        self.recording_name = None
        self.recording_commands = []
        
        return name
    
    def get_route_length(self, name: str) -> int:
        """Возвращает длину маршрута"""
        if self.route_memory:
            route = self.route_memory.get_route(name)
            return len(route) if route else 0
        return 0
    
    async def execute_route(self, commands: List[Dict]):
        """Выполняет последовательность команд маршрута"""
        logger.info(f"🚶 Выполнение маршрута из {len(commands)} команд")
        
        for i, cmd in enumerate(commands):
            # Проверяем, не остановлен ли агент
            if not self.agent.is_running() or self.agent.is_frozen():
                logger.warning(f"⏸️ Выполнение маршрута прервано на шаге {i+1}")
                break
            
            action = cmd.get('action')
            params = cmd.get('params', {})
            
            # Выполняем команду через агента
            result = await self.agent.execute_tool(action, params)
            
            if result and "ошибка" in str(result).lower():
                logger.error(f"❌ Ошибка выполнения {action}: {result}")
                await self.agent.speak("Ошибка при выполнении маршрута")
                return
            
            # Небольшая пауза между командами
            await asyncio.sleep(0.3)
        
        logger.info("✅ Маршрут выполнен")
        await self.agent.speak("Маршрут выполнен")
    
    # ==================== ОБРАБОТКА РЕФЛЕКСОВ ====================
    
    async def on_message(self, msg_type: str, data: Dict[str, Any]):
        """Обработка сообщений от TinyML"""
        
        # Рефлекс от TinyML (экстренная остановка)
        if msg_type == 'reflex' or data.get('capability') == 'tinyml.reflex':
            reflex = data.get('reflex', data.get('data', {}))
            reflex_type = reflex.get('type', 'unknown')
            distance = reflex.get('distance_cm', 0)
            action_taken = reflex.get('action', 'stop')
            
            logger.warning(f"🚨 РЕФЛЕКС: {reflex_type} на {distance}см → {action_taken}")
            
            # Сохраняем в эпизодическую память
            memory_plugin = self.agent.get_plugin("memory")
            if memory_plugin:
                memory_plugin.add_reflex(
                    reflex_type=reflex_type,
                    distance_cm=distance,
                    action_taken=action_taken,
                    context={"raw": reflex, "timestamp": time.time()}
                )
            
            # Прерываем активную задачу
            tasks_plugin = self.agent.get_plugin("tasks")
            if tasks_plugin and tasks_plugin.has_active_task():
                tasks_plugin.fail_task(f"рефлекс {reflex_type}")
    
    # ==================== ИНСТРУМЕНТЫ ====================
    
    def get_tools(self) -> List[Tool]:
        """Возвращает инструменты движения и маршрутов"""
        return [
            MoveForwardTool(self.agent, self),
            MoveBackwardTool(self.agent, self),
            TurnLeftTool(self.agent, self),
            TurnRightTool(self.agent, self),
            StopTool(self.agent),
            WaitTool(),
            SetLightTool(self.agent),
            RecordRouteTool(self.agent, self),
            ExecuteRouteTool(self.agent, self),
        ]
    
    # ==================== СТАТИСТИКА ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику плагина"""
        routes_count = 0
        if self.route_memory:
            routes_count = len(self.route_memory.list_routes())
        
        return {
            "name": self.name,
            "version": self.version,
            "is_recording": self.is_recording,
            "recording_name": self.recording_name,
            "recorded_commands": len(self.recording_commands),
            "saved_routes": routes_count
        }