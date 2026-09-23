#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/network/__init__.py — ПЛАГИН СЕТИ (WEBSOCKET)                        ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Управляет WebSocket соединением с сервером                               ║
║   • Автоматическое переподключение при разрыве                               ║
║   • Маршрутизация входящих сообщений к другим плагинам                       ║
║   • Отправка команд (движение, захват кадра, и т.д.)                         ║
║                                                                              ║
║ МАРШРУТИЗАЦИЯ СООБЩЕНИЙ:                                                     ║
║   • reflex → MotionPlugin (аварийная остановка) + MemoryPlugin              ║
║   • frame_data → VisionPlugin (кадр для VLM)                                 ║
║   • human_query → LLMPlugin (обработка речи)                                 ║
║   • motor_command → MotionPlugin (команда от оператора)                      ║
║   • freeze_agent → напрямую в агента                                         ║
║   • debug_prompt → напрямую в агента                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, Callable, List

from plugins.plugin import Plugin

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ websockets library not installed")

logger = logging.getLogger(__name__)

# ================================================================
# WEBSOCKET КЛИЕНТ
# ================================================================

class WebSocketClient:
    """
    WebSocket клиент с автопереподключением.
    
    Отправляет и принимает сообщения асинхронно.
    При разрыве соединения автоматически переподключается.
    """
    
    def __init__(self, url: str, on_message: Callable, reconnect_delay: float = 3.0):
        self.url = url
        self.on_message = on_message
        self.reconnect_delay = reconnect_delay
        
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def connect(self):
        """Запускает клиент (подключается и слушает)"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
    
    async def _run(self):
        """Основной цикл: подключение и переподключение"""
        while self._running:
            try:
                logger.info(f"🔌 Connecting to {self.url}...")
                self.ws = await websockets.connect(self.url)
                self.connected = True
                logger.info("✅ WebSocket connected")
                
                # Регистрируемся как агент
                await self.send({"type": "register", "role": "agent"})
                
                # Слушаем сообщения
                async for message in self.ws:
                    try:
                        data = json.loads(message)
                        await self.on_message(data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON: {message[:100]}")
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")
                
            except websockets.ConnectionClosed:
                logger.warning("Connection closed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.connected = False
                self.ws = None
                
                if self._running:
                    logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)
    
    async def send(self, data: Dict[str, Any]):
        """Отправляет сообщение"""
        if self.ws and self.connected:
            try:
                await self.ws.send(json.dumps(data))
                logger.debug(f"📤 Sent: {data.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"Send error: {e}")
                self.connected = False
        else:
            logger.warning("Not connected, message not sent")
    
    async def stop(self):
        """Останавливает клиент"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self.ws:
            await self.ws.close()
        
        logger.info("WebSocket client stopped")
    
    def is_connected(self) -> bool:
        return self.connected and self.ws is not None


# ================================================================
# ПЛАГИН СЕТИ
# ================================================================

class NetworkPlugin(Plugin):
    """
    Плагин сети — управляет WebSocket соединением и маршрутизацией сообщений.
    
    Все входящие сообщения распределяются по другим плагинам.
    Агент получает только критически важные сообщения (freeze, debug).
    """
    
    name = "network"
    version = "1.0"
    dependencies = []  # загружается первым, т.к. другие плагины могут зависеть от сети
    
    def __init__(self, agent):
        super().__init__(agent)
        
        self.ws_client: Optional[WebSocketClient] = None
        
        # Очередь для отложенной отправки (если нужно)
        self._send_queue: asyncio.Queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None
        
        # URL сервера из конфига
        self.ws_url = None
    
    async def on_load(self):
        """Инициализация WebSocket клиента"""
        
        self.ws_url = self.agent.get_config('WS_URL', 'ws://127.0.0.1:5002')
        
        self.ws_client = WebSocketClient(
            url=self.ws_url,
            on_message=self._on_message,
            reconnect_delay=3.0
        )
        
        await self.ws_client.connect()
        
        # Запускаем обработчик очереди отправки
        self._send_task = asyncio.create_task(self._process_send_queue())
        
        logger.info(f"✅ NetworkPlugin loaded (server: {self.ws_url})")
        await super().on_load()
    
    async def on_unload(self):
        """Остановка клиента"""
        if self.ws_client:
            await self.ws_client.stop()
        
        if self._send_task:
            self._send_task.cancel()
        
        await super().on_unload()
    
    # ==================== ОТПРАВКА СООБЩЕНИЙ ====================
    
    async def send(self, data: Dict[str, Any]):
        """Отправляет сообщение (асинхронно, не блокирует)"""
        if self.ws_client:
            await self.ws_client.send(data)
    
    async def send_command(self, capability: str, data: Dict = None, target: str = "esp"):
        """Отправляет команду в формате протокола v5.0"""
        from core.protocol_v5 import RobotProtocolV5
        
        msg = RobotProtocolV5.create_message(
            source="agent",
            capability=capability,
            data=data or {},
            target=target,
            source_type="agent"
        )
        await self.send(msg)
    
    async def send_motor_command(self, left: int, right: int,
                                  duration: float = None,
                                  distance: float = None,
                                  angle: float = None):
        """Отправляет команду движения"""
        from core.protocol_v5 import RobotProtocolV5
        
        msg = RobotProtocolV5.create_motor_command(
            left=left, right=right,
            duration=duration,
            distance=distance,
            angle=angle,
            source="agent"
        )
        await self.send(msg)
    
    async def capture_frame(self) -> asyncio.Future:
        """
        Запрашивает кадр с камеры.
        Возвращает Future, который завершится при получении frame_data.
        """
        frame_future = asyncio.Future()
        self._pending_frame = frame_future
        await self.send({"type": "capture_frame"})
        return frame_future
    
    async def _process_send_queue(self):
        """Обрабатывает очередь отправки (для rate limiting)"""
        while self._loaded:
            try:
                data = await self._send_queue.get()
                if self.ws_client:
                    await self.ws_client.send(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send queue error: {e}")
    
    # ==================== ПРИЁМ СООБЩЕНИЙ ====================
    
    async def _on_message(self, data: Dict[str, Any]):
        """
        Обработчик входящих сообщений.
        Маршрутизирует сообщения по плагинам.
        """
        msg_type = data.get('type')
        capability = data.get('capability')
        
        # === КРИТИЧЕСКИЕ СООБЩЕНИЯ (в агент) ===
        
        # Заморозка / разморозка агента
        if msg_type == 'freeze_agent':
            self.agent.freeze()
            await self.send({"type": "agent_frozen"})
            return
        
        if msg_type == 'resume_agent':
            self.agent.resume()
            await self.send({"type": "agent_resumed"})
            return
        
        # Отладочный промпт от дашборда
        if msg_type == 'debug_prompt':
            asyncio.create_task(self._handle_debug_prompt(data))
            return
        
        # === МАРШРУТИЗАЦИЯ ПО ПЛАГИНАМ ===
        
        # Рефлексы TinyML → MotionPlugin + MemoryPlugin
        if msg_type == 'reflex' or capability == 'tinyml.reflex':
            motion_plugin = self.agent.get_plugin("motion")
            if motion_plugin:
                await motion_plugin.on_message(msg_type, data)
            
            memory_plugin = self.agent.get_plugin("memory")
            if memory_plugin:
                await memory_plugin.on_message(msg_type, data)
            
            # Также уведомляем агента (для статистики)
            self.agent.inc_stat("reflexes_received")
            return
        
        # Кадр с камеры → VisionPlugin
        if msg_type == 'frame_data':
            vision_plugin = self.agent.get_plugin("vision")
            if vision_plugin:
                await vision_plugin.on_message(msg_type, data)
            
            # Если есть ожидающий Future — завершаем его
            if hasattr(self, '_pending_frame') and not self._pending_frame.done():
                self._pending_frame.set_result(data.get('image', ''))
                delattr(self, '_pending_frame')
            return
        
        # Голосовой запрос от оператора → SpeechPlugin → LLMPlugin
        if msg_type == 'human_query':
            text = data.get('text', '')
            if text:
                logger.info(f"👤 Human query: {text}")
                
                # Обновляем таймер простоя в TasksPlugin
                tasks_plugin = self.agent.get_plugin("tasks")
                if tasks_plugin:
                    tasks_plugin.on_user_command()
                
                # Передаём в SpeechPlugin (который вызовет LLM)
                speech_plugin = self.agent.get_plugin("speech")
                if speech_plugin:
                    # Эмулируем распознанную речь
                    await speech_plugin._on_speech(text, wake=False, interactive=False)
                else:
                    # Fallback: напрямую в агента
                    await self.agent.on_human_command(text, wake=False, interactive=False)
            return
        
        # Команда движения от оператора → MotionPlugin
        if msg_type == 'motor_command':
            motion_plugin = self.agent.get_plugin("motion")
            if motion_plugin:
                await motion_plugin.on_message(msg_type, data)
            return
        
        # Результат выполнения команды
        if msg_type == 'execution_result':
            # Пока просто логируем, в будущем можно маршрутизировать
            command_id = data.get('in_response_to')
            result = data.get('data', {})
            logger.debug(f"Execution result for {command_id}: {result.get('executed', False)}")
            return
        
        # Подтверждение регистрации
        if msg_type == 'registered':
            logger.info("✅ Registered on server")
            self.agent.connected = True
            return
        
        # Батарея (системная телеметрия)
        if msg_type == 'battery':
            level = data.get('level', 0)
            charging = data.get('charging', False)
            logger.info(f"🔋 Battery: {level}% {'(charging)' if charging else ''}")
            
            # Автоматическое управление зарядкой (по желанию)
            if level < 20 and not charging:
                # Отправляем команду на зарядку
                await self.send_command("power.charge", {"enable": True})
            return
        
        # Одометрия → SensorsPlugin
        if capability == 'sensor.odometry':
            sensors_plugin = self.agent.get_plugin("sensors")
            if sensors_plugin:
                await sensors_plugin.on_message(msg_type, data)
            return
        
        # Лидар → SensorsPlugin
        if capability == 'sensor.lidar.scan':
            sensors_plugin = self.agent.get_plugin("sensors")
            if sensors_plugin:
                await sensors_plugin.on_message(msg_type, data)
            return
        
        # Неизвестное сообщение
        logger.debug(f"Unhandled message: {msg_type or capability}")
    
    async def _handle_debug_prompt(self, data: Dict):
        """Обрабатывает отладочный промпт от дашборда"""
        prompt = data.get('prompt', '')
        if not prompt:
            return
        
        logger.info(f"🐛 Debug prompt: {prompt[:100]}...")
        
        # Собираем контекст
        sensor_summary = ""
        sensor_memory = self.agent.get_sensor_memory()
        if sensor_memory:
            summaries = sensor_memory.get_summaries()
            sensor_summary = "\n".join([f"{k}: {v['summary']}" for k, v in summaries.items()])
        
        full_prompt = f"СИТУАЦИЯ:\n{sensor_summary}\n\nВОПРОС ОТ РАЗРАБОТЧИКА:\n{prompt}\n\nОтветь подробно, на человеческом языке."
        
        # Вызываем LLM
        llm_plugin = self.agent.get_plugin("llm")
        if llm_plugin:
            response = await llm_plugin.call(full_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
        else:
            response_text = "LLM plugin not available"
        
        # Отправляем ответ
        await self.send({
            "type": "debug_response",
            "prompt": prompt,
            "response": response_text
        })
    
    # ==================== API ДЛЯ АГЕНТА ====================
    
    def is_connected(self) -> bool:
        """Проверяет соединение с сервером"""
        return self.ws_client is not None and self.ws_client.is_connected()
    
    # ==================== СТАТИСТИКА ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика плагина"""
        return {
            "name": self.name,
            "version": self.version,
            "connected": self.is_connected(),
            "server_url": self.ws_url
        }