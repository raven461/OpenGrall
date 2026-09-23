#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/vision/__init__.py — ПЛАГИН КОМПЬЮТЕРНОГО ЗРЕНИЯ (VLM)               ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Фоновое сканирование камеры через VLM (асинхронно)                       ║
║   • Предоставляет инструменты для работы со зрением                          ║
║   • Фокусировка на объектах (focus_on)                                       ║
║   • Ответы на вопросы о кадре (ask_vlm)                                      ║
║                                                                              ║
║ ВАЖНО: ВСЕ ЗАПРОСЫ К VLM НА АНГЛИЙСКОМ!                                      ║
║   VLM плохо понимают русский, поэтому все промпты на английском.             ║
║   Ответы VLM тоже приходят на английском.                                    ║
║                                                                              ║
║ ИНСТРУМЕНТЫ:                                                                 ║
║   • focus_on(target) — найти объект в кадре, вернуть координаты              ║
║   • ask_vlm(prompt) — задать произвольный вопрос на английском               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import base64
import cv2
import numpy as np
import time
import logging
import json
import re
from typing import Optional, Dict, Any, List
from collections import deque

from plugins.plugin import Plugin
from orchestration.tool import Tool

logger = logging.getLogger(__name__)


# ================================================================
# VLM КЛИЕНТ (ЗАГРУЖАЕТ МОДЕЛЬ)
# ================================================================

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ transformers не установлен, VLM недоступен")


class VLMClient:
    """
    Клиент для визуальной языковой модели.
    Загружает модель один раз при старте.
    ВСЕ ПРОМПТЫ НА АНГЛИЙСКОМ!
    """
    
    def __init__(self, model_name: str = "vikhr/vikhr-vl-2b",
                 load_in_4bit: bool = True,
                 max_new_tokens: int = 100):
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers не установлен")
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Автоопределение устройства
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🔄 Загружаю VLM: {model_name} на {self.device}...")
        
        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        
        if load_in_4bit and self.device == "cuda":
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
            logger.info("   📦 4-bit quantization enabled")
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        if self.device == "cpu" and not load_in_4bit:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"✅ VLM загружена ({self.device})")
    
    async def analyze_scene(self, image: np.ndarray, prompt: str) -> str:
        """
        Анализирует сцену с заданным промптом.
        Промпт ДОЛЖЕН быть на английском!
        """
        return await self._generate(image, prompt)
    
    async def analyze_with_focus(self, image: np.ndarray, target: str) -> Dict:
        """
        Фокусировка на объекте.
        target — название объекта на английском (person, chair, remote, etc.)
        """
        prompt = f"""You are a robot's vision system. Look at the image and find the {target}.
Answer ONLY in JSON format:
{{
  "found": true/false,
  "object": "name of object found or null",
  "distance_cm": number (estimated distance in cm) or null,
  "offset_x_deg": number (horizontal offset from center, negative=left, positive=right) or null,
  "offset_y_deg": number (vertical offset from center, negative=down, positive=up) or null,
  "orientation": "left/right/front/back/unknown" or null,
  "confidence": number from 0 to 1
}}
If object not found, return: {{"found": false, "confidence": 0}}
NO extra text, ONLY JSON."""
        
        response = await self._generate(image, prompt)
        return self._parse_focus_response(response)
    
    async def answer_question(self, image: np.ndarray, question: str) -> str:
        """
        Отвечает на вопрос о кадре.
        Вопрос ДОЛЖЕН быть на английском!
        """
        prompt = f"""{question}

Answer concisely based ONLY on what you see in the image.
If you cannot determine the answer, say "unknown".
Reply with a short phrase or single word. No JSON, no extra text."""
        
        return await self._generate(image, prompt)
    
    async def _generate(self, image: np.ndarray, prompt: str) -> str:
        """Внутренний метод генерации ответа VLM"""
        # Конвертируем BGR (OpenCV) в RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        image_pil = Image.fromarray(image_rgb)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=0.0,
            )
        
        output_text = self.processor.decode(
            generated_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return output_text.strip()
    
    def _parse_focus_response(self, response: str) -> Dict:
        """Парсит JSON ответ от VLM в режиме фокусировки"""
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "found": data.get("found", False),
                    "object": data.get("object"),
                    "distance_cm": data.get("distance_cm"),
                    "offset_x_deg": data.get("offset_x_deg"),
                    "offset_y_deg": data.get("offset_y_deg"),
                    "orientation": data.get("orientation"),
                    "confidence": data.get("confidence", 0.0),
                    "raw_response": response
                }
            except json.JSONDecodeError:
                pass
        
        return {
            "found": False,
            "object": None,
            "distance_cm": None,
            "offset_x_deg": None,
            "offset_y_deg": None,
            "orientation": None,
            "confidence": 0.0,
            "raw_response": response
        }


# ================================================================
# VLM СКАНЕР (ФОНОВЫЙ)
# ================================================================

class VLMScanner:
    """
    Фоновый сканер VLM.
    Асинхронно анализирует кадры и обновляет SensorMemory.
    """
    
    def __init__(self, vlm_client: VLMClient, ws_client,
                 scan_interval: float = 0.5,
                 idle_threshold: float = 10.0,
                 frame_timeout: float = 2.0):
        
        self.vlm = vlm_client
        self.ws = ws_client
        self.scan_interval = scan_interval
        self.idle_threshold = idle_threshold
        self.frame_timeout = frame_timeout
        
        self.latest_result: Optional[Dict] = None
        self.result_history = deque(maxlen=10)
        self.last_scan_time = 0
        self.is_running = False
        self.is_active = False
        self.last_activity_time = 0
        self._pending_question: Optional[str] = None
        
        self.sensor_memory = None
        self.weight_calculator = None
        
        self.stats = {
            "scans_completed": 0,
            "scans_failed": 0,
            "questions_answered": 0
        }
        
        logger.info("✅ VLMScanner initialized")
    
    def set_sensor_memory(self, memory):
        self.sensor_memory = memory
    
    def set_weight_calculator(self, calc):
        self.weight_calculator = calc
    
    def set_pending_question(self, question: str):
        """Устанавливает вопрос от LLM (на английском!)"""
        self._pending_question = question
        logger.debug(f"📝 Pending question: {question[:50]}...")
    
    def on_movement(self):
        """Активирует сканер при движении"""
        self.last_activity_time = time.time()
        if not self.is_active:
            self.is_active = True
    
    def on_command(self):
        """Активирует сканер при команде"""
        self.last_activity_time = time.time()
        if not self.is_active:
            self.is_active = True
    
    def get_latest(self) -> Optional[Dict]:
        """Возвращает последний результат сканирования"""
        if self.latest_result:
            age = time.time() - self.last_scan_time
            self.latest_result["age"] = age
        return self.latest_result
    
    async def start(self):
        self.is_running = True
        asyncio.create_task(self._scan_loop())
        logger.info("🔄 VLMScanner started")
    
    async def stop(self):
        self.is_running = False
    
    async def _scan_loop(self):
        while self.is_running:
            if self.is_active or self._pending_question:
                await self._perform_scan()
                await asyncio.sleep(self.scan_interval)
            else:
                idle = time.time() - self.last_activity_time
                if idle > self.idle_threshold:
                    self.is_active = False
                await asyncio.sleep(1.0)
    
    async def _perform_scan(self):
        try:
            start_time = time.time()
            
            # Захват кадра
            frame = await self._capture_frame()
            if frame is None:
                self.stats["scans_failed"] += 1
                return
            
            result = {"timestamp": time.time(), "source_type": "vlm", "data": {}}
            
            # Ответ на вопрос (если есть)
            if self._pending_question:
                question = self._pending_question
                self._pending_question = None
                
                answer = await self.vlm.answer_question(frame, question)
                result["data"]["answer_llm"] = answer
                self.stats["questions_answered"] += 1
                logger.debug(f"❓ Q: {question[:50]}... → A: {answer[:50]}...")
            
            # Стандартный анализ сцены (всегда)
            scene_prompt = """Analyze the image. Return ONLY JSON:
{
  "scene": "what is this place (corridor, kitchen, living_room, office, street, unknown)",
  "objects": [{"name": "object name in English", "distance": number in meters, "position": "left/right/center/front"}],
  "path_status": "free/occupied/unknown",
  "free_space": {"front": number, "left": number, "right": number}
}
If unsure, use "unknown". NO extra text, ONLY JSON."""
            
            scene_text = await self.vlm.analyze_scene(frame, scene_prompt)
            parsed = self._parse_scene_response(scene_text)
            result["data"].update(parsed)
            
            # Обогащаем весами
            if self.weight_calculator:
                enriched = self.weight_calculator.process_with_meta(
                    source_type="vlm",
                    data=result["data"],
                    timestamp=result["timestamp"],
                    priority=4,
                    confidence=0.7,
                    latency=time.time() - start_time
                )
                result["data"] = enriched
                result["weight"] = enriched.get("_meta", {}).get("weight", 0.7)
            
            # Сохраняем в SensorMemory
            if self.sensor_memory:
                self.sensor_memory.update(
                    source="vlm",
                    data=result["data"],
                    weight=result.get("weight", 0.7),
                    meta={"scan_timestamp": result["timestamp"]}
                )
            
            self.latest_result = result
            self.last_scan_time = time.time()
            self.stats["scans_completed"] += 1
            
        except Exception as e:
            logger.error(f"VLM scan error: {e}")
            self.stats["scans_failed"] += 1
    
    async def _capture_frame(self) -> Optional[np.ndarray]:
        """Захватывает кадр через WebSocket"""
        try:
            frame_future = asyncio.Future()
            if hasattr(self.ws, 'pending_frame'):
                self.ws.pending_frame = frame_future
            else:
                self.ws._temp_frame_future = frame_future
            
            await self.ws.send({"type": "capture_frame"})
            image_b64 = await asyncio.wait_for(frame_future, timeout=self.frame_timeout)
            
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
        except asyncio.TimeoutError:
            logger.warning(f"Frame capture timeout ({self.frame_timeout}s)")
            return None
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
        finally:
            if hasattr(self.ws, '_temp_frame_future'):
                delattr(self.ws, '_temp_frame_future')
            if hasattr(self.ws, 'pending_frame'):
                self.ws.pending_frame = None
    
    def _parse_scene_response(self, text: str) -> Dict:
        """Парсит JSON ответ VLM о сцене"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "scene": data.get("scene", "unknown"),
                    "room_type": data.get("room_type", data.get("scene", "unknown")),
                    "objects": data.get("objects", []),
                    "path_status": data.get("path_status", "unknown"),
                    "free_space": data.get("free_space", {"front": 10.0, "left": 10.0, "right": 10.0})
                }
            except json.JSONDecodeError:
                pass
        
        return {
            "scene": "unknown",
            "room_type": "unknown",
            "objects": [],
            "path_status": "unknown",
            "free_space": {"front": 10.0, "left": 10.0, "right": 10.0}
        }
    
    def get_stats(self) -> Dict:
        return {
            "scans_completed": self.stats["scans_completed"],
            "scans_failed": self.stats["scans_failed"],
            "questions_answered": self.stats["questions_answered"],
            "is_active": self.is_active
        }


# ================================================================
# ИНСТРУМЕНТЫ ЗРЕНИЯ (исправлены на **kwargs)
# ================================================================

class FocusTool(Tool):
    """Фокусировка на объекте"""
    name = "focus_on"
    description = "Find an object in the camera frame. Target name in ENGLISH (person, chair, remote, etc.)"
    latency = 1.5

    def __init__(self, agent, vision_plugin):
        self.agent = agent
        self.plugin = vision_plugin

    async def forward(self, **kwargs) -> str:
        target = kwargs.get('target')
        if not target:
            return "Error: missing 'target' parameter. Example: focus_on(target='person')"
        
        if not self.plugin.vlm:
            return "Error: VLM not available"

        frame = await self.plugin.capture_frame()
        if frame is None:
            return "Failed to capture frame"

        try:
            result = await self.plugin.vlm.analyze_with_focus(frame, target)
        except Exception as e:
            logger.error(f"focus_on error: {e}")
            return f"Analysis error: {e}"

        if not result.get("found"):
            return f"Object '{target}' not found (confidence: {result.get('confidence', 0):.2f})"

        obj = result.get("object", target)
        distance = result.get("distance_cm", "?")
        offset_x = result.get("offset_x_deg", 0)
        offset_y = result.get("offset_y_deg", 0)

        dir_x = "center" if abs(offset_x) < 5 else ("left" if offset_x < 0 else "right")
        dir_y = "center" if abs(offset_y) < 5 else ("down" if offset_y < 0 else "up")

        return (f"Object '{obj}' found. Distance: {distance} cm. "
                f"Position: {dir_x} ({offset_x:.1f}°), {dir_y} ({offset_y:.1f}°). "
                f"Confidence: {result.get('confidence', 0):.2f}")


class AskVLMTool(Tool):
    """Задать вопрос VLM (на английском!)"""
    name = "ask_vlm"
    description = "Ask VLM a question about the current camera frame. Question MUST be in ENGLISH!"
    latency = 0.05

    def __init__(self, agent, vision_plugin):
        self.agent = agent
        self.plugin = vision_plugin

    async def forward(self, **kwargs) -> str:
        prompt = kwargs.get('prompt')
        if not prompt:
            return "Error: missing 'prompt' parameter. Example: ask_vlm(prompt='What color is the wall?')"
        
        if not self.plugin.vlm_scanner:
            return "Error: VLM Scanner not available"

        # Проверяем, что промпт на английском (грубая проверка)
        if any(c in prompt for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
            logger.warning(f"⚠️ ask_vlm got Russian prompt: {prompt[:50]}...")
            return "Please ask your question in ENGLISH. VLM does not understand Russian well."

        self.plugin.vlm_scanner.set_pending_question(prompt)
        logger.info(f"🔍 ask_vlm: {prompt[:50]}...")
        return f"Question sent: '{prompt[:50]}...'. Answer will appear in the next frame."


class RememberObjectTool(Tool):
    """Запомнить видимый объект"""
    name = "remember_object"
    description = "Remember the currently visible object"
    latency = 1.0

    def __init__(self, agent, vision_plugin):
        self.agent = agent
        self.plugin = vision_plugin

    async def forward(self, **kwargs) -> str:
        name = kwargs.get('name')
        if not name:
            return "Error: missing 'name' parameter. Example: remember_object(name='my_cup')"
        
        frame = await self.plugin.capture_frame()
        if frame is None:
            return "Failed to capture frame"

        memory_plugin = self.agent.get_plugin("memory")
        if memory_plugin and memory_plugin.get_visual_memory():
            memory_plugin.get_visual_memory().save_object(name, frame)
            return f"Object '{name}' remembered"
        
        return "Visual memory not available"


class FindObjectTool(Tool):
    """Найти запомненный объект"""
    name = "find_object"
    description = "Find previously remembered object"
    latency = 1.0

    def __init__(self, agent, vision_plugin):
        self.agent = agent
        self.plugin = vision_plugin

    async def forward(self, **kwargs) -> str:
        name = kwargs.get('name')
        if not name:
            return "Error: missing 'name' parameter. Example: find_object(name='my_cup')"
        
        frame = await self.plugin.capture_frame()
        if frame is None:
            return "Failed to capture frame"

        memory_plugin = self.agent.get_plugin("memory")
        if memory_plugin and memory_plugin.get_visual_memory():
            result = memory_plugin.get_visual_memory().find_object(name, frame)
            if result.get("found"):
                return f"Object '{name}' found! (method: {result.get('method')}, confidence: {result.get('confidence', 0):.2f})"
            return f"Object '{name}' not found"
        
        return "Visual memory not available"


class SearchByTextTool(Tool):
    """Поиск объектов по текстовому описанию"""
    name = "search_by_text"
    description = "Search remembered objects by text description"
    latency = 0.3

    def __init__(self, agent, vision_plugin):
        self.agent = agent
        self.plugin = vision_plugin

    async def forward(self, **kwargs) -> str:
        query = kwargs.get('query')
        if not query:
            return "Error: missing 'query' parameter. Example: search_by_text(query='red cup')"
        
        memory_plugin = self.agent.get_plugin("memory")
        if memory_plugin and memory_plugin.get_visual_memory():
            results = memory_plugin.get_visual_memory().search_by_text(query)
            if results:
                names = [r["name"] for r in results[:3]]
                return f"Found: {', '.join(names)}"
            return "Nothing found"
        
        return "Visual memory not available"


# ================================================================
# ПЛАГИН ЗРЕНИЯ
# ================================================================

class VisionPlugin(Plugin):
    """
    Плагин компьютерного зрения.
    
    Управляет VLM: фоновое сканирование, фокусировка, ответы на вопросы.
    ВСЕ ЗАПРОСЫ К VLM НА АНГЛИЙСКОМ!
    """
    
    name = "vision"
    version = "1.0"
    dependencies = []  # независим
    
    def __init__(self, agent):
        super().__init__(agent)
        
        self.vlm: Optional[VLMClient] = None
        self.vlm_scanner: Optional[VLMScanner] = None
        self._pending_frame: Optional[asyncio.Future] = None
    
    async def on_load(self):
        """Инициализация VLM и сканера"""
        
        model_name = self.agent.get_config('VLM_MODEL')
        if not model_name or not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ VLM not configured, vision features disabled")
            await super().on_load()
            return
        
        try:
            load_in_4bit = self.agent.get_config('VLM_LOAD_IN_4BIT', True)
            self.vlm = VLMClient(model_name=model_name, load_in_4bit=load_in_4bit)
            
            scan_interval = self.agent.get_config('VLM_SCAN_INTERVAL', 0.5)
            idle_threshold = self.agent.get_config('VLM_IDLE_THRESHOLD', 10.0)
            frame_timeout = self.agent.get_config('VLM_FRAME_TIMEOUT', 2.0)
            
            # Получаем WebSocket клиент из NetworkPlugin
            network_plugin = self.agent.get_plugin("network")
            ws_client = network_plugin.ws_client if network_plugin else None
            
            self.vlm_scanner = VLMScanner(
                vlm_client=self.vlm,
                ws_client=ws_client,
                scan_interval=scan_interval,
                idle_threshold=idle_threshold,
                frame_timeout=frame_timeout
            )
            
            # Подключаем к SensorMemory и WeightCalculator
            if hasattr(self.agent, 'sensor_memory'):
                self.vlm_scanner.set_sensor_memory(self.agent.sensor_memory)
            if hasattr(self.agent, 'weight_calculator'):
                self.vlm_scanner.set_weight_calculator(self.agent.weight_calculator)
            
            await self.vlm_scanner.start()
            logger.info("✅ VLM Scanner started")
            
        except Exception as e:
            logger.warning(f"⚠️ VLM initialization failed: {e}")
            self.vlm = None
            self.vlm_scanner = None
        
        await super().on_load()
    
    async def on_unload(self):
        """Остановка сканера"""
        if self.vlm_scanner:
            await self.vlm_scanner.stop()
        
        await super().on_unload()
    
    # ==================== ОСНОВНЫЕ МЕТОДЫ ====================
    
    def get_latest_scan(self) -> Optional[Dict]:
        """Возвращает последний скан VLM"""
        if self.vlm_scanner:
            return self.vlm_scanner.get_latest()
        return None
    
    def on_movement(self):
        """Уведомляет сканер о движении"""
        if self.vlm_scanner:
            self.vlm_scanner.on_movement()
    
    def on_command(self):
        """Уведомляет сканер о команде"""
        if self.vlm_scanner:
            self.vlm_scanner.on_command()
    
    async def capture_frame(self) -> Optional[np.ndarray]:
        """Захватывает текущий кадр с камеры"""
        network_plugin = self.agent.get_plugin("network")
        if not network_plugin or not network_plugin.ws_client:
            return None
        
        try:
            self._pending_frame = asyncio.Future()
            await network_plugin.ws_client.send({"type": "capture_frame"})
            image_b64 = await asyncio.wait_for(self._pending_frame, timeout=2.0)
            
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
        except asyncio.TimeoutError:
            logger.warning("Frame capture timeout")
            return None
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
        finally:
            self._pending_frame = None
    
    # ==================== ОБРАБОТКА СООБЩЕНИЙ ====================
    
    async def on_message(self, msg_type: str, data: Dict[str, Any]):
        """Обработка входящих сообщений (кадры от камеры)"""
        if msg_type == 'frame_data' and self._pending_frame:
            image = data.get('image', '')
            if image:
                self._pending_frame.set_result(image)
    
    # ==================== ИНСТРУМЕНТЫ ====================
    
    def get_tools(self) -> List[Tool]:
        """Возвращает инструменты зрения"""
        tools = [
            FocusTool(self.agent, self),
            AskVLMTool(self.agent, self),
        ]
        
        # Добавляем инструменты визуальной памяти, если есть Memory-плагин
        memory_plugin = self.agent.get_plugin("memory")
        if memory_plugin and memory_plugin.get_visual_memory():
            tools.extend([
                RememberObjectTool(self.agent, self),
                FindObjectTool(self.agent, self),
                SearchByTextTool(self.agent, self),
            ])
        
        return tools
    
    # ==================== СТАТИСТИКА ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику плагина"""
        stats = {
            "name": self.name,
            "version": self.version,
            "vlm_available": self.vlm is not None,
        }
        
        if self.vlm_scanner:
            stats.update(self.vlm_scanner.get_stats())
        
        return stats