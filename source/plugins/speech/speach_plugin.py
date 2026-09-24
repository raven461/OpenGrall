#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/speech/__init__.py — ПЛАГИН РЕЧИ (СЛУХ И ГОЛОС)                      ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Распознавание речи через Vosk (офлайн, с wake word)                      ║
║   • Синтез речи через RHVoice / eSpeak (роботизированный голос)              ║
║   • Управление режимами (обычный / интерактивный)                            ║
║   • Аварийная остановка по голосовой команде                                 ║
║                                                                              ║
║ ИНСТРУМЕНТЫ:                                                                 ║
║   • speak(text, wait) — произнести текст, опционально ждать ответа          ║
║   • ask_human(question) — задать вопрос и запомнить ответ                    ║
║                                                                              ║
║ РЕЖИМЫ РАБОТЫ:                                                               ║
║   • Обычный — ждёт wake word «Гралл»                                         ║
║   • Интерактивный — слушает всё, сам инициирует диалог                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
import queue
import threading
import json
import numpy as np
import subprocess
import shutil
from typing import Optional, Dict, Any, List, Callable

from plugins.plugin import Plugin
from orchestration.tool import Tool

logger = logging.getLogger(__name__)

# ================================================================
# ПРОВЕРКА ДОСТУПНОСТИ БИБЛИОТЕК
# ================================================================

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("⚠️ Vosk не установлен, распознавание речи отключено")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False
    logger.warning("⚠️ PyAudio не установлен, микрофон недоступен")


# ================================================================
# ИНСТРУМЕНТЫ РЕЧИ (исправлены на **kwargs)
# ================================================================

class SpeakTool(Tool):
    """Инструмент: произнести текст"""
    name = "speak"
    description = "Произнести текст. Если wait=True — ждать ответа человека."
    latency = 0.5

    def __init__(self, agent, speech_plugin):
        self.agent = agent
        self.plugin = speech_plugin

    async def forward(self, **kwargs) -> str:
        text = kwargs.get('text')
        if not text:
            return "Error: missing 'text' parameter. Example: speak(text='Hello world')"
        
        wait = kwargs.get('wait', False)

        if wait:
            human_nearby = await self.plugin.is_human_nearby()
            if not human_nearby:
                return "Никого рядом нет"

        await self.plugin.speak(text)

        if wait:
            try:
                answer = await asyncio.wait_for(self.plugin.wait_for_speech(), timeout=10.0)
                if answer:
                    logger.info(f"👤 Человек ответил: {answer}")
                    
                    # Сохраняем в эпизодическую память
                    memory_plugin = self.agent.get_plugin("memory")
                    if memory_plugin:
                        memory_plugin.add_conversation(text, answer)
                    
                    return f"Человек ответил: {answer}"
                return "Человек не ответил"
            except asyncio.TimeoutError:
                return "Человек не ответил"

        return f"Сказано: {text[:50]}..."


class AskHumanTool(Tool):
    """Инструмент: спросить человека и запомнить ответ"""
    name = "ask_human"
    description = "Спросить человека о чём-то и запомнить ответ для будущего"
    latency = 0.5

    def __init__(self, agent, speech_plugin):
        self.agent = agent
        self.plugin = speech_plugin

    async def forward(self, **kwargs) -> str:
        question = kwargs.get('question')
        if not question:
            return "Error: missing 'question' parameter. Example: ask_human(question='Как тебя зовут?')"
        
        human_nearby = await self.plugin.is_human_nearby()
        if not human_nearby:
            return "Никого рядом нет"

        await self.plugin.speak(question)
        logger.info(f"🤖 Робот спрашивает: {question}")

        try:
            answer = await asyncio.wait_for(self.plugin.wait_for_speech(), timeout=10.0)
            if answer:
                logger.info(f"👤 Человек ответил: {answer}")
                
                # Сохраняем инструкцию в эпизодическую память
                memory_plugin = self.agent.get_plugin("memory")
                if memory_plugin:
                    memory_plugin.add_human_instruction(
                        question=question,
                        answer=answer,
                        context={
                            "asked_at": time.time(),
                            "intent": self.agent.get_primary_intent() if hasattr(self.agent, 'get_primary_intent') else None
                        }
                    )
                return f"Человек ответил: {answer}. Запомнил."
            return "Человек не ответил"
        except asyncio.TimeoutError:
            return "Человек не ответил"


# ================================================================
# TTS ДВИЖОК (СИНТЕЗ РЕЧИ)
# ================================================================

class TTSEngine:
    """Синтезатор речи с автовыбором движка (RHVoice / eSpeak)"""
    
    def __init__(self, voice: str = "ru", speed: int = 110, pitch: int = 45, amplitude: int = 100):
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.amplitude = amplitude
        self.engine = self._detect_engine()
        
        logger.info(f"✅ TTS движок: {self.engine}")
    
    def _detect_engine(self) -> str:
        """Определяет доступный TTS движок"""
        if shutil.which("RHVoice-client") or shutil.which("rhvoice-client"):
            return "rhvoice"
        if shutil.which("espeak"):
            return "espeak"
        logger.warning("⚠️ Нет TTS движка (ни RHVoice, ни eSpeak)")
        return "none"
    
    async def speak(self, text: str):
        """Произносит текст асинхронно"""
        if not text or self.engine == "none":
            return
        
        log_text = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"🗣️ Робот говорит: {log_text}")
        
        if self.engine == "rhvoice":
            await self._speak_rhvoice(text)
        elif self.engine == "espeak":
            await self._speak_espeak(text)
    
    async def _speak_rhvoice(self, text: str):
        """RHVoice (Александр)"""
        rh_rate = max(0.5, min(2.0, self.speed / 120))
        rh_pitch = max(-3, min(3, int((self.pitch - 50) / 10)))
        
        tmp_file = "/tmp/tts_text.txt"
        with open(tmp_file, "w") as f:
            f.write(text)
        
        client_cmd = None
        if shutil.which("RHVoice-client"):
            client_cmd = ["RHVoice-client", "-s", "Aleksandr"]
        elif shutil.which("rhvoice-client"):
            client_cmd = ["rhvoice-client", "-s", "Aleksandr"]
        
        if client_cmd:
            cmd = client_cmd + [
                "--rate", str(rh_rate),
                "--pitch", str(rh_pitch),
                "--volume", str(self.amplitude / 100),
                "-i", tmp_file
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                await process.wait()
            except Exception as e:
                logger.error(f"Ошибка RHVoice: {e}")
                await self._speak_espeak(text)
    
    async def _speak_espeak(self, text: str):
        """Fallback: eSpeak"""
        voice = self.voice if self.voice in ["ru", "en", "mb-en1"] else "ru"
        cmd = [
            "espeak",
            "-v", voice,
            "-s", str(self.speed),
            "-p", str(self.pitch),
            "-a", str(self.amplitude),
            text
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await process.wait()
        except Exception as e:
            logger.error(f"Ошибка eSpeak: {e}")


# ================================================================
# VOSK РАСПОЗНАВАТЕЛЬ РЕЧИ
# ================================================================

class VoiceListener:
    """Распознавание речи через Vosk с поддержкой wake word"""
    
    def __init__(self, model_path: str, threshold: int = 500,
                 wake_words: list = None, active_timeout: float = 15.0):
        
        if not VOSK_AVAILABLE or not PYAUDIO_AVAILABLE:
            raise RuntimeError("Vosk или PyAudio не установлены")
        
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.threshold = threshold
        self.wake_words = wake_words or ["гралл", "робот", "эй"]
        self.active_timeout = active_timeout
        
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_active = False
        self.active_until = 0
        self.interactive_mode = False
        
        self.callback: Optional[Callable] = None
        self._speech_future: Optional[asyncio.Future] = None
        
        logger.info(f"🎙️ VoiceListener инициализирован (wake: {self.wake_words})")
    
    def set_callback(self, callback: Callable):
        self.callback = callback
    
    def set_interactive_mode(self, enabled: bool):
        self.interactive_mode = enabled
        if enabled:
            self.is_active = True
            logger.info("🎙️ Интерактивный режим ВКЛЮЧЁН")
        else:
            self.is_active = False
            logger.info("🎙️ Интерактивный режим ВЫКЛЮЧЁН")
    
    def set_speech_future(self, future: asyncio.Future):
        self._speech_future = future
    
    def cancel_speech_future(self):
        if self._speech_future and not self._speech_future.done():
            self._speech_future.cancel()
        self._speech_future = None
    
    def activate(self, duration: float = None):
        if duration is None:
            duration = self.active_timeout
        self.is_active = True
        self.active_until = time.time() + duration
    
    def deactivate(self):
        self.is_active = False
        self.active_until = 0
    
    def is_emergency_stop(self, text: str) -> bool:
        stop_words = ["стоп", "стой", "stop", "halt", "тормози"]
        return any(word in text.lower() for word in stop_words)
    
    def contains_wake_word(self, text: str) -> bool:
        text_lower = text.lower()
        return any(word in text_lower for word in self.wake_words)
    
    def remove_wake_word(self, text: str) -> str:
        text_lower = text.lower()
        for word in self.wake_words:
            if word in text_lower:
                return text_lower.replace(word, "").strip()
        return text
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        if np.max(np.abs(audio_array)) > self.threshold:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    async def start(self):
        self.is_listening = True
        loop = asyncio.get_event_loop()
        threading.Thread(target=self._listen_thread, args=(loop,), daemon=True).start()
        logger.info("🎙️ VoiceListener запущен")
    
    def stop(self):
        self.is_listening = False
    
    def _listen_thread(self, loop):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4000,
            stream_callback=self.audio_callback
        )
        stream.start_stream()
        
        frames = []
        silence_frames = 0
        max_silence = 20
        min_voice_frames = 10
        
        while self.is_listening:
            try:
                data = self.audio_queue.get(timeout=0.1)
                frames.append(data)
                silence_frames = 0
            except queue.Empty:
                if frames:
                    silence_frames += 1
                    if silence_frames > max_silence and len(frames) >= min_voice_frames:
                        audio_data = b''.join(frames)
                        if self.recognizer.AcceptWaveform(audio_data):
                            result = json.loads(self.recognizer.Result())
                            text = result.get('text', '').strip()
                            if text:
                                asyncio.run_coroutine_threadsafe(
                                    self._process_text(text), loop
                                )
                        frames = []
                        silence_frames = 0
                
                if self.is_active and not self.interactive_mode:
                    if time.time() > self.active_until:
                        self.is_active = False
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    async def _process_text(self, text: str):
        logger.info(f"🎙️ Распознано: {text}")
        
        # Аварийная остановка — всегда
        if self.is_emergency_stop(text):
            logger.warning(f"🚨 АВАРИЙНАЯ ОСТАНОВКА: {text}")
            if self.callback:
                await self.callback(text, wake=False, emergency_stop=True)
            return
        
        # Если ждём ответ на вопрос
        if self._speech_future and not self._speech_future.done():
            self._speech_future.set_result(text)
            return
        
        # Интерактивный режим
        if self.interactive_mode:
            if not self.is_active:
                self.activate()
            if self.callback:
                await self.callback(text, wake=False, interactive=True)
            return
        
        # Обычный режим — нужен wake word
        if self.contains_wake_word(text):
            self.activate()
            clean_text = self.remove_wake_word(text)
            if self.callback:
                await self.callback(clean_text, wake=True)
        elif self.is_active:
            if self.callback:
                await self.callback(text, wake=False)


# ================================================================
# ПЛАГИН РЕЧИ
# ================================================================

class SpeechPlugin(Plugin):
    """
    Плагин речи — слух и голос робота.
    
    Управляет распознаванием речи (Vosk) и синтезом (TTS).
    Поддерживает обычный и интерактивный режимы.
    """
    
    name = "speech"
    version = "1.0"
    dependencies = []  # независим
    
    def __init__(self, agent):
        super().__init__(agent)
        
        # TTS движок
        self.tts: Optional[TTSEngine] = None
        
        # Voice listener
        self.listener: Optional[VoiceListener] = None
        
        # Состояние
        self.interactive_mode = False
        self._speech_future: Optional[asyncio.Future] = None
    
    async def on_load(self):
        """Инициализация TTS и VoiceListener"""
        
        # 1. TTS
        voice = self.agent.get_config('TTS_VOICE', 'ru')
        speed = self.agent.get_config('TTS_SPEED', 110)
        pitch = self.agent.get_config('TTS_PITCH', 45)
        amplitude = self.agent.get_config('TTS_AMPLITUDE', 100)
        
        self.tts = TTSEngine(voice=voice, speed=speed, pitch=pitch, amplitude=amplitude)
        logger.info("✅ TTS инициализирован")
        
        # 2. VoiceListener (если есть модель Vosk)
        model_path = self.agent.get_config('VOSK_MODEL_PATH')
        if model_path and VOSK_AVAILABLE and PYAUDIO_AVAILABLE:
            try:
                threshold = self.agent.get_config('SOUND_THRESHOLD', 500)
                wake_words = self.agent.get_config('WAKE_WORDS', ["гралл", "робот", "эй"])
                active_timeout = self.agent.get_config('VOICE_ACTIVE_TIMEOUT', 15.0)
                
                self.listener = VoiceListener(
                    model_path=model_path,
                    threshold=threshold,
                    wake_words=wake_words,
                    active_timeout=active_timeout
                )
                self.listener.set_callback(self._on_speech)
                await self.listener.start()
                
                logger.info("✅ VoiceListener инициализирован")
            except Exception as e:
                logger.warning(f"⚠️ VoiceListener не запущен: {e}")
                self.listener = None
        else:
            logger.warning("⚠️ Vosk не настроен, голосовое управление отключено")
        
        await super().on_load()
    
    async def on_unload(self):
        """Остановка слушателя"""
        if self.listener:
            self.listener.stop()
        
        await super().on_unload()
    
    # ==================== ОСНОВНЫЕ МЕТОДЫ ====================
    
    async def speak(self, text: str):
        """Произнести текст"""
        if self.tts:
            await self.tts.speak(text)
        
        # Сохраняем в диалоговый контекст
        memory_plugin = self.agent.get_plugin("memory")
        if memory_plugin:
            memory_plugin.add_dialog_turn("", text, intent="conversation", source="agent")
    
    async def wait_for_speech(self) -> Optional[str]:
        """Ожидает ответа человека"""
        self._speech_future = asyncio.Future()
        if self.listener:
            self.listener.set_speech_future(self._speech_future)
        try:
            return await self._speech_future
        finally:
            self._speech_future = None
            if self.listener:
                self.listener.cancel_speech_future()
    
    async def is_human_nearby(self) -> bool:
        """Проверяет, есть ли человек поблизости"""
        # Проверяем через VLM
        vision_plugin = self.agent.get_plugin("vision")
        if vision_plugin:
            latest = vision_plugin.get_latest_scan()
            if latest:
                objects = latest.get("data", {}).get("objects", [])
                for obj in objects:
                    if obj.get("name") == "человек" and obj.get("distance", 10) < 3.0:
                        return True
        
        # Проверяем через лидар
        sensor_memory = self.agent.get_sensor_memory()
        if sensor_memory:
            lidar = sensor_memory.get("lidar")
            if lidar:
                clusters = lidar.data.get("clusters", [])
                for c in clusters:
                    if c.get("type") == "human" and c.get("min_distance", 10) < 3.0:
                        return True
        
        return False
    
    def get_interactive_mode(self) -> bool:
        """Возвращает состояние интерактивного режима"""
        return self.interactive_mode
    
    def set_interactive_mode(self, enabled: bool):
        """Включает/выключает интерактивный режим"""
        self.interactive_mode = enabled
        if self.listener:
            self.listener.set_interactive_mode(enabled)
        logger.info(f"🔛 Интерактивный режим: {'ВКЛ' if enabled else 'ВЫКЛ'}")
    
    # ==================== ОБРАБОТКА РЕЧИ ====================
    
    async def _on_speech(self, text: str, wake: bool = False,
                         emergency_stop: bool = False, interactive: bool = False):
        """Колбэк от VoiceListener — передаёт команду агенту"""
        
        if emergency_stop:
            # Аварийная остановка
            motion_plugin = self.agent.get_plugin("motion")
            if motion_plugin:
                # Находим StopTool
                for tool in self.agent.tools:
                    if tool.name == "stop":
                        await tool.forward()
                        break
            
            # Отменяем активную задачу
            tasks_plugin = self.agent.get_plugin("tasks")
            if tasks_plugin and tasks_plugin.has_active_task():
                tasks_plugin.cancel_task()
            
            await self.speak("Останавливаюсь")
            return
        
        # Проверяем похвалу через tasks-плагин
        tasks_plugin = self.agent.get_plugin("tasks")
        if tasks_plugin and tasks_plugin.check_praise(text):
            # Похвала — будет обработана в tasks-плагине
            pass
        
        # Передаём команду агенту
        if hasattr(self.agent, 'on_human_command'):
            await self.agent.on_human_command(text, wake, interactive)
        else:
            # Если у агента нет метода — логируем
            logger.info(f"👤 Команда: {text} (wake={wake})")
    
    # ==================== ИНСТРУМЕНТЫ ====================
    
    def get_tools(self) -> List[Tool]:
        """Возвращает инструменты речи"""
        return [
            SpeakTool(self.agent, self),
            AskHumanTool(self.agent, self),
        ]
    
    # ==================== СТАТИСТИКА ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику плагина"""
        return {
            "name": self.name,
            "version": self.version,
            "tts_engine": self.tts.engine if self.tts else "none",
            "voice_listener": self.listener is not None,
            "interactive_mode": self.interactive_mode
        }