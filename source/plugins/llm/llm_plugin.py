#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/llm/__init__.py — LLM-ПЛАГИН (РЕЖИМЫ ПИЛОТА И ИНЖЕНЕРА)               ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Управляет режимами LLM (local/cloud)                                     ║
║   • Определяет нужный режим по ключевым словам                               ║
║   • Переключает сессии с сохранением истории                                 ║
║   • Предоставляет инструменты для LLM                                        ║
║                                                                              ║
║ ИСПОЛЬЗУЕТ API АГЕНТА (НЕ ЛЕЗЕТ ВО ВНУТРЕННОСТИ):                             ║
║   • agent.add_to_conversation(role, content) — добавить в историю диалога    ║
║   • agent.rebuild_tools() — перестроить инструменты после смены режима       ║
║   • agent.inc_stat(key) — увеличить счётчик статистики                       ║
║   • agent.get_config(key, default) — получить настройку                      ║
║   • agent.speak(text) — озвучить текст                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import subprocess
import logging
from typing import Dict, Any, Optional, List

from plugins.plugin import Plugin

# Импорты клиентов LLM
try:
    from agents.llm_client import LocalLLM
except ImportError:
    LocalLLM = None
    logging.warning("⚠️ agents.llm_client не найден")

try:
    from agents.yandex_gpt_client import YandexGPTClient
except ImportError:
    YandexGPTClient = None
    logging.warning("⚠️ agents.yandex_gpt_client не найден")

from core.prompts import get_system_prompt, CLOUD_ENGINEER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMPlugin(Plugin):
    """
    Плагин управления LLM — локальный пилот и облачный инженер
    
    Вся логика определения режима, переключения и вызова LLM находится здесь,
    а не в теле агента.
    """
    
    name = "llm"
    version = "1.0"
    dependencies = []
    
    def __init__(self, agent):
        super().__init__(agent)
        
        # Текущий режим
        self.active_mode = "local"      # "local" или "cloud"
        self.cloud_session_active = False
        self.local_llm_available = False
        
        # История сессий (хранится в плагине, не в агенте)
        self._local_conversation: List[Dict] = []
        self._cloud_conversation: List[Dict] = []
        
        # Ссылки на LLM-клиенты (создаются в on_load)
        self.local_llm = None
        self.yandex_client = None
        
        # Ключевые слова для переключения режимов
        self.cloud_trigger_keywords = [
            "настрой", "откалибруй", "обнови конфиг", "исправь файл",
            "замени в файле", "выполни код", "сгенерируй модуль",
            "прочитай grall_self", "покажи мои характеристики",
            "создай файл", "запиши файл", "сохрани в файл",
            "прочти свои файлы", "посмотри свои файлы", "список файлов",
            "отредактируй", "поправь", "добавь в конфиг", "напиши код",
            "создай инструмент", "добавь модуль", "скачай драйвер"
        ]
        
        self.local_trigger_keywords = [
            "включи локальный режим", "локальный режим", "работай локально",
            "отключи облако", "выключи облако", "без облака", "вернись в локальный"
        ]
    
    # ==================== ЖИЗНЕННЫЙ ЦИКЛ ====================
    
    async def on_load(self):
        """Инициализация LLM-клиентов при загрузке плагина"""
        
        # 1. Инициализируем локальную LLM (Ollama)
        await self._init_local_llm()
        
        # 2. Инициализируем облачную LLM (YandexGPT) если есть конфиг
        await self._init_cloud_llm()
        
        # 3. Если локальная недоступна, но есть облако — переключаемся
        if not self.local_llm_available and self.yandex_client:
            self.active_mode = "cloud"
            self.cloud_session_active = True
            logger.info("☁️ Работа только в облачном режиме")
        
        await super().on_load()
    
    async def on_unload(self):
        """Закрытие соединений при выгрузке плагина"""
        if self.local_llm:
            await self.local_llm.close()
        if self.yandex_client:
            await self.yandex_client.close()
        
        await super().on_unload()
    
    # ==================== ИНИЦИАЛИЗАЦИЯ LLM ====================
    
    async def _init_local_llm(self):
        """Создаёт кастомную модель Ollama с системным промптом"""
        if LocalLLM is None:
            logger.warning("⚠️ LocalLLM не импортирован")
            return
        
        try:
            model_name = await self._setup_ollama_model()
            
            ollama_url = self.agent.get_config('OLLAMA_URL', 'http://localhost:11434')
            max_history = self.agent.get_config('LLM_MAX_HISTORY', 100)
            
            self.local_llm = LocalLLM(
                model=model_name,
                base_url=ollama_url,
                max_history_messages=max_history
            )
            
            # Проверяем соединение
            test_response = await self.local_llm.generate([{"role": "user", "content": "ping"}])
            if test_response:
                self.local_llm_available = True
                logger.info("✅ Локальная LLM доступна")
            else:
                self.local_llm_available = False
                logger.warning("⚠️ Локальная LLM недоступна")
                
        except Exception as e:
            logger.warning(f"⚠️ Локальная LLM недоступна: {e}")
            self.local_llm = None
            self.local_llm_available = False
    
    async def _setup_ollama_model(self) -> str:
        """Создаёт кастомную модель Ollama с системным промптом"""
        # Генерируем список инструментов для системного промпта
        tools_list = """
move_forward(speed, duration, distance), move_backward(speed, duration, distance),
turn_left(speed, duration, angle), turn_right(speed, duration, angle),
stop(), wait(seconds), set_light(state),
speak(text, wait), ask_human(question),
search_web(query), web_fetch(url),
remember_object(name), find_object(name), search_by_text(query),
record_route(action, name), execute_route(name),
compose_plan(goal),
write_file(path, content, append), read_file(path), list_files(subdir),
focus_on(target), ask_vlm(prompt),
navigate_to(target), get_room_map()
"""
        
        system_prompt = get_system_prompt(tools_list)
        
        model = self.agent.get_config('LLM_MODEL', 'qwen2.5:1.5b')
        temperature = self.agent.get_config('LLM_TEMPERATURE', 0.7)
        top_p = self.agent.get_config('LLM_TOP_P', 0.9)
        
        modelfile = f"""FROM {model}
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
SYSTEM \"\"\"
{system_prompt}
\"\"\"
"""
        modelfile_path = "/tmp/GrallModelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile)
        
        try:
            subprocess.run(["ollama", "create", "grall-robot", "-f", modelfile_path],
                          check=True, capture_output=True, timeout=30)
            logger.info("✅ Модель grall-robot создана")
            return "grall-robot"
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка создания модели: {e.stderr.decode() if e.stderr else str(e)}")
            return model
        except subprocess.TimeoutExpired:
            logger.error("Таймаут создания модели")
            return model
    
    async def _init_cloud_llm(self):
        """Инициализирует YandexGPT клиент"""
        if YandexGPTClient is None:
            logger.debug("⚠️ YandexGPTClient не импортирован")
            return
        
        folder_id = self.agent.get_config('YANDEX_FOLDER_ID')
        api_key = self.agent.get_config('YANDEX_API_KEY')
        
        if folder_id and api_key:
            try:
                self.yandex_client = YandexGPTClient(
                    folder_id=folder_id,
                    api_key=api_key
                )
                logger.info("✅ YandexGPT клиент активирован")
            except Exception as e:
                logger.warning(f"⚠️ YandexGPT недоступен: {e}")
                self.yandex_client = None
        else:
            logger.debug("ℹ️ YandexGPT не настроен (пропуск)")
    
    # ==================== ОПРЕДЕЛЕНИЕ РЕЖИМА ====================
    
    def detect_mode(self, text: str, context: Dict = None) -> str:
        """
        Определяет, какой режим LLM нужен для запроса.
        Возвращает "local" или "cloud".
        """
        text_lower = text.lower()
        
        # Явные команды переключения на локальный режим
        for kw in self.local_trigger_keywords:
            if kw in text_lower:
                logger.info(f"🔑 ЛОКАЛЬНЫЙ режим: '{kw}'")
                return "local"
        
        # Обнаруживаем необходимость облака
        for kw in self.cloud_trigger_keywords:
            if kw in text_lower:
                logger.info(f"🔑 ОБЛАЧНЫЙ режим: '{kw}'")
                return "cloud"
        
        # Проверяем intent из контекста диалога
        if context:
            intent = context.get("current_intent", "")
            if intent in ["configure", "calibrate", "setup", "file_operation", "code_execution"]:
                return "cloud"
        
        # Если локальная LLM недоступна — всегда облако
        if not self.local_llm_available and self.yandex_client:
            return "cloud"
        
        # По умолчанию — текущий режим
        return self.active_mode
    
    async def switch_mode(self, new_mode: str, reason: str = "") -> bool:
        """
        Переключает режим LLM с сохранением истории.
        
        Args:
            new_mode: "local" или "cloud"
            reason: причина переключения (для логов)
        
        Returns:
            True — успешно, False — ошибка
        """
        if new_mode == self.active_mode:
            return True
        
        if new_mode == "cloud":
            # Проверяем доступность
            if not self.yandex_client:
                await self.agent.speak("Облачный режим недоступен. Проверьте настройки YandexGPT.")
                return False
            
            # Сохраняем локальную историю (через API агента)
            self._local_conversation = self._get_current_conversation()
            
            # Переключаемся
            self.active_mode = "cloud"
            self.cloud_session_active = True
            self._restore_conversation(self._cloud_conversation)
            
            logger.info(f"☁️ Облачный режим: {reason}")
            await self.agent.speak("Переключаюсь в облачный режим.")
            
        elif new_mode == "local":
            # Проверяем доступность
            if not self.local_llm_available:
                await self.agent.speak("Локальный режим недоступен. Ollama не отвечает.")
                return False
            
            # Сохраняем облачную историю
            self._cloud_conversation = self._get_current_conversation()
            
            # Переключаемся
            self.active_mode = "local"
            self.cloud_session_active = False
            self._restore_conversation(self._local_conversation)
            
            logger.info(f"💻 Локальный режим: {reason}")
            await self.agent.speak("Возвращаюсь в локальный режим.")
        
        # Перестраиваем инструменты через API агента
        self.agent.rebuild_tools()
        
        return True
    
    # ==================== ОСНОВНОЙ МЕТОД ВЫЗОВА LLM ====================
    
    async def call(self, text: str, context: Dict = None) -> Any:
        """
        Вызывает LLM в текущем режиме.
        
        Args:
            text: запрос пользователя
            context: контекст (сенсоры, намерение и т.д.)
        
        Returns:
            Response объект с полями content, action, text
        """
        if self.active_mode == "cloud":
            return await self._call_cloud(text, context)
        else:
            return await self._call_local(text, context)
    
    # ==================== ЛОКАЛЬНАЯ LLM ====================
    
    async def _call_local(self, text: str, context: Dict = None):
        """
        Вызывает локальную LLM (Ollama).
        Использует API агента для работы с историей и статистикой.
        """
        if not self.local_llm:
            return self._error_response("Локальная LLM недоступна")
        
        # Получаем сенсорное сообщение из контекста
        sensor_message = ""
        if context:
            sensor_message = context.get("sensor_message", "")
        
        # Добавляем запрос в историю через API агента
        self.agent.add_to_conversation("user", text)
        if sensor_message:
            self.agent.add_to_conversation("user", sensor_message)
        
        # Ограничиваем историю
        self._trim_conversation()
        
        # Вызываем LLM
        response = await self.local_llm.generate(self._get_current_conversation())
        
        # Добавляем ответ в историю
        self.agent.add_to_conversation("assistant", response.content)
        
        # Увеличиваем счётчик
        self.agent.inc_stat("llm_calls")
        
        return response
    
    # ==================== ОБЛАЧНАЯ LLM ====================
    
    async def _call_cloud(self, text: str, context: Dict = None):
        """
        Вызывает облачную LLM (YandexGPT).
        Использует API агента для работы с историей и статистикой.
        """
        if not self.yandex_client:
            return self._error_response("Облачная LLM недоступна")
        
        # Формируем сообщения для облачной LLM
        messages = []
        
        # Добавляем системный контекст
        messages.append({"role": "system", "content": CLOUD_ENGINEER_SYSTEM_PROMPT})
        
        # Добавляем историю (последние 10 сообщений)
        history = self._get_current_conversation()
        if history:
            messages.extend(history[-10:])
        
        # Добавляем сенсорный контекст
        sensor_message = ""
        if context:
            sensor_message = context.get("sensor_message", "")
        
        full_prompt = f"СИТУАЦИЯ:\n{sensor_message}\n\nЗАПРОС:\n{text}" if sensor_message else text
        messages.append({"role": "user", "content": full_prompt})
        
        # Вызываем LLM
        raw_response = await self.yandex_client.generate(messages)
        
        # Нормализуем ответ
        response = self._normalize_cloud_response(raw_response)
        
        # Добавляем в историю через API агента
        self.agent.add_to_conversation("user", text)
        self.agent.add_to_conversation("assistant", response.content)
        
        # Увеличиваем счётчик
        self.agent.inc_stat("llm_cloud_calls")
        
        return response
    
    def _normalize_cloud_response(self, raw_response):
        """Нормализует ответ от YandexGPT в единый формат"""
        
        class CloudResponse:
            __slots__ = ('content', 'action', 'text')
            def __init__(self, content: str, action: Any = None, text: str = None):
                self.content = content
                self.action = action
                self.text = text if text is not None else content[:200]
        
        content = ""
        action = None
        text = ""
        
        if hasattr(raw_response, 'content'):
            content = raw_response.content
            action = getattr(raw_response, 'action', None)
            text = getattr(raw_response, 'text', None)
        elif isinstance(raw_response, dict):
            content = raw_response.get('content', '')
            action = raw_response.get('action')
            text = raw_response.get('text')
        elif isinstance(raw_response, str):
            content = raw_response
            text = content[:200]
        else:
            content = str(raw_response)
            text = content[:200]
        
        # Нормализуем action в единый формат
        if action and isinstance(action, str):
            action = {"action": action, "params": {}, "reasoning": ""}
        elif action and isinstance(action, (list, tuple)) and len(action) >= 2:
            action = {"action": action[0], "params": action[1] if len(action) > 1 else {}, "reasoning": ""}
        
        return CloudResponse(content=content, action=action, text=text)
    
    # ==================== РАБОТА С ИСТОРИЕЙ (через API агента) ====================
    
    def _get_current_conversation(self) -> List[Dict]:
        """
        Получает текущую историю диалога.
        Использует внутреннее хранилище плагина, а не агента.
        """
        # История хранится в плагине, а не в агенте
        if self.active_mode == "cloud":
            return self._cloud_conversation
        return self._local_conversation
    
    def _restore_conversation(self, conversation: List[Dict]):
        """
        Восстанавливает историю диалога.
        Очищает историю агента и добавляет сохранённые сообщения.
        """
        # Очищаем историю агента через API
        self.agent.clear_conversation()
        
        # Добавляем сохранённые сообщения
        for msg in conversation:
            self.agent.add_to_conversation(msg.get("role"), msg.get("content", ""))
    
    def _trim_conversation(self, max_messages: int = 20):
        """Ограничивает историю диалога (последние N сообщений)"""
        conv = self._get_current_conversation()
        if len(conv) > max_messages:
            trimmed = conv[-max_messages:]
            if self.active_mode == "cloud":
                self._cloud_conversation = trimmed
            else:
                self._local_conversation = trimmed
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def _error_response(self, error_msg: str) -> Any:
        """Создаёт ответ с ошибкой"""
        class ErrorResponse:
            def __init__(self, content, text):
                self.content = content
                self.action = None
                self.text = text
        return ErrorResponse(content=error_msg, text=error_msg)
    
    # ==================== ИНСТРУМЕНТЫ ====================
    
    def get_tools(self) -> List:
        """
        Возвращает инструменты, которые предоставляет плагин.
        LLM-плагин не предоставляет инструментов — они в других плагинах.
        """
        return []
    
    def get_mode(self) -> str:
        """Возвращает текущий режим"""
        return self.active_mode
    
    def is_cloud_available(self) -> bool:
        """Доступен ли облачный режим"""
        return self.yandex_client is not None
    
    def is_local_available(self) -> bool:
        """Доступен ли локальный режим"""
        return self.local_llm_available