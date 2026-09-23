#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ agent.py — ГЛАВНЫЙ КЛАСС РОБОТА (ЧИСТАЯ ОБОЛОЧКА)                         ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «сознание» Гралла. Агент — чистая оболочка, которая загружает         ║
║   плагины и предоставляет им API. Вся логика вынесена в плагины.             ║
║                                                                              ║
║ АРХИТЕКТУРА:                                                                 ║
║   • Ядро — SensorMemory, WeightCalculator, ContextBuilder                    ║
║   • Плагины — вся остальная логика                                           ║
║   • Агент предоставляет API для плагинов:                                    ║
║     - add_to_conversation(), get_conversation()                              ║
║     - execute_tool(), send_command()                                         ║
║     - inc_stat(), get_config()                                               ║
║     - is_running(), is_frozen(), freeze(), resume()                          ║
║                                                                              ║
║ ПЛАГИНЫ (автоматически загружаются из папки plugins/):                       ║
║   • network  — WebSocket, маршрутизация сообщений                            ║
║   • memory   — эпизодическая память, маршруты, диалог                        ║
║   • sensors  — лидар, одометрия                                              ║
║   • vision   — VLM, фокусировка, вопросы                                     ║
║   • speech   — распознавание и синтез речи                                   ║
║   • motion   — движение, маршруты, рефлексы                                  ║
║   • llm      — управление LLM (локальный/облачный)                           ║
║   • tasks    — автономные задачи, стратегии                                  ║
║   • slam     — SLAM, карты, навигация                                        ║
║   • files    — работа с файлами (песочница/полный доступ)                    ║
║                                                                              ║
║ КАК ЗАПУСТИТЬ:                                                               ║
║   python agent_v7.py                                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
import os
from typing import Optional, Dict, Any, List

# Ядро (остаётся)
from config import *
from core.sensor_memory import SensorMemory
from core.weight_calculator import WeightCalculator
from core.context_builder import ContextBuilder
from core.feedback_learner import FeedbackLearner

# Плагины
from plugins.loader import PluginLoader

logger = logging.getLogger(__name__)


class RobotAgentV7:
    """
    ГЛАВНЫЙ КЛАСС РОБОТА — ЧИСТАЯ ОБОЛОЧКА
    
    Внутри только:
    - Ядро (SensorMemory, WeightCalculator, ContextBuilder)
    - Состояние (running, frozen)
    - Статистика
    - API для плагинов
    
    Вся остальная логика — в плагинах.
    """

    def __init__(self):
        # ========== СОСТОЯНИЕ ==========
        self.running = True
        self.frozen = False
        self.connected = False
        self.last_command_time = time.time()

        # ========== ЯДРО (всегда здесь) ==========
        self.sensor_memory = SensorMemory(max_age=SENSOR_MAX_AGE)
        self.weight_calculator = WeightCalculator()
        self.context_builder = ContextBuilder(self.weight_calculator, self.sensor_memory)
        self.feedback_learner = FeedbackLearner(FEEDBACK_FILE)

        # ========== СОСТОЯНИЕ ДЛЯ ПЛАГИНОВ ==========
        self.tools = []           # собирается из плагинов
        self.conversation = []    # история диалога (используется LLM-плагином)
        self.interactive_mode = INTERACTIVE_MODE_DEFAULT

        # ========== СТАТИСТИКА ==========
        self.stats = {
            "start_time": time.time(),
            "messages_received": 0,
            "llm_calls": 0,
            "llm_cloud_calls": 0,
            "reflexes_received": 0,
            "praise_received": 0,
            "emergency_stops": 0,
            "vlm_scans": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "strategy_hits": 0,
            "strategy_success": 0,
        }

        # ========== ПЛАГИНЫ ==========
        self.plugin_loader = PluginLoader(self)
        self.plugins = {}  # будет заполнен после загрузки

        logger.info("🤖 RobotAgentV7 создан")

    # ================================================================
    # ИНИЦИАЛИЗАЦИЯ И ЗАПУСК
    # ================================================================

    async def setup(self):
        """Загружает плагины и запускает агента"""
        logger.info("🚀 Запуск RobotAgentV7")

        # 1. Обнаруживаем и загружаем все плагины
        self.plugin_loader.discover()
        await self.plugin_loader.load_all()
        self.plugins = self.plugin_loader.plugins

        # 2. Собираем инструменты из всех плагинов
        self._collect_tools()

        # 3. Запускаем фоновые задачи ядра
        asyncio.create_task(self._stats_logger())
        asyncio.create_task(self._sensor_memory_cleanup())

        # 4. Приветствие (через Speech-плагин)
        await self.speak("Агент активирован. Все плагины загружены.")

        logger.info("✅ Агент v7.0 готов")

    def _collect_tools(self):
        """Собирает инструменты из всех плагинов"""
        self.tools = []
        for plugin in self.plugins.values():
            self.tools.extend(plugin.get_tools())
        logger.info(f"🔧 Собрано {len(self.tools)} инструментов из {len(self.plugins)} плагинов")

    async def run(self):
        """Основной цикл"""
        await self.setup()
        logger.info("🤖 Агент запущен")
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Остановка по Ctrl+C")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Корректное завершение"""
        logger.info("🛑 Завершение работы...")
        self.running = False

        # Выгружаем плагины
        await self.plugin_loader.unload_all()

        # Сохраняем ядро
        self.feedback_learner.save()
        self.sensor_memory.save_to_file(SENSOR_MEMORY_FILE)

        uptime = time.time() - self.stats["start_time"]
        logger.info(f"👋 Агент остановлен. Время работы: {uptime:.0f}с")

    # ================================================================
    # API ДЛЯ ПЛАГИНОВ
    # ================================================================

    # --- Состояние ---
    def is_running(self) -> bool:
        return self.running

    def is_frozen(self) -> bool:
        return self.frozen

    def freeze(self):
        self.frozen = True
        logger.info("🧊 Агент заморожен")

    def resume(self):
        self.frozen = False
        logger.info("▶ Агент разморожен")

    def get_interactive_mode(self) -> bool:
        return self.interactive_mode

    def set_interactive_mode(self, enabled: bool):
        self.interactive_mode = enabled
        # Оповещаем Speech-плагин
        speech = self.get_plugin("speech")
        if speech:
            speech.set_interactive_mode(enabled)

    # --- Статистика ---
    def inc_stat(self, key: str, delta: int = 1):
        self.stats[key] = self.stats.get(key, 0) + delta

    def get_stat(self, key: str) -> int:
        return self.stats.get(key, 0)

    # --- Конфигурация ---
    def get_config(self, key: str, default=None):
        """Получает значение из config.py (глобальный namespace)"""
        try:
            import config
            return getattr(config, key, default)
        except:
            return default

    # --- Инструменты ---
    async def execute_tool(self, tool_name: str, params: dict = None) -> Any:
        """Выполняет инструмент по имени"""
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.forward(**(params or {}))
        logger.warning(f"⚠️ Инструмент {tool_name} не найден")
        return None

    def rebuild_tools(self):
        """Перестраивает инструменты (после смены режима LLM)"""
        self._collect_tools()

    # --- Диалог ---
    def add_to_conversation(self, role: str, content: str):
        """Добавляет сообщение в историю диалога"""
        self.conversation.append({"role": role, "content": content})
        # Ограничиваем размер
        if len(self.conversation) > 100:
            self.conversation = self.conversation[-80:]

    def get_conversation(self) -> List[Dict]:
        """Возвращает историю диалога"""
        return self.conversation.copy()

    def clear_conversation(self):
        """Очищает историю диалога"""
        self.conversation = []

    # --- Речь ---
    async def speak(self, text: str):
        """Произносит текст через Speech-плагин"""
        speech = self.get_plugin("speech")
        if speech:
            await speech.speak(text)

    # --- Команды ---
    async def send_command(self, msg: Dict):
        """Отправляет команду через Network-плагин"""
        network = self.get_plugin("network")
        if network:
            await network.send(msg)

    # --- Память ---
    def get_sensor_memory(self):
        return self.sensor_memory

    def get_episodic_memory(self):
        memory = self.get_plugin("memory")
        if memory:
            return memory.get_episodic_memory()
        return None

    def get_route_memory(self):
        memory = self.get_plugin("memory")
        if memory:
            return memory.get_route_memory()
        return None

    def get_vision_memory(self):
        memory = self.get_plugin("memory")
        if memory:
            return memory.get_visual_memory()
        return None

    # --- Плагины ---
    def get_plugin(self, name: str):
        """Возвращает плагин по имени"""
        return self.plugins.get(name)

    # --- Обработка команд от пользователя ---
    async def on_human_command(self, text: str, wake: bool = False, interactive: bool = False):
        """Обрабатывает голосовую команду (вызывается из Speech-плагина)"""
        # Обновляем таймер простоя
        tasks = self.get_plugin("tasks")
        if tasks:
            tasks.on_user_command()

        # Вызываем LLM
        llm = self.get_plugin("llm")
        if llm:
            # Собираем контекст
            sensor_data = await self._collect_sensor_data()
            context = self.context_builder.build_context(
                dialog_context={},
                sensor_data=sensor_data,
                active_reflexes=[],
                current_intent="query"
            )
            sensor_message = self.context_builder.format_for_llm(context)

            # Добавляем interactive hint если нужно
            if self.interactive_mode and not interactive:
                from core.prompts import INTERACTIVE_MODE_HINT
                sensor_message = f"{sensor_message}\n{INTERACTIVE_MODE_HINT}"

            # Вызываем LLM
            response = await llm.call(text, {"sensor_message": sensor_message, **context})

            # Выполняем действие
            if hasattr(response, 'action') and response.action:
                await self.execute_decision(response.action)
            elif hasattr(response, 'text') and response.text:
                await self.speak(response.text)

    async def _collect_sensor_data(self) -> List[Dict]:
        """Собирает данные из SensorMemory для контекста"""
        summaries = self.sensor_memory.get_summaries(min_weight=0.3)
        return [{"source_type": k, "data": v} for k, v in summaries.items()]

    async def execute_decision(self, decision: Dict):
        """Выполняет решение LLM"""
        action = decision.get("action")
        params = decision.get("params", {})

        # Регистрируем действие для оценки
        tasks = self.get_plugin("tasks")
        if tasks:
            action_id = f"{action}_{int(time.time()*1000)}"
            tasks.register_action(action_id, decision.get("reasoning", ""))

        # Выполняем
        result = await self.execute_tool(action, params)

        # Озвучиваем результат если нужно
        if result and isinstance(result, str):
            await self.speak(result[:200])

    async def execute_next_step(self):
        """Выполняет следующий шаг задачи (вызывается из Tasks-плагина)"""
        tasks = self.get_plugin("tasks")
        if not tasks:
            return

        step = tasks.get_current_step()
        if step:
            await self.on_human_command(step, wake=False, interactive=False)

    # ================================================================
    # ФОНОВЫЕ ЗАДАЧИ
    # ================================================================

    async def _stats_logger(self):
        """Логирование статистики каждые 60 секунд"""
        while self.running:
            await asyncio.sleep(60)
            uptime = time.time() - self.stats["start_time"]
            logger.info(f"📊 Статистика за {uptime:.0f}с: LLM={self.stats['llm_calls']}, "
                       f"Cloud={self.stats['llm_cloud_calls']}, "
                       f"Reflexes={self.stats['reflexes_received']}")

    async def _sensor_memory_cleanup(self):
        """Очистка устаревших сенсорных данных"""
        while self.running:
            await asyncio.sleep(30)
            self.sensor_memory._cleanup_old()


# ================================================================
# ТОЧКА ВХОДА
# ================================================================

async def main():
    agent = RobotAgentV7()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())