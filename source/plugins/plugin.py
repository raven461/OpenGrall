#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/plugin.py — БАЗОВЫЙ КЛАСС ДЛЯ ВСЕХ ПЛАГИНОВ                          ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Единый интерфейс для всех расширений агента.                               ║
║   Каждый плагин — это независимый модуль, который:                           ║
║     • Предоставляет инструменты (get_tools)                                  ║
║     • Обрабатывает сообщения (on_message)                                   ║
║     • Имеет свой жизненный цикл (on_load, on_unload)                        ║
║                                                                              ║
║ КАК СОЗДАТЬ СВОЙ ПЛАГИН:                                                     ║
║   1. Создайте папку plugins/my_plugin/                                       ║
║   2. Создайте __init__.py с классом, наследующим Plugin                      ║
║   3. Переопределите нужные методы                                            ║
║   4. Плагин будет автоматически обнаружен и загружен                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Plugin:
    """
    Базовый класс для всех плагинов агента.
    
    Жизненный цикл:
    1. __init__() — создание, получение ссылки на агента
    2. on_load() — загрузка, инициализация ресурсов
    3. Работа: on_message(), get_tools(), фоновые задачи
    4. on_unload() — выгрузка, освобождение ресурсов
    """
    
    name: str = "base_plugin"
    version: str = "1.0"
    dependencies: List[str] = []  # имена плагинов, от которых зависит
    
    def __init__(self, agent):
        """
        Args:
            agent: экземпляр RobotAgentV5 (мозг робота)
        """
        self.agent = agent
        self._loaded = False
        logger.debug(f"📦 Плагин {self.name} v{self.version} создан")
    
    async def on_load(self):
        """
        Вызывается при загрузке плагина.
        Здесь инициализируются ресурсы, запускаются фоновые задачи.
        """
        self._loaded = True
        logger.info(f"✅ Плагин {self.name} v{self.version} загружен")
    
    async def on_unload(self):
        """
        Вызывается при выгрузке плагина.
        Здесь освобождаются ресурсы, останавливаются фоновые задачи.
        """
        self._loaded = False
        logger.info(f"🛑 Плагин {self.name} выгружен")
    
    async def on_message(self, msg_type: str, data: Dict[str, Any]):
        """
        Обработка входящих WebSocket-сообщений.
        Вызывается для каждого сообщения от сервера.
        
        Args:
            msg_type: тип сообщения ('reflex', 'battery', 'frame_data', ...)
            data: полные данные сообщения
        """
        pass
    
    def get_tools(self) -> List:
        """
        Возвращает список инструментов (Tool), которые предоставляет плагин.
        Эти инструменты будут добавлены в agent.tools.
        
        Returns:
            List[Tool]: список инструментов
        """
        return []
    
    def get_handlers(self) -> Dict[str, callable]:
        """
        Возвращает обработчики событий.
        Ключ — тип события, значение — async функция-обработчик.
        
        Returns:
            Dict[str, callable]: словарь обработчиков
        """
        return {}
    
    @property
    def is_loaded(self) -> bool:
        """Загружен ли плагин"""
        return self._loaded
    
    async def _execute_tool(self, tool_name: str, params: dict = None):
        """
        Вспомогательный метод: вызывает инструмент агента по имени.
        Все плагины используют этот метод для взаимодействия с агентом.
        """
        for tool in self.agent.tools:
            if tool.name == tool_name:
                return await tool.forward(**(params or {}))
        
        logger.warning(f"⚠️ [{self.name}] Инструмент {tool_name} не найден")
        return None