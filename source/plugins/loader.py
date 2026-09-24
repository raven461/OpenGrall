#!/usr/bin/env python3
"""Загрузчик плагинов"""

import importlib
import pkgutil
import logging
from typing import Dict, List

import plugins
from .plugin import Plugin

logger = logging.getLogger(__name__)


class PluginLoader:
    def __init__(self, agent):
        self.agent = agent
        self.plugins: Dict[str, Plugin] = {}
        self._loaded = False

    def discover(self) -> List[Plugin]:
        """Находит все классы-наследники Plugin"""
        result = []
        for module_info in pkgutil.iter_modules(plugins.__path__):
            if module_info.name == 'plugin' or module_info.name == 'loader':
                continue
            try:
                module = importlib.import_module(f"plugins.{module_info.name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, Plugin) and 
                        attr != Plugin):
                        plugin = attr(self.agent)
                        self.plugins[plugin.name] = plugin
                        result.append(plugin)
                        logger.debug(f"📦 Найден плагин: {plugin.name}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки плагина {module_info.name}: {e}")
        return result

    async def load_all(self):
        """Загружает плагины с учётом зависимостей"""
        # Сначала загружаем без зависимостей
        for plugin in self.plugins.values():
            if not plugin.dependencies:
                await plugin.on_load()

        # Затем зависимые (несколько проходов)
        for _ in range(5):
            for plugin in self.plugins.values():
                if plugin.is_loaded:
                    continue
                deps_loaded = all(d in self.plugins and self.plugins[d].is_loaded 
                                 for d in plugin.dependencies)
                if deps_loaded:
                    await plugin.on_load()

        # Логируем результат
        loaded = [p.name for p in self.plugins.values() if p.is_loaded]
        failed = [p.name for p in self.plugins.values() if not p.is_loaded]
        logger.info(f"✅ Загружены плагины: {loaded}")
        if failed:
            logger.warning(f"⚠️ Не загружены: {failed}")

    async def unload_all(self):
        """Выгружает все плагины в обратном порядке"""
        for plugin in reversed(list(self.plugins.values())):
            if plugin.is_loaded:
                await plugin.on_unload()