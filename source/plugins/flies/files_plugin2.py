#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/files/__init__.py — ПЛАГИН ФАЙЛОВОЙ СИСТЕМЫ И ПОИСКА                 ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Предоставляет инструменты для работы с файлами                          ║
║   • Предоставляет инструменты для поиска в интернете                        ║
║   • ЛОКАЛЬНЫЙ РЕЖИМ (пилот): только песочница data/agent_files/             ║
║   • ОБЛАЧНЫЙ РЕЖИМ (инженер): полный доступ к проекту                       ║
║                                                                              ║
║ ИНСТРУМЕНТЫ:                                                                 ║
║   • read_file(path) — прочитать файл                                         ║
║   • write_file(path, content, append) — записать/добавить в файл            ║
║   • list_files(subdir) — список файлов в директории                         ║
║   • search_web(query) — поиск в интернете (погода, новости, факты)          ║
║   • web_fetch(url) — получить содержимое веб-страницы                       ║
║   • apply_patch(path, find, replace) — точное редактирование (только облако) ║
║   • execute_code(code) — выполнить Python-код (только облако)               ║
║                                                                              ║
║ РЕЖИМЫ:                                                                      ║
║   • Локальный (local) — песочница, безопасно для автономной работы          ║
║   • Облачный (cloud) — полный доступ, для настройки и калибровки            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import shutil
import tempfile
import asyncio
import subprocess
import logging
import time
import aiohttp
import re
from typing import Optional, Dict, Any, List

from plugins.plugin import Plugin
from orchestration.tool import Tool

logger = logging.getLogger(__name__)


# ================================================================
# БАЗОВЫЙ КЛАСС ДЛЯ ФАЙЛОВЫХ ИНСТРУМЕНТОВ
# ================================================================

class BaseFileTool(Tool):
    """Базовый класс для файловых инструментов с проверкой пути"""
    
    def __init__(self, base_path: str, agent):
        self.base_path = os.path.abspath(base_path)
        self.agent = agent
        os.makedirs(self.base_path, exist_ok=True)
    
    def _safe_path(self, path: str) -> str:
        """Проверяет, что путь внутри разрешённой директории"""
        full_path = os.path.abspath(os.path.join(self.base_path, path))
        if not full_path.startswith(self.base_path):
            raise PermissionError(f"Path {path} is outside allowed directory")
        return full_path
    
    def _create_backup(self, file_path: str) -> Optional[str]:
        """Создаёт бэкап файла"""
        if not os.path.exists(file_path):
            return None
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(file_path), ".backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Backup created: {backup_path}")
        return backup_path
    
    def _get_param(self, kwargs: dict, name: str, required: bool = True) -> Optional[str]:
        """Безопасное извлечение параметра из kwargs"""
        value = kwargs.get(name)
        if required and value is None:
            raise ValueError(f"Missing required parameter: '{name}'")
        return value


# ================================================================
# ЛОКАЛЬНЫЕ ИНСТРУМЕНТЫ (ПЕСОЧНИЦА)
# ================================================================

class LocalFileReadTool(BaseFileTool):
    """Чтение файла из песочницы"""
    name = "read_file"
    description = "Read a file from the sandbox (data/agent_files/)"
    latency = 0.03
    
    async def forward(self, **kwargs) -> str:
        try:
            path = self._get_param(kwargs, 'path')
            full_path = self._safe_path(path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content if content else "(file is empty)"
        except ValueError as e:
            return f"Parameter error: {e}"
        except FileNotFoundError:
            return f"File {kwargs.get('path', '?')} not found"
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            return f"Read error: {e}"


class LocalFileWriteTool(BaseFileTool):
    """Запись файла в песочницу"""
    name = "write_file"
    description = "Create or overwrite a file in the sandbox (data/agent_files/)"
    latency = 0.05
    
    async def forward(self, **kwargs) -> str:
        try:
            path = self._get_param(kwargs, 'path')
            content = self._get_param(kwargs, 'content')
            append = kwargs.get('append', False)
            
            full_path = self._safe_path(path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            if not append and os.path.exists(full_path):
                self._create_backup(full_path)
            
            mode = "a" if append else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)
            
            action = "appended to" if append else "written to"
            return f"File {path} successfully {action} ({len(content)} chars)"
        
        except ValueError as e:
            return f"Parameter error: {e}"
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            return f"Write error: {e}"


class LocalFileListTool(BaseFileTool):
    """Список файлов в песочнице"""
    name = "list_files"
    description = "List files in the sandbox directory (data/agent_files/)"
    latency = 0.02
    
    async def forward(self, **kwargs) -> str:
        try:
            subdir = kwargs.get('subdir', '')
            full_path = self._safe_path(subdir)
            
            if not os.path.exists(full_path):
                return f"Directory {subdir} not found"
            
            items = os.listdir(full_path)
            if not items:
                return f"Directory {subdir or '.'} is empty"
            
            result = [f"Contents of {subdir or 'sandbox'}:"]
            for item in sorted(items):
                if item == ".backups":
                    continue
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    result.append(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    result.append(f"  📄 {item} ({size} bytes)")
            
            return "\n".join(result)
        
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            return f"List error: {e}"


# ================================================================
# ОБЛАЧНЫЕ ИНСТРУМЕНТЫ (ПОЛНЫЙ ДОСТУП)
# ================================================================

class CloudFileReadTool(BaseFileTool):
    """Чтение любого файла проекта (только облачный режим)"""
    name = "read_file"
    description = "Read ANY file in the project (cloud mode only)"
    latency = 0.03
    
    async def forward(self, **kwargs) -> str:
        try:
            path = self._get_param(kwargs, 'path')
            full_path = os.path.abspath(os.path.join(self.base_path, path))
            
            # Блокируем доступ к системным файлам
            forbidden = ["/etc/", "/proc/", "/sys/", "/boot/", "/dev/", "~/.ssh", "~/.gnupg"]
            for fb in forbidden:
                if fb in full_path:
                    return f"Access denied: system files are protected"
            
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Ограничиваем размер ответа
            if len(content) > 50000:
                content = content[:50000] + "\n... (truncated, file too large)"
            
            return content if content else "(file is empty)"
        
        except ValueError as e:
            return f"Parameter error: {e}"
        except FileNotFoundError:
            return f"File {kwargs.get('path', '?')} not found"
        except PermissionError:
            return f"Permission denied: {kwargs.get('path', '?')}"
        except Exception as e:
            return f"Read error: {e}"


class CloudFileWriteTool(BaseFileTool):
    """Запись/редактирование любого файла проекта (только облачный режим)"""
    name = "write_file"
    description = "Create or modify ANY file in the project (cloud mode only)"
    latency = 0.05
    
    async def forward(self, **kwargs) -> str:
        try:
            path = self._get_param(kwargs, 'path')
            content = self._get_param(kwargs, 'content')
            append = kwargs.get('append', False)
            
            full_path = os.path.abspath(os.path.join(self.base_path, path))
            
            # Блокируем доступ к системным файлам
            forbidden = ["/etc/", "/proc/", "/sys/", "/boot/", "/dev/", "~/.ssh", "~/.gnupg"]
            for fb in forbidden:
                if fb in full_path:
                    return f"Access denied: system files are protected"
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            if not append and os.path.exists(full_path):
                self._create_backup(full_path)
            
            mode = "a" if append else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)
            
            action = "appended to" if append else "written to"
            logger.info(f"📝 Cloud: {path} {action}")
            return f"File {path} successfully {action} ({len(content)} chars)"
        
        except ValueError as e:
            return f"Parameter error: {e}"
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            return f"Write error: {e}"


class CloudFileListTool(BaseFileTool):
    """Список файлов в любой директории проекта (только облачный режим)"""
    name = "list_files"
    description = "List files in ANY directory of the project (cloud mode only)"
    latency = 0.02
    
    async def forward(self, **kwargs) -> str:
        try:
            subdir = kwargs.get('subdir', '')
            full_path = os.path.abspath(os.path.join(self.base_path, subdir))
            
            # Блокируем доступ к системным директориям
            forbidden = ["/etc", "/proc", "/sys", "/boot", "/dev", "/root", "/home"]
            for fb in forbidden:
                if full_path.startswith(fb):
                    return f"Access denied: system directories are protected"
            
            if not os.path.exists(full_path):
                return f"Directory {subdir} not found"
            
            items = os.listdir(full_path)
            if not items:
                return f"Directory {subdir or '.'} is empty"
            
            result = [f"Contents of {subdir or 'project root'}:"]
            for item in sorted(items):
                if item == ".backups" or item == "__pycache__" or item.startswith("."):
                    continue
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    result.append(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    result.append(f"  📄 {item} ({size} bytes)")
            
            return "\n".join(result)
        
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            return f"List error: {e}"


class ApplyPatchTool(BaseFileTool):
    """
    ТОЧНОЕ РЕДАКТИРОВАНИЕ ФАЙЛА (FIND/REPLACE)
    Только для облачного режима (инженер)
    """
    name = "apply_patch"
    description = "Replace text in a file using find/replace (cloud mode only)"
    latency = 0.05
    
    async def forward(self, **kwargs) -> str:
        try:
            path = self._get_param(kwargs, 'path')
            find = self._get_param(kwargs, 'find')
            replace = self._get_param(kwargs, 'replace')
            
            full_path = os.path.abspath(os.path.join(self.base_path, path))
            
            # Блокируем системные файлы
            forbidden = ["/etc/", "/proc/", "/sys/", "/boot/", "/dev/"]
            for fb in forbidden:
                if fb in full_path:
                    return f"Access denied: system files are protected"
            
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if find not in content:
                return f"Text to replace not found in {path}"
            
            self._create_backup(full_path)
            new_content = content.replace(find, replace)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            logger.info(f"📝 Patch applied to {path}")
            return f"File {path} successfully patched"
        
        except ValueError as e:
            return f"Parameter error: {e}"
        except FileNotFoundError:
            return f"File {kwargs.get('path', '?')} not found"
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            return f"Patch error: {e}"


class CodeExecutionTool(Tool):
    """
    ВЫПОЛНЕНИЕ PYTHON-КОДА
    Только для облачного режима (инженер)
    Безопасная песочница с ограничениями
    """
    name = "execute_code"
    description = "Execute Python code in a sandbox (cloud mode only)"
    latency = 5.0
    
    def __init__(self, agent, timeout: int = 10):
        self.agent = agent
        self.timeout = timeout
    
    async def forward(self, **kwargs) -> str:
        try:
            code = kwargs.get('code')
            if not code:
                return "Error: missing 'code' parameter"
            
            # Безопасность: запрещаем опасные операции
            forbidden_keywords = [
                "import os", "import subprocess", "import sys", "import shutil",
                "__import__", "eval(", "exec(", "compile(", "open(", "file(",
                "__builtins__", "__class__", "__bases__", "__subclasses__"
            ]
            
            code_lower = code.lower()
            for keyword in forbidden_keywords:
                if keyword in code_lower:
                    return f"Security error: '{keyword}' is not allowed"
            
            # Записываем код во временный файл
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                # Добавляем ограничения
                safe_code = f"""
import sys
import io

# Перехватываем stdout
old_stdout = sys.stdout
sys.stdout = io.StringIO()

# Разрешённые импорты
import math
import random
import time
import json

# Пользовательский код
{code}

# Собираем вывод
result = sys.stdout.getvalue()
sys.stdout = old_stdout
"""
                f.write(safe_code)
                temp_path = f.name
            
            try:
                process = await asyncio.create_subprocess_exec(
                    "python3", temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                
                if stderr:
                    error_msg = stderr.decode()[:500]
                    return f"Execution error:\n{error_msg}"
                
                output = stdout.decode().strip()
                logger.info(f"⚡ Code executed successfully")
                
                if not output:
                    return "Code executed successfully (no output)"
                
                # Ограничиваем размер ответа
                if len(output) > 2000:
                    output = output[:2000] + "\n... (truncated)"
                
                return output
            
            except asyncio.TimeoutError:
                return f"Code execution timeout ({self.timeout} seconds)"
            except Exception as e:
                return f"Execution error: {e}"
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        except Exception as e:
            return f"Execution error: {e}"


# ================================================================
# ИНСТРУМЕНТЫ ПОИСКА В ИНТЕРНЕТЕ
# ================================================================

class SearchWebTool(Tool):
    """Поиск в интернете через YandexGPT"""
    name = "search_web"
    description = "Search the internet for information (weather, news, exchange rates, facts)"
    latency = 2.0

    def __init__(self, agent):
        self.agent = agent

    async def forward(self, **kwargs) -> str:
        query = kwargs.get('query')
        if not query:
            return "Error: missing 'query' parameter"
        
        # Проверяем наличие YandexGPT клиента
        yandex_client = getattr(self.agent, 'yandex_client', None)
        if not yandex_client:
            return "Internet search not available (YandexGPT not configured)"

        try:
            logger.info(f"🔍 Web search: {query}")
            answer = await yandex_client.search_web(query)
            return answer if answer else "No information found"
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search error: {e}"


class WebFetchTool(Tool):
    """Получение содержимого веб-страницы"""
    name = "web_fetch"
    description = "Fetch and read text content from a webpage by URL"
    latency = 1.5

    async def forward(self, **kwargs) -> str:
        url = kwargs.get('url')
        if not url:
            return "Error: missing 'url' parameter"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return f"HTTP error: {resp.status}"

                    html = await resp.text()
                    # Удаляем скрипты и стили
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()

                    if len(text) > 5000:
                        text = text[:5000] + "\n... (truncated)"
                    return text if text else "(page is empty)"
                    
        except asyncio.TimeoutError:
            return "Page load timeout"
        except aiohttp.ClientError as e:
            return f"Network error: {e}"
        except Exception as e:
            logger.error(f"Web fetch error: {e}")
            return f"Error: {e}"


# ================================================================
# ПЛАГИН ФАЙЛОВОЙ СИСТЕМЫ И ПОИСКА
# ================================================================

class FilesPlugin(Plugin):
    """
    Плагин файловой системы и поиска в интернете.
    
    Предоставляет инструменты для работы с файлами и поиска в интернете.
    Режим определяется автоматически через LLM-плагин:
    - local: только песочница data/agent_files/
    - cloud: полный доступ к проекту + apply_patch + execute_code
    """
    
    name = "files"
    version = "1.0"
    dependencies = ["llm"]  # зависит от LLM-плагина для определения режима
    
    def __init__(self, agent):
        super().__init__(agent)
        
        # Пути
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.sandbox_path = os.path.join(self.project_root, "data", "agent_files")
        
        # YandexGPT клиент (будет получен из агента)
        self.yandex_client = None
    
    async def on_load(self):
        """Инициализация"""
        os.makedirs(self.sandbox_path, exist_ok=True)
        
        # Получаем YandexGPT клиент из агента
        self.yandex_client = getattr(self.agent, 'yandex_client', None)
        
        logger.info(f"✅ FilesPlugin loaded (sandbox: {self.sandbox_path})")
        await super().on_load()
    
    def _get_current_mode(self) -> str:
        """Определяет текущий режим из LLM-плагина"""
        llm_plugin = self.agent.get_plugin("llm")
        if llm_plugin:
            return llm_plugin.get_mode()
        return "local"  # fallback
    
    def get_tools(self) -> List[Tool]:
        """
        Возвращает инструменты в зависимости от текущего режима.
        Поиск в интернете доступен всегда (если есть YandexGPT).
        """
        mode = self._get_current_mode()
        
        tools = []
        
        # ========== ИНСТРУМЕНТЫ ПОИСКА (ВСЕГДА, ЕСЛИ ЕСТЬ YANDEX) ==========
        if self.yandex_client:
            tools.extend([
                SearchWebTool(self.agent),
                WebFetchTool(),
            ])
        
        # ========== ФАЙЛОВЫЕ ИНСТРУМЕНТЫ (В ЗАВИСИМОСТИ ОТ РЕЖИМА) ==========
        if mode == "cloud":
            # Облачный режим — полный доступ к проекту
            tools.extend([
                CloudFileReadTool(self.project_root, self.agent),
                CloudFileWriteTool(self.project_root, self.agent),
                CloudFileListTool(self.project_root, self.agent),
                ApplyPatchTool(self.project_root, self.agent),
                CodeExecutionTool(self.agent, timeout=10),
            ])
            logger.debug("🔧 FilesPlugin: cloud mode (full access)")
        else:
            # Локальный режим — только песочница
            tools.extend([
                LocalFileReadTool(self.sandbox_path, self.agent),
                LocalFileWriteTool(self.sandbox_path, self.agent),
                LocalFileListTool(self.sandbox_path, self.agent),
            ])
            logger.debug("🔧 FilesPlugin: local mode (sandbox only)")
        
        return tools
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика плагина"""
        mode = self._get_current_mode()
        
        # Подсчёт файлов в песочнице
        sandbox_files = 0
        if os.path.exists(self.sandbox_path):
            try:
                sandbox_files = len([f for f in os.listdir(self.sandbox_path) 
                                    if os.path.isfile(os.path.join(self.sandbox_path, f))])
            except:
                pass
        
        return {
            "name": self.name,
            "version": self.version,
            "mode": mode,
            "sandbox_path": self.sandbox_path,
            "sandbox_files": sandbox_files,
            "project_root": self.project_root if mode == "cloud" else "(restricted)",
            "web_search_available": self.yandex_client is not None
        }