#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/memory/__init__.py — ПЛАГИН ПАМЯТИ                                    ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Управляет всеми видами памяти робота                                     ║
║   • Эпизодическая память (события, инструкции человека)                      ║
║   • Память маршрутов                                                         ║
║   • Визуальная память (запоминание объектов)                                 ║
║   • Диалоговый контекст (кратковременная память разговора)                   ║
║                                                                              ║
║ ИСПОЛЬЗУЕТ API АГЕНТА:                                                        ║
║   • agent.get_config(key, default) — настройки                                ║
║   • agent.get_sensor_memory() — сенсорная память                              ║
║   • agent.speak(text) — озвучить                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List

from plugins.plugin import Plugin

logger = logging.getLogger(__name__)


class MemoryPlugin(Plugin):
    """
    Плагин памяти.
    
    Управляет всеми видами памяти робота:
    - Эпизодическая память (события, рефлексы, инструкции)
    - Память маршрутов
    - Визуальная память
    - Диалоговый контекст
    """
    
    name = "memory"
    version = "1.0"
    dependencies = []  # независим от других плагинов
    
    def __init__(self, agent):
        super().__init__(agent)
        
        # Компоненты памяти (инициализируются в on_load)
        self.episodic_memory = None
        self.route_memory = None
        self.visual_memory = None
        self.dialog_context = None
        
        # Пути к файлам
        self.episodes_file = None
        self.routes_file = None
        self.visual_file = None
        
        # Фоновая задача сохранения
        self._save_task: Optional[asyncio.Task] = None
    
    # ==================== ЖИЗНЕННЫЙ ЦИКЛ ====================
    
    async def on_load(self):
        """Инициализация всех видов памяти"""
        
        # 1. Эпизодическая память
        await self._init_episodic_memory()
        
        # 2. Память маршрутов
        await self._init_route_memory()
        
        # 3. Визуальная память
        await self._init_visual_memory()
        
        # 4. Диалоговый контекст
        await self._init_dialog_context()
        
        # 5. Запускаем фоновое сохранение
        self._save_task = asyncio.create_task(self._auto_save_loop())
        
        logger.info("✅ Memory-плагин загружен")
        await super().on_load()
    
    async def on_unload(self):
        """Сохранение всех данных при выгрузке"""
        
        # Останавливаем фоновое сохранение
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        
        # Сохраняем все данные
        await self._save_all()
        
        logger.info("🛑 Memory-плагин выгружен")
        await super().on_unload()
    
    # ==================== ИНИЦИАЛИЗАЦИЯ ====================
    
    async def _init_episodic_memory(self):
        """Инициализирует эпизодическую память"""
        try:
            from memory.episodic_memory import EpisodicMemory
            
            self.episodes_file = self.agent.get_config('EPISODES_FILE', 'data/episodes.json')
            self.episodic_memory = EpisodicMemory(storage_path=self.episodes_file)
            
            logger.info(f"✅ EpisodicMemory инициализирована ({self.episodes_file})")
            
        except ImportError as e:
            logger.warning(f"⚠️ EpisodicMemory недоступна: {e}")
            self.episodic_memory = None
    
    async def _init_route_memory(self):
        """Инициализирует память маршрутов"""
        try:
            from memory.route_memory import RouteMemory
            
            self.routes_file = self.agent.get_config('ROUTES_FILE', 'data/routes.json')
            self.route_memory = RouteMemory(self.routes_file)
            
            logger.info(f"✅ RouteMemory инициализирована ({self.routes_file})")
            
        except ImportError as e:
            logger.warning(f"⚠️ RouteMemory недоступна: {e}")
            self.route_memory = None
    
    async def _init_visual_memory(self):
        """Инициализирует визуальную память"""
        try:
            from vision.visual_memory import VisualMemory
            
            self.visual_file = self.agent.get_config('VISUAL_MEMORY_FILE', 'data/visual_memory.json')
            self.visual_memory = VisualMemory(self.visual_file)
            
            logger.info(f"✅ VisualMemory инициализирована ({self.visual_file})")
            
        except ImportError as e:
            logger.warning(f"⚠️ VisualMemory недоступна: {e}")
            self.visual_memory = None
    
    async def _init_dialog_context(self):
        """Инициализирует диалоговый контекст"""
        try:
            from memory.dialog_context import DialogContext
            
            max_turns = self.agent.get_config('DIALOG_MAX_TURNS', 20)
            self.dialog_context = DialogContext(max_turns=max_turns)
            
            logger.info(f"✅ DialogContext инициализирован (max_turns={max_turns})")
            
        except ImportError as e:
            logger.warning(f"⚠️ DialogContext недоступен: {e}")
            self.dialog_context = None
    
    # ==================== API ДЛЯ АГЕНТА ====================
    
    # --- Эпизодическая память ---
    
    def get_episodic_memory(self):
        """Возвращает эпизодическую память"""
        return self.episodic_memory
    
    def add_reflex(self, reflex_type: str, distance_cm: float, 
                   action_taken: str, context: Dict = None):
        """Добавляет рефлекс в память"""
        if self.episodic_memory:
            return self.episodic_memory.add_reflex(
                reflex_type=reflex_type,
                distance_cm=distance_cm,
                action_taken=action_taken,
                context=context
            )
        return None
    
    def add_command(self, command: str, params: Dict, 
                    success: bool, context: Dict = None):
        """Добавляет выполнение команды в память"""
        if self.episodic_memory:
            return self.episodic_memory.add_command(
                command=command,
                params=params,
                success=success,
                context=context
            )
        return None
    
    def add_conversation(self, user_input: str, agent_response: str, context: Dict = None):
        """Добавляет разговор в память"""
        if self.episodic_memory:
            return self.episodic_memory.add_conversation(
                user_input=user_input,
                agent_response=agent_response,
                context=context
            )
        return None
    
    def add_observation(self, observation: str, importance: float = 0.3, context: Dict = None):
        """Добавляет наблюдение в память"""
        if self.episodic_memory:
            return self.episodic_memory.add_observation(
                observation=observation,
                importance=importance,
                context=context
            )
        return None
    
    def add_human_instruction(self, question: str, answer: str, 
                              context: Dict, importance: float = 0.8):
        """Добавляет инструкцию человека в память (ключевая функция обучения)"""
        if self.episodic_memory:
            return self.episodic_memory.add_human_instruction(
                question=question,
                answer=answer,
                context=context,
                importance=importance
            )
        return None
    
    def get_instruction_for_context(self, context: Dict, similarity_threshold: float = 0.6):
        """Находит инструкцию для похожей ситуации"""
        if self.episodic_memory:
            return self.episodic_memory.get_instruction_for_context(
                context=context,
                similarity_threshold=similarity_threshold
            )
        return None
    
    def recall_episodes(self, **kwargs) -> List:
        """Поиск эпизодов по критериям"""
        if self.episodic_memory:
            return self.episodic_memory.recall(**kwargs)
        return []
    
    # --- Память маршрутов ---
    
    def get_route_memory(self):
        """Возвращает память маршрутов"""
        return self.route_memory
    
    def save_route(self, name: str, commands: List[Dict]):
        """Сохраняет маршрут"""
        if self.route_memory:
            self.route_memory.save_route(name, commands)
    
    def get_route(self, name: str) -> Optional[List[Dict]]:
        """Получает маршрут по имени"""
        if self.route_memory:
            return self.route_memory.get_route(name)
        return None
    
    def list_routes(self) -> List[str]:
        """Список всех маршрутов"""
        if self.route_memory:
            return self.route_memory.list_routes()
        return []
    
    def delete_route(self, name: str):
        """Удаляет маршрут"""
        if self.route_memory:
            self.route_memory.delete_route(name)
    
    # --- Визуальная память ---
    
    def get_visual_memory(self):
        """Возвращает визуальную память"""
        return self.visual_memory
    
    def remember_object(self, name: str, image):
        """Запоминает объект"""
        if self.visual_memory:
            return self.visual_memory.save_object(name, image)
        return None
    
    def find_object(self, name: str, image) -> Dict:
        """Ищет объект на изображении"""
        if self.visual_memory:
            return self.visual_memory.find_object(name, image)
        return {"found": False, "confidence": 0}
    
    def search_by_text(self, query: str) -> List:
        """Поиск объектов по текстовому описанию"""
        if self.visual_memory:
            return self.visual_memory.search_by_text(query)
        return []
    
    # --- Диалоговый контекст ---
    
    def get_dialog_context(self):
        """Возвращает диалоговый контекст"""
        return self.dialog_context
    
    def add_dialog_turn(self, human_input: str, agent_output: str, 
                        intent: str = None, confidence: float = 1.0,
                        source: str = "human"):
        """Добавляет реплику в диалог"""
        if self.dialog_context:
            self.dialog_context.add_turn(
                human_input=human_input,
                agent_output=agent_output,
                intent=intent,
                confidence=confidence,
                source=source
            )
    
    def get_primary_intent(self) -> Optional[str]:
        """Возвращает текущее намерение человека"""
        if self.dialog_context:
            return self.dialog_context.get_primary_intent()
        return None
    
    def get_dialog_summary(self) -> str:
        """Возвращает текстовую сводку диалога"""
        if self.dialog_context:
            return self.dialog_context.get_context_text()
        return ""
    
    def clear_dialog(self):
        """Очищает историю диалога"""
        if self.dialog_context:
            self.dialog_context.clear()
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    async def _save_all(self):
        """Сохраняет все виды памяти"""
        
        if self.episodic_memory:
            try:
                self.episodic_memory.save_to_file(self.episodes_file)
                logger.debug("💾 EpisodicMemory сохранена")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения EpisodicMemory: {e}")
        
        if self.route_memory:
            try:
                self.route_memory.save()
                logger.debug("💾 RouteMemory сохранена")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения RouteMemory: {e}")
        
        if self.visual_memory:
            try:
                self.visual_memory.save()
                logger.debug("💾 VisualMemory сохранена")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения VisualMemory: {e}")
    
    async def _auto_save_loop(self):
        """Автоматическое сохранение каждые 60 секунд"""
        while self._loaded and self.agent.is_running():
            await asyncio.sleep(60)
            await self._save_all()
    
    # ==================== ОБРАБОТКА СООБЩЕНИЙ ====================
    
    async def on_message(self, msg_type: str, data: Dict[str, Any]):
        """Обработка входящих сообщений для памяти"""
        
        # Обработка рефлексов (запоминаем их)
        if msg_type == 'reflex' or data.get('capability') == 'tinyml.reflex':
            reflex = data.get('reflex', data.get('data', {}))
            reflex_type = reflex.get('type', 'unknown')
            distance = reflex.get('distance_cm', 0)
            action_taken = reflex.get('action', 'stop')
            
            self.add_reflex(
                reflex_type=reflex_type,
                distance_cm=distance,
                action_taken=action_taken,
                context={"raw": reflex, "timestamp": time.time()}
            )
        
        # Обработка команд оператора
        elif msg_type == 'motor_command':
            self.add_command(
                command="motor_command",
                params={"left": data.get('left'), "right": data.get('right')},
                success=True,
                context={"source": "operator"}
            )
    
    # ==================== СТАТИСТИКА ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику памяти"""
        stats = {
            "name": self.name,
            "version": self.version,
        }
        
        if self.episodic_memory:
            ep_stats = self.episodic_memory.get_stats()
            stats["episodic"] = {
                "total_episodes": ep_stats.get("total_episodes", 0),
                "active_episodes": ep_stats.get("active_episodes", 0),
                "human_instructions": ep_stats.get("human_instructions", 0)
            }
        
        if self.route_memory:
            stats["routes"] = {
                "total": len(self.route_memory.list_routes()) if self.route_memory else 0
            }
        
        if self.visual_memory:
            stats["visual"] = {
                "objects": self.visual_memory.count_objects() if hasattr(self.visual_memory, 'count_objects') else 0
            }
        
        if self.dialog_context:
            stats["dialog"] = {
                "turns": len(self.dialog_context.turns) if hasattr(self.dialog_context, 'turns') else 0,
                "current_intent": self.get_primary_intent()
            }
        
        return stats


# ================================================================
# КОМАНДЫ ДЛЯ LLM (инструменты памяти)
# ================================================================

# Эти инструменты будут зарегистрированы отдельно
# или могут быть встроены в плагин через get_tools()

class RememberInstructionTool:
    """Инструмент для запоминания инструкций человека"""
    name = "remember_instruction"
    description = "Запомнить инструкцию человека для будущего использования"
    
    def __init__(self, agent, memory_plugin):
        self.agent = agent
        self.memory = memory_plugin
    
    async def forward(self, question: str, answer: str, context: str = ""):
        """Запоминает инструкцию"""
        ctx = {"question": question, "answer": answer}
        if context:
            ctx["additional_context"] = context
        
        self.memory.add_human_instruction(
            question=question,
            answer=answer,
            context=ctx,
            importance=0.9
        )
        
        return f"Запомнил: {question[:50]}... → {answer[:50]}..."

class RecallInstructionTool:
    """Инструмент для поиска инструкций в памяти"""
    name = "recall_instruction"
    description = "Вспомнить, что человек говорил в похожей ситуации"
    
    def __init__(self, agent, memory_plugin):
        self.agent = agent
        self.memory = memory_plugin
    
    async def forward(self, situation: str):
        """Ищет инструкцию для похожей ситуации"""
        context = {"description": situation}
        
        instruction = self.memory.get_instruction_for_context(context)
        if instruction:
            return f"Вспомнил: {instruction['question']} → {instruction['answer']}"
        return "Не помню, чтобы кто-то говорил что-то похожее"

class ListRoutesTool:
    """Инструмент для просмотра сохранённых маршрутов"""
    name = "list_routes"
    description = "Показать список сохранённых маршрутов"
    
    def __init__(self, agent, memory_plugin):
        self.agent = agent
        self.memory = memory_plugin
    
    async def forward(self):
        routes = self.memory.list_routes()
        if routes:
            return f"Сохранённые маршруты: {', '.join(routes)}"
        return "Нет сохранённых маршрутов"