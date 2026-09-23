#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ plugins/tasks/__init__.py — ПЛАГИН АВТОНОМНЫХ ЗАДАЧ                          ║
║                                                                              ║
║ ЧТО ДЕЛАЕТ:                                                                  ║
║   • Автономное целеполагание в простое                                       ║
║   • Выполнение многошаговых планов (LLMDecisionMemory)                       ║
║   • Эволюционное обучение (StrategyLearner)                                  ║
║   • Оценка результатов (OutcomeEvaluator)                                    ║
║   • Планирование сложных задач (compose_plan)                                ║
║                                                                              ║
║ ИНСТРУМЕНТЫ:                                                                 ║
║   • compose_plan(goal) — составить план действий для сложной задачи         ║
║                                                                              ║
║ ИСПОЛЬЗУЕТ API АГЕНТА:                                                       ║
║   • agent.inc_stat(key) — статистика                                         ║
║   • agent.get_config(key, default) — настройки                               ║
║   • agent.speak(text) — озвучить                                             ║
║   • agent.get_plugin("llm") — LLM-плагин                                     ║
║   • agent.get_sensor_memory() — сенсорная память                             ║
║   • agent.get_episodic_memory() — эпизодическая память                       ║
║   • agent.get_tools() — список инструментов                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import re
import time
import random
import logging
from typing import Optional, Dict, Any, List

from plugins.plugin import Plugin
from orchestration.tool import Tool

logger = logging.getLogger(__name__)


# ================================================================
# ИНСТРУМЕНТ ПЛАНИРОВАНИЯ
# ================================================================

class ComposePlanTool(Tool):
    """
    Инструмент: compose_plan
    
    Составляет план действий для сложной задачи.
    Использует LLM для разбиения цели на последовательность шагов.
    """
    name = "compose_plan"
    description = "Create a multi-step plan for complex tasks. Returns plan with steps."
    latency = 0.5

    def __init__(self, agent, tasks_plugin):
        self.agent = agent
        self.plugin = tasks_plugin

    async def forward(self, **kwargs) -> str:
        """
        Составляет план для достижения цели.
        
        Args:
            **kwargs: должен содержать 'goal' (обязательно) и 'context' (опционально)
        """
        goal = kwargs.get('goal')
        if not goal:
            return "Error: missing 'goal' parameter"
        
        context = kwargs.get('context', {})
        
        from core.prompts import get_compose_plan_prompt

        # Собираем информацию о текущей ситуации
        sensor_summary = {}
        if self.agent.get_sensor_memory():
            summaries = self.agent.get_sensor_memory().get_summaries()
            sensor_summary = summaries

        available_tools = [t.name for t in self.agent.tools]

        prompt = get_compose_plan_prompt(
            goal=goal,
            sensor_summary=sensor_summary,
            available_tools=', '.join(available_tools)
        )

        # Вызываем LLM
        llm_plugin = self.agent.get_plugin("llm")
        if not llm_plugin:
            return "LLM not available"

        response = await llm_plugin.call(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Парсим JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group())
                if plan.get("steps"):
                    steps_text = []
                    for s in plan["steps"]:
                        if isinstance(s, dict):
                            steps_text.append(s.get("description", s.get("action", "")))
                        else:
                            steps_text.append(str(s))
                    
                    self.plugin.set_task(
                        steps=steps_text,
                        reasoning=plan.get("reasoning", ""),
                        task_name=goal[:50]
                    )
                    return f"Plan created: {len(plan['steps'])} steps.\nReasoning: {plan.get('reasoning', '')[:100]}..."
            except json.JSONDecodeError:
                pass

        return "Failed to create plan. Please rephrase your goal."


# ================================================================
# ПЛАГИН АВТОНОМНЫХ ЗАДАЧ
# ================================================================

class TasksPlugin(Plugin):
    """
    Плагин автономных задач.
    
    В простое генерирует себе задачи, выполняет многошаговые планы,
    учится на ошибках через StrategyLearner.
    """
    
    name = "tasks"
    version = "1.0"
    dependencies = ["llm"]  # нужен LLM-плагин для вызова модели
    
    def __init__(self, agent):
        super().__init__(agent)
        
        # Время последней команды (для определения простоя)
        self.last_command_time = time.time()
        
        # Состояние
        self._conversation_active = False
        self._idle_task: Optional[asyncio.Task] = None
        
        # Компоненты (инициализируются в on_load)
        self.strategy_learner = None
        self.decision_memory = None
        self.outcome_evaluator = None
        
        # Для оценки действий
        self._last_action_id: Optional[str] = None
        self._last_strategy_id: Optional[str] = None
        self._last_task_type: Optional[str] = None
        
        # Статистика
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "strategies_evolved": 0,
            "self_tasks_generated": 0
        }
    
    # ==================== ЖИЗНЕННЫЙ ЦИКЛ ====================
    
    async def on_load(self):
        """Инициализация компонентов и запуск фоновых задач"""
        
        # 1. Инициализируем LLMDecisionMemory
        await self._init_decision_memory()
        
        # 2. Инициализируем StrategyLearner
        await self._init_strategy_learner()
        
        # 3. Запускаем фоновый цикл обучения в простое
        self._idle_task = asyncio.create_task(self._idle_learning_loop())
        
        logger.info("✅ TasksPlugin loaded")
        await super().on_load()
    
    async def on_unload(self):
        """Остановка фоновых задач и сохранение данных"""
        
        if self._idle_task:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        
        if self.strategy_learner:
            await self.strategy_learner.stop_evaluator()
            self.strategy_learner.save()
        
        if self.decision_memory:
            decisions_file = self.agent.get_config('DECISIONS_FILE')
            if decisions_file:
                self.decision_memory.save_to_file(decisions_file)
        
        logger.info("🛑 TasksPlugin unloaded")
        await super().on_unload()
    
    # ==================== ИНИЦИАЛИЗАЦИЯ ====================
    
    async def _init_decision_memory(self):
        """Инициализирует LLMDecisionMemory"""
        try:
            from core.llm_decision_memory import LLMDecisionMemory
            
            self.decision_memory = LLMDecisionMemory()
            
            decisions_file = self.agent.get_config('DECISIONS_FILE')
            if decisions_file:
                import os
                if os.path.exists(decisions_file):
                    self.decision_memory.load_from_file(decisions_file)
            
            logger.info("✅ LLMDecisionMemory initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️ LLMDecisionMemory not available: {e}")
            self.decision_memory = None
    
    async def _init_strategy_learner(self):
        """Инициализирует StrategyLearner"""
        try:
            from core.strategy_learner import StrategyLearner
            
            # Получаем LLM из LLM-плагина
            llm_plugin = self.agent.get_plugin("llm")
            if not llm_plugin or not llm_plugin.local_llm:
                logger.warning("⚠️ LLM plugin not loaded or local LLM not available")
                return
            
            strategies_file = self.agent.get_config('STRATEGIES_FILE', 'data/strategies.json')
            
            self.strategy_learner = StrategyLearner(llm_plugin.local_llm, strategies_file)
            self.strategy_learner.evaluator.set_agent(self.agent)
            await self.strategy_learner.start_evaluator()
            
            logger.info("✅ StrategyLearner initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️ StrategyLearner not available: {e}")
            self.strategy_learner = None
    
    # ==================== API ДЛЯ АГЕНТА ====================
    
    def on_user_command(self):
        """Сбрасывает таймер простоя при получении команды"""
        self.last_command_time = time.time()
    
    def get_idle_time(self) -> float:
        """Возвращает время простоя в секундах"""
        return time.time() - self.last_command_time
    
    def has_active_task(self) -> bool:
        """Есть ли активная задача"""
        return self.decision_memory is not None and self.decision_memory.has_active_task()
    
    def get_current_step(self) -> Optional[str]:
        """Возвращает текущий шаг активной задачи"""
        if self.decision_memory:
            return self.decision_memory.get_current_step()
        return None
    
    def get_task_progress(self) -> Dict[str, Any]:
        """Возвращает прогресс текущей задачи"""
        if self.decision_memory:
            return self.decision_memory.get_task_progress()
        return {"has_task": False}
    
    def advance_step(self, result: Any = None) -> bool:
        """
        Переходит к следующему шагу задачи.
        Возвращает True если есть следующий шаг, False если задача завершена.
        """
        if not self.decision_memory:
            return False
        
        has_next = self.decision_memory.advance_step(result)
        
        if not has_next:
            progress = self.decision_memory.get_task_progress()
            if progress.get("has_task") == False:
                self.stats["tasks_completed"] += 1
        
        return has_next
    
    def set_task(self, steps: List[str], reasoning: str = "", task_name: str = ""):
        """Устанавливает новую задачу"""
        if self.decision_memory:
            self.decision_memory.set_task(
                steps=steps,
                reasoning=reasoning,
                task_name=task_name or "автономная задача"
            )
            logger.info(f"📋 New task: {task_name[:50]}...")
            self.stats["self_tasks_generated"] += 1
    
    def cancel_task(self):
        """Отменяет текущую задачу"""
        if self.decision_memory:
            self.decision_memory.cancel_task()
            logger.info("🛑 Current task cancelled")
    
    def fail_task(self, reason: str):
        """Помечает текущую задачу как проваленную"""
        if self.decision_memory:
            self.decision_memory.fail_task(reason)
            self.stats["tasks_failed"] += 1
            logger.warning(f"❌ Task failed: {reason}")
    
    # ==================== ОЦЕНКА ДЕЙСТВИЙ ====================
    
    def register_action(self, action_id: str, intent: str,
                       strategy_id: str = None, task_type: str = None,
                       context: Dict = None):
        """
        Регистрирует действие для последующей оценки.
        Вызывается агентом перед выполнением действия.
        """
        self._last_action_id = action_id
        self._last_strategy_id = strategy_id
        self._last_task_type = task_type
        
        if self.strategy_learner:
            self.strategy_learner.register_action_for_evaluation(
                action_id=action_id,
                intent=intent,
                strategy_id=strategy_id,
                task_type=task_type or "unknown",
                context=context or {}
            )
    
    async def schedule_evaluation(self, action_id: str, delay: float = 2.0):
        """Отложенная оценка действия"""
        await asyncio.sleep(delay)
        if self.strategy_learner:
            await self.strategy_learner.evaluator.evaluation_queue.put(action_id)
            self.agent.inc_stat("evaluations_completed")
    
    def check_praise(self, text: str) -> bool:
        """Проверяет, содержит ли текст похвалу"""
        if not self.strategy_learner:
            return False
        
        praise_phrases = [
            "молодец", "хорошо", "отлично", "умница", "спасибо",
            "класс", "супер", "замечательно", "правильно", "так держать",
            "good", "nice", "well done", "great", "excellent", "thanks"
        ]
        
        text_lower = text.lower()
        for phrase in praise_phrases:
            if phrase in text_lower:
                return True
        return False
    
    async def evaluate_with_praise(self, action_id: str):
        """Оценивает действие с похвалой от человека"""
        if self.strategy_learner:
            await self.strategy_learner.evaluator.evaluate(action_id, human_praise=True)
            self.agent.inc_stat("praise_received")
    
    # ==================== ВЫБОР СТРАТЕГИИ ====================
    
    def select_strategy(self, intent: str):
        """Выбирает стратегию для выполнения задачи"""
        if not self.strategy_learner:
            return None
        return self.strategy_learner.select_strategy(intent)
    
    async def execute_strategy(self, strategy, user_input: str) -> bool:
        """Выполняет стратегию"""
        if not self.strategy_learner:
            return False
        
        try:
            logger.info(f"🔧 Strategy: {strategy.name} (score={strategy.get_current_score()})")
            
            namespace = {
                "self": self.agent,
                "user_input": user_input,
                "asyncio": asyncio,
                "logger": logger,
                "time": time
            }
            exec(strategy.code, namespace)
            
            if "execute" in namespace:
                result = await namespace["execute"](self.agent, user_input=user_input)
                return result if isinstance(result, bool) else True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy execution error: {e}")
            return False
    
    def update_strategy_score(self, strategy, success: bool):
        """Обновляет счёт стратегии"""
        if self.strategy_learner:
            self.strategy_learner.update_score(strategy, success)
            if success:
                self.agent.inc_stat("strategy_success")
    
    # ==================== АВТОНОМНОЕ ЦЕЛЕПОЛАГАНИЕ ====================
    
    async def generate_self_task(self) -> Optional[Dict]:
        """Генерирует задачу для автономного выполнения"""
        if not self.strategy_learner:
            return None
        
        # 1. Собираем сводку сенсоров
        sensor_text = "no data"
        sensor_memory = self.agent.get_sensor_memory()
        if sensor_memory:
            summaries = sensor_memory.get_summaries()
            sensor_text = ", ".join([f"{k}: {v['summary'][:100]}" for k, v in summaries.items()])
        
        # 2. Получаем список доступных инструментов
        tools = self.agent.tools
        available_tools = [t.name for t in tools] if tools else []
        
        # 3. Данные из эпизодической памяти
        explored = []
        interesting = []
        recent = []
        
        episodic_memory = self.agent.get_episodic_memory()
        if episodic_memory:
            explored = [ep.description for ep in episodic_memory.recall(
                episode_type="observation", limit=5, min_weight=0.3
            )]
            interesting = [ep.context.get("object", "") for ep in episodic_memory.recall(
                tags=["interesting"], limit=5
            )]
            recent = [ep.description for ep in episodic_memory.recall(
                limit=3, min_weight=0.5, max_age=300
            )]
        
        # 4. Генерируем задачу
        return await self.strategy_learner.generate_self_task(
            sensor_summary=sensor_text,
            explored_areas=explored or ["no data"],
            interesting_objects=interesting or ["none"],
            recent_events=recent or ["none"],
            available_tools=available_tools or ["none"],
            role="autonomous agent",
            self_description="I can move, observe, speak, search the web and interact"
        )
    
    # ==================== ОБРАБОТКА СООБЩЕНИЙ ====================
    
    async def on_message(self, msg_type: str, data: Dict[str, Any]):
        """Обработка входящих сообщений"""
        
        # Обновляем таймер при командах от человека
        if msg_type == 'human_query':
            self.on_user_command()
            
            text = data.get('text', '')
            if text and self.check_praise(text):
                if self._last_action_id:
                    await self.evaluate_with_praise(self._last_action_id)
                    await self.agent.speak("Спасибо! Я стараюсь.")
        
        # При рефлексе — прерываем задачу
        elif msg_type == 'reflex':
            if self.has_active_task():
                reflex_type = data.get('reflex', {}).get('type', 'unknown')
                self.fail_task(f"reflex {reflex_type}")
    
    # ==================== ФОНОВЫЙ ЦИКЛ ====================
    
    async def _idle_learning_loop(self):
        """Обучение в простое и автономное целеполагание"""
        while self._loaded and self.agent.is_running():
            idle = self.get_idle_time()
            
            # 1. Обучение стратегий (каждые 20-25 секунд простоя)
            if 20 < idle < 25:
                logger.info(f"📚 Idle {idle:.0f}s, learning strategies...")
                if self.strategy_learner:
                    await self.strategy_learner.learn_in_idle(idle)
                await asyncio.sleep(30)
            
            # 2. Автономное целеполагание
            elif (self.agent.get_interactive_mode() and
                  not self.agent.is_frozen() and
                  not self.has_active_task() and
                  not self._conversation_active):
                
                idle_threshold = self.agent.get_config('AUTO_TASK_IDLE_TIME', 120)
                
                if idle > idle_threshold:
                    random_time = random.randint(40, 180)
                    
                    if idle > random_time:
                        logger.info(f"🎯 Idle {idle:.0f}s, generating self task...")
                        
                        task = await self.generate_self_task()
                        if task and task.get("steps"):
                            self.set_task(
                                steps=task["steps"],
                                reasoning=task.get("reasoning", ""),
                                task_name=task.get("task_name", "автономная задача")
                            )
                            
                            # Запускаем выполнение через API агента
                            if hasattr(self.agent, 'execute_next_step'):
                                await self.agent.execute_next_step()
                        
                        cooldown = self.agent.get_config('AUTO_TASK_COOLDOWN', 60)
                        await asyncio.sleep(cooldown)
            
            await asyncio.sleep(5)
    
    # ==================== ИНСТРУМЕНТЫ ====================
    
    def get_tools(self) -> List[Tool]:
        """Возвращает инструменты планирования"""
        return [
            ComposePlanTool(self.agent, self),
        ]
    
    # ==================== СТАТИСТИКА ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику плагина"""
        stats = {
            "name": self.name,
            "version": self.version,
            "idle_time": self.get_idle_time(),
            "has_active_task": self.has_active_task(),
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "self_tasks_generated": self.stats["self_tasks_generated"]
        }
        
        if self.strategy_learner:
            strategy_stats = self.strategy_learner.get_stats()
            stats.update(strategy_stats)
        
        if self.decision_memory:
            mem_stats = self.decision_memory.get_stats()
            stats["cache_hit_rate"] = mem_stats.get("hit_rate", 0)
            stats["cache_hits"] = mem_stats.get("cache_hits", 0)
            stats["cache_misses"] = mem_stats.get("cache_misses", 0)
        
        return stats