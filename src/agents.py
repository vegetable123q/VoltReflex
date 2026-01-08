"""
Agents Module
定义 Agent: RuleAgent, SimpleLLMAgent, CoTAgent, MetaReflexionAgent
"""
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 从环境变量获取默认模型和 API 配置
DEFAULT_MODEL = os.getenv('OPENAI_MODEL', 'Kimi-K2')
DEFAULT_BASE_URL = os.getenv('OPENAI_API_BASE', 'https://llmapi.paratera.com/v1')

from .prompts import (
    format_simple_prompt,
)


# ============================================================
# Base Agent
# ============================================================
class BaseAgent(ABC):
    """Agent 基类"""
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.total_llm_calls = 0
    
    @abstractmethod
    def decide(self, observation: Dict) -> str:
        """
        根据观测做出决策
        
        Args:
            observation: 包含 price, soc, hour 等信息的字典
            
        Returns:
            动作字符串: "CHARGE", "DISCHARGE", or "HOLD"
        """
        pass
    
    def reset(self):
        """重置 Agent 状态"""
        pass
    
    def end_of_day(self, daily_records: List[Dict]) -> Optional[str]:
        """
        每日结束时的回调（可选实现）
        
        Args:
            daily_records: 当天的交易记录列表
            
        Returns:
            反思总结（如果有）
        """
        return None


# ============================================================
# Rule-Based Agent (Baseline)
# ============================================================
class RuleAgent(BaseAgent):
    """
    基于规则的基线 Agent
    
    规则:
    - IF price < charge_threshold AND soc < 90% THEN CHARGE
    - IF price > discharge_threshold AND soc > 10% THEN DISCHARGE
    - ELSE HOLD
    """
    
    def __init__(
        self,
        charge_threshold: float = 0.025,
        discharge_threshold: float = 0.035,
        max_soc: float = 90,
        min_soc: float = 10,
    ):
        super().__init__(name="RuleAgent")
        self.charge_threshold = charge_threshold  # 低于此价格充电
        self.discharge_threshold = discharge_threshold  # 高于此价格放电
        self.max_soc = max_soc  # 最大充电 SOC (%)
        self.min_soc = min_soc  # 最小放电 SOC (%)
    
    def decide(self, observation: Dict) -> str:
        price = observation['price']
        soc = observation['soc']
        
        if price < self.charge_threshold and soc < self.max_soc:
            return "CHARGE"
        elif price > self.discharge_threshold and soc > self.min_soc:
            return "DISCHARGE"
        else:
            return "HOLD"


# ============================================================
# Simple LLM Agent (Zero-Shot, No Memory)
# ============================================================
class SimpleLLMAgent(BaseAgent):
    """
    简单的 LLM Agent，无记忆
    每次决策独立调用 LLM
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.1, base_url: str = None):
        super().__init__(name="SimpleLLMAgent")
        model_name = model_name or DEFAULT_MODEL
        base_url = base_url or DEFAULT_BASE_URL
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
    
    def decide(self, observation: Dict) -> str:
        prompt = format_simple_prompt(
            hour=observation['hour'],
            price=observation['price'],
            soc=observation['soc']
        )
        
        self.total_llm_calls += 1
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            action = self._parse_action(response.content)
            return action
        except Exception as e:
            print(f"LLM Error: {e}")
            return "HOLD"
    
    def _parse_action(self, response: str) -> str:
        """从 LLM 响应中解析动作"""
        response_upper = response.upper().strip()
        
        if "CHARGE" in response_upper and "DISCHARGE" not in response_upper:
            return "CHARGE"
        elif "DISCHARGE" in response_upper:
            return "DISCHARGE"
        else:
            return "HOLD"


# ============================================================
# CoT Agent (自适应 Chain-of-Thought 智能体)
# ============================================================
class CoTAgent(BaseAgent):
    """
    自适应 Chain-of-Thought Agent
    
    核心改进:
    1. 动态学习价格阈值：观察历史价格自动判断高/低价
    2. 每日反思更新：根据当天交易结果调整策略参数
    3. 无需人工设定阈值：LLM 自主分析并给出阈值建议
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.3, base_url: str = None):
        super().__init__(name="CoTAgent")
        model_name = model_name or DEFAULT_MODEL
        base_url = base_url or DEFAULT_BASE_URL
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
        
        # 自适应状态
        self.price_history: List[float] = []  # 观察到的价格历史
        self.daily_prices: List[float] = []   # 当天的价格
        self.strategy_notes: str = ""         # LLM 生成的策略笔记
        self.learned_thresholds: Dict = {     # 学习到的阈值（初始为 None，让 LLM 自己判断）
            "charge_threshold": None,
            "discharge_threshold": None
        }
        self.daily_records: List[Dict] = []   # 当天的交易记录
    
    def reset(self):
        """重置 Agent 状态"""
        self.price_history = []
        self.daily_prices = []
        self.strategy_notes = ""
        self.learned_thresholds = {"charge_threshold": None, "discharge_threshold": None}
        self.daily_records = []
        self.total_llm_calls = 0
    
    def decide(self, observation: Dict) -> str:
        hour = observation['hour']
        price = observation['price']
        soc = observation['soc']
        
        # 记录价格历史
        self.price_history.append(price)
        self.daily_prices.append(price)
        
        # 1. 构造自适应 CoT Prompt
        user_msg = self._build_adaptive_prompt(hour, price, soc)
        
        messages = [
            SystemMessage(content=self._get_adaptive_system_prompt()),
            HumanMessage(content=user_msg)
        ]
        
        self.total_llm_calls += 1
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            # 解析动作
            action = self._parse_cot_action(content)
            
            # 规则护栏
            if action == "CHARGE" and soc >= 95:
                return "HOLD"
            if action == "DISCHARGE" and soc <= 5:
                return "HOLD"
            
            return action
            
        except Exception as e:
            print(f"CoT Agent Error: {e}")
            return "HOLD"
    
    def _get_adaptive_system_prompt(self) -> str:
        """生成包含学习到的策略的系统提示"""
        base_prompt = """You are an expert energy trading agent that LEARNS from experience.
Your goal is to MAXIMIZE PROFIT through battery arbitrage.

BATTERY PHYSICS:
- Capacity: 13.5 kWh
- Max Charge/Discharge: 5 kW per hour
- Efficiency: 90% round-trip (~95% one-way)
- SOC Limits: Keep between 10%-90% for safety

CORE STRATEGY:
1. ARBITRAGE: Buy electricity when CHEAP, sell when EXPENSIVE
2. You must ANALYZE the price data to determine what counts as cheap/expensive
3. CONSTRAINT: Never CHARGE if SOC > 90%, never DISCHARGE if SOC < 10%
"""
        
        # 如果有学习到的策略，添加到提示中
        if self.strategy_notes:
            base_prompt += f"""

YOUR LEARNED STRATEGY (from previous experience):
{self.strategy_notes}
"""
        
        return base_prompt
    
    def _build_adaptive_prompt(self, hour: int, price: float, soc: float) -> str:
        """构建自适应提示，包含价格统计信息"""
        # 计算价格统计
        if len(self.price_history) >= 24:
            recent_prices = self.price_history[-24:]
            price_min = min(recent_prices)
            price_max = max(recent_prices)
            price_mean = sum(recent_prices) / len(recent_prices)
            price_context = f"""
Recent Price Statistics (last 24 hours):
- Min: ${price_min:.4f}, Max: ${price_max:.4f}, Mean: ${price_mean:.4f}
- Current price ${price:.4f} is {'BELOW' if price < price_mean else 'ABOVE'} average
"""
        elif len(self.price_history) > 0:
            price_min = min(self.price_history)
            price_max = max(self.price_history)
            price_mean = sum(self.price_history) / len(self.price_history)
            price_context = f"""
Price Statistics (from {len(self.price_history)} observations):
- Min: ${price_min:.4f}, Max: ${price_max:.4f}, Mean: ${price_mean:.4f}
- Current price ${price:.4f} is {'BELOW' if price < price_mean else 'ABOVE'} average
"""
        else:
            price_context = """
Note: This is the first observation. No price history available yet.
Be conservative and consider HOLD until you gather more price data.
"""
        
        # 添加阈值建议（如果已学习）
        threshold_hint = ""
        if self.learned_thresholds["charge_threshold"] is not None:
            threshold_hint = f"""
Your learned thresholds (you can adjust based on current data):
- CHARGE when price < ${self.learned_thresholds['charge_threshold']:.4f}
- DISCHARGE when price > ${self.learned_thresholds['discharge_threshold']:.4f}
"""
        
        return f"""Current State:
- Hour: {hour}:00
- Price: ${price:.4f}/kWh
- Current SOC: {soc}%
{price_context}{threshold_hint}
Step-by-step Reasoning Task:
1. **Analyze Price**: Based on the statistics, is ${price:.4f} relatively LOW, MODERATE, or HIGH?
2. **Check SOC Constraints**: 
   - CHARGE would add ~35% SOC → New SOC ≈ {soc + 35}%
   - DISCHARGE would remove ~39% SOC → New SOC ≈ {soc - 39}%
   - Is the intended action safe?
3. **Make Decision**: What action maximizes expected profit?

OUTPUT FORMAT:
THOUGHT: [Your analysis of price level and SOC constraints]
ACTION: [CHARGE / DISCHARGE / HOLD]
"""
    
    def _parse_cot_action(self, text: str) -> str:
        """提取 ACTION: 后的关键词"""
        match = re.search(r'ACTION:\s*(CHARGE|DISCHARGE|HOLD)', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        text_upper = text.upper()
        if "CHARGE" in text_upper and "DISCHARGE" not in text_upper:
            return "CHARGE"
        if "DISCHARGE" in text_upper:
            return "DISCHARGE"
        return "HOLD"
    
    def record_transaction(self, info: Dict):
        """记录交易信息（供反思使用）"""
        self.daily_records.append(info)
    
    def end_of_day(self, daily_records: List[Dict]) -> Optional[str]:
        """每日结束时进行反思，更新策略"""
        if not daily_records:
            self.daily_prices = []
            return None
        
        # 使用传入的记录或自己收集的记录
        records = daily_records if daily_records else self.daily_records
        
        # 计算当天统计
        total_profit = sum(r['reward'] for r in records)
        prices = [r['price'] for r in records]
        charge_prices = [r['price'] for r in records if r['action'] == 'CHARGE']
        discharge_prices = [r['price'] for r in records if r['action'] == 'DISCHARGE']
        
        # 构建反思提示
        reflection_prompt = self._build_reflection_prompt(
            total_profit, prices, charge_prices, discharge_prices, records
        )
        
        messages = [
            SystemMessage(content="""You are analyzing battery trading performance to improve strategy.
Your task: Based on today's results, suggest SPECIFIC price thresholds for tomorrow.
Be data-driven: look at actual prices where actions were profitable vs unprofitable."""),
            HumanMessage(content=reflection_prompt)
        ]
        
        self.total_llm_calls += 1
        
        try:
            response = self.llm.invoke(messages)
            reflection = response.content
            
            # 提取建议的阈值
            self._update_thresholds_from_reflection(reflection, prices)
            
            # 更新策略笔记
            self._update_strategy_notes(reflection)
            
            # 清空当天数据
            self.daily_prices = []
            self.daily_records = []
            
            return reflection
            
        except Exception as e:
            print(f"Reflection Error: {e}")
            self.daily_prices = []
            self.daily_records = []
            return None
    
    def _build_reflection_prompt(
        self, 
        total_profit: float, 
        prices: List[float],
        charge_prices: List[float],
        discharge_prices: List[float],
        records: List[Dict]
    ) -> str:
        """构建反思提示"""
        price_min = min(prices) if prices else 0
        price_max = max(prices) if prices else 0
        price_mean = sum(prices) / len(prices) if prices else 0
        
        avg_charge_price = sum(charge_prices) / len(charge_prices) if charge_prices else 0
        avg_discharge_price = sum(discharge_prices) / len(discharge_prices) if discharge_prices else 0
        
        # 找出盈利和亏损的交易
        profitable_charges = [r for r in records if r['action'] == 'CHARGE' and r['reward'] < 0]  # 充电成本
        profitable_discharges = [r for r in records if r['action'] == 'DISCHARGE' and r['reward'] > 0]
        
        return f"""Today's Performance:
- Total Profit: ${total_profit:.4f}
- Price Range: ${price_min:.4f} - ${price_max:.4f} (Mean: ${price_mean:.4f})

Trading Statistics:
- Charged {len(charge_prices)} times at avg price ${avg_charge_price:.4f}
- Discharged {len(discharge_prices)} times at avg price ${avg_discharge_price:.4f}
- Spread captured: ${avg_discharge_price - avg_charge_price:.4f}/kWh

Current Thresholds: 
- Charge: {f"${self.learned_thresholds['charge_threshold']:.4f}" if self.learned_thresholds['charge_threshold'] else "Not set"}
- Discharge: {f"${self.learned_thresholds['discharge_threshold']:.4f}" if self.learned_thresholds['discharge_threshold'] else "Not set"}

Based on this data, provide:
1. ANALYSIS: What worked and what didn't?
2. RECOMMENDED THRESHOLDS:
   - CHARGE_THRESHOLD: $X.XXXX (charge when price below this)
   - DISCHARGE_THRESHOLD: $X.XXXX (discharge when price above this)
3. STRATEGY_NOTE: One sentence advice for tomorrow
"""
    
    def _update_thresholds_from_reflection(self, reflection: str, prices: List[float]):
        """从反思中提取并更新阈值"""
        import re
        
        # 尝试提取 CHARGE_THRESHOLD
        charge_match = re.search(r'CHARGE_THRESHOLD:\s*\$?([\d.]+)', reflection, re.IGNORECASE)
        if charge_match:
            try:
                self.learned_thresholds["charge_threshold"] = float(charge_match.group(1))
            except ValueError:
                pass
        
        # 尝试提取 DISCHARGE_THRESHOLD
        discharge_match = re.search(r'DISCHARGE_THRESHOLD:\s*\$?([\d.]+)', reflection, re.IGNORECASE)
        if discharge_match:
            try:
                self.learned_thresholds["discharge_threshold"] = float(discharge_match.group(1))
            except ValueError:
                pass
        
        # 如果没有提取到，使用价格统计自动设定
        if prices and self.learned_thresholds["charge_threshold"] is None:
            sorted_prices = sorted(prices)
            # 充电阈值：25 分位数
            self.learned_thresholds["charge_threshold"] = sorted_prices[len(sorted_prices) // 4]
            # 放电阈值：75 分位数
            self.learned_thresholds["discharge_threshold"] = sorted_prices[3 * len(sorted_prices) // 4]
    
    def _update_strategy_notes(self, reflection: str):
        """从反思中提取策略笔记"""
        match = re.search(r'STRATEGY_NOTE:\s*(.+?)(?:\n|$)', reflection, re.IGNORECASE | re.DOTALL)
        if match:
            new_note = match.group(1).strip()
            # 保留最近的策略笔记
            if self.strategy_notes:
                lines = self.strategy_notes.split('\n')[-2:]  # 保留最近2条
                lines.append(f"- {new_note}")
                self.strategy_notes = '\n'.join(lines)
            else:
                self.strategy_notes = f"- {new_note}"
    
    def get_memory_summary(self) -> str:
        """返回当前的记忆摘要"""
        summary = f"Learned Thresholds:\n"
        summary += f"  - Charge when < ${self.learned_thresholds['charge_threshold']:.4f}\n" if self.learned_thresholds['charge_threshold'] else "  - Charge threshold: Not learned yet\n"
        summary += f"  - Discharge when > ${self.learned_thresholds['discharge_threshold']:.4f}\n" if self.learned_thresholds['discharge_threshold'] else "  - Discharge threshold: Not learned yet\n"
        if self.strategy_notes:
            summary += f"\nStrategy Notes:\n{self.strategy_notes}"
        return summary


# ============================================================
# Meta-Reflexion Agent (AGA 架构 - Agent Generates Agent)
# ============================================================
class MetaReflexionAgent(BaseAgent):
    """
    元反思智能体 (Agent-Generates-Agent)
    
    核心创新:
    1. 不再输出文本策略，而是生成 Python 代码
    2. 生成的代码直接在环境中执行
    3. 根据执行结果迭代优化代码
    4. 最终生成超越人工编写规则的策略
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.3,
        base_url: str = None,
        max_fix_attempts: int = 3
    ):
        super().__init__(name="MetaReflexionAgent")
        model_name = model_name or DEFAULT_MODEL
        base_url = base_url or DEFAULT_BASE_URL
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
        
        # 代码执行器
        from .code_executor import StrategyLoader, GeneratedStrategyBase
        self.loader = StrategyLoader()
        
        # 当前生成的策略代理
        self.current_strategy = None
        self.current_code = ""
        
        # 最佳策略记录
        self.best_code = ""
        self.best_profit = float('-inf')
        
        # 策略历史
        self.code_history: List[Dict] = []  # [{code, profit, day}]
        
        # 当天数据收集
        self.daily_records: List[Dict] = []
        self.price_history: List[float] = []
        
        # 配置
        self.max_fix_attempts = max_fix_attempts
        self.current_day = 0
        
        # 导入 prompts
        from .prompts import (
            META_CODER_SYSTEM_PROMPT,
            META_CODER_INITIAL_PROMPT,
            format_meta_coder_feedback,
            format_meta_code_fix
        )
        self.system_prompt = META_CODER_SYSTEM_PROMPT
        self.initial_prompt = META_CODER_INITIAL_PROMPT
        self.format_feedback = format_meta_coder_feedback
        self.format_fix = format_meta_code_fix
    
    def reset(self):
        """重置 Agent 状态"""
        self.current_strategy = None
        self.current_code = ""
        self.best_code = ""
        self.best_profit = float('-inf')
        self.code_history = []
        self.daily_records = []
        self.price_history = []
        self.current_day = 0
        self.total_llm_calls = 0
    
    def _generate_initial_strategy(self):
        """生成初始策略代码"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.initial_prompt)
        ]
        
        self.total_llm_calls += 1
        response = self.llm.invoke(messages)
        return response.content
    
    def _generate_improved_strategy(self, feedback_prompt: str) -> str:
        """基于反馈生成改进的策略"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=feedback_prompt)
        ]
        
        self.total_llm_calls += 1
        response = self.llm.invoke(messages)
        return response.content
    
    def _fix_code(self, code: str, error: str) -> str:
        """修复有错误的代码"""
        fix_prompt = self.format_fix(code, error)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=fix_prompt)
        ]
        
        self.total_llm_calls += 1
        response = self.llm.invoke(messages)
        return response.content
    
    def _load_strategy_with_retry(self, code_string: str) -> bool:
        """
        尝试加载策略，如果失败则让 LLM 修复
        
        Returns:
            是否成功加载
        """
        current_code = code_string
        
        for attempt in range(self.max_fix_attempts):
            agent, error = self.loader.load_strategy(current_code)
            
            if agent is not None:
                # 测试策略
                success, test_error = self.loader.test_strategy(agent)
                if success:
                    self.current_strategy = agent
                    self.current_code = current_code
                    return True
                else:
                    error = f"Strategy test failed: {test_error}"
            
            # 尝试修复
            if attempt < self.max_fix_attempts - 1:
                print(f"  [MetaAgent] Attempt {attempt + 1} failed: {error[:100]}...")
                current_code = self._fix_code(current_code, error)
        
        print(f"  [MetaAgent] Failed to load strategy after {self.max_fix_attempts} attempts")
        return False
    
    def _ensure_strategy_exists(self):
        """确保有可用的策略"""
        if self.current_strategy is None:
            print("  [MetaAgent] Generating initial strategy...")
            code = self._generate_initial_strategy()
            
            if not self._load_strategy_with_retry(code):
                # 回退到简单规则
                print("  [MetaAgent] Using fallback rule-based strategy")
                self.current_strategy = self._create_fallback_strategy()
                self.current_code = "# Fallback strategy"
    
    def _create_fallback_strategy(self):
        """创建回退策略"""
        from .code_executor import GeneratedStrategyBase
        
        class FallbackStrategy(GeneratedStrategyBase):
            def __init__(self):
                super().__init__()
                self.name = "FallbackStrategy"
            
            def decide(self, observation):
                price = observation["price"]
                soc = observation["soc"]
                
                if price < 0.03 and soc < 90:
                    return "CHARGE"
                elif price > 0.04 and soc > 10:
                    return "DISCHARGE"
                return "HOLD"
        
        return FallbackStrategy()
    
    def decide(self, observation: Dict) -> str:
        """
        使用生成的策略进行决策
        """
        # 确保有策略
        self._ensure_strategy_exists()
        
        # 记录价格
        self.price_history.append(observation['price'])
        
        # 调用生成的策略
        try:
            action = self.current_strategy.decide(observation)
            
            # 验证并清理动作
            action = action.upper().strip() if isinstance(action, str) else "HOLD"
            if action not in ["CHARGE", "DISCHARGE", "HOLD"]:
                action = "HOLD"
            
            # 规则护栏
            soc = observation['soc']
            if action == "CHARGE" and soc >= 90:
                action = "HOLD"
            if action == "DISCHARGE" and soc <= 10:
                action = "HOLD"
            
            return action
            
        except Exception as e:
            print(f"  [MetaAgent] Strategy execution error: {e}")
            return "HOLD"
    
    def record_transaction(self, info: Dict):
        """记录交易"""
        self.daily_records.append(info)
    
    def end_of_day(self, daily_records: List[Dict]) -> Optional[str]:
        """
        每日结束时进行元反思，生成新策略
        """
        records = daily_records if daily_records else self.daily_records
        
        if not records:
            self.daily_records = []
            self.current_day += 1
            return None
        
        # 计算统计
        total_profit = sum(r['reward'] for r in records)
        prices = [r['price'] for r in records]
        charge_prices = [r['price'] for r in records if r['action'] == 'CHARGE']
        discharge_prices = [r['price'] for r in records if r['action'] == 'DISCHARGE']
        
        price_stats = {
            'min': min(prices) if prices else 0,
            'max': max(prices) if prices else 0,
            'mean': sum(prices) / len(prices) if prices else 0
        }
        
        action_stats = {
            'charge': len(charge_prices),
            'discharge': len(discharge_prices),
            'hold': len(records) - len(charge_prices) - len(discharge_prices),
            'avg_charge_price': sum(charge_prices) / len(charge_prices) if charge_prices else 0,
            'avg_discharge_price': sum(discharge_prices) / len(discharge_prices) if discharge_prices else 0
        }
        
        # 分析策略问题
        analysis = self._analyze_performance(records, action_stats)
        
        # 记录历史
        self.code_history.append({
            'code': self.current_code,
            'profit': total_profit,
            'day': self.current_day
        })
        
        # 更新最佳策略
        if total_profit > self.best_profit:
            self.best_profit = total_profit
            self.best_code = self.current_code
            print(f"  [MetaAgent] New best strategy! Profit: ${total_profit:.4f}")
        
        # 生成改进的策略
        feedback_prompt = self.format_feedback(
            previous_code=self.current_code,
            total_profit=total_profit,
            num_days=1,
            price_stats=price_stats,
            action_stats=action_stats,
            analysis=analysis
        )
        
        new_code = self._generate_improved_strategy(feedback_prompt)
        
        # 尝试加载新策略
        if self._load_strategy_with_retry(new_code):
            print(f"  [MetaAgent] Day {self.current_day + 1}: Strategy updated successfully")
        else:
            print(f"  [MetaAgent] Day {self.current_day + 1}: Keeping previous strategy")
        
        # 清空当天数据
        self.daily_records = []
        self.current_day += 1
        
        return f"Day {self.current_day} profit: ${total_profit:.4f}, Strategy updated."
    
    def _analyze_performance(self, records: List[Dict], action_stats: Dict) -> str:
        """分析策略表现"""
        analysis_points = []
        
        avg_charge = action_stats['avg_charge_price']
        avg_discharge = action_stats['avg_discharge_price']
        
        # 检查买卖价差
        if avg_charge > 0 and avg_discharge > 0:
            spread = avg_discharge - avg_charge
            if spread < 0:
                analysis_points.append(f"CRITICAL: Buying high (${avg_charge:.4f}) and selling low (${avg_discharge:.4f})! Thresholds are inverted.")
            elif spread < 0.01:
                analysis_points.append(f"WARNING: Spread too small (${spread:.4f}). Need wider threshold gap.")
        
        # 检查动作分布
        total = action_stats['charge'] + action_stats['discharge'] + action_stats['hold']
        if action_stats['hold'] > total * 0.8:
            analysis_points.append("Too many HOLD actions. Thresholds may be too aggressive.")
        
        if action_stats['charge'] == 0:
            analysis_points.append("No CHARGE actions. Charge threshold too low.")
        
        if action_stats['discharge'] == 0:
            analysis_points.append("No DISCHARGE actions. Discharge threshold too high.")
        
        # 检查高价充电和低价放电
        for r in records:
            if r['action'] == 'CHARGE' and r['price'] > 0.05:
                analysis_points.append(f"Charged at high price ${r['price']:.4f} at hour {r['hour']}")
                break
        
        for r in records:
            if r['action'] == 'DISCHARGE' and r['price'] < 0.03:
                analysis_points.append(f"Discharged at low price ${r['price']:.4f} at hour {r['hour']}")
                break
        
        return "\n".join(analysis_points) if analysis_points else "Strategy performing reasonably."
    
    def get_memory_summary(self) -> str:
        """获取当前记忆摘要"""
        summary = f"Meta-Reflexion Agent Status:\n"
        summary += f"  - Current Day: {self.current_day}\n"
        summary += f"  - Best Profit: ${self.best_profit:.4f}\n"
        summary += f"  - Strategies Tried: {len(self.code_history)}\n"
        summary += f"  - Total LLM Calls: {self.total_llm_calls}\n"
        
        if self.current_code and self.current_code != "# Fallback strategy":
            # 提取策略参数
            summary += f"\nCurrent Strategy Code (truncated):\n"
            code_lines = self.current_code.split('\n')[:20]
            summary += '\n'.join(code_lines)
            if len(self.current_code.split('\n')) > 20:
                summary += "\n... (truncated)"
        
        return summary
    
    def get_best_strategy_code(self) -> str:
        """返回最佳策略代码"""
        return self.best_code


# ============================================================
# Agent Factory
# ============================================================
def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Agent 工厂函数
    
    Args:
        agent_type: "rule", "simple_llm", "cot", or "meta"
        **kwargs: Agent 特定参数
        
    Returns:
        Agent 实例
    """
    agents = {
        "rule": RuleAgent,
        "simple_llm": SimpleLLMAgent,
        "cot": CoTAgent,
        "meta": MetaReflexionAgent,
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from {list(agents.keys())}")
    
    return agents[agent_type](**kwargs)
