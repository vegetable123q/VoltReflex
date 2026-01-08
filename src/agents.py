"""
Agents Module
定义三种 Agent: RuleAgent, SimpleLLMAgent, ReflexionAgent
"""
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# 从环境变量获取默认模型和 API 配置
DEFAULT_MODEL = os.getenv('OPENAI_MODEL', 'Kimi-K2')
DEFAULT_BASE_URL = os.getenv('OPENAI_API_BASE', 'https://llmapi.paratera.com/v1')

from .prompts import (
    DECISION_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    format_decision_prompt,
    format_reflection_prompt,
    format_simple_prompt,
)


# ============================================================
# Type Definitions
# ============================================================
class AgentState(TypedDict, total=False):
    """Reflexion Agent 的状态定义"""
    short_term_memory: List[Dict]  # 当天的 (State, Action, Reward) 记录
    long_term_memory: str  # 每日反思日记 (Insights)
    current_obs: Optional[Dict]  # 当前观测
    current_day: int  # 当前天数
    # 临时字段（用于节点间传递）
    _last_response: str  # LLM 决策响应
    _reflection: str  # LLM 反思响应


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
# Reflexion Agent (with Memory and Daily Reflection)
# ============================================================
class ReflexionAgent(BaseAgent):
    """
    带反思机制的 Agent
    
    特点:
    - 短期记忆: 记录当天的交易
    - 长期记忆: 存储每日反思总结
    - 每日结束时进行反思，更新策略
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.1,
        reflection_temperature: float = 0.3,
        base_url: str = None
    ):
        super().__init__(name="ReflexionAgent")
        model_name = model_name or DEFAULT_MODEL
        base_url = base_url or DEFAULT_BASE_URL
        
        # 决策用 LLM（低温度，更确定性）
        self.decision_llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
        
        # 反思用 LLM（稍高温度，更有创造性）
        self.reflection_llm = ChatOpenAI(
            model=model_name,
            temperature=reflection_temperature,
            base_url=base_url
        )
        
        # 初始化状态
        self.state: AgentState = {
            "short_term_memory": [],
            "long_term_memory": "",
            "current_obs": None,
            "current_day": 0
        }
        
        # 构建 LangGraph
        self.decision_graph = self._build_decision_graph()
        self.reflection_graph = self._build_reflection_graph()
    
    def reset(self):
        """重置 Agent 状态"""
        self.state = {
            "short_term_memory": [],
            "long_term_memory": "",
            "current_obs": None,
            "current_day": 0
        }
        self.total_llm_calls = 0
    
    def _build_decision_graph(self) -> StateGraph:
        """构建决策图"""
        workflow = StateGraph(AgentState)
        
        # 添加决策节点
        workflow.add_node("make_decision", self._decision_node)
        
        # 设置入口和出口
        workflow.set_entry_point("make_decision")
        workflow.add_edge("make_decision", END)
        
        return workflow.compile()
    
    def _build_reflection_graph(self) -> StateGraph:
        """构建反思图"""
        workflow = StateGraph(AgentState)
        
        # 添加反思节点
        workflow.add_node("reflect", self._reflection_node)
        workflow.add_node("update_memory", self._update_memory_node)
        
        # 设置流程
        workflow.set_entry_point("reflect")
        workflow.add_edge("reflect", "update_memory")
        workflow.add_edge("update_memory", END)
        
        return workflow.compile()
    
    def _decision_node(self, state: AgentState) -> AgentState:
        """决策节点：调用 LLM 做出交易决策"""
        obs = state["current_obs"]
        
        prompt = format_decision_prompt(
            hour=obs['hour'],
            day=obs.get('day', 0),
            price=obs['price'],
            soc=obs['soc'],
            load=obs.get('load', 0),
            long_term_memory=state["long_term_memory"]
        )
        
        messages = [
            SystemMessage(content=DECISION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        self.total_llm_calls += 1
        response = self.decision_llm.invoke(messages)
        
        # 存储响应以便后续解析
        state["_last_response"] = response.content
        
        return state
    
    def _reflection_node(self, state: AgentState) -> AgentState:
        """反思节点：分析当天表现并生成洞察"""
        records = state["short_term_memory"]
        
        if not records:
            state["_reflection"] = "No transactions to reflect on."
            return state
        
        # 计算统计
        total_profit = sum(r['reward'] for r in records)
        total_cost = sum(r['grid_cost'] for r in records)
        total_revenue = sum(r['grid_revenue'] for r in records)
        charge_count = sum(1 for r in records if r['action'] == 'CHARGE')
        discharge_count = sum(1 for r in records if r['action'] == 'DISCHARGE')
        hold_count = len(records) - charge_count - discharge_count
        
        prompt = format_reflection_prompt(
            day=state["current_day"],
            total_profit=total_profit,
            total_cost=total_cost,
            total_revenue=total_revenue,
            charge_count=charge_count,
            discharge_count=discharge_count,
            hold_count=hold_count,
            records=records,
            previous_insights=state["long_term_memory"]
        )
        
        messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        self.total_llm_calls += 1
        response = self.reflection_llm.invoke(messages)
        
        state["_reflection"] = response.content
        
        return state
    
    def _update_memory_node(self, state: AgentState) -> AgentState:
        """更新记忆节点：将反思结果存入长期记忆"""
        reflection = state.get("_reflection", "")
        
        # 提取策略笔记
        strategy_note = self._extract_strategy_note(reflection)
        
        # 更新长期记忆（保留最近的洞察，避免太长）
        day = state["current_day"]
        new_entry = f"[Day {day}] {strategy_note}"
        
        # 保留最近 5 天的记忆
        existing = state["long_term_memory"]
        if existing:
            lines = existing.split("\n")
            lines = [l for l in lines if l.strip()][-4:]  # 保留最近4条
            lines.append(new_entry)
            state["long_term_memory"] = "\n".join(lines)
        else:
            state["long_term_memory"] = new_entry
        
        # 清空短期记忆，准备新的一天
        state["short_term_memory"] = []
        state["current_day"] += 1
        
        return state
    
    def _extract_strategy_note(self, reflection: str) -> str:
        """从反思响应中提取策略笔记"""
        # 尝试匹配 "STRATEGY NOTE:" 后的内容
        match = re.search(r'STRATEGY NOTE:\s*(.+?)(?:\n|$)', reflection, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 如果没找到，返回整个反思的最后一段
        lines = [l.strip() for l in reflection.split('\n') if l.strip()]
        return lines[-1] if lines else "Continue optimizing buy-low-sell-high strategy."
    
    def decide(self, observation: Dict) -> str:
        """做出决策"""
        self.state["current_obs"] = observation
        
        # 运行决策图
        result = self.decision_graph.invoke(self.state)
        
        # 解析动作
        response = result.get("_last_response", "HOLD")
        action = self._parse_action(response)
        
        return action
    
    def _parse_action(self, response: str) -> str:
        """从 LLM 响应中解析动作"""
        response_upper = response.upper()
        
        # 尝试匹配 "ACTION: XXX" 格式
        match = re.search(r'ACTION:\s*(CHARGE|DISCHARGE|HOLD)', response_upper)
        if match:
            return match.group(1)
        
        # 回退到简单匹配
        if "CHARGE" in response_upper and "DISCHARGE" not in response_upper:
            return "CHARGE"
        elif "DISCHARGE" in response_upper:
            return "DISCHARGE"
        else:
            return "HOLD"
    
    def record_transaction(self, info: Dict):
        """记录交易到短期记忆"""
        self.state["short_term_memory"].append(info)
    
    def end_of_day(self, daily_records: List[Dict]) -> Optional[str]:
        """
        每日结束时进行反思
        
        Args:
            daily_records: 当天的交易记录
            
        Returns:
            反思总结
        """
        # 更新短期记忆
        self.state["short_term_memory"] = daily_records
        
        # 运行反思图
        result = self.reflection_graph.invoke(self.state)
        
        # 更新状态
        self.state = result
        
        return result.get("_reflection", "")
    
    def get_memory_summary(self) -> str:
        """获取当前记忆摘要"""
        return self.state["long_term_memory"]


# ============================================================
# Agent Factory
# ============================================================
def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Agent 工厂函数
    
    Args:
        agent_type: "rule", "simple_llm", or "reflexion"
        **kwargs: Agent 特定参数
        
    Returns:
        Agent 实例
    """
    agents = {
        "rule": RuleAgent,
        "simple_llm": SimpleLLMAgent,
        "reflexion": ReflexionAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from {list(agents.keys())}")
    
    return agents[agent_type](**kwargs)
