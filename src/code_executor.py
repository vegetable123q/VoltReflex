"""
Code Executor Module
安全执行 LLM 生成的 Python 策略代码

提供代码沙箱功能，将字符串形式的 Python 代码转化为可运行的 Agent 对象。
"""
import re
import traceback
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod


class GeneratedStrategyBase(ABC):
    """
    生成策略的基类接口
    所有 LLM 生成的策略必须继承此类
    """
    
    def __init__(self):
        self.name = "GeneratedAgent"
        self.total_llm_calls = 0
    
    @abstractmethod
    def decide(self, observation: Dict) -> str:
        """
        根据观测做出决策
        
        Args:
            observation: 包含以下字段的字典
                - price: float, 当前电价 ($/kWh)
                - soc: float, 当前电量百分比 (0-100)
                - hour: int, 当前小时 (0-23)
                
        Returns:
            str: "CHARGE", "DISCHARGE", 或 "HOLD"
        """
        pass
    
    def reset(self):
        """重置 Agent 状态"""
        pass
    
    def end_of_day(self, daily_records) -> Optional[str]:
        """每日结束回调（可选）"""
        return None


# 提供给 LLM 的代码模板
STRATEGY_TEMPLATE = '''
class GeneratedAgent(GeneratedStrategyBase):
    """
    由 Meta-Agent 生成的交易策略
    """
    
    def __init__(self):
        super().__init__()
        self.name = "GeneratedAgent"
        # 在此定义策略参数
        self.charge_threshold = 0.025    # 低于此价格充电
        self.discharge_threshold = 0.035  # 高于此价格放电
        self.max_soc = 90  # 最大充电 SOC (%)
        self.min_soc = 10  # 最小放电 SOC (%)
    
    def decide(self, observation: Dict) -> str:
        """
        Args:
            observation: {"price": float, "soc": float, "hour": int}
        Returns:
            "CHARGE", "DISCHARGE", or "HOLD"
        """
        price = observation["price"]
        soc = observation["soc"]
        hour = observation["hour"]
        
        # 策略逻辑
        if price < self.charge_threshold and soc < self.max_soc:
            return "CHARGE"
        elif price > self.discharge_threshold and soc > self.min_soc:
            return "DISCHARGE"
        else:
            return "HOLD"
'''


class StrategyLoader:
    """
    策略加载器
    负责将 LLM 生成的代码字符串转化为可执行的 Agent 实例
    """
    
    def __init__(self):
        self.last_error = None
        self.last_code = None
    
    def _extract_code(self, code_string: str) -> str:
        """
        从 LLM 输出中提取纯净的 Python 代码
        处理 markdown 代码块标记
        """
        # 尝试提取 ```python ... ``` 代码块
        pattern = r'```(?:python)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, code_string)
        
        if matches:
            # 取最长的代码块（通常是主要代码）
            code = max(matches, key=len)
        else:
            # 没有代码块标记，直接使用原文
            code = code_string
        
        return code.strip()
    
    def _validate_code_structure(self, code: str) -> Tuple[bool, str]:
        """
        验证代码结构是否符合要求
        """
        # 检查是否定义了 GeneratedAgent 类
        if 'class GeneratedAgent' not in code:
            return False, "Code must define a class named 'GeneratedAgent'"
        
        # 检查是否有 decide 方法
        if 'def decide' not in code:
            return False, "GeneratedAgent must implement 'decide' method"
        
        # 检查返回值是否合法（简单检查）
        valid_returns = ['CHARGE', 'DISCHARGE', 'HOLD']
        has_valid_return = any(ret in code for ret in valid_returns)
        if not has_valid_return:
            return False, "decide() must return 'CHARGE', 'DISCHARGE', or 'HOLD'"
        
        return True, ""
    
    def load_strategy(self, code_string: str) -> Tuple[Optional[GeneratedStrategyBase], Optional[str]]:
        """
        加载并实例化策略代码
        
        Args:
            code_string: LLM 生成的代码字符串
            
        Returns:
            (agent_instance, error_message)
            - 成功: (GeneratedAgent实例, None)
            - 失败: (None, 错误信息)
        """
        self.last_code = code_string
        self.last_error = None
        
        # 1. 提取代码
        try:
            clean_code = self._extract_code(code_string)
        except Exception as e:
            self.last_error = f"Code extraction failed: {str(e)}"
            return None, self.last_error
        
        # 2. 验证代码结构
        is_valid, error_msg = self._validate_code_structure(clean_code)
        if not is_valid:
            self.last_error = f"Code validation failed: {error_msg}"
            return None, self.last_error
        
        # 3. 准备执行环境
        # 注入必要的类型和基类
        global_scope = {
            'GeneratedStrategyBase': GeneratedStrategyBase,
            'Dict': Dict,
            'Optional': Optional,
            'Any': Any,
            '__builtins__': __builtins__,  # 允许基本内置函数
        }
        local_scope = {}
        
        # 4. 执行代码
        try:
            exec(clean_code, global_scope, local_scope)
        except SyntaxError as e:
            self.last_error = f"Syntax error at line {e.lineno}: {e.msg}"
            return None, self.last_error
        except Exception as e:
            self.last_error = f"Code execution failed: {str(e)}\n{traceback.format_exc()}"
            return None, self.last_error
        
        # 5. 检查并实例化
        if 'GeneratedAgent' not in local_scope:
            self.last_error = "GeneratedAgent class not found after execution"
            return None, self.last_error
        
        try:
            agent_class = local_scope['GeneratedAgent']
            agent_instance = agent_class()
            
            # 验证 decide 方法可调用
            if not callable(getattr(agent_instance, 'decide', None)):
                self.last_error = "GeneratedAgent.decide is not callable"
                return None, self.last_error
            
            return agent_instance, None
            
        except Exception as e:
            self.last_error = f"Agent instantiation failed: {str(e)}"
            return None, self.last_error
    
    def test_strategy(self, agent: GeneratedStrategyBase, test_cases: list = None) -> Tuple[bool, str]:
        """
        对生成的策略进行基本测试
        
        Args:
            agent: 策略实例
            test_cases: 测试用例列表，每个用例是 (observation, expected_action_type)
            
        Returns:
            (success, error_message)
        """
        if test_cases is None:
            # 默认测试用例
            test_cases = [
                # (observation, 期望返回类型为str且在有效范围内)
                {"price": 0.01, "soc": 50, "hour": 3},   # 低价低SOC，应该能处理
                {"price": 0.10, "soc": 50, "hour": 12},  # 中等价格
                {"price": 0.50, "soc": 50, "hour": 18},  # 高价
                {"price": 0.02, "soc": 95, "hour": 3},   # 低价高SOC
                {"price": 0.08, "soc": 5, "hour": 20},   # 中价低SOC
            ]
        
        valid_actions = {"CHARGE", "DISCHARGE", "HOLD"}
        
        for i, obs in enumerate(test_cases):
            try:
                action = agent.decide(obs)
                
                if not isinstance(action, str):
                    return False, f"Test case {i}: decide() returned non-string: {type(action)}"
                
                action = action.upper().strip()
                if action not in valid_actions:
                    return False, f"Test case {i}: Invalid action '{action}', must be CHARGE/DISCHARGE/HOLD"
                    
            except Exception as e:
                return False, f"Test case {i} failed: {str(e)}"
        
        return True, ""


def get_strategy_template() -> str:
    """获取策略代码模板"""
    return STRATEGY_TEMPLATE


def get_api_documentation() -> str:
    """获取 API 文档供 LLM 参考"""
    return """
## Battery Trading Strategy API Documentation

### Input: observation (Dict)
- `observation["price"]`: float - Current electricity price in $/kWh (typical range: $0.01 - $0.50)
- `observation["soc"]`: float - Current battery state of charge in % (0-100)
- `observation["hour"]`: int - Current hour of day (0-23)

### Output: Action (str)
Must return one of:
- "CHARGE" - Buy electricity to charge battery
- "DISCHARGE" - Sell electricity from battery
- "HOLD" - Do nothing

### Battery Constraints
- Capacity: 13.5 kWh
- Max charge/discharge power: 5 kW per hour
- Efficiency: 90% round-trip (~95% one-way)
- Safe SOC range: 10% - 90%

### Strategy Tips
- Charge when price is LOW and SOC is not too high
- Discharge when price is HIGH and SOC is not too low
- Consider time-of-day patterns (off-peak: night, peak: evening)
- Avoid charging above 90% SOC or discharging below 10% SOC
"""
