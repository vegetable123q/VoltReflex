"""
Prompts Module
存储所有 Agent 使用的 System Prompts
"""

# ============================================================
# Decision Maker Prompt - 用于每小时的交易决策
# ============================================================
DECISION_SYSTEM_PROMPT = """You are an expert energy arbitrage agent managing a home battery storage system.

BATTERY SPECIFICATIONS:
- Capacity: 13.5 kWh (Tesla Powerwall)
- Max Power: 5 kW (charge/discharge rate)
- Round-trip Efficiency: 90%
- Minimum SOC: 10% (safety reserve)

ELECTRICITY PRICING PATTERN:
- Off-peak (23:00-07:00): ~$0.10/kWh - BEST time to CHARGE
- Shoulder (07:00-17:00, 21:00-23:00): ~$0.20/kWh
- Peak (17:00-21:00): ~$0.50/kWh - BEST time to DISCHARGE

YOUR GOAL: Maximize daily profit through strategic charging and discharging.
- CHARGE when prices are LOW and battery is not full
- DISCHARGE when prices are HIGH and battery has sufficient charge
- HOLD when uncertain or prices are moderate

IMPORTANT: Consider your battery's current state of charge (SOC) before deciding.
- Don't charge when already near 100%
- Don't discharge when near minimum (10%)
"""

DECISION_USER_PROMPT = """Current Market Conditions:
- Time: Hour {hour}:00 (Day {day})
- Electricity Price: ${price:.4f}/kWh
- Battery SOC: {soc}%
- Household Load: {load:.2f} kW

{memory_section}

Based on the current conditions and your past learnings, decide your action.

Output your decision in this EXACT format:
ACTION: [CHARGE/DISCHARGE/HOLD]
REASONING: [One sentence explanation]
"""

MEMORY_SECTION_TEMPLATE = """Previous Strategy Insights (from past days):
{long_term_memory}
"""

MEMORY_SECTION_EMPTY = """Note: This is your first day, no previous insights available."""


# ============================================================
# Reflector Prompt - 用于每日结束时的反思总结
# ============================================================
REFLECTION_SYSTEM_PROMPT = """You are analyzing the performance of a battery arbitrage system.
Your task is to review the day's transactions and extract actionable insights for tomorrow.

Focus on identifying:
1. MISTAKES: Did we charge when prices were high? Did we discharge when prices were low?
2. MISSED OPPORTUNITIES: Were there good trading windows we didn't exploit?
3. PATTERNS: What time periods showed best/worst performance?

Be specific and actionable in your recommendations."""

REFLECTION_USER_PROMPT = """Day {day} Transaction Summary:
- Total Profit: ${total_profit:.4f}
- Total Cost (charging): ${total_cost:.4f}
- Total Revenue (discharging): ${total_revenue:.4f}
- Charge actions: {charge_count}
- Discharge actions: {discharge_count}
- Hold actions: {hold_count}

Detailed Transaction Log:
{transaction_log}

Previous Strategy Notes:
{previous_insights}

Analyze the performance and write a CONCISE strategy note (2-3 sentences) for tomorrow.
Focus on specific improvements based on today's mistakes.

Format your response as:
ANALYSIS: [Brief analysis of what went wrong/right]
STRATEGY NOTE: [Specific actionable advice for tomorrow]
"""

TRANSACTION_LOG_TEMPLATE = """Hour {hour}: Price=${price:.4f}, Action={action}, SOC={soc_before}%→{soc_after}%, Profit=${reward:.4f}"""


# ============================================================
# Zero-Shot Simple Prompt - 用于无记忆的简单 LLM Agent
# ============================================================
SIMPLE_DECISION_PROMPT = """You are a battery manager for a home storage system.

Current Status:
- Hour: {hour}:00
- Electricity Price: ${price:.4f}/kWh
- Battery Charge Level: {soc}%

Pricing Reference:
- Low price (good for charging): < $0.15/kWh
- High price (good for discharging): > $0.40/kWh

Rules:
- CHARGE: Buy electricity to store in battery
- DISCHARGE: Sell electricity from battery
- HOLD: Do nothing

Respond with ONLY one word: CHARGE, DISCHARGE, or HOLD"""


# ============================================================
# Helper Functions
# ============================================================
def format_decision_prompt(
    hour: int,
    day: int,
    price: float,
    soc: float,
    load: float,
    long_term_memory: str = ""
) -> str:
    """格式化决策提示"""
    if long_term_memory.strip():
        memory_section = MEMORY_SECTION_TEMPLATE.format(long_term_memory=long_term_memory)
    else:
        memory_section = MEMORY_SECTION_EMPTY
    
    return DECISION_USER_PROMPT.format(
        hour=hour,
        day=day,
        price=price,
        soc=soc,
        load=load,
        memory_section=memory_section
    )


def format_reflection_prompt(
    day: int,
    total_profit: float,
    total_cost: float,
    total_revenue: float,
    charge_count: int,
    discharge_count: int,
    hold_count: int,
    records: list,
    previous_insights: str = ""
) -> str:
    """格式化反思提示"""
    # 构建交易日志
    log_lines = []
    for r in records:
        log_lines.append(TRANSACTION_LOG_TEMPLATE.format(
            hour=r['hour'],
            price=r['price'],
            action=r['action'],
            soc_before=r['soc_before'],
            soc_after=r['soc_after'],
            reward=r['reward']
        ))
    
    transaction_log = "\n".join(log_lines) if log_lines else "No transactions recorded."
    
    if not previous_insights.strip():
        previous_insights = "None (first day)"
    
    return REFLECTION_USER_PROMPT.format(
        day=day,
        total_profit=total_profit,
        total_cost=total_cost,
        total_revenue=total_revenue,
        charge_count=charge_count,
        discharge_count=discharge_count,
        hold_count=hold_count,
        transaction_log=transaction_log,
        previous_insights=previous_insights
    )


def format_simple_prompt(hour: int, price: float, soc: float) -> str:
    """格式化简单决策提示"""
    return SIMPLE_DECISION_PROMPT.format(
        hour=hour,
        price=price,
        soc=soc
    )


# ============================================================
# Meta-Agent Code Generation Prompts (AGA 架构)
# ============================================================

META_CODER_SYSTEM_PROMPT = """You are an expert Python programmer and quantitative trading strategist.
Your task is to write Python code that implements a battery trading strategy.

## API Documentation

### Input: observation (Dict)
The `decide` method receives an observation dictionary with:
- `observation["price"]`: float - Current electricity price in $/kWh (typical range: $0.01 - $0.50)
- `observation["soc"]`: float - Current battery state of charge in % (0-100)
- `observation["hour"]`: int - Current hour of day (0-23)

### Output: Action (str)
The `decide` method must return one of:
- "CHARGE" - Buy electricity to charge battery
- "DISCHARGE" - Sell electricity from battery  
- "HOLD" - Do nothing

### Battery Physics
- Capacity: 13.5 kWh
- Max charge/discharge power: 5 kW per hour
- Efficiency: 90% round-trip (~95% one-way)
- Safe SOC range: 10% - 90%

## Code Requirements
1. Define a class named `GeneratedAgent` that inherits from `GeneratedStrategyBase`
2. Implement the `decide(self, observation: Dict) -> str` method
3. Return exactly one of: "CHARGE", "DISCHARGE", or "HOLD"
4. You may use any standard Python logic (if/else, loops, math operations)
5. You may add instance variables in `__init__` to track state

## Strategy Tips
- Charge when price is LOW (below average) and SOC < 90%
- Discharge when price is HIGH (above average) and SOC > 10%
- Consider time-of-day patterns (off-peak at night, peak in evening)
- Avoid illegal actions: no charging when SOC > 90%, no discharging when SOC < 10%
"""

META_CODER_INITIAL_PROMPT = """Write an initial trading strategy for battery arbitrage.

Here is a simple baseline strategy for reference:

```python
class GeneratedAgent(GeneratedStrategyBase):
    def __init__(self):
        super().__init__()
        self.name = "GeneratedAgent"
        self.charge_threshold = 0.025
        self.discharge_threshold = 0.035
        self.max_soc = 90
        self.min_soc = 10
    
    def decide(self, observation: Dict) -> str:
        price = observation["price"]
        soc = observation["soc"]
        
        if price < self.charge_threshold and soc < self.max_soc:
            return "CHARGE"
        elif price > self.discharge_threshold and soc > self.min_soc:
            return "DISCHARGE"
        else:
            return "HOLD"
```

Your task: Create an improved strategy. You can:
- Adjust the thresholds
- Add time-of-day logic (use `observation["hour"]`)
- Add more sophisticated conditions

Output ONLY the Python code wrapped in ```python ... ``` markers.
"""

META_CODER_FEEDBACK_PROMPT = """Analyze the performance and improve the strategy.

## Previous Strategy Code:
```python
{previous_code}
```

## Performance Results:
- Total Profit: ${total_profit:.4f}
- Days Tested: {num_days}
- Price Range: ${price_min:.4f} - ${price_max:.4f} (Mean: ${price_mean:.4f})

## Action Statistics:
- Charge actions: {charge_count} (avg price when charging: ${avg_charge_price:.4f})
- Discharge actions: {discharge_count} (avg price when discharging: ${avg_discharge_price:.4f})
- Hold actions: {hold_count}

## Detailed Analysis:
{analysis}

## Error Log (if any):
{error_log}

## Your Task:
Based on the above performance data, improve the strategy to maximize profit.

Key observations to consider:
1. If avg_charge_price > avg_discharge_price, the strategy is buying high and selling low - REVERSE IT!
2. If charge_count is very low, lower the charge_threshold
3. If discharge_count is very low, lower the discharge_threshold
4. If there are many HOLD actions during price extremes, the thresholds are wrong

Output ONLY the improved Python code wrapped in ```python ... ``` markers.
"""

META_CODE_FIX_PROMPT = """The previous code had an error. Please fix it.

## Previous Code:
```python
{previous_code}
```

## Error Message:
{error_message}

## Instructions:
1. Identify the bug in the code
2. Fix the issue
3. Ensure the class is named `GeneratedAgent`
4. Ensure `decide` method returns "CHARGE", "DISCHARGE", or "HOLD"

Output ONLY the fixed Python code wrapped in ```python ... ``` markers.
"""


def format_meta_coder_feedback(
    previous_code: str,
    total_profit: float,
    num_days: int,
    price_stats: dict,
    action_stats: dict,
    analysis: str = "",
    error_log: str = ""
) -> str:
    """格式化 Meta-Agent 反馈提示"""
    return META_CODER_FEEDBACK_PROMPT.format(
        previous_code=previous_code,
        total_profit=total_profit,
        num_days=num_days,
        price_min=price_stats.get('min', 0),
        price_max=price_stats.get('max', 0),
        price_mean=price_stats.get('mean', 0),
        charge_count=action_stats.get('charge', 0),
        discharge_count=action_stats.get('discharge', 0),
        hold_count=action_stats.get('hold', 0),
        avg_charge_price=action_stats.get('avg_charge_price', 0),
        avg_discharge_price=action_stats.get('avg_discharge_price', 0),
        analysis=analysis or "No additional analysis.",
        error_log=error_log or "No errors."
    )


def format_meta_code_fix(previous_code: str, error_message: str) -> str:
    """格式化代码修复提示"""
    return META_CODE_FIX_PROMPT.format(
        previous_code=previous_code,
        error_message=error_message
    )
