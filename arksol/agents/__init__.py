"""四个角色。

| 角色 | 线程 | 职责 |
|---|---|---|
| Intake | 单线程 | 把客户原话变成结构化需求；信息不足时先追问，不出方案 |
| Evidence | **并行 · 只读** | 每个实例负责一摊取证，只贡献信息、不产生动作 |
| Writer | **单线程** | 全系统唯一允许「写入」的环节，每条断言挂证据 |
| Critic | 独立 | 规则层 + 模型层，不合格退回重写（最多 2 轮） |
"""

from arksol.agents.base import Agent
from arksol.agents.critic import CriticAgent
from arksol.agents.evidence import EvidenceAgent
from arksol.agents.intake import IntakeAgent
from arksol.agents.writer import MemoryPlanAgent, WriterAgent

__all__ = ["Agent", "CriticAgent", "EvidenceAgent", "IntakeAgent", "MemoryPlanAgent", "WriterAgent"]
