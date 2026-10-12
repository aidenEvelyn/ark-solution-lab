"""统一异常类型。

把「可以预期的失败」都做成明确的类型，而不是让它们变成一段普通的报错文本 ——
因为编排器需要**按类型**决定动作：契约不合法是本地重试，预算超了是直接停。
"""

from __future__ import annotations


class ArksolError(Exception):
    """本项目所有自定义异常的基类。"""


class ContractError(ArksolError):
    """交接契约校验失败：上一环的产出不符合下一环要求的形状。"""

    def __init__(self, schema_name: str, path: str, message: str) -> None:
        self.schema_name = schema_name
        self.path = path
        self.message = message
        super().__init__(f"[{schema_name}] {path}: {message}")


class BudgetExceeded(ArksolError):
    """预算熔断：token / 费用 / 步数 任一超限。"""

    def __init__(self, kind: str, used: float, limit: float) -> None:
        self.kind = kind
        self.used = used
        self.limit = limit
        super().__init__(f"预算熔断（{kind}）：已用 {used:g}，上限 {limit:g}")


class RepeatActionDetected(ArksolError):
    """同一个动作（工具+参数指纹）重复太多次 —— 典型的「沉默失败」。

    这类失败在日志里没有 error，轨迹看起来还在活动，实际上只是在重复同一个无效动作。
    """

    def __init__(self, fingerprint: str, count: int) -> None:
        self.fingerprint = fingerprint
        self.count = count
        super().__init__(f"重复动作熔断：{fingerprint} 已出现 {count} 次")


class ProviderError(ArksolError):
    """模型提供方返回错误（网络、鉴权、限流、模型不存在等）。"""


class KBError(ArksolError):
    """知识库加载或完整性校验失败。"""
