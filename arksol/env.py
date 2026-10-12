"""极简 .env 读取（刻意不引入 python-dotenv）。

为什么需要它：让密钥有一个**不会进版本库**的落脚点。
`.env` 已经在 `.gitignore` 里；把 key 写在代码里、写在命令行里、或者粘在聊天记录里，都比它差。

支持的范围（够用即可）：`KEY=VALUE`、`#` 注释、可选引号、`export ` 前缀、等号两边空格。
**只读返回值，不改 os.environ** —— 库不该有这种副作用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

ENV_FILENAME = ".env"


def parse_dotenv(text: str) -> Dict[str, str]:
    """把 .env 的正文解析成字典。解析不了的行走忽略，不报错。"""
    values: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def find_dotenv(start: Optional[Path | str] = None) -> Optional[Path]:
    """从 start（默认当前目录）往上找 .env —— 在子目录里跑命令也能找到。"""
    base = Path(start) if start is not None else Path.cwd()
    for directory in (base, *base.parents):
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def read_dotenv(path: Optional[Path | str] = None) -> Dict[str, str]:
    """读 .env。找不到就返回空字典（不报错 —— 没配 .env 是完全正常的情况）。"""
    dotenv_path = Path(path) if path is not None else find_dotenv()
    if dotenv_path is None or not dotenv_path.is_file():
        return {}
    try:
        return parse_dotenv(dotenv_path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def effective_environ(dotenv_path: Optional[Path | str] = None) -> Dict[str, str]:
    """合并后的配置来源：`.env` 打底，**真实环境变量优先**。

    真实环境变量优先是有意的：部署环境（容器、CI）里设的值，不该被仓库里一个忘记删的
    .env 文件盖掉。
    """
    merged = dict(read_dotenv(dotenv_path))
    merged.update(os.environ)
    return merged
