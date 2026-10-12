"""配置文件读取测试。

这一层保护的是**密钥不会因为图方便而写进代码**：
给它一个不进版本库的落脚点（`.env`），并且保证真实环境变量优先。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arksol.env import effective_environ, find_dotenv, parse_dotenv, read_dotenv
from arksol.errors import ProviderError
from arksol.llm import OpenAICompatProvider


class TestParseDotenv(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(parse_dotenv("ARK_API_KEY=abc123"), {"ARK_API_KEY": "abc123"})

    def test_ignores_comments_and_blank_lines(self) -> None:
        text = "# 注释\n\nARK_MODEL=x\n   \n# 又一行注释\n"
        self.assertEqual(parse_dotenv(text), {"ARK_MODEL": "x"})

    def test_strips_quotes(self) -> None:
        text = "A='单引号值'\nB=\"双引号值\"\n"
        self.assertEqual(parse_dotenv(text), {"A": "单引号值", "B": "双引号值"})

    def test_strips_spaces_around_equals(self) -> None:
        self.assertEqual(parse_dotenv("  KEY  =  value  "), {"KEY": "value"})

    def test_supports_export_prefix(self) -> None:
        self.assertEqual(parse_dotenv("export KEY=value"), {"KEY": "value"})

    def test_skips_lines_without_equals(self) -> None:
        self.assertEqual(parse_dotenv("这不是一行配置\nKEY=1"), {"KEY": "1"})

    def test_value_may_contain_equals(self) -> None:
        self.assertEqual(parse_dotenv("URL=https://x.test/a?b=c"), {"URL": "https://x.test/a?b=c"})

    def test_crlf_input(self) -> None:
        self.assertEqual(parse_dotenv("A=1\r\nB=2\r\n"), {"A": "1", "B": "2"})


class TestReadDotenv(unittest.TestCase):
    def test_reads_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("ARK_API_KEY=from-file\n", encoding="utf-8")
            self.assertEqual(read_dotenv(path), {"ARK_API_KEY": "from-file"})

    def test_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_dotenv(Path(tmp) / "不存在"), {})

    def test_find_dotenv_walks_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("A=1\n", encoding="utf-8")
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            self.assertEqual(find_dotenv(nested), root / ".env")

    def test_find_dotenv_returns_none_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_dotenv(Path(tmp)))


class TestEffectiveEnviron(unittest.TestCase):
    def test_real_env_wins_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("ARK_MODEL=from-dotenv\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"ARK_MODEL": "from-real-env"}, clear=True):
                merged = effective_environ(path)
            self.assertEqual(merged["ARK_MODEL"], "from-real-env")

    def test_dotenv_fills_the_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("ARK_MODEL=from-dotenv\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                merged = effective_environ(path)
            self.assertEqual(merged["ARK_MODEL"], "from-dotenv")


class TestProviderReadsDotenv(unittest.TestCase):
    def test_key_can_come_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "ARK_API_KEY=key-from-file\nARK_MODEL=doubao-from-file\nARG_IGNORED=1\n", encoding="utf-8"
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                provider = OpenAICompatProvider.from_env(dotenv_path=path)
        self.assertEqual(provider.model, "doubao-from-file")

    def test_explicit_env_is_used_verbatim(self) -> None:
        provider = OpenAICompatProvider.from_env(env={"ARK_API_KEY": "k", "ARK_MODEL": "m"})
        self.assertEqual(provider.model, "m")

    def test_no_key_anywhere_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ProviderError) as ctx:
                    OpenAICompatProvider.from_env(dotenv_path=Path(tmp) / "不存在的.env")
        self.assertIn("API Key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
