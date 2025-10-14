#!/usr/bin/env python3
"""
Telegram bot entrypoint for controlling the trading bot lifecycle and
fetching performance snapshots.

Environment variables:
  TELEGRAM_BOT_TOKEN            Telegram bot token (required)
  TELEGRAM_ALLOWED_CHAT_IDS     Comma-separated chat IDs allowed to interact (optional)
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "run_scripts"
ANALYTICS_FILE = PROJECT_ROOT / "analytics" / "performance_snapshot.json"

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8205925913:AAFHam14nD2AimwH94oi7ae9U-DjJIPzHKk")
ALLOWED_CHAT_IDS = {
    int(cid)
    for cid in filter(None, (part.strip() for part in os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",")))
}


async def send_text(update: Update, text: str, **kwargs) -> None:
    if update.message:
        await update.message.reply_text(text, **kwargs)
    elif update.effective_chat:
        await update.effective_chat.send_message(text, **kwargs)


class ScriptError(RuntimeError):
    """Raised when a managed script exits with a non-zero return code."""


def ensure_token() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the Telegram bot.")


def is_authorized(chat_id: int) -> bool:
    return not ALLOWED_CHAT_IDS or chat_id in ALLOWED_CHAT_IDS


async def guard_authorization(update: Update) -> bool:
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None or is_authorized(chat_id):
        return True
    await send_text(update, "🚫 이 챗은 허용되지 않았습니다.")
    return False


def run_shell_script(script_name: str, extra_args: Iterable[str] | None = None) -> str:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"스크립트를 찾을 수 없습니다: {script_path}")

    cmd: List[str] = ["bash", str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    message_parts = [
        f"$ {' '.join(shlex.quote(part) for part in cmd)}",
        stdout or "(stdout 비어있음)",
    ]

    if stderr:
        message_parts.append(f"(stderr)\n{stderr}")

    if completed.returncode != 0:
        raise ScriptError("\n\n".join(message_parts))

    return "\n\n".join(message_parts)


def format_performance_snapshot() -> str:
    if not ANALYTICS_FILE.exists():
        raise FileNotFoundError(f"파일이 없습니다: {ANALYTICS_FILE}")

    data = json.loads(ANALYTICS_FILE.read_text())
    total = data.get("total", {})
    window = data.get("window", {})

    def pack(section_name: str, section: dict) -> str:
        if not section:
            return f"*{section_name}*\n데이터가 없습니다."
        return (
            f"*{section_name}*\n"
            f"- 거래 수: {section.get('trades', 'N/A')} (승 {section.get('wins', 'N/A')}, 패 {section.get('losses', 'N/A')}, 무 {section.get('ties', 0)})\n"
            f"- 승률: {section.get('win_rate', 0):.2%}\n"
            f"- 순손익: {section.get('net_pnl', 0):.4f}\n"
            f"- 손익비: {section.get('profit_factor', 0):.4f}\n"
            f"- 평균 수익률: {section.get('avg_return_pct', 0):.4%}"
        )

    segments = [
        "*거래 성과 요약*",
        pack("전체", total),
        pack("최근 구간", window),
    ]
    return "\n\n".join(segments)


async def handle_script(update: Update, context: ContextTypes.DEFAULT_TYPE, script_name: str) -> None:
    if not await guard_authorization(update):
        return

    try:
        result = await asyncio.to_thread(run_shell_script, script_name, context.args)
        await send_text(update, result)
    except FileNotFoundError as exc:
        await send_text(update, f"❌ {exc}")
    except ScriptError as exc:
        await send_text(update, f"⚠️ 스크립트 실행 실패:\n{exc}")
    except Exception as exc:  # pragma: no cover - defensive
        await send_text(update, f"❌ 알 수 없는 오류: {exc}")


async def handle_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_script(update, context, "run_bot.sh")


async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_script(update, context, "stop_bot.sh")


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_script(update, context, "status_bot.sh")


async def handle_restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_script(update, context, "restart_bot.sh")


async def handle_performance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await guard_authorization(update):
        return

    try:
        summary = await asyncio.to_thread(format_performance_snapshot)
        await send_text(update, summary, parse_mode=ParseMode.MARKDOWN)
    except FileNotFoundError as exc:
        await send_text(update, f"❌ {exc}")
    except json.JSONDecodeError:
        await send_text(update, "❌ 성과 파일 JSON 파싱에 실패했습니다.")
    except Exception as exc:  # pragma: no cover - defensive
        await send_text(update, f"❌ 알 수 없는 오류: {exc}")


def build_application() -> Application:
    ensure_token()
    return ApplicationBuilder().token(BOT_TOKEN).build()


def main() -> None:
    ensure_token()
    app = build_application()
    app.add_handler(CommandHandler("run", handle_run))
    app.add_handler(CommandHandler("stop", handle_stop))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CommandHandler("restart", handle_restart))
    app.add_handler(CommandHandler(["performance", "stats"], handle_performance))

    print("✅ Telegram bot 초기화 완료.")
    print("🤖 Telegram bot이 시작되었습니다. 종료하려면 Ctrl+C 입력.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
