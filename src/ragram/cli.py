"""RagRam command line interface."""

from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .app import build_start_plan, index_stored_messages, ingest_text_messages
from .config import ChannelConfig, default_config, ensure_app_layout, load_config, save_config, scaffold_status
from .embeddings import EMBEDDING_MODEL_CHOICES, SentenceTransformerEmbeddingProvider
from .llm import ANSWER_MODEL_CHOICES, OllamaClient, pull_commands_for_missing_models
from .models import ChannelRecord, IndexScope
from .progress import RichProgressReporter
from .storage import SQLiteStore
from .telegram_client import (
    TelegramLoginCodeInvalid,
    TelegramLoginCodeResendRequested,
    TelegramDialog,
    create_telegram_client,
    ensure_telegram_login,
    filter_dialogs,
    list_accessible_dialogs,
    resolve_dialog_choice,
)
from .ui import launch_streamlit_ui
from .vector_store import ChromaVectorStore

console = Console()
app = typer.Typer(
    name="ragram",
    help="Local-first Telegram channel RAG powered by local models.",
    no_args_is_help=True,
    invoke_without_command=True,
)


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _load_inquirer() -> Any:
    try:
        from InquirerPy import inquirer
    except ImportError as exc:  # pragma: no cover - dependency is part of base install.
        raise RuntimeError("InquirerPy is required for interactive setup; reinstall RagRam.") from exc
    return inquirer


def _print_telegram_credentials_help() -> None:
    """Explain where users get Telegram MTProto app credentials."""

    console.print("Telegram MTProto access needs api_id, api_hash, and phone number.")
    console.print("Get api_id/api_hash here: https://my.telegram.org/apps")
    console.print(
        "Log in with your Telegram phone number, create an app if needed, then copy "
        "App api_id and App api_hash. Telegram sends the confirmation code in Telegram, not SMS."
    )
    console.print("RagRam stores these values only in your local ~/.ragram/config.toml.")


async def _prompt_value(prompt: Any) -> Any:
    """Execute an InquirerPy prompt without nesting asyncio.run().

    InquirerPy's synchronous ``execute()`` starts its own event loop. RagRam's
    interactive Telegram flow already runs inside ``asyncio.run()``, so real
    terminal prompts must use ``execute_async()`` when available. Tests keep
    simple fake prompts with only ``execute()``.
    """

    execute_async = getattr(prompt, "execute_async", None)
    if execute_async is not None:
        return await execute_async()
    return prompt.execute()


def _dialog_label(dialog: TelegramDialog) -> str:
    username = f"@{dialog.username}" if dialog.username else "no username"
    date = dialog.last_message_date.isoformat() if dialog.last_message_date else "unknown date"
    unread = "?" if dialog.unread_count is None else str(dialog.unread_count)
    return f"{dialog.title} — {username} — {dialog.kind} — {date} — unread {unread}"


def _filter_dialog_choices(dialogs: list[TelegramDialog], query: str) -> list[TelegramDialog]:
    """Return dialogs matching the optional interactive search text."""

    return filter_dialogs(dialogs, query) if query.strip() else dialogs


def _scope_from_config(scope_type: str, scope_value: str | None) -> IndexScope:
    if scope_type == "all":
        return IndexScope.all()
    if scope_type == "from_year":
        return IndexScope.from_year(int(scope_value or "2024"))
    if scope_type == "from_date":
        return IndexScope.from_date(scope_value or "2024-01-01")
    return IndexScope.last_n(int(scope_value or "1000"))


def _telegram_config_complete(config: Any) -> bool:
    return config.telegram.api_id is not None and bool(config.telegram.api_hash) and bool(config.telegram.phone)


def _session_path_candidates(paths: Any, session_name: str) -> tuple[Any, Any]:
    session_path = paths.sessions_dir / session_name
    return session_path, session_path.with_suffix(".session")


def _has_telegram_session(paths: Any, session_name: str) -> bool:
    return any(path.exists() for path in _session_path_candidates(paths, session_name))


def _reset_telegram_credentials(config: Any, paths: Any) -> None:
    for session_path in _session_path_candidates(paths, config.telegram.session_name):
        if session_path.exists():
            session_path.unlink()
    config.telegram.api_id = None
    config.telegram.api_hash = None
    config.telegram.phone = None


async def _disconnect_if_supported(client: Any) -> None:
    disconnect = getattr(client, "disconnect", None)
    if disconnect is None:
        return
    maybe_awaitable = disconnect()
    if hasattr(maybe_awaitable, "__await__"):
        await maybe_awaitable


async def _run_interactive_start(config, paths) -> None:
    """Run guided local setup, Telegram ingestion, Chroma indexing, and model checks."""

    inquirer = _load_inquirer()
    store = SQLiteStore(paths.sqlite_path)
    store.initialize()

    if _telegram_config_complete(config) and (
        config.channel.entity_id is None or not _has_telegram_session(paths, config.telegram.session_name)
    ):
        console.print(f"Saved Telegram phone/config found: {config.telegram.phone}")
        use_saved = await _prompt_value(
            inquirer.confirm(
                message="Use saved Telegram credentials? Choose No to re-enter api_id/api_hash/phone.",
                default=True,
            )
        )
        if not use_saved:
            _reset_telegram_credentials(config, paths)
            save_config(config, paths.config_path)

    if not _telegram_config_complete(config):
        _print_telegram_credentials_help()
        api_id = await _prompt_value(inquirer.text(message="Telegram api_id:"))
        api_hash = await _prompt_value(inquirer.secret(message="Telegram api_hash:"))
        phone = await _prompt_value(inquirer.text(message="Telegram phone number (international format, e.g. +15551234567):"))
        config.telegram.api_id = int(str(api_id).strip())
        config.telegram.api_hash = str(api_hash).strip()
        config.telegram.phone = str(phone).strip()
        save_config(config, paths.config_path)

    client = create_telegram_client(config.telegram, paths)
    selected: TelegramDialog | None = None
    try:
        async def ask_login_code() -> str:
            console.print(
                "Telegram sent a login code to your Telegram app/session for this phone number "
                "(usually the official 'Telegram' chat or a login notification), not to this terminal."
            )
            console.print("If it does not arrive, type 'r' to resend or 'q' to quit and re-check the phone number.")
            value = str(
                await _prompt_value(
                    inquirer.text(
                        message="Telegram login code (or r=resend, q=quit):",
                    )
                )
            ).strip()
            if value.casefold() in {"r", "resend"}:
                console.print("Requesting a new Telegram login code...")
                raise TelegramLoginCodeResendRequested()
            if value.casefold() in {"q", "quit", "exit"}:
                raise typer.Exit(1)
            return value

        async def ask_2fa_password() -> str:
            return str(await _prompt_value(inquirer.secret(message="Telegram 2FA password:")))

        try:
            login = await ensure_telegram_login(
                client,
                phone=config.telegram.phone or "",
                code_callback=ask_login_code,
                password_callback=ask_2fa_password,
            )
        except TelegramLoginCodeInvalid as exc:
            console.print(f"[red]{exc}[/red]")
            console.print("Tip: use the newest code from Telegram. If the phone/api_id/api_hash is wrong, run ragram restart --reconfigure or choose No when asked to use saved credentials.")
            raise typer.Exit(1) from exc
        console.print("Telegram session reused." if login.reused_session else "Telegram session saved locally.")

        dialogs = await list_accessible_dialogs(client, limit=200)
        if not dialogs:
            raise typer.BadParameter("No accessible Telegram channels/groups were found for this account.")

        query = await _prompt_value(inquirer.text(message="Filter channels/groups by title (optional):", default=""))
        visible_dialogs = _filter_dialog_choices(dialogs, str(query))
        if not visible_dialogs:
            console.print("No matching dialogs; showing the recent list instead.")
            visible_dialogs = dialogs
        choices = [{"name": _dialog_label(dialog), "value": dialog} for dialog in visible_dialogs[:100]]
        choices.append({"name": "Custom input (username, URL, exact title, or entity id)", "value": "__custom__"})
        selection = await _prompt_value(inquirer.select(message="Choose one Telegram channel/group:", choices=choices))
        if selection == "__custom__":
            custom = await _prompt_value(inquirer.text(message="Channel username, URL, exact title, or entity id:"))
            selected = await resolve_dialog_choice(client, str(custom), dialogs)
        else:
            selected = selection

        assert selected is not None
        store.upsert_channel(
            ChannelRecord(
                entity_id=selected.entity_id,
                title=selected.title,
                username=selected.username,
                kind=selected.kind,
                last_message_date=selected.last_message_date,
                unread_count=selected.unread_count,
                raw={"label": _dialog_label(selected)},
            )
        )
        config.channel = ChannelConfig(entity_id=selected.entity_id, title=selected.title, username=selected.username)

        scope_choice = await _prompt_value(inquirer.select(
            message="Indexing scope:",
            choices=[
                {"name": "Last N messages (default 1000, safe/fast)", "value": "last_n"},
                {"name": "All messages", "value": "all"},
                {"name": "From year", "value": "from_year"},
                {"name": "From exact date (YYYY-MM-DD)", "value": "from_date"},
            ],
            default="last_n",
        ))
        if scope_choice == "last_n":
            scope_value = str(await _prompt_value(inquirer.text(message="How many recent messages?", default="1000")))
        elif scope_choice == "from_year":
            scope_value = str(await _prompt_value(inquirer.text(message="Start year:", default="2024")))
        elif scope_choice == "from_date":
            scope_value = str(await _prompt_value(inquirer.text(message="Start date (YYYY-MM-DD):", default="2024-01-01")))
        else:
            scope_value = None

        config.indexing.scope_type = str(scope_choice)
        config.indexing.scope_value = "" if scope_value is None else scope_value
        config.indexing.embedding_model = await _prompt_value(inquirer.select(
            message="Embedding model:",
            choices=list(EMBEDDING_MODEL_CHOICES),
            default=config.indexing.embedding_model,
        ))
        config.indexing.answer_model = await _prompt_value(inquirer.select(
            message="Answer model (Ollama):",
            choices=list(ANSWER_MODEL_CHOICES),
            default=config.indexing.answer_model,
        ))
        config.indexing.summarization_model = await _prompt_value(inquirer.select(
            message="Summarization model (Ollama):",
            choices=list(ANSWER_MODEL_CHOICES),
            default=config.indexing.summarization_model or config.indexing.answer_model,
        ))
        save_config(config, paths.config_path)

        ollama_status = OllamaClient().model_status([config.indexing.answer_model, config.indexing.summarization_model])
        if not ollama_status.running:
            console.print("Ollama is not running. Start it locally, then pull the selected models if needed:")
        if ollama_status.missing_models:
            for command in pull_commands_for_missing_models(ollama_status.missing_models):
                console.print(f"  {command}")

        scope = _scope_from_config(config.indexing.scope_type, config.indexing.scope_value)
        console.print("Fetching Telegram messages...")
        await ingest_text_messages(
            client,
            store,
            entity=selected.raw,
            entity_id=selected.entity_id,
            channel_username=selected.username,
            scope=scope,
            run_id=str(uuid.uuid4()),
            embedding_model=config.indexing.embedding_model,
            progress=RichProgressReporter("Fetching messages"),
        )

        console.print("Chunking, embedding, and writing to Chroma...")
        vector_store = ChromaVectorStore(
            embedding_provider=SentenceTransformerEmbeddingProvider(config.indexing.embedding_model),
            persist_directory=paths.chroma_dir,
        )
        result = index_stored_messages(
            store,
            vector_store=vector_store,
            entity_id=selected.entity_id,
            embedding_model=config.indexing.embedding_model,
            progress=RichProgressReporter("Embedding chunks"),
        )
        console.print(f"Indexed {result.chunks_indexed} chunks from {result.raw_messages} raw messages.")
    finally:
        await _disconnect_if_supported(client)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show RagRam version and exit."),
) -> None:
    """RagRam CLI root."""

    if version:
        console.print(f"RagRam {__version__}")
        raise typer.Exit()


@app.command()
def status() -> None:
    """Show local status without contacting external paid services."""

    current = scaffold_status()
    paths = ensure_app_layout(current.app_home)
    config = load_config(paths.config_path)
    store = SQLiteStore(paths.sqlite_path)
    store.initialize()
    entity_id = config.channel.entity_id
    raw_messages = store.message_count(entity_id) if entity_id is not None else store.message_count()
    chunks = store.list_chunks(entity_id=entity_id, embedding_model=config.indexing.embedding_model) if entity_id is not None else []
    session_base = paths.sessions_dir / config.telegram.session_name
    has_session = session_base.exists() or session_base.with_suffix(".session").exists()

    table = Table(title="RagRam status")
    table.add_column("Item", style="bold")
    table.add_column("Value")
    table.add_row("App home", str(paths.home))
    table.add_row("Config", "present" if paths.config_path.exists() else "missing")
    table.add_row("SQLite", f"{paths.sqlite_path} ({'present' if paths.sqlite_path.exists() else 'missing'})")
    table.add_row("Telegram session", "present" if has_session else "missing")
    table.add_row("Selected channel", config.channel.title or (str(entity_id) if entity_id is not None else "not configured"))
    table.add_row("Raw messages", str(raw_messages))
    table.add_row("Indexed chunks", str(len(chunks)))
    table.add_row("Embedding model", config.indexing.embedding_model)
    table.add_row("Answer model", config.indexing.answer_model)
    required_models = tuple(dict.fromkeys([config.indexing.answer_model, config.indexing.summarization_model]))
    ollama_status = OllamaClient().model_status(required_models)
    if ollama_status.running and ollama_status.missing_models:
        ollama_value = "running; missing " + ", ".join(ollama_status.missing_models)
    elif ollama_status.running:
        ollama_value = "running; selected models available"
    else:
        ollama_value = "not running; run " + "; ".join(pull_commands_for_missing_models(required_models))

    table.add_row("Summarization model", config.indexing.summarization_model)
    table.add_row("Ollama", ollama_value)
    table.add_row("Missing local models", ", ".join(ollama_status.missing_models) if ollama_status.missing_models else "none")
    table.add_row("UI URL", f"http://localhost:{config.ui.port}")
    table.add_row("No paid APIs", "yes")
    table.add_row("External services contacted", "no")
    console.print(table)


@app.command()
def start(
    no_ui: bool = typer.Option(False, "--no-ui", help="Run setup/indexing without launching the local UI."),
    ui_port: int = typer.Option(8501, "--ui-port", min=1, max=65535, help="Local UI port."),
) -> None:
    """Start RagRam's guided local app flow."""

    plan = build_start_plan(no_ui=no_ui, ui_port=ui_port)
    config = load_config(plan.paths.config_path)
    console.print("[bold]RagRam start[/bold]")
    console.print("Local product flow is ready; no paid APIs are configured or contacted.")
    console.print(f"App home: {plan.status.app_home}")

    if config.telegram.api_id is None or not config.telegram.api_hash or not config.telegram.phone:
        if not _is_interactive():
            _print_telegram_credentials_help()
            console.print("Run this command in an interactive terminal to enter them securely; api_hash/password prompts are hidden.")

    if _is_interactive():
        asyncio.run(_run_interactive_start(config, plan.paths))
        config = load_config(plan.paths.config_path)

    if config.channel.entity_id is not None:
        console.print(f"Selected channel: {config.channel.title or config.channel.entity_id}")
    else:
        console.print("Selected channel: not configured yet")

    if plan.no_ui:
        console.print("UI: disabled (--no-ui)")
        return

    if config.channel.entity_id is None:
        console.print("UI not started: choose and index a Telegram channel first by running ragram start in an interactive terminal.")
        return

    launch_plan = launch_streamlit_ui(paths=plan.paths, port=plan.ui_port)
    console.print(f"Open RagRam: {launch_plan.url}")


@app.command()
def restart(
    clear_data: bool = typer.Option(False, "--clear-data", help="Clear local raw SQLite data and Chroma index after confirmation."),
    reconfigure: bool = typer.Option(False, "--reconfigure", help="Guided local reconfiguration of session/channel/scope/model choices."),
) -> None:
    """Restart/reconfigure RagRam with confirmation before destructive resets."""

    paths = ensure_app_layout()
    if not paths.config_path.exists():
        save_config(default_config(), paths.config_path)
    config = load_config(paths.config_path)

    if clear_data:
        confirmed = typer.confirm("Clear local raw data and Chroma index? This cannot be undone.")
        if not confirmed:
            console.print("Restart cancelled; no local data was changed.")
            return
        if paths.sqlite_path.exists():
            paths.sqlite_path.unlink()
        if paths.chroma_dir.exists():
            shutil.rmtree(paths.chroma_dir)
        paths.chroma_dir.mkdir(parents=True, exist_ok=True)
        SQLiteStore(paths.sqlite_path).initialize()
        console.print("Local raw data and Chroma index cleared.")

    if reconfigure:
        if typer.confirm("Reset Telegram account/session config?", default=False):
            config.telegram.api_id = None
            config.telegram.api_hash = None
            config.telegram.phone = None
            for session_path in (paths.sessions_dir / config.telegram.session_name, (paths.sessions_dir / config.telegram.session_name).with_suffix(".session")):
                if session_path.exists():
                    session_path.unlink()
            console.print("Telegram account/session config reset.")
        if typer.confirm("Reconfigure local model choices?", default=False):
            config.indexing.embedding_model = typer.prompt("Embedding model", default=config.indexing.embedding_model)
            config.indexing.answer_model = typer.prompt("Answer model", default=config.indexing.answer_model)
            config.indexing.summarization_model = typer.prompt("Summarization model", default=config.indexing.summarization_model or config.indexing.answer_model)
            console.print("Reconfigured local model choices.")
        if typer.confirm("Reset selected channel/group so start can choose a new one?", default=False):
            config.channel = ChannelConfig()
            console.print("Selected channel/group reset.")
        if typer.confirm("Reconfigure indexing scope?", default=False):
            scope_type = typer.prompt("Indexing scope (last_n/all/from_year/from_date)", default=config.indexing.scope_type)
            if scope_type == "all":
                scope_value = ""
            elif scope_type == "from_year":
                scope_value = typer.prompt("Start year", default=config.indexing.scope_value if config.indexing.scope_type == "from_year" else "2024")
            elif scope_type == "from_date":
                scope_value = typer.prompt("Start date (YYYY-MM-DD)", default=config.indexing.scope_value if config.indexing.scope_type == "from_date" else "2024-01-01")
            else:
                scope_type = "last_n"
                scope_value = typer.prompt("How many recent messages?", default=config.indexing.scope_value or "1000")
            config.indexing.scope_type = scope_type
            config.indexing.scope_value = scope_value
            console.print("Reconfigured indexing scope.")
        save_config(config, paths.config_path)

    console.print("RagRam restart")
    console.print(
        "You can reconfigure Telegram account/session, selected channel, indexing scope, "
        "embedding model, answer model, and summarization model on the next interactive start."
    )
