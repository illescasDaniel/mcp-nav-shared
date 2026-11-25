from __future__ import annotations

import asyncio
from pathlib import Path

from mcp_nav_shared.lsp_client import LspClient


def test_sync_and_diagnostics_hold_reader_lease_during_open_and_diagnostics(tmp_path: Path, monkeypatch):
	client = LspClient(workspace_root=tmp_path, command=[], language_id="python")

	async def run():
		opened = asyncio.Event()
		release = asyncio.Event()
		preview_entered = asyncio.Event()
		calls = []

		async def ensure(path):
			calls.append(("open", path, client._readers))
			opened.set()
			await release.wait()
			return path

		async def diagnostic(path):
			calls.append(("diagnostics", path, client._readers))
			return []

		async def preview():
			async with client.overlay({}):
				preview_entered.set()

		monkeypatch.setattr(client, "ensure_open", ensure)
		monkeypatch.setattr(client, "diagnostics", diagnostic)
		task = asyncio.create_task(client.diagnostics_with_open_files(["importer.py"], open_first=["new.py"]))
		await opened.wait()
		preview_task = asyncio.create_task(preview())
		await asyncio.sleep(0)
		assert not preview_entered.is_set()
		release.set()
		assert await task == {"importer.py": []}
		await preview_task
		assert preview_entered.is_set()
		assert calls == [("open", "new.py", 1), ("diagnostics", "importer.py", 1)]

	asyncio.run(run())


def test_real_diagnostics_is_reentrant_and_blocks_overlay(tmp_path, monkeypatch):
	client = LspClient(workspace_root=tmp_path, command=[], language_id="python")
	(tmp_path / "new.py").write_text("VALUE = 1\n")
	(tmp_path / "user.py").write_text("from new import VALUE\n")

	async def run():
		entered = asyncio.Event()
		release = asyncio.Event()
		preview_entered = asyncio.Event()
		notifications = []

		def notify(method, params):
			notifications.append((method, params))

		async def request(method, params):
			assert method == "textDocument/diagnostic"
			assert client._readers == 1
			entered.set()
			await release.wait()
			return {"result": {"kind": "full", "items": []}}

		async def preview():
			async with client.overlay({}):
				preview_entered.set()

		monkeypatch.setattr(client, "_notify", notify)
		monkeypatch.setattr(client, "_request", request)
		task = asyncio.create_task(client.diagnostics_with_open_files(["user.py"], open_first=["new.py"]))
		await asyncio.wait_for(entered.wait(), timeout=1)
		preview_task = asyncio.create_task(preview())
		await asyncio.sleep(0)
		assert not preview_entered.is_set()
		release.set()
		assert await asyncio.wait_for(task, timeout=1) == {"user.py": []}
		await asyncio.wait_for(preview_task, timeout=1)
		assert preview_entered.is_set()
		opened = [data["textDocument"]["uri"] for method, data in notifications if method == "textDocument/didOpen"]
		assert opened == [(tmp_path / "new.py").as_uri(), (tmp_path / "user.py").as_uri()]
		assert client._readers == 0

	asyncio.run(run())


def test_failed_prerequisite_or_diagnostic_aborts_and_releases_lease(tmp_path, monkeypatch):
	import pytest

	client = LspClient(workspace_root=tmp_path, command=[], language_id="python")
	(tmp_path / "new.py").write_text("VALUE = 1\n")
	(tmp_path / "user.py").write_text("from new import VALUE\n")
	requests = []
	monkeypatch.setattr(client, "_notify", lambda *_: None)

	async def request(method, params):
		requests.append(method)
		raise OSError("cannot obtain diagnostics")

	monkeypatch.setattr(client, "_request", request)

	async def run():
		with pytest.raises(FileNotFoundError):
			await client.diagnostics_with_open_files(["user.py"], open_first=["missing.py"])
		assert requests == []
		assert client._readers == 0
		with pytest.raises(OSError, match="cannot obtain diagnostics"):
			await client.diagnostics_with_open_files(["user.py"], open_first=["new.py"])
		assert requests == ["textDocument/diagnostic"]
		assert client._readers == 0
		async with client.overlay({}):
			pass

	asyncio.run(run())
