"""QA block B23 (shared): preview cleanup never deletes user content; previews never create files.

Covers the overlay cleanup hardening (atomic steal-and-decide) and the manifest sweep's
warn-only behaviour for the preview-creation path (RC-091, RC-029's cleanup half).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mcp_nav_shared import lsp_client
from mcp_nav_shared.lsp_client import LspClient


# A minimal language server: answers initialize/shutdown and accepts document notifications.
_FAKE_SERVER = textwrap.dedent(
	"""
	import json, sys, threading
	out_lock = threading.Lock()
	def send(obj):
		body = json.dumps(obj).encode()
		with out_lock:
			sys.stdout.buffer.write(b"Content-Length: %d\\r\\n\\r\\n" % len(body) + body)
			sys.stdout.buffer.flush()
	def read():
		n = 0
		while True:
			line = sys.stdin.buffer.readline()
			if not line:
				return None
			if line.lower().startswith(b"content-length"):
				n = int(line.split(b":")[1])
			if line in (b"\\r\\n", b"\\n"):
				return json.loads(sys.stdin.buffer.read(n))
	while True:
		msg = read()
		if msg is None:
			break
		m = msg.get("method")
		if m == "initialize":
			send({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {}}})
		elif m == "shutdown":
			send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
	"""
)


def _fake_command(tmp_path: Path) -> list[str]:
	script = tmp_path / "fake_server.py"
	script.write_text(_FAKE_SERVER, encoding="utf-8")
	return [sys.executable, str(script)]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
	ws = tmp_path / "ws"
	ws.mkdir()
	tmp = tmp_path / "tmp"
	tmp.mkdir()
	monkeypatch.setenv("TMPDIR", str(tmp))
	monkeypatch.setattr("tempfile.tempdir", str(tmp))
	return ws


def test_rc_091_normal_preview_cleanup_removes_the_transient(workspace, tmp_path):
	# given a preview that materialises a new file in a new directory
	target = workspace / "pkg" / "new_mod.py"
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def _run() -> None:
		await client.start()
		try:
			async with client.overlay({target: "PREVIEW\n"}):
				assert target.read_text(encoding="utf-8") == "PREVIEW\n"
		finally:
			await client.stop()

	# when the preview ends
	asyncio.run(_run())
	# then the file and the directory it needed are gone, and the manifest is empty
	assert not target.exists()
	assert not (workspace / "pkg").exists()
	assert client._manifest_read() == []


def test_rc_029_user_write_between_check_and_unlink_is_never_deleted(workspace, tmp_path, monkeypatch):
	# given a preview of a new file, and a user write that lands while cleanup verifies the content
	target = workspace / "pkg" / "new_mod.py"
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")
	real_sha = lsp_client._sha1
	armed = {"on": False}

	def spy_sha(data: bytes) -> str:
		result = real_sha(data)
		if armed["on"]:
			armed["on"] = False
			# the concurrent write lands after the content check, before the deletion
			target.write_text("USER\n", encoding="utf-8")
		return result

	monkeypatch.setattr(lsp_client, "_sha1", spy_sha)

	async def _run() -> None:
		await client.start()
		try:
			async with client.overlay({target: "PREVIEW\n"}):
				armed["on"] = True  # arm after creation: the manifest add hashes too
		finally:
			await client.stop()

	asyncio.run(_run())
	# then the user's file is untouched and no private cleanup file is left behind
	assert target.read_text(encoding="utf-8") == "USER\n"
	assert not list(target.parent.glob(".*mcp-nav-preview*"))


def test_rc_091_sweep_never_deletes_a_preview_that_still_matches(workspace):
	# given a dead process's manifest entry whose file content is byte-identical to the recorded preview
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	victim = workspace / "adopted.py"
	victim.write_text("PREVIEW\n", encoding="utf-8")
	dead = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
	dead.wait()
	client._manifest_write(
		[{"path": str(victim), "sha1": lsp_client._sha1(b"PREVIEW\n"), "pid": dead.pid, "created_dirs": []}]
	)
	# when the sweep runs
	client._sweep_materialized()
	# then the matching file is kept for manual inspection (never deleted on a guess), and the entry is dropped
	assert victim.read_text(encoding="utf-8") == "PREVIEW\n"
	assert client._manifest_read() == []


def test_rc_029_restore_never_clobbers_a_file_recreated_during_cleanup(workspace, monkeypatch):
	# given a preview path whose content changed in place (adopted: no longer ours) and whose
	# identity is stale, and a user file that is created at the path during the cleanup check
	target = workspace / "pkg" / "new_mod.py"
	target.parent.mkdir(parents=True)
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	target.write_text("OURS\n", encoding="utf-8")
	identity = (target.stat().st_dev, target.stat().st_ino, lsp_client._sha1(b"OURS\n"))
	target.write_text("ADOPTED\n", encoding="utf-8")  # same inode, different bytes: not ours any more
	real_sha = lsp_client._sha1
	armed = {"on": False}

	def spy_sha(data: bytes) -> str:
		result = real_sha(data)
		if armed["on"]:
			armed["on"] = False
			# the concurrent write lands after the steal, while the stolen copy is verified
			target.write_text("USER2\n", encoding="utf-8")
		return result

	monkeypatch.setattr(lsp_client, "_sha1", spy_sha)
	armed["on"] = True
	# when the cleanup decides the file's fate
	removed = client._discard_if_ours(target, identity)
	# then nothing was removed, the user's recreated file survives untouched, and the transient
	# (the user's in-place edit) is kept as a private sibling instead of being renamed over USER2
	assert removed is False
	assert target.read_text(encoding="utf-8") == "USER2\n"
	siblings = list(target.parent.glob(".*mcp-nav-preview*"))
	assert len(siblings) == 1
	assert siblings[0].read_text(encoding="utf-8") == "ADOPTED\n"


def test_rc_091_sweep_removes_the_empty_directories_a_dead_preview_created(workspace):
	# given a dead process's manifest entry whose file is gone but whose created directories remain
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	dead = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
	dead.wait()
	empty = workspace / "pkg" / "sub"
	empty.mkdir(parents=True)
	occupied = workspace / "used"
	occupied.mkdir()
	(occupied / "keep.py").write_text("VALUE = 1\n", encoding="utf-8")  # non-empty: rmdir must refuse it
	client._manifest_write(
		[
			{
				"path": str(workspace / "pkg" / "sub" / "new_mod.py"),
				"sha1": "x",
				"pid": dead.pid,
				"created_dirs": [str(empty), str(occupied)],
			}
		]
	)
	# when the sweep runs
	client._sweep_materialized()
	# then the empty created directories are removed, the non-empty one is kept, and the entry is dropped
	assert not empty.exists()
	assert occupied.is_dir()
	assert client._manifest_read() == []
