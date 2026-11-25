"""Regression tests for QA block B2: language-server lifecycle (RC-029, 047, 091, 093, 094, 154, 158)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from mcp_nav_shared import lsp_client
from mcp_nav_shared.lsp_client import LanguageServerExitedError, LspClient


# A tiny language server: answers initialize/shutdown, tracks document text, and answers `hover` with the
# document's text AT REPLY TIME after a delay (like a server computing on whatever text it holds).
# It starts a `sleep 300` child (pid written to $FAKE_CHILD_PID_FILE) like a `uvx` wrapper would.
_FAKE_SERVER = textwrap.dedent(
	"""
	import json, os, subprocess, sys, threading, time
	child = subprocess.Popen(["sleep", "300"])
	if os.environ.get("FAKE_CHILD_PID_FILE"):
		open(os.environ["FAKE_CHILD_PID_FILE"], "w").write(str(child.pid))
	if os.environ.get("FAKE_HANG"):
		time.sleep(300)
	docs = {}
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
	def hover(msg):
		time.sleep(float(os.environ.get("FAKE_HOVER_DELAY", "0")))
		uri = msg["params"]["textDocument"]["uri"]
		send({"jsonrpc": "2.0", "id": msg["id"], "result": {"contents": docs.get(uri, "")}})
	while True:
		msg = read()
		if msg is None:
			break
		m = msg.get("method")
		if m == "initialize":
			send({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {}}})
		elif m == "shutdown":
			send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
		elif m == "textDocument/hover":
			threading.Thread(target=hover, args=(msg,), daemon=True).start()
		elif m == "textDocument/didOpen":
			d = msg["params"]["textDocument"]
			docs[d["uri"]] = d["text"]
		elif m == "textDocument/didChange":
			docs[msg["params"]["textDocument"]["uri"]] = msg["params"]["contentChanges"][0]["text"]
		elif m == "exit":
			time.sleep(float(os.environ.get("FAKE_EXIT_DELAY", "0")))
	"""
)


def _fake_command(tmp_path: Path) -> list[str]:
	script = tmp_path / "fake_server.py"
	script.write_text(_FAKE_SERVER, encoding="utf-8")
	return [sys.executable, str(script)]


def _pid_gone(pid: int, wait: float = 3.0) -> bool:
	deadline = time.monotonic() + wait
	while time.monotonic() < deadline:
		try:
			with open(f"/proc/{pid}/stat") as fh:
				if fh.read().split()[2] == "Z":
					return True  # a zombie is dead; its parent (init, or us) just hasn't reaped it
		except OSError:
			return True
		time.sleep(0.05)
	return False


@pytest.fixture
def workspace(tmp_path, monkeypatch):
	ws = tmp_path / "ws"
	ws.mkdir()
	tmp = tmp_path / "tmp"
	tmp.mkdir()
	monkeypatch.setenv("TMPDIR", str(tmp))
	monkeypatch.setattr("tempfile.tempdir", str(tmp))
	return ws


def test_rc_029_inflight_read_is_answered_on_disk_text_not_overlay(workspace, tmp_path, monkeypatch):
	# given a slow hover in flight and a preview (overlay) of the same file started meanwhile
	monkeypatch.setenv("FAKE_HOVER_DELAY", "0.6")
	target = workspace / "a.py"
	target.write_text("DISK", encoding="utf-8")
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def _run() -> tuple[dict, str]:
		await client.start()
		try:
			hover = asyncio.create_task(client.hover("a.py", 1, 1))
			await asyncio.sleep(0.2)  # the request is with the server now
			seen_during = ""
			async with client.overlay({target: "OVERLAY"}):
				seen_during = (await client.hover("a.py", 1, 1))["contents"]
			return await hover, seen_during
		finally:
			await client.stop()

	# when
	first, during = asyncio.run(_run())
	# then the first reply was computed on the disk text; a read inside the preview sees the preview
	assert first["contents"] == "DISK"
	assert during == "OVERLAY"


def test_rc_029_read_started_during_overlay_waits_for_it(workspace, tmp_path, monkeypatch):
	# given
	target = workspace / "a.py"
	target.write_text("DISK", encoding="utf-8")
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def _run() -> str:
		await client.start()
		inside, release = asyncio.Event(), asyncio.Event()

		async def _preview() -> None:
			async with client.overlay({target: "OVERLAY"}):
				inside.set()
				await release.wait()

		try:
			preview = asyncio.create_task(_preview())
			await inside.wait()
			outsider = asyncio.create_task(client.hover("a.py", 1, 1))  # another tool call, not the preview's
			await asyncio.sleep(0.2)
			assert not outsider.done()  # still waiting for the preview to end
			release.set()
			await preview
			return (await outsider)["contents"]
		finally:
			await client.stop()

	# when / then it only ran after the preview ended, on the disk text
	assert asyncio.run(_run()) == "DISK"


def test_rc_091_sigkill_during_preview_preserves_ambiguous_file_after_next_start(workspace, tmp_path, monkeypatch):
	# given a process that is killed (SIGKILL) while a new file is materialised for a preview
	monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(tmp_path / "orphan.pid"))
	cmd = _fake_command(tmp_path)
	child_script = tmp_path / "child.py"
	child_script.write_text(
		textwrap.dedent(
			f"""
			import asyncio, sys, time
			from pathlib import Path
			from mcp_nav_shared.lsp_client import LspClient
			async def main():
				client = LspClient(workspace_root=Path({str(workspace)!r}), command={cmd!r}, language_id="python")
				await client.start()
				async with client.overlay({{Path({str(workspace / "pkg" / "material_new.py")!r}): "x = 1\\n"}}):
					print("ready", flush=True)
					time.sleep(300)
			asyncio.run(main())
			"""
		),
		encoding="utf-8",
	)
	env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "TMPDIR": os.environ["TMPDIR"]}
	proc = subprocess.Popen([sys.executable, str(child_script)], stdout=subprocess.PIPE, env=env)
	assert proc.stdout is not None
	assert proc.stdout.readline().strip() == b"ready"
	assert (workspace / "pkg" / "material_new.py").exists()
	proc.send_signal(signal.SIGKILL)
	proc.wait()
	orphan_child = int((tmp_path / "orphan.pid").read_text())
	assert (workspace / "pkg" / "material_new.py").exists()  # the finally never ran
	_kill_strays(tmp_path)  # the orphaned fake server of the killed process, and its `sleep` child
	with contextlib.suppress(OSError, ValueError):
		os.kill(orphan_child, signal.SIGKILL)

	# when a new client starts for the same workspace
	client = LspClient(workspace_root=workspace, command=cmd, language_id="python")

	async def _run() -> None:
		await client.start()
		await client.stop()

	asyncio.run(_run())
	# then the ambiguous preview is retained for manual inspection
	assert (workspace / "pkg" / "material_new.py").exists()


def test_rc_091_sweep_keeps_a_file_the_user_edited(workspace, tmp_path):
	# given a dead process's manifest entry whose file has since been edited
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	victim = workspace / "mine.py"
	victim.write_text("user work\n", encoding="utf-8")
	dead = subprocess.Popen([sys.executable, "-c", "pass"])
	dead.wait()
	client._manifest_write(
		[{"path": str(victim), "sha1": lsp_client._sha1(b"preview text"), "pid": dead.pid, "created_dirs": []}]
	)
	# when
	client._sweep_materialized()
	# then
	assert victim.read_text(encoding="utf-8") == "user work\n"
	assert client._manifest_read() == []


def test_rc_093_same_size_same_mtime_rewrite_is_reread(workspace, tmp_path):
	# given an open file, then a same-size rewrite whose mtime is put back
	target = workspace / "c.py"
	target.write_text("def c2(): ...\n", encoding="utf-8")
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	notes: list[tuple[str, dict]] = []
	client._notify = lambda method, params: notes.append((method, params))  # type: ignore[method-assign]

	async def _run() -> None:
		await client.ensure_open(str(target))
		before = target.stat()
		time.sleep(0.02)
		target.write_text("def c3(): ...\n", encoding="utf-8")
		os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
		assert target.stat().st_size == before.st_size and target.stat().st_mtime_ns == before.st_mtime_ns
		await client.ensure_open(str(target))

	# when
	asyncio.run(_run())
	# then the server was told about the new text
	assert [m for m, _ in notes] == ["textDocument/didOpen", "textDocument/didChange"]
	assert notes[1][1]["contentChanges"][0]["text"] == "def c3(): ...\n"


def test_rc_154_timeout_at_start_kills_the_whole_process_group(workspace, tmp_path, monkeypatch):
	# given a "uvx" that never answers and has a long-lived child
	pid_file = tmp_path / "child.pid"
	monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(pid_file))
	monkeypatch.setenv("FAKE_HANG", "1")
	client = LspClient(
		workspace_root=workspace, command=_fake_command(tmp_path), language_id="python", start_timeout=1.0
	)

	# when start times out
	with pytest.raises(TimeoutError):
		asyncio.run(client.start())
	# then neither the wrapper nor its child survives, and the client can be started again
	child_pid = int(pid_file.read_text())
	assert _pid_gone(child_pid)
	assert client._proc is None
	assert client.is_alive is False


def test_rc_154_stop_kills_children_that_outlive_the_server(workspace, tmp_path, monkeypatch):
	# given a server that ignores `exit` for a while and whose child would live 300 s
	pid_file = tmp_path / "child.pid"
	monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(pid_file))
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def _run() -> None:
		await client.start()
		await client.stop()

	# when
	asyncio.run(_run())
	# then
	assert _pid_gone(int(pid_file.read_text()))


def test_rc_094_missing_command_is_not_reported_as_a_missing_file(workspace):
	# given
	client = LspClient(workspace_root=workspace, command=["no-such-uvx-binary", "ty", "server"], language_id="python")
	# when
	with pytest.raises(LanguageServerExitedError) as caught:
		asyncio.run(client.start())
	# then
	assert str(caught.value) == (
		"could not start the language server (command not found: no-such-uvx-binary); install ty "
		"(for example `pip install ty`) or make sure uv/uvx is on PATH"
	)


def test_rc_094_message_does_not_leak_an_absolute_command_path(workspace):
	# given
	client = LspClient(workspace_root=workspace, command=["/nonexistent/dir/ty", "server"], language_id="python")
	# when
	with pytest.raises(LanguageServerExitedError) as caught:
		asyncio.run(client.start())
	# then
	assert "/nonexistent" not in str(caught.value)
	assert "command not found: ty)" in str(caught.value)


def test_rc_047_exit_message_has_no_command_line_or_stderr_but_detail_has(workspace):
	# given a server that prints why it can't run, then exits with code 2
	script = "import sys; sys.stderr.write('FATAL: ty at /abs/path cannot start\\n'); sys.exit(2)"
	client = LspClient(workspace_root=workspace, command=[sys.executable, "-c", script], language_id="python")
	# when
	with pytest.raises(LanguageServerExitedError) as caught:
		asyncio.run(client.start())
	# then the message is short and safe; the technical part is in `.detail`
	message = str(caught.value)
	assert message == "the language server exited unexpectedly (exit code 2); the next call restarts it"
	assert sys.executable not in message and "FATAL" not in message
	assert "FATAL: ty at /abs/path cannot start" in caught.value.detail
	assert sys.executable in caught.value.detail


def test_rc_158_restart_counter(workspace, tmp_path):
	# given
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def _run() -> int:
		await client.start()
		try:
			assert client.restarts == 0
			await client.restart()
			await client.restart()
			return client.restarts
		finally:
			await client.stop()

	# when / then
	assert asyncio.run(_run()) == 2


def _kill_strays(tmp_path: Path) -> None:
	"""Kill leftover fake servers / `sleep` children of a test (anything whose command line names tmp_path)."""
	for entry in Path("/proc").iterdir():
		if not entry.name.isdigit() or int(entry.name) == os.getpid():
			continue
		with contextlib.suppress(OSError):
			if str(tmp_path) in (entry / "cmdline").read_text().replace("\0", " "):
				os.kill(int(entry.name), signal.SIGKILL)


@pytest.fixture(autouse=True)
def _reap_strays(tmp_path):
	yield
	_kill_strays(tmp_path)


def test_rc_154_request_timeout_kills_server_and_child(workspace, tmp_path, monkeypatch):
	pid_file = tmp_path / "child.pid"
	monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(pid_file))
	monkeypatch.setenv("FAKE_HOVER_DELAY", "5")
	(workspace / "a.py").write_text("x = 1\n")
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def run():
		await client.start()
		await client.ensure_open("a.py")
		with pytest.raises(TimeoutError):
			await client._request(
				"textDocument/hover", {"textDocument": {"uri": (workspace / "a.py").as_uri()}}, timeout=0.2
			)
		assert client._proc is None

	asyncio.run(run())
	assert _pid_gone(int(pid_file.read_text()))


def test_rc_091_clients_have_independent_manifests(workspace):
	one = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	two = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	one._manifest_add(workspace / "one.py", "one", [])
	two._manifest_add(workspace / "two.py", "two", [])
	assert one._manifest_path() != two._manifest_path()
	one._manifest_drop([workspace / "one.py"])
	assert two._manifest_read()[0]["path"] == str(workspace / "two.py")
	two._manifest_drop([workspace / "two.py"])


def test_rc_091_manifest_failure_prevents_materialisation(workspace, monkeypatch):
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	target = workspace / "new.py"

	def fail(_entries):
		raise OSError("manifest unavailable")

	monkeypatch.setattr(client, "_manifest_write", fail)

	async def run():
		with pytest.raises(OSError, match="manifest unavailable"):
			async with client.overlay({target: "x = 1\n"}):
				pass

	asyncio.run(run())
	assert not target.exists()


def test_overlay_preserves_user_edited_transient(workspace, tmp_path):
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")
	target = workspace / "new.py"

	async def run():
		await client.start()
		try:
			async with client.overlay({target: "preview\n"}):
				target.write_text("user work\n")
		finally:
			await client.stop()

	asyncio.run(run())
	assert target.read_text() == "user work\n"


def test_overlay_creation_race_does_not_remove_user_file(workspace, tmp_path, monkeypatch):
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")
	target = workspace / "new.py"
	original = client._materialize

	def race(path, text):
		path.write_text("user work\n")
		return original(path, text)

	monkeypatch.setattr(client, "_materialize", race)

	async def run():
		await client.start()
		try:
			with pytest.raises(FileExistsError):
				async with client.overlay({target: "preview\n"}):
					pass
		finally:
			await client.stop()

	asyncio.run(run())
	assert target.read_text() == "user work\n"


def test_sweep_ignores_forged_manifest_outside_private_directory(workspace, tmp_path):
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	victim = tmp_path / "victim.py"
	victim.write_text("valuable\n")
	dead = subprocess.Popen([sys.executable, "-c", "pass"])
	dead.wait()
	import json

	forged = Path(os.environ["TMPDIR"]) / f"{client._manifest_prefix()}forged.json"
	forged.write_text(json.dumps([{"path": str(victim), "pid": dead.pid, "sha1": lsp_client._sha1(b"valuable\n")}]))
	client._sweep_materialized()
	assert victim.read_text() == "valuable\n"


def test_overlay_timeout_preserves_timeout_and_releases_lease(workspace, tmp_path, monkeypatch):
	monkeypatch.setenv("FAKE_HOVER_DELAY", "5")
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")
	target = workspace / "new.py"

	async def run():
		await client.start()
		with pytest.raises(TimeoutError):
			async with client.overlay({target: "preview\n"}):
				await client._request("textDocument/hover", {"textDocument": {"uri": target.as_uri()}}, timeout=0.1)
		assert not lsp_client._OVERLAY_HOLDERS.get().get(id(client), [False])[0]

	asyncio.run(run())
	assert not target.exists()


def test_inherited_overlay_lease_expires_for_child_task(workspace, tmp_path):
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")
	target = workspace / "a.py"
	target.write_text("disk")

	async def run():
		await client.start()
		proceed = asyncio.Event()
		try:
			async with client.overlay({target: "preview"}):

				async def child():
					await proceed.wait()
					async with client._reading():
						return client._readers

				task = asyncio.create_task(child())
			proceed.set()
			assert await task == 1
		finally:
			await client.stop()

	asyncio.run(run())


def test_inherited_reader_does_not_outlive_parent_untracked(workspace, tmp_path):
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def run():
		ready = asyncio.Event()
		finish = asyncio.Event()
		entered = asyncio.Event()

		async def child():
			async with client._reading():
				ready.set()
				await finish.wait()

		async def preview():
			async with client.overlay({}):
				entered.set()

		async with client._reading():
			task = asyncio.create_task(child())
			await ready.wait()
		preview_task = asyncio.create_task(preview())
		await asyncio.sleep(0)
		assert not entered.is_set()
		finish.set()
		await task
		await preview_task
		assert entered.is_set()

	asyncio.run(run())


def test_child_cannot_inherit_active_overlay_access(workspace, tmp_path):
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def run():
		async with client.overlay({}):

			async def child():
				async with client._reading():
					pass

			with pytest.raises(RuntimeError, match="child task"):
				await asyncio.create_task(child())

	asyncio.run(run())


def test_child_reader_refuses_instead_of_deadlocking_with_waiting_overlay(workspace, tmp_path):
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")

	async def run():
		async def preview():
			async with client.overlay({}):
				pass

		async def child():
			async with client._reading():
				pass

		async with client._reading():
			preview_task = asyncio.create_task(preview())
			await asyncio.sleep(0)
			with pytest.raises(RuntimeError, match="overlay waits"):
				await asyncio.wait_for(asyncio.create_task(child()), 1)
		await preview_task

	asyncio.run(run())


def test_materialize_partial_write_failure_reports_and_retains_ambiguous_file(workspace, monkeypatch):
	notices = []
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python", on_notice=notices.append)
	target = workspace / "newdir" / "new.py"
	original_open = Path.open

	class BrokenWrite:
		def __init__(self, inner):
			self.inner = inner

		def __enter__(self):
			return self

		def __exit__(self, *args):
			return self.inner.__exit__(*args)

		def fileno(self):
			return self.inner.fileno()

		def write(self, data):
			self.inner.write(data[:3])
			self.inner.flush()
			raise OSError("disk full")

	def open_file(path, mode="r", *args, **kwargs):
		f = original_open(path, mode, *args, **kwargs)
		return BrokenWrite(f) if path == target and mode == "xb" else f

	monkeypatch.setattr(Path, "open", open_file)

	async def run():
		with pytest.raises(OSError, match="disk full"):
			async with client.overlay({target: "preview text"}):
				pass

	asyncio.run(run())
	assert target.read_text() == "pre"
	assert len(client._manifest_read()) == 1
	assert len(notices) == 1 and "newdir/new.py" in notices[0]
	assert str(workspace) not in notices[0]


def test_materialize_failure_preserves_replaced_user_file(workspace, monkeypatch):
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	target = workspace / "new.py"
	original_open = Path.open

	class ReplacedWrite:
		def __init__(self, inner):
			self.inner = inner

		def __enter__(self):
			return self

		def __exit__(self, *args):
			return self.inner.__exit__(*args)

		def fileno(self):
			return self.inner.fileno()

		def write(self, data):
			replacement = workspace / "replacement.py"
			replacement.write_text("valuable user data")
			os.replace(replacement, target)
			raise OSError("disk full")

	def open_file(path, mode="r", *args, **kwargs):
		f = original_open(path, mode, *args, **kwargs)
		return ReplacedWrite(f) if path == target and mode == "xb" else f

	monkeypatch.setattr(Path, "open", open_file)

	async def run():
		with pytest.raises(OSError, match="disk full"):
			async with client.overlay({target: "preview"}):
				pass

	asyncio.run(run())
	assert target.read_text() == "valuable user data"
	assert len(client._manifest_read()) == 1


def test_crash_notice_is_relative_and_only_once(workspace):
	notices = []
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python", on_notice=notices.append)
	target = workspace / "possibly-preview.py"
	target.write_text("user data")
	dead = subprocess.Popen([sys.executable, "-c", "pass"])
	dead.wait()
	client._manifest_write([{"path": str(target), "pid": dead.pid}])
	client._sweep_materialized()
	client._sweep_materialized()
	assert len(notices) == 1
	assert "possibly-preview.py" in notices[0]
	assert str(workspace) not in notices[0]
	assert target.read_text() == "user data"
	assert client._manifest_read() == []


def test_overlay_preserves_replacement_after_materialize_returns(workspace, tmp_path, monkeypatch):
	client = LspClient(workspace_root=workspace, command=_fake_command(tmp_path), language_id="python")
	target = workspace / "new.py"
	original = client._materialize

	def replace_after_write(path, text):
		identity = original(path, text)
		replacement = workspace / "replacement.py"
		replacement.write_text(text)
		os.replace(replacement, path)
		return identity

	monkeypatch.setattr(client, "_materialize", replace_after_write)

	async def run():
		await client.start()
		try:
			async with client.overlay({target: "preview\n"}):
				pass
		finally:
			await client.stop()

	asyncio.run(run())
	assert target.read_text() == "preview\n"


def test_materialize_open_failure_drops_manifest(workspace, monkeypatch):
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python")
	target = workspace / "newdir" / "new.py"

	def fail(_path, _text):
		raise PermissionError("denied")

	monkeypatch.setattr(client, "_materialize", fail)

	async def run():
		with pytest.raises(PermissionError):
			async with client.overlay({target: "preview"}):
				pass

	asyncio.run(run())
	assert not target.exists()
	assert not target.parent.exists()
	assert client._manifest_read() == []


def test_materialize_failure_preserves_in_place_matching_prefix_edit(workspace, monkeypatch):
	notices = []
	client = LspClient(workspace_root=workspace, command=["true"], language_id="python", on_notice=notices.append)
	target = workspace / "new.py"
	original_open = Path.open

	class UserEditOnFailedWrite:
		def __init__(self, inner):
			self.inner = inner

		def __enter__(self):
			return self

		def __exit__(self, *args):
			return self.inner.__exit__(*args)

		def fileno(self):
			return self.inner.fileno()

		def write(self, data):
			self.inner.write(data[:7])
			self.inner.flush()
			with original_open(target, "r+b") as user:
				user.truncate(3)
			raise OSError("disk full")

	def open_file(path, mode="r", *args, **kwargs):
		f = original_open(path, mode, *args, **kwargs)
		return UserEditOnFailedWrite(f) if path == target and mode == "xb" else f

	monkeypatch.setattr(Path, "open", open_file)

	async def run():
		with pytest.raises(OSError, match="disk full"):
			async with client.overlay({target: "preview text"}):
				pass

	asyncio.run(run())
	assert target.read_text() == "pre"
	assert len(client._manifest_read()) == 1
	assert len(notices) == 1 and "new.py" in notices[0]
	assert str(workspace) not in notices[0]
