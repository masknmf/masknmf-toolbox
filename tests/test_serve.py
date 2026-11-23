"""A figure on the http canvas, served to the fake browser in ``asgi_client`` as uvicorn would serve it."""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("uvicorn")

sys.path.insert(0, str(Path(__file__).parent))
from asgi_client import Browser, Lifespan, frame_of_size, get, request, ws_reply  # noqa: E402

from masknmf.visualization.canvas_server import SESSION_KEY_HEADER, BrowserCanvas, CanvasServer, client_js, page_html  # noqa: E402

KEY = [(SESSION_KEY_HEADER.encode(), b"sesame")]


class FigureVis:
    """The least a CanvasServer serves: a figure, shown and closed."""

    def __init__(self, canvas):
        import fastplotlib as fpl

        self.figure = fpl.Figure(size=(900, 640), canvas=canvas)
        self.figure[0, 0].add_image(np.random.default_rng(0).random((64, 64), dtype=np.float32))

    def show(self):
        self.figure.show()

    def close(self):
        self.figure.close()


def test_the_page_escapes_its_title_and_the_client_follows_https():
    assert "<title>a &lt;b&gt; &amp; c</title>" in page_html("a <b> & c", "Opening ...")
    assert "location.protocol === 'https:' ? 'wss://'" in client_js()


async def stream(server):
    status, _ = await get(server.asgi, "/")
    assert status == 403
    status, body = await get(server.asgi, "/", KEY)
    assert status == 200 and "figure" in body.decode()
    assert await ws_reply(server.asgi, "/") == {"type": "websocket.close", "code": 1008}

    await Lifespan(server.asgi).start()
    browser = Browser(server.asgi, headers=KEY)
    await browser.connect()
    # a HiDPI browser gets frames at one pixel per css pixel
    await browser.event(type="resize", width=800, height=600, pwidth=1600, pheight=1200, ratio=2)
    await frame_of_size(browser, (800, 600))

    # once the view stops changing, one lossless frame follows, then nothing
    while True:
        msg, _ = await browser.next_frame(timeout=15)
        if msg["mimetype"] == "image/png":
            break
    with pytest.raises(TimeoutError):
        await browser.next_frame(timeout=1.0)
    await browser.disconnect()
    server.close()


def test_a_served_figure_streams_frames_only_to_the_key_and_goes_still():
    canvas = BrowserCanvas(size=(900, 640), max_fps=60.0)
    server = CanvasServer(FigureVis(canvas), title="figure", waiting="Opening ...", host="127.0.0.1", port=0, key="sesame")
    asyncio.run(stream(server))


async def stream_launcher(server, served):
    await Lifespan(server.asgi).start()
    browser = Browser(server.asgi, headers=KEY)
    await browser.connect()
    await browser.resize(1000, 700)
    await frame_of_size(browser, (1000, 700))
    # Esc on the Overview closes the window on a desktop; a served page stays
    served.launcher.quit()
    await frame_of_size(browser, (1000, 700))
    assert not served.figure.canvas.get_closed()
    await browser.disconnect()
    server.close()


def test_the_served_launcher_streams_frames_and_keeps_its_page_open():
    from rendercanvas.http import loop

    from masknmf.launcher import ServedLauncher

    served = ServedLauncher(canvas=BrowserCanvas(size=(1000, 700), update_mode="continuous", max_fps=60.0), loop=loop)
    server = CanvasServer(served, title="masknmf", waiting="Opening ...", host="127.0.0.1", port=0, key="sesame")
    asyncio.run(stream_launcher(server, served))


def write_results(path, groups):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for group in groups:
            f.create_group(group)


def test_the_picker_lists_one_row_per_results_file_by_folder_newest_run_first(tmp_path, monkeypatch):
    from masknmf.io import group_name_compression, group_name_demixing
    from masknmf.visualization import results_picker

    stages = [group_name_compression(), group_name_demixing()]
    old = tmp_path / "mouse" / "fov1" / "20260101T000000_two-photon-calcium"
    new = tmp_path / "mouse" / "fov1" / "20260202T000000_two-photon-calcium"
    write_results(old / "results.hdf5", stages)
    write_results(old / "results.20260101T010000.curated.hdf5", stages)
    write_results(new / "results.hdf5", stages[:1])
    write_results(tmp_path / "mouse" / "other.results.hdf5", [])
    for skipped in ("_archive/run", "store.zarr", "movie.n5", "node_modules/pkg", "remote"):
        write_results(tmp_path / skipped / "results.hdf5", stages)
    write_results(tmp_path / "unnamed_store" / "results.hdf5", stages)
    (tmp_path / "unnamed_store" / "zarr.json").write_text("{}")
    monkeypatch.setattr(results_picker, "fuse_mounts", lambda: {str(tmp_path / "remote")})

    files = results_picker.find_results(tmp_path)
    assert [(f.folder, f.run, f.name) for f in files] == [
        ("mouse/fov1", new.name, "results.hdf5"),
        ("mouse/fov1", old.name, "results.20260101T010000.curated.hdf5"),
    ]
    assert files[0].stages == ["compression"] and files[1].stages == ["compression", "demixing"]
    assert files[1].num_curated == 1 and files[0].pipeline == "two-photon-calcium"

    viewer = open_viewer(files[1].path, FakeProcess())
    page = results_picker.picker_html(str(tmp_path), files, [viewer], error="a <b>")
    assert page.count("class='file") == 2 and page.count("class='file open'") == 1
    assert "a &lt;b&gt;" in page and "curated (1)" in page
    # mouse holds only fov1, so the tree shows them as one folder, collapsed
    assert "<details class='surface' data-name='mouse/fov1'" in page and "id='recent'" not in page
    assert re.search(r"<details[^>]* open[ >]", page) is None
    viewers = page[page.index("id='viewers'") : page.index("id='tree'")]
    assert "href='v/7/'" in viewers and "action='stop?id=7'" in viewers and "stop process" in viewers


def test_the_picker_tree_sorts_folders_by_number_and_lists_the_newest_files_on_top():
    from masknmf.visualization.results_picker import ResultsFile, folder_tree, picker_html

    files = [
        ResultsFile(f"/data/sweep/epochs_{n}/results.hdf5", f"sweep/epochs_{n}", f"epochs_{n}", ["compression"], 0, 1, float(n))
        for n in (100, 10, 30, 5, 50, 1000, 20, 40, 60)
    ]
    tree = folder_tree(files)
    assert list(tree["folders"]) == ["sweep"]
    page = picker_html("/data", files)
    order = [page.index(f"<summary><b>epochs_{n}</b>") for n in (5, 10, 20, 30, 40, 50, 60, 100, 1000)]
    assert order == sorted(order)
    recent = page[page.index("id='recent'") : page.index("id='tree'")]
    assert recent.count("class='file") == 8 and "epochs_1000/results.hdf5" in recent and "epochs_5/" not in recent


class FakeProcess:
    """Stands in for a viewer's process: running until stopped, or exited with ``code``."""

    def __init__(self, code=None):
        self.code = code
        self.pid = os.getpid()

    def poll(self):
        return self.code

    def terminate(self):
        self.code = -15

    def wait(self, timeout=None):
        return self.code


class SleepLauncher:
    """Stands in for ``cli.launch_viewer``: keeps what each process was started with and starts one that sleeps."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def open_viewer(path, process, filepath_log=os.devnull):
    from masknmf.visualization.results_picker import ViewerProcess, free_port

    return ViewerProcess(id=7, path=str(path), port=free_port(), key="sesame", process=process, filepath_log=str(filepath_log), time_started=0.0)


def test_the_picker_lists_and_opens_miniali_results_beside_masknmf_ones(tmp_path):
    import h5py
    from test_adapter_sources import write_miniali_file, write_miniali_split

    from masknmf.io import group_name_demixing
    from masknmf.visualization import results_picker

    write_results(tmp_path / "rec" / "20260101T000000_one-photon-culture" / "results.hdf5", [group_name_demixing()])
    (tmp_path / "Analysis" / "rec").mkdir(parents=True)
    write_miniali_file(tmp_path / "Analysis" / "rec" / "ALI_Result_new.mat")
    write_miniali_split(tmp_path / "Analysis" / "rec2")
    with h5py.File(tmp_path / "rec" / "output_data.mat", "w") as f:
        f["x"] = 1

    files = results_picker.find_results(tmp_path)
    assert [(f.folder, f.pipeline, f.name) for f in files] == [
        ("Analysis/rec", "miniali", "ALI_Result_new.mat"),
        ("Analysis/rec2", "miniali", "signals.h5"),
        ("rec", "one-photon-culture", "results.hdf5"),
    ]
    assert files[0].stages == ["miniali: 2 cells, 40 frames, no Background, recomputed"]

    launcher = SleepLauncher()
    server = results_picker.PickerServer(tmp_path, options=["--device", "cpu"], launch_viewer=launcher, host="127.0.0.1", port=0)
    viewer = server.open(files[0].path)
    assert launcher.calls[0]["argv"] == ["serve", files[0].path, "--host", "127.0.0.1", "--port", str(viewer.port), "--device", "cpu"]
    assert launcher.calls[0]["env_extra"] == {"CANVAS_SESSION_KEY": viewer.key} and launcher.calls[0]["detach"] is False
    # a file already open goes back to its viewer
    assert server.open(files[0].path) is viewer and len(launcher.calls) == 1
    server.close()
    assert viewer.process.poll() is not None and server.viewers == {}


async def picker_requests(server, path):
    status, headers, _ = await request(server.asgi, "GET", "/")
    assert status == 303 and headers[b"location"] == b"files"
    status, headers, body = await request(server.asgi, "GET", "/files")
    assert status == 200 and "results.hdf5" in body.decode() and headers[b"cache-control"] == b"no-store"
    status, body = await get(server.asgi, "/open?path=/etc/passwd")
    assert status == 200 and "is not a results file below" in body.decode()

    status, headers, _ = await request(server.asgi, "GET", f"/open?path={path}")
    assert status == 303 and headers[b"location"] == b"v/1/"
    status, body = await get(server.asgi, "/v/1/")
    assert status == 200 and "href='../../files'" in body.decode() and "fetch('status.json'" in body.decode()
    status, body = await get(server.asgi, "/v/1/status.json")
    assert json.loads(body) == {"status": "opening", "reason": ""}
    status, body = await get(server.asgi, "/v/1/renderview-client.js")
    assert body.decode() == client_js()
    status, body = await get(server.asgi, "/viewers.json")
    states = json.loads(body)
    assert [(state["id"], state["status"]) for state in states] == [(1, "opening")] and states[0]["memory"].endswith("B")

    status, _ = await get(server.asgi, "/stop?id=1")
    assert status == 404 and 1 in server.viewers
    process = server.viewers[1].process
    status, headers, _ = await request(server.asgi, "POST", "/stop?id=1")
    assert status == 303 and headers[b"location"] == b"files"
    assert server.viewers == {} and process.poll() is not None
    status, body = await get(server.asgi, "/v/1/status.json")
    assert json.loads(body)["status"] == "stopped"

    # uvicorn's shutdown stops the viewers still open
    lifespan = Lifespan(server.asgi)
    await lifespan.start()
    viewer = server.open(str(path))
    await lifespan.stop()
    assert server.viewers == {} and viewer.process.poll() is not None


def test_the_picker_starts_a_viewer_process_per_pick_and_stops_it(tmp_path):
    from masknmf.io import group_name_demixing
    from masknmf.visualization.results_picker import PickerServer

    write_results(tmp_path / "mouse" / "results.hdf5", [group_name_demixing()])
    server = PickerServer(tmp_path, options=[], launch_viewer=SleepLauncher(), host="127.0.0.1", port=0)

    asyncio.run(picker_requests(server, tmp_path / "mouse" / "results.hdf5"))
    server.close()


async def echo_with_a_frame(connection):
    """A viewer's websocket: refuses a connection without the key, sends back each message and then a frame's bytes."""
    if connection.request.headers.get(SESSION_KEY_HEADER) != "sesame":
        await connection.close()
        return
    async for message in connection:
        await connection.send(message)
        await connection.send(b"\x00frame")


async def relay_both_ways(server, viewer):
    from websockets.asyncio.server import serve

    browser = Browser(server.asgi, path="/v/7/")
    await browser.inbox.put({"type": "websocket.connect"})
    # accepted before the viewer listens; what the browser sends meanwhile waits for it
    await browser.connect()
    await browser.event(type="resize")
    await asyncio.sleep(0.5)
    async with serve(echo_with_a_frame, "127.0.0.1", viewer.port):
        message = await asyncio.wait_for(browser.outbox.get(), 10)
        assert json.loads(message["text"])["type"] == "resize"
        message = await asyncio.wait_for(browser.outbox.get(), 10)
        assert message["bytes"] == b"\x00frame"
        await browser.disconnect()


async def relay_exited(server):
    browser = Browser(server.asgi, path="/v/7/")
    await browser.connect()
    message = await asyncio.wait_for(browser.outbox.get(), 10)
    assert message["type"] == "websocket.close" and message["reason"] == "the viewer exited: RuntimeError: boom"
    await asyncio.wait_for(browser.task, 5)


def test_the_picker_relays_a_viewers_websocket_and_says_why_one_exited(tmp_path):
    from masknmf.visualization.results_picker import PickerServer

    server = PickerServer(tmp_path, options=[], launch_viewer=None, host="127.0.0.1", port=0)
    viewer = server.viewers[7] = open_viewer(tmp_path / "results.hdf5", FakeProcess())
    asyncio.run(relay_both_ways(server, viewer))

    filepath_log = tmp_path / "viewer.log"
    filepath_log.write_text("Traceback ...\nRuntimeError: boom\n\n")
    server.viewers[7] = open_viewer(tmp_path / "results.hdf5", FakeProcess(code=1), filepath_log)
    asyncio.run(relay_exited(server))
    assert "<td class='reason'" in server.page() and "RuntimeError: boom" in server.page()


@pytest.mark.parametrize(
    "cmdline",
    [
        ["/env/bin/python", "-m", "masknmf.cli", "serve", "results.hdf5", "--port", "8811"],
        ["/env/bin/python", "/env/bin/masknmf", "serve", "results.hdf5", "--port", "8811"],
        [str(Path("env") / "Scripts" / "masknmf.exe"), "serve", "results.hdf5", "--port", "8811"],
    ],
)
def test_serving_port_reads_each_way_masknmf_serve_starts(cmdline):
    from masknmf import cli

    assert cli.serving_port(cmdline=cmdline) == 8811
    assert cli.serving_port(cmdline=[*cmdline[:-2], "--kill", "8811"]) is None
    assert cli.serving_port(cmdline=[c.replace("serve", "view") for c in cmdline]) is None
