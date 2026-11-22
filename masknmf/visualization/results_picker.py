"""Serve a folder of results files to a browser: a page listing them, and a demixing viewer on each one picked.

``masknmf serve FOLDER`` runs this. ``files`` lists every results file below the folder, and every other pipeline's
results an adapter finds by file name (miniALI's ALI_Result_new.mat, suite2p's stat.npy), as a tree of folders that
starts collapsed, the runs in each newest first, under the most recently changed files; a filter box narrows it and a
click or Enter opens a file. Each file opens in a ``masknmf serve FILE`` process of its own on a free localhost port,
which this server passes through at ``v/<id>/``, so only its own port needs a tunnel. The files page lists the open
viewers on top, to go back to one or stop its process.
"""

from __future__ import annotations

import asyncio
import fnmatch
import html
import json
import os
import re
import secrets
import socket
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Callable
from urllib.parse import parse_qs, quote

import psutil
from rendercanvas.http import resources

from masknmf.demixing._adapters import adapter_for, adapter_patterns
from masknmf.demixing.curation import CURATED_SUFFIX, latest_results, results_stem
from masknmf.io import stage_groups
from masknmf.visualization.canvas_server import SESSION_KEY_ENV, SESSION_KEY_HEADER, client_js, page_html, require

__all__ = ["PickerServer", "ResultsFile", "ViewerProcess", "find_results", "picker_html"]

PATTERNS_RESULTS = ("*results*.hdf5", "*results*.h5")
# a run folder is named <timestamp>_<pipeline>
PATTERN_RUN = re.compile(r"^(\d{8}T\d{6})_(.+)$")
# a viewer's pages and websocket, ``v/<id>/<resource>``
PATTERN_VIEWER = re.compile(r"/v/(\d+)/([^/]*)$")
# folders of chunked arrays, code or caches: thousands of entries and never a results file
SUFFIXES_STORE = (".zarr", ".n5")
NAMES_SKIPPED = {"node_modules", "__pycache__", "site-packages"}
MARKERS_ZARR = {".zgroup", ".zarray", "zarr.json"}
NUM_RECENT = 8
NUM_PIPELINES_SHOWN = 3
# runhub's dates: day and minute within a week, else the day
NUM_DAYS_WITH_TIME = 7
SECONDS_STOP = 5.0

_PICKER = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>$title</title>
    <style>
        /* runhub's Material 3 dark roles */
        :root { color-scheme: dark; --paper: #141218; --card: #1d1b20; --raised: #2b2930; --ink: #e6e0e9;
                --muted: #cac4d0; --line: #49454f; --link: #d0bcff; --selected: #4a4458; --on-selected: #e8def8;
                --bad: #f2b8b5; --good: #b5e3b0; --hover: rgba(230, 224, 233, 0.05); }
        body { margin: 0; padding: 0 24px 32px; background: var(--paper); color: var(--ink);
               font: 14px Roboto, 'Google Sans', system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
        a { color: var(--link); text-decoration: none; }
        a:hover { text-decoration: underline; }
        [hidden] { display: none !important; }
        header { position: sticky; top: 0; z-index: 1; padding: 18px 0 4px; background: var(--paper); }
        h1 { margin: 0; font-size: 22px; font-weight: 400; }
        .meta { color: var(--muted); font-size: 12px; font-weight: 400; }
        .root { font-family: monospace; }
        #filter { box-sizing: border-box; width: 100%; max-width: 576px; height: 44px; margin: 14px 0 10px; padding: 0 20px;
                  background: var(--raised); color: var(--ink); border: none; border-radius: 9999px; outline: none; font: inherit; }
        #filter::placeholder { color: rgba(202, 196, 208, 0.7); }
        #filter:focus { box-shadow: 0 0 0 2px var(--link); }
        .controls { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin-bottom: 12px; }
        .chip { display: inline-flex; box-sizing: border-box; height: 28px; align-items: center; padding: 0 12px;
                background: none; color: var(--muted); border: 1px solid var(--line); border-radius: 8px;
                font: 500 12px Roboto, system-ui, sans-serif; white-space: nowrap; cursor: pointer; }
        .chip:hover { background: rgba(230, 224, 233, 0.08); color: var(--ink); text-decoration: none; }
        .chip.on { background: var(--selected); color: var(--on-selected); border-color: transparent; }
        .chip.small { height: 20px; padding: 0 8px; margin-left: 8px; font-size: 11px; cursor: inherit; }
        .surface { margin-bottom: 8px; background: var(--card); border-radius: 12px; overflow: hidden; }
        .error { padding: 10px 16px; margin-bottom: 12px; background: #4a2326; border-radius: 12px; }
        .title { padding: 12px 16px 4px; }
        h2 { margin: 24px 0 10px; font-size: 16px; font-weight: 500; }
        summary { list-style: none; padding: 12px 16px; cursor: pointer; user-select: none; }
        summary::-webkit-details-marker { display: none; }
        summary::before { content: '\\25B8'; display: inline-block; width: 16px; color: var(--muted); }
        details[open] > summary::before { content: '\\25BE'; }
        summary:hover { background: var(--hover); }
        summary .meta { margin-left: 10px; }
        details details { border-top: 1px solid rgba(73, 69, 79, 0.6); }
        details details > summary { padding: 8px 16px; }
        .children { padding-left: 16px; }
        .scroll { overflow-x: auto; }
        table { width: 100%; min-width: 860px; border-collapse: collapse; table-layout: fixed; }
        th { padding: 4px 12px 4px 0; color: var(--muted); font-size: 12px; font-weight: 400; text-align: left;
             border-bottom: 1px solid var(--line); }
        td { padding: 6px 12px 6px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
             border-bottom: 1px solid rgba(73, 69, 79, 0.6); }
        tr:last-child td { border-bottom: none; }
        th:first-child, td:first-child { padding-left: 16px; }
        .when { width: 110px; } .pipeline { width: 170px; } .stages { width: 240px; } .curated { width: 90px; }
        .num { width: 70px; text-align: right; font-variant-numeric: tabular-nums; } .folder { width: 220px; }
        .status { width: 90px; } .actions { width: 210px; text-align: right; }
        td.when, td.pipeline, td.stages, td.curated, td.num, td.folder { color: var(--muted); }
        tr.file { cursor: pointer; }
        tr.file:hover { background: var(--hover); }
        tr.file.active { background: rgba(230, 224, 233, 0.08); }
        tr.file.active td:first-child { box-shadow: inset 2px 0 var(--link); }
        td.status.ready { color: var(--good); } td.status.opening { color: var(--muted); }
        td.status.exited, td.reason { color: var(--bad); }
        td.actions form { display: inline; margin-left: 4px; }
        body.opening { cursor: progress; opacity: 0.6; }
    </style>
</head>
<body>
    <header>
        <h1>masknmf results</h1>
        <span class='meta'><span class='root'>$root</span> · $count</span>
        <br><input id='filter' placeholder='type to filter: folder, run, pipeline, stage; arrows and Enter open' autofocus>
    </header>
    $error
    $viewers
    $recent
    $tree
    <script>
        const filter = document.getElementById('filter')
        const recent = document.getElementById('recent')
        const folders = [...document.querySelectorAll('#tree details')]
        let active = null

        function visibleRows () {
            // NOTE: offsetParent stays set inside a closed details in Chrome, so ancestors are checked instead
            return [...document.querySelectorAll('tr.file')].filter((row) => row.closest('[hidden], details:not([open])') === null)
        }
        function setActive (row) {
            if (active !== null) { active.classList.remove('active') }
            active = row === undefined ? null : row
            if (active !== null) {
                active.classList.add('active')
                active.scrollIntoView({ block: 'nearest' })
            }
        }
        function openRow (row) {
            document.body.classList.add('opening')
            window.location = row.dataset.href
        }
        function sortFolders (key) {
            for (const container of document.querySelectorAll('#tree, #tree .children')) {
                const children = [...container.children].filter((child) => child.matches('details'))
                children.sort((a, b) => key === 'newest'
                    ? Number(b.dataset.time) - Number(a.dataset.time)
                    : a.dataset.name.localeCompare(b.dataset.name, undefined, { numeric: true, sensitivity: 'base' }))
                container.append(...children)
            }
            for (const chip of document.querySelectorAll('[data-sort]')) { chip.classList.toggle('on', chip.dataset.sort === key) }
            try { localStorage.setItem('masknmf-picker-sort', key) } catch (e) {}
        }

        filter.addEventListener('input', () => {
            const words = filter.value.toLowerCase().split(/\\s+/).filter(Boolean)
            if (recent !== null) { recent.hidden = words.length > 0 }
            for (const row of document.querySelectorAll('#tree tr.file')) {
                row.hidden = !words.every((word) => row.dataset.text.includes(word))
            }
            for (const folder of folders) {
                const hit = folder.querySelector('tr.file:not([hidden])') !== null
                folder.hidden = !hit
                folder.open = words.length > 0 && hit
            }
            setActive(visibleRows()[0])
        })
        filter.addEventListener('keydown', (event) => {
            const rows = visibleRows()
            const index = rows.indexOf(active)
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault()
                const step = event.key === 'ArrowDown' ? 1 : -1
                setActive(rows[Math.min(rows.length - 1, Math.max(0, index + step))])
            } else if (event.key === 'Enter' && rows.length > 0) {
                openRow(index >= 0 ? active : rows[0])
            } else if (event.key === 'Escape') {
                filter.value = ''
                filter.dispatchEvent(new Event('input'))
            }
        })
        document.addEventListener('keydown', (event) => {
            if (event.key === '/' && document.activeElement !== filter) {
                event.preventDefault()
                filter.focus()
            }
        })
        for (const button of document.querySelectorAll('[data-expand]')) {
            button.addEventListener('click', () => folders.forEach((folder) => { folder.open = button.dataset.expand === '1' }))
        }
        for (const button of document.querySelectorAll('[data-sort]')) {
            button.addEventListener('click', () => sortFolders(button.dataset.sort))
        }
        for (const row of document.querySelectorAll('tr.file')) {
            row.addEventListener('click', () => openRow(row))
        }
        try {
            if (localStorage.getItem('masknmf-picker-sort') === 'newest') { sortFolders('newest') }
        } catch (e) {}

        const rowsViewer = [...document.querySelectorAll('#viewers tr[data-viewer]')]
        async function refreshViewers () {
            if (document.hidden) { return }
            const response = await fetch('viewers.json', { cache: 'no-store' })
            const viewers = new Map((await response.json()).map((viewer) => [String(viewer.id), viewer]))
            for (const row of rowsViewer) {
                const viewer = viewers.get(row.dataset.viewer)
                if (viewer === undefined) { row.hidden = true; continue }
                const status = row.querySelector('td.status')
                status.textContent = viewer.status
                status.className = 'status ' + viewer.status.split(' ')[0]
                row.querySelector('td.memory').textContent = viewer.memory
            }
        }
        if (rowsViewer.length > 0) { setInterval(refreshViewers, 2000) }
        // NOTE: chrome restores a no-store page from its back-forward cache too, with the viewers listed before the pick
        window.addEventListener('pageshow', (event) => { if (event.persisted) { location.reload() } })
        // the folders start collapsed again, so the old scroll offset would only hide the open viewers under the header
        history.scrollRestoration = 'manual'
    </script>
</body>
</html>
""")


@dataclass
class ResultsFile:
    """One results file below the served folder, with what its row on the page shows.

    ``folder`` is the folder it is listed under, relative to the served one: the folder holding its run folder,
    or the folder holding the file when that is not a run folder. ``adapter`` names the pipeline of results an
    adapter opens, empty for masknmf's.
    """

    path: str
    folder: str
    run: str
    stages: list[str]
    num_curated: int
    num_bytes: int
    time_modified: float
    adapter: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def is_curated(self) -> bool:
        return self.path.endswith(CURATED_SUFFIX)

    @property
    def time_run(self) -> float:
        """When the run started, from its folder's name; the file's modification time for any other folder."""
        match = PATTERN_RUN.match(self.run)
        if match is None:
            return self.time_modified
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").timestamp()

    @property
    def pipeline(self) -> str:
        match = PATTERN_RUN.match(self.run)
        return self.adapter or ("" if match is None else match.group(2))


@dataclass
class ViewerProcess:
    """One results file open in a ``masknmf serve FILE`` process of its own, listening on ``port`` behind ``key``."""

    id: int
    path: str
    port: int
    key: str
    process: subprocess.Popen
    filepath_log: str
    time_started: float

    @property
    def status(self) -> str:
        """``opening`` until the viewer listens on its port, then ``ready``; ``exited <code>`` once the process ends."""
        code = self.process.poll()
        if code is not None:
            return f"exited {code}"
        return "ready" if port_listening(port=self.port) else "opening"

    @property
    def num_bytes(self) -> int:
        """The process's resident memory; 0 once it has exited."""
        try:
            return psutil.Process(self.process.pid).memory_info().rss
        except psutil.NoSuchProcess:
            return 0

    def last_line(self) -> str:
        """The last line the process wrote to its log, which names the error of one that exited."""
        lines = [line for line in Path(self.filepath_log).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        return lines[-1] if lines else ""


def find_results(folder: str | os.PathLike) -> list[ResultsFile]:
    """
    Find the masknmf results files below a folder, one per results file: its newest curated file when it has one;
    and the files of other pipelines' results an adapter opens, by the adapters' file patterns.

    Not entered: hidden folders, folders starting with ``_`` (archives), zarr and n5 stores (by name or by a
    zarr's own marker files), ``node_modules``, ``__pycache__``, ``site-packages`` and FUSE mounts such as sshfs,
    where a sleeping remote blocks the listing. Symlinked folders are not followed.

    Parameters
    ----------
    folder : str | os.PathLike
        The folder to search.

    Returns
    -------
    list[ResultsFile]
        Sorted by folder, then newest run first.
    """
    folder = Path(folder).expanduser().resolve()
    mounts_fuse = fuse_mounts()
    patterns_adapter = adapter_patterns()
    paths = []
    paths_adapter = []
    for dirpath, dirnames, filenames in os.walk(folder):
        if MARKERS_ZARR.intersection(filenames):
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith((".", "_"))
            and not name.endswith(SUFFIXES_STORE)
            and name not in NAMES_SKIPPED
            and os.path.join(dirpath, name) not in mounts_fuse
        ]
        paths += [
            os.path.join(dirpath, name)
            for name in filenames
            if any(fnmatch.fnmatch(name, pattern) for pattern in PATTERNS_RESULTS)
        ]
        paths_adapter += [
            os.path.join(dirpath, name)
            for name in filenames
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns_adapter)
        ]
    num_curated = {}
    for path in paths:
        key = (os.path.dirname(path), results_stem(path))
        num_curated[key] = num_curated.get(key, 0) + path.endswith(CURATED_SUFFIX)

    files = []
    for path in latest_results(paths):
        stages = stage_groups(path)
        if len(stages) == 0:
            continue
        folder_run = Path(path).parent
        is_run = PATTERN_RUN.match(folder_run.name) is not None
        folder_listed = folder_run.parent if is_run and folder_run != folder else folder_run
        stat = os.stat(path)
        files.append(
            ResultsFile(
                path=path,
                folder=folder_listed.relative_to(folder).as_posix(),
                run=folder_run.name,
                stages=[stage_label(stage) for stage in stages],
                num_curated=num_curated[(os.path.dirname(path), results_stem(path))],
                num_bytes=stat.st_size,
                time_modified=stat.st_mtime,
            )
        )
    for path in paths_adapter:
        adapter = adapter_for(path)
        if adapter is None:
            continue
        stat = os.stat(path)
        files.append(
            ResultsFile(
                path=path,
                folder=Path(path).parent.relative_to(folder).as_posix(),
                run=Path(path).parent.name,
                stages=[adapter.describe(Path(path))],
                num_curated=0,
                num_bytes=stat.st_size,
                time_modified=stat.st_mtime,
                adapter=adapter.pipeline,
            )
        )
    return sorted(files, key=lambda file: (file.folder, -file.time_run, file.name))


def fuse_mounts() -> set[str]:
    """Returns the mount points of FUSE filesystems (sshfs, rclone and the like)."""
    return {partition.mountpoint for partition in psutil.disk_partitions(all=True) if "fuse" in partition.fstype}


def stage_label(stage: str) -> str:
    """A stage group's name as the page shows it: ``RigidRegistrationArray`` is registration, ``global/DemixingResults`` global/demixing."""
    prefix, _, name = stage.rpartition("/")
    label = next(word for word in ("registration", "compression", "demixing") if word in name.lower())
    return f"{prefix}/{label}" if prefix else label


def format_bytes(num_bytes: int) -> str:
    """A file size in the largest unit that keeps it at or above 1."""
    size = float(num_bytes)
    for unit in ("B", "kB", "MB", "GB"):
        if size < 1000:
            return f"{size:.0f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"


def folder_tree(files: list[ResultsFile]) -> dict:
    """
    Nest files by their folder, merging a folder whose only content is one folder into it (``a/b``).

    Parameters
    ----------
    files : list[ResultsFile]
        What ``find_results`` returned.

    Returns
    -------
    dict
        ``{"folders": {name: subtree}, "files": [...]}``; the served folder's own files are at the top.
    """
    root = {"folders": {}, "files": []}
    for file in files:
        node = root
        for part in Path(file.folder).parts:
            node = node["folders"].setdefault(part, {"folders": {}, "files": []})
        node["files"].append(file)
    return merge_chains(root)


def merge_chains(node: dict) -> dict:
    """Returns ``node`` with every child folder that holds only one folder merged with it."""
    folders = {}
    for name, child in node["folders"].items():
        child = merge_chains(child)
        # NOTE: one merge suffices; merge_chains(child) has already merged child's own children
        if len(child["folders"]) == 1 and len(child["files"]) == 0:
            name_sub, child = next(iter(child["folders"].items()))
            name = f"{name}/{name_sub}"
        folders[name] = child
    return {"folders": folders, "files": node["files"]}


def files_below(node: dict) -> list[ResultsFile]:
    """Returns the files of a tree node and of every folder below it."""
    return node["files"] + [file for child in node["folders"].values() for file in files_below(child)]


def natural_key(name: str) -> list:
    """Sorts ``epochs_10`` before ``epochs_100``: digit runs compare as numbers."""
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def format_time(timestamp: float) -> str:
    """A timestamp as runhub shows it: ``Nov 14 16:20`` within a week of now, else ``2026-11-01``."""
    time_then = datetime.fromtimestamp(timestamp)
    if (datetime.now() - time_then).days < NUM_DAYS_WITH_TIME:
        return time_then.strftime("%b %d %H:%M")
    return time_then.strftime("%Y-%m-%d")


def rows_html(files: list[ResultsFile], paths_open: set[str], with_folder: bool) -> str:
    """A table with one row per file, a file with a viewer open marked; ``with_folder`` adds the folder each is listed under."""
    head_folder = "<th class='folder'>folder</th>" if with_folder else ""
    rows = []
    for file in files:
        curated = f"curated ({file.num_curated})" if file.is_curated else ""
        text = " ".join([file.folder, file.run, file.pipeline, file.name, *file.stages, curated]).lower()
        cell_folder = f"<td class='folder'>{html.escape(file.folder)}</td>" if with_folder else ""
        chip_open = "<span class='chip on small'>open</span>" if file.path in paths_open else ""
        rows.append(
            f"<tr class='file{' open' if chip_open else ''}' data-text='{html.escape(text, quote=True)}' "
            f"data-href='open?path={quote(file.path, safe='')}' title='{html.escape(file.path, quote=True)}'>"
            f"{cell_folder}<td class='when'>{format_time(file.time_run)}</td>"
            f"<td class='pipeline'>{html.escape(file.pipeline)}</td>"
            f"<td class='name'>{html.escape(file.name)}{chip_open}</td>"
            f"<td class='stages'>{html.escape(', '.join(file.stages))}</td><td class='curated'>{curated}</td>"
            f"<td class='num'>{format_bytes(file.num_bytes)}</td></tr>"
        )
    head = (
        f"<tr>{head_folder}<th class='when'>run</th><th class='pipeline'>pipeline</th><th>file</th>"
        "<th class='stages'>stages</th><th class='curated'></th><th class='num'>size</th></tr>"
    )
    return f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


def folder_meta(files: list[ResultsFile], paths_open: set[str]) -> str:
    """A folder's summary line: how many results, curated and open, its pipelines and its newest change."""
    counts_pipeline = Counter(file.pipeline for file in files if file.pipeline)
    parts = [f"{len(files)} result{'s' if len(files) != 1 else ''}"]
    num_curated = sum(file.is_curated for file in files)
    if num_curated:
        parts.append(f"{num_curated} curated")
    if counts_pipeline:
        parts.append(", ".join(name for name, _ in counts_pipeline.most_common(NUM_PIPELINES_SHOWN)))
    parts.append(f"latest {format_time(max(file.time_modified for file in files))}")
    num_open = sum(file.path in paths_open for file in files)
    chip = f"<span class='chip on small'>{num_open} open</span>" if num_open else ""
    return f"<span class='meta'>{html.escape(' · '.join(parts))}</span>{chip}"


def folder_html(name: str, node: dict, paths_open: set[str], depth: int) -> str:
    """
    One folder of the tree, collapsed: a header with its file count, pipelines and newest change, then its runs
    and subfolders.

    Parameters
    ----------
    name : str
        The folder's name, or ``a/b`` for merged folders.
    node : dict
        Its subtree from ``folder_tree``.
    paths_open : set[str]
        The files with a viewer open; their rows and the folders above them are marked.
    depth : int
        0 for a folder of the served one, which shows as a card.

    Returns
    -------
    str
        The html.
    """
    below = files_below(node)
    files = sorted(node["files"], key=lambda file: (-file.time_run, file.name))
    table = rows_html(files, paths_open, with_folder=False) if files else ""
    children = "".join(
        folder_html(name_child, child, paths_open, depth=depth + 1)
        for name_child, child in sorted(node["folders"].items(), key=lambda item: natural_key(item[0]))
    )
    newest = max(file.time_modified for file in below)
    card = " class='surface'" if depth == 0 else ""
    return (
        f"<details{card} data-name='{html.escape(name, quote=True)}' data-time='{newest:.0f}'>"
        f"<summary><b>{html.escape(name)}</b>{folder_meta(below, paths_open)}</summary>"
        f"<div class='children'>{table}{children}</div></details>"
    )


def viewers_html(viewers: list[ViewerProcess], folder: str) -> str:
    """The open viewers, each with its status and memory, a link back to it and a button that stops its process."""
    if len(viewers) == 0:
        return ""
    rows = []
    for viewer in viewers:
        status = viewer.status
        is_exited = status.startswith("exited")
        place = Path(viewer.path).parent.relative_to(folder).as_posix()
        cell_place = (
            f"<td class='reason' title='log: {html.escape(viewer.filepath_log, quote=True)}'>{html.escape(viewer.last_line())}</td>"
            if is_exited
            else f"<td class='folder'>{html.escape(place)}</td>"
        )
        link = f"<a class='chip' href='v/{viewer.id}/'>view</a>" if not is_exited else ""
        rows.append(
            f"<tr data-viewer='{viewer.id}'><td class='name' title='{html.escape(viewer.path, quote=True)}'>"
            f"<a href='v/{viewer.id}/'>{html.escape(Path(viewer.path).name)}</a></td>{cell_place}"
            f"<td class='status {status.split()[0]}'>{status}</td><td class='when'>{format_time(viewer.time_started)}</td>"
            f"<td class='num memory'>{format_bytes(viewer.num_bytes) if not is_exited else ''}</td>"
            f"<td class='actions'>{link}<form method='post' action='stop?id={viewer.id}'>"
            f"<button class='chip'>{'remove' if is_exited else 'stop process'}</button></form></td></tr>"
        )
    head = (
        "<tr><th>file</th><th>folder</th><th class='status'>status</th><th class='when'>opened</th>"
        "<th class='num'>memory</th><th class='actions'></th></tr>"
    )
    return (
        f"<section id='viewers' class='surface'><div class='title'><b>open</b>"
        f"<span class='meta'> · {len(viewers)} viewer process{'es' if len(viewers) != 1 else ''}</span></div>"
        f"<div class='scroll'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def picker_html(folder: str, files: list[ResultsFile], viewers: list[ViewerProcess] = (), error: str = "") -> str:
    """
    The page listing the results files: the open viewers, the most recently changed files, then all of them as a
    collapsed folder tree.

    Parameters
    ----------
    folder : str
        The served folder, shown in the header.
    files : list[ResultsFile]
        What ``find_results`` returned.
    viewers : list[ViewerProcess]
        The viewer processes this server started and has not stopped.
    error : str
        Why the last pick did not open.

    Returns
    -------
    str
        The html.
    """
    paths_open = {viewer.path for viewer in viewers if viewer.process.poll() is None}
    tree = folder_tree(files)
    recent = ""
    if len(files) > NUM_RECENT:
        newest = sorted(files, key=lambda file: -file.time_modified)[:NUM_RECENT]
        recent = (
            f"<details id='recent' class='surface' open><summary><b>recent</b>"
            f"<span class='meta'>the {NUM_RECENT} files changed last</span></summary>{rows_html(newest, paths_open, with_folder=True)}</details>"
        )
    blocks = [f"<div class='surface'>{rows_html(tree['files'], paths_open, with_folder=False)}</div>"] if tree["files"] else []
    blocks += [
        folder_html(name, child, paths_open, depth=0)
        for name, child in sorted(tree["folders"].items(), key=lambda item: natural_key(item[0]))
    ]
    controls = (
        "<div class='controls'><span class='meta'>sort by&nbsp;</span>"
        "<button class='chip on' data-sort='name'>name</button><button class='chip' data-sort='newest'>newest</button>"
        "<span class='meta'>&nbsp;·&nbsp;</span><button class='chip' data-expand='1'>expand all</button>"
        "<button class='chip' data-expand='0'>collapse all</button></div>"
    )
    return _PICKER.substitute(
        title=f"{Path(folder).name} - masknmf",
        root=html.escape(folder),
        count=f"{len(files)} files",
        error=f"<div class='error'>{html.escape(error)}</div>" if error else "",
        viewers=viewers_html(list(viewers), folder),
        recent=recent,
        tree=(
            f"<h2>folders</h2>{controls}<section id='tree'>{''.join(blocks)}</section>"
            if files
            else "<p>No masknmf results files below this folder.</p>"
        ),
    )


def free_port() -> int:
    """A localhost port nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_listening(port: int) -> bool:
    """Whether something accepts connections on a localhost port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def stop_process(process: subprocess.Popen) -> None:
    """Ask a process to stop, and kill it when it has not within ``SECONDS_STOP``."""
    process.terminate()
    try:
        process.wait(timeout=SECONDS_STOP)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class PickerServer:
    """
    The results files below a folder at ``files``, and a viewer process per picked file at ``v/<id>/``.

    Parameters
    ----------
    folder : str | os.PathLike
        The folder whose results files are listed; only those can be opened.
    options : list[str]
        The ``masknmf serve`` options every viewer process takes, after its file, host and port.
    launch_viewer : Callable
        Starts ``masknmf <argv>``: ``launch_viewer(argv=, filepath_log=, env_extra=, detach=)`` returns its ``subprocess.Popen``.
    host, port : str, int
        Where ``serve`` listens.
    """

    def __init__(
        self, folder: str | os.PathLike, options: list[str], launch_viewer: Callable, *, host: str, port: int
    ):
        self.folder = str(Path(folder).expanduser().resolve())
        self.options = list(options)
        self.launch_viewer = launch_viewer
        self.host = str(host)
        self.port = int(port)
        self.viewers: dict[int, ViewerProcess] = {}
        self.dir_logs = Path(tempfile.gettempdir()) / "masknmf-serve"
        self.resources = {**resources, "renderview-client.js": ("text/javascript", client_js())}
        self._num_started = 0
        self._server = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def open(self, path: str) -> ViewerProcess:
        """The viewer process of the results file at ``path``, started when none is running; it has to be one ``find_results`` lists."""
        if path not in [file.path for file in find_results(folder=self.folder)]:
            raise ValueError(f"{path} is not a results file below {self.folder}")
        for viewer in list(self.viewers.values()):
            if viewer.path != path:
                continue
            if viewer.process.poll() is None:
                return viewer
            del self.viewers[viewer.id]
        self._num_started += 1
        self.dir_logs.mkdir(exist_ok=True)
        filepath_log = self.dir_logs / f"{datetime.now():%Y%m%dT%H%M%S}_{self._num_started}_{Path(path).stem}.log"
        port, key = free_port(), secrets.token_urlsafe(16)
        # NOTE: not detached, so closing this terminal or ctrl+c here stops the viewers with this server
        process = self.launch_viewer(
            argv=["serve", path, "--host", "127.0.0.1", "--port", str(port), *self.options],
            filepath_log=filepath_log,
            env_extra={SESSION_KEY_ENV: key},
            detach=False,
        )
        viewer = ViewerProcess(
            id=self._num_started, path=path, port=port, key=key, process=process, filepath_log=str(filepath_log), time_started=time.time()
        )
        self.viewers[viewer.id] = viewer
        return viewer

    def stop_viewer(self, id_viewer: int) -> None:
        """Stop a viewer's process and drop it from the list."""
        stop_process(self.viewers.pop(id_viewer).process)

    def page(self, error: str = "") -> str:
        """Returns the files page as it stands now; ``error`` says why the last pick did not open."""
        return picker_html(self.folder, find_results(self.folder), list(self.viewers.values()), error)

    async def asgi(self, scope, receive, send):
        """
        ``files`` lists, ``open?path=`` starts or finds a viewer and sends the browser to it, ``stop?id=`` (a post)
        stops one, ``viewers.json`` gives their status, and ``v/<id>/`` is a viewer's page, resources and websocket.
        """
        if scope["type"] == "lifespan":
            await self.lifespan(receive, send)
            return
        match = PATTERN_VIEWER.search(scope["path"])
        viewer = None if match is None else self.viewers.get(int(match.group(1)))
        if scope["type"] == "websocket":
            if viewer is None:
                await send({"type": "websocket.close", "code": 1008})
                return
            await relay(viewer, receive, send)
            return
        if match is not None:
            await self.respond_viewer(send, viewer, match.group(2))
            return
        name = scope["path"].rsplit("/", 1)[-1]
        query = parse_qs(scope.get("query_string", b"").decode())
        if name == "open":
            path = query.get("path", [""])[0]
            try:
                viewer = self.open(path)
            except (OSError, ValueError) as e:
                await respond(send, self.page(f"{path}: {e}"))
                return
            await redirect(send, f"v/{viewer.id}/")
        elif name == "stop" and scope["method"] == "POST":
            id_viewer = int(query.get("id", ["0"])[0])
            if id_viewer in self.viewers:
                await asyncio.to_thread(self.stop_viewer, id_viewer)
            await redirect(send, "files")
        elif name == "viewers.json":
            states = [
                {"id": viewer.id, "status": viewer.status, "memory": format_bytes(viewer.num_bytes) if viewer.process.poll() is None else ""}
                for viewer in self.viewers.values()
            ]
            await respond(send, json.dumps(states), content_type=b"application/json")
        elif name == "files":
            await respond(send, self.page())
        elif name == "":
            await redirect(send, "files")
        else:
            await respond(send, "Not Found", status=404, content_type=b"text/plain")

    async def lifespan(self, receive, send) -> None:
        """Stops every viewer process when uvicorn shuts down."""
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                # NOTE: uvicorn re-raises the signal that stopped it once it has shut down, so a finally after run() never runs
                await asyncio.to_thread(self.close)
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def respond_viewer(self, send, viewer: ViewerProcess | None, name: str) -> None:
        """A viewer's page, served here so it shows while the process still opens the file, or one of its resources."""
        if viewer is None and name == "status.json":
            await respond(send, json.dumps({"status": "stopped", "reason": "this viewer was stopped"}), content_type=b"application/json")
        elif viewer is None:
            await redirect(send, "../../files")
        elif name == "":
            filename = Path(viewer.path).name
            body = page_html(
                f"{filename} - masknmf", f"Opening {filename} ...", home="../../files", home_label="files", wait_url="status.json"
            )
            await respond(send, body)
        elif name == "status.json":
            status = viewer.status
            reason = f"the viewer exited: {viewer.last_line()}" if status.startswith("exited") else ""
            await respond(send, json.dumps({"status": status, "reason": reason}), content_type=b"application/json")
        elif name in self.resources:
            content_type, body = self.resources[name]
            await respond(send, body, content_type=content_type.encode())
        else:
            await respond(send, "Not Found", status=404, content_type=b"text/plain")

    def serve(self) -> None:
        """Run the server in this thread until it is interrupted; its shutdown stops every viewer process."""
        uvicorn = require("uvicorn")
        config = uvicorn.Config(
            self.asgi, host=self.host, port=self.port, log_level="warning", interface="asgi3", lifespan="on", ws_per_message_deflate=False
        )
        self._server = uvicorn.Server(config)
        try:
            self._server.run()
        finally:
            self.close()

    def close(self) -> None:
        for id_viewer in list(self.viewers):
            self.stop_viewer(id_viewer)


async def relay(viewer: ViewerProcess, receive, send) -> None:
    """Pass a browser's websocket through to its viewer process's, once the process listens; the browser's messages wait meanwhile."""
    client = require("websockets.asyncio.client")
    exceptions = require("websockets.exceptions")
    await send({"type": "websocket.accept"})
    upstream = None
    while upstream is None and viewer.process.poll() is None:
        try:
            # frames are already compressed and a png frame exceeds the default 1 MiB; the viewer's loop may block
            # for longer than a ping timeout while it computes
            upstream = await client.connect(
                f"ws://127.0.0.1:{viewer.port}/",
                additional_headers={SESSION_KEY_HEADER: viewer.key},
                compression=None,
                max_size=None,
                ping_interval=None,
                proxy=None,
            )
        except OSError:
            await asyncio.sleep(0.25)
    if upstream is None:
        reason = f"the viewer exited: {viewer.last_line()}".encode()[:120].decode(errors="ignore")
        await send({"type": "websocket.close", "code": 1011, "reason": reason})
        return
    to_viewer = asyncio.create_task(browser_to_viewer(receive, upstream, exceptions.ConnectionClosed))
    to_browser = asyncio.create_task(viewer_to_browser(upstream, send, exceptions.ConnectionClosed))
    done, pending = await asyncio.wait({to_viewer, to_browser}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await upstream.close()
    if to_browser in done and to_browser.result():
        await send({"type": "websocket.close", "code": 1001, "reason": "the viewer stopped"})


async def browser_to_viewer(receive, upstream, closed: type[Exception]) -> None:
    """Forward the browser's messages until it disconnects or the viewer closes."""
    while True:
        message = await receive()
        if message["type"] == "websocket.disconnect":
            return
        # NOTE: uvicorn's first message is websocket.connect, still queued since the accept went out unasked
        if message["type"] != "websocket.receive":
            continue
        try:
            await upstream.send(message["text"] if message.get("text") is not None else message["bytes"])
        except closed:
            return


async def viewer_to_browser(upstream, send, closed: type[Exception]) -> bool:
    """Forward the viewer's messages; True when the viewer closed, False when the browser was gone."""
    try:
        async for message in upstream:
            await send({"type": "websocket.send", "text": message} if isinstance(message, str) else {"type": "websocket.send", "bytes": message})
    except closed:
        return True
    except OSError:
        return False
    return True


async def respond(send, body: str | bytes, status: int = 200, content_type: bytes = b"text/html; charset=utf-8") -> None:
    """Send one response, never cached, so going back to the files page shows the viewers open now."""
    headers = [(b"content-type", content_type), (b"cache-control", b"no-store")]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body.encode() if isinstance(body, str) else body})


async def redirect(send, location: str) -> None:
    """Send the browser to ``location``, relative to the page it asked for."""
    await send({"type": "http.response.start", "status": 303, "headers": [(b"location", location.encode())]})
    await send({"type": "http.response.body", "body": b""})
