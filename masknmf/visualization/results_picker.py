"""Serve a folder of results files to a browser: a page listing them, and the demixing viewer on the one picked.

``masknmf serve FOLDER`` runs this. ``files`` lists every results file below the folder, and every other pipeline's
results an adapter finds by file name (miniALI's ALI_Result_new.mat, suite2p's stat.npy), as a tree of folders that
starts collapsed, the runs in each newest first, under the most recently changed files; a filter box narrows it and a
click or Enter opens a file in the one viewer this process holds, at ``/``. The first pick builds the viewer, later
picks load into it in place.
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import os
import re
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
from masknmf.visualization.canvas_server import BrowserCanvas, CanvasServer, page_html

__all__ = ["PickerServer", "ResultsFile", "find_results", "picker_html"]

PATTERNS_RESULTS = ("*results*.hdf5", "*results*.h5")
# a run folder is named <timestamp>_<pipeline>
PATTERN_RUN = re.compile(r"^(\d{8}T\d{6})_(.+)$")
# folders of chunked arrays, code or caches: thousands of entries and never a results file
SUFFIXES_STORE = (".zarr", ".n5")
NAMES_SKIPPED = {"node_modules", "__pycache__", "site-packages"}
MARKERS_ZARR = {".zgroup", ".zarray", "zarr.json"}
NUM_RECENT = 8
NUM_PIPELINES_SHOWN = 3
# runhub's dates: day and minute within a week, else the day
NUM_DAYS_WITH_TIME = 7

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
        $viewer
        <br><input id='filter' placeholder='type to filter: folder, run, pipeline, stage; arrows and Enter open' autofocus>
    </header>
    $error
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


def picker_html(folder: str, files: list[ResultsFile], path_open: str | None = None, error: str = "") -> str:
    """
    The page listing the results files: the most recently changed on top, then all of them as a collapsed folder tree.

    Parameters
    ----------
    folder : str
        The served folder, shown in the header.
    files : list[ResultsFile]
        What ``find_results`` returned.
    path_open : str, optional
        The file the viewer holds; its row is marked and the header links to the viewer.
    error : str
        Why the last pick did not open.

    Returns
    -------
    str
        The html.
    """
    paths_open = set() if path_open is None else {path_open}
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
        viewer="" if path_open is None else f"<a class='chip' href='./'>viewer: {html.escape(Path(path_open).name)}</a>",
        recent=recent,
        tree=(
            f"<h2>folders</h2>{controls}<section id='tree'>{''.join(blocks)}</section>"
            if files
            else "<p>No masknmf results files below this folder.</p>"
        ),
    )


class PickedViewer:
    """What a CanvasServer shows and closes: nothing until a file is picked, then that file's demixing viewer."""

    def __init__(self):
        self.viewer = None

    @property
    def figure(self):
        return self.viewer.figure

    def show(self):
        if self.viewer is not None:
            self.viewer.show()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class PickerServer(CanvasServer):
    """
    The results files below a folder at ``files``, and the demixing viewer on the picked one at ``/``.

    One viewer, so one file open at a time: a pick replaces what every connected browser sees.

    Parameters
    ----------
    folder : str | os.PathLike
        The folder whose results files are listed; only those can be opened.
    open_viewer : Callable
        Builds the viewer of the first pick: ``open_viewer(args=, filepath_results=, names_present=, canvas=)``.
    args : argparse.Namespace
        The parsed ``serve`` options, handed to ``open_viewer``.
    host, port : str, int
        Where ``serve`` listens.
    """

    def __init__(self, folder: str | os.PathLike, open_viewer: Callable, args: argparse.Namespace, *, host: str, port: int):
        self.folder = str(Path(folder).expanduser().resolve())
        self.open_viewer = open_viewer
        self.args = args
        self.path_open = None
        self._canvas = BrowserCanvas(size=(1600, 1000), max_fps=60.0)
        super().__init__(
            PickedViewer(), title=f"{Path(self.folder).name} - masknmf", waiting="Opening ...", host=host, port=port
        )

    @property
    def canvas(self):
        return self._canvas

    def open(self, path: str) -> None:
        """Show the results file at ``path`` in the viewer; it has to be one ``find_results`` lists."""
        if path not in [file.path for file in find_results(folder=self.folder)]:
            raise ValueError(f"{path} is not a results file below {self.folder}")
        if self.vis.viewer is None:
            adapter = adapter_for(path)
            names_present = stage_groups(path) if adapter is None else [adapter.describe(Path(path))]
            self.vis.viewer = self.open_viewer(
                args=self.args, filepath_results=path, names_present=names_present, canvas=self.canvas
            )
            self.vis.show()
        else:
            self.vis.viewer.load_results(path)
        self.path_open = path
        name = Path(path).name
        resources["index.html"] = "text/html", page_html(f"{name} - masknmf", f"Opening {name} ...", home="files", home_label="files")

    async def asgi(self, scope, receive, send):
        """The picker's pages, then the canvas: ``files`` lists, ``open?path=`` picks, ``/`` is the viewer."""
        if scope["type"] != "http":
            await super().asgi(scope, receive, send)
            return
        name = scope["path"].rsplit("/", 1)[-1]
        if name == "open":
            path = parse_qs(scope.get("query_string", b"").decode()).get("path", [""])[0]
            try:
                self.open(path)
            # NOTE: SystemExit is how the cli's checks end a command; here it ends the pick, not the server
            except (OSError, ValueError, SystemExit) as e:
                reason = str(e) if not isinstance(e, SystemExit) else "it could not be opened; the server's terminal says why"
                await respond(send, picker_html(self.folder, find_results(self.folder), self.path_open, f"{path}: {reason}"))
                return
            await redirect(send, "./")
        elif name == "files":
            await respond(send, picker_html(self.folder, find_results(self.folder), self.path_open))
        elif name == "" and self.path_open is None:
            await redirect(send, "files")
        else:
            await super().asgi(scope, receive, send)


async def respond(send, body: str) -> None:
    """Send one html page."""
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/html; charset=utf-8")]})
    await send({"type": "http.response.body", "body": body.encode()})


async def redirect(send, location: str) -> None:
    """Send the browser to ``location``, relative to the page it asked for."""
    await send({"type": "http.response.start", "status": 303, "headers": [(b"location", location.encode())]})
    await send({"type": "http.response.body", "body": b""})
