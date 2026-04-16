"""Persistent MuJoCo viewer window for live robot design review.

Keeps a single viewer window open across design iterations. When the user
refines the robot, the old window closes and a new one opens immediately
with the updated model — no manual restart needed.

Physics runs continuously on a background thread so the user can interact
with the simulation (camera, drag-to-apply-forces) while chatting.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import NamedTuple

log = logging.getLogger(__name__)

_STOP_SENTINEL = object()


class _ShowRequest(NamedTuple):
    mjcf_xml: str        # XML string (empty string if mjcf_path is set)
    title: str
    mjcf_path: str = ""  # File path — used instead of mjcf_xml for catalog robots


class SessionViewer:
    """Persistent MuJoCo viewer across robot design iterations.

    Usage::

        viewer = SessionViewer()
        viewer.show(mjcf_xml, title="cart_pole v1")  # opens window
        # ... user chats, Claude refines ...
        viewer.show(new_xml, title="cart_pole v2")   # window refreshes
        viewer.close()                                # on app exit
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def show(self, mjcf_xml: str, title: str = "Robo Garden — Design Preview") -> None:
        """Open or refresh the viewer with a new robot model (from XML string)."""
        self._enqueue(_ShowRequest(mjcf_xml=mjcf_xml, title=title))

    def show_path(self, mjcf_path: str, title: str = "Robo Garden — Design Preview") -> None:
        """Open or refresh the viewer with a robot model loaded from a file path.

        Use this for catalog robots where mesh assets live next to the MJCF file.
        """
        self._enqueue(_ShowRequest(mjcf_xml="", title=title, mjcf_path=mjcf_path))

    def _enqueue(self, req: _ShowRequest) -> None:
        """Replace any pending (unseen) update with the latest, then start thread."""
        # Replace any pending (unseen) update with the latest
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(req)

        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._viewer_loop, daemon=True, name="mujoco-viewer"
                )
                self._thread.start()
                log.info("SessionViewer: background thread started")


    def close(self) -> None:
        """Signal the viewer thread to exit. Returns immediately."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(_STOP_SENTINEL)
        log.info("SessionViewer: stop requested")

    def _viewer_loop(self) -> None:
        """Background thread: dequeue robots and open a viewer for each."""
        import mujoco
        import mujoco.viewer as mjviewer

        while True:
            # Block until a new robot (or stop) arrives
            item = self._queue.get()

            if item is _STOP_SENTINEL:
                log.info("SessionViewer: stop received, exiting loop")
                return

            req: _ShowRequest = item

            try:
                if req.mjcf_path:
                    model = mujoco.MjModel.from_xml_path(req.mjcf_path)
                else:
                    model = mujoco.MjModel.from_xml_string(req.mjcf_xml)
            except Exception as exc:
                log.warning(f"SessionViewer: failed to compile MJCF — {exc}")
                continue

            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            log.info(f"SessionViewer: opening viewer — {req.title}")

            try:
                with mjviewer.launch_passive(model, data) as handle:
                    handle.cam.distance = max(1.0, model.stat.extent * 3.0)
                    while handle.is_running():
                        # Check for a new robot or stop signal (non-blocking)
                        try:
                            next_item = self._queue.get_nowait()
                            if next_item is _STOP_SENTINEL:
                                log.info("SessionViewer: stop while viewing, closing")
                                return
                            # New robot queued — put it back and break to restart
                            try:
                                self._queue.get_nowait()  # drain any double-queued
                            except queue.Empty:
                                pass
                            self._queue.put(next_item)
                            break
                        except queue.Empty:
                            pass

                        mujoco.mj_step(model, data)
                        handle.sync()
                        # ~500Hz physics, viewer syncs at its own display rate
            except Exception as exc:
                log.warning(f"SessionViewer: viewer error — {exc}")
                # Brief pause before retrying to avoid tight error loop
                time.sleep(0.5)
