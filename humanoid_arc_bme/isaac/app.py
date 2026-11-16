"""Start Kit for a headless run: log on a chosen volume (errors only) and a bounded number of worker threads.

Kit starts one task thread per core of the machine (24 here) however few cores the process may use; on a shared machine those
threads show up in the load average even when a CPU quota throttles them. $HUMANOID_ARC_KIT_THREADS (default 4) caps them."""

from __future__ import annotations

import os
from pathlib import Path


def make_app(name: str = "kit", headless: bool = True, extra: tuple = (), lidar: bool = False):
    """SimulationApp with the log at $HUMANOID_ARC_KIT_LOGS/<name>_<pid>.log (default ~/.cache/humanoid_arc_bme/kit_logs).
    `lidar=True` (any run with simulated sensors) makes the package's RTX lidar profiles visible and renders synchronously. Import NOTHING that pulls numpy or
    scipy before calling this: Kit must load its own copies first (a scipy import chain broke otherwise)."""
    if lidar:
        # Synchronous rendering: with Kit's default (asynchronous) an annotator read after an update returns the previous or a
        # repeated frame; for a sensor that must be stamped with simulation time that is a wrong measurement (measured: lidar slices
        # arrived twice or not at all).
        extra = (*extra, f'--/app/sensors/nv/lidar/profileBaseFolder=["{Path(__file__).resolve().parent / "lidar_profiles"}/"]',
                 "--/app/asyncRendering=false", "--/app/asyncRenderingLowLatency=false", "--/app/hydraEngine/waitIdle=true")
    from isaacsim import SimulationApp

    logs = Path(os.environ.get("HUMANOID_ARC_KIT_LOGS", Path.home() / ".cache" / "humanoid_arc_bme" / "kit_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    app = SimulationApp({"headless": headless, "extra_args": [f"--/log/file={logs}/{name}_{os.getpid()}.log", "--/log/fileLogLevel=error", *thread_args(), *extra]})
    fast_physics_io()
    return app


def fast_physics_io() -> None:
    """Stop PhysX writing every body's transform and velocity back to USD each step; the renderer reads them from Fabric instead.
    Measured with the G1 alone: 3.3 ms -> 0.39 ms per physics step. Everything in this package reads poses through the tensor API,
    never from USD; scripts/isaac/g1_sensor_gate.py proves the cameras and the lidar still follow the moving robot."""
    import carb
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.physx.fabric")
    st = carb.settings.get_settings()
    for key in ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd", "/physics/updateForceSensorsToUsd", "/physics/updateParticlesToUsd"):
        st.set(key, False)
    st.set("/physics/fabricEnabled", True)


def thread_args() -> list[str]:
    n = int(os.environ.get("HUMANOID_ARC_KIT_THREADS", "4"))
    return [f"--/plugins/carb.tasking.plugin/threadCount={n}", f"--/persistent/physics/numThreads={max(1, n - 1)}", f"--/physics/numThreads={max(1, n - 1)}"]
