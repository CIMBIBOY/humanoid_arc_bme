"""Tests of the public package. Plain tests need only IsaacLab's python (gymnasium, torch).

Tests marked `isaac` launch a headless Isaac Sim app and only run with
--isaac, through the launcher (numpy overlay for Pink IK):

    ../IsaacLab/_isaac_sim/python.sh -m pytest tests              # fast, no Isaac
    scripts/launch.sh -m pytest tests --isaac [--robot_base fixed]        # + Isaac smoke test

The smoke test builds one env per process (a second env after env.close()
hangs on this install), so each base is its own run.
"""

import os
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--isaac", action="store_true", help="also run tests that launch Isaac Sim")
    parser.addoption("--robot_base", default="agile_legs", help="base plug-in the Isaac smoke test builds")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--isaac"):
        return
    skip = pytest.mark.skip(reason="needs --isaac (launches Isaac Sim)")
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip)


_app = None


@pytest.fixture(scope="session")
def isaac_app():
    """One headless app for the whole session (Kit can only start once per process)."""
    global _app
    if _app is None:
        import pinocchio  # noqa: F401 - must be imported before Kit loads its own copy
        from isaaclab.app import AppLauncher

        _app = AppLauncher(headless=True, enable_pinocchio=True).app
    return _app


@pytest.fixture(scope="session")
def robot_base(request):
    return request.config.getoption("--robot_base")


_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    # simulation_app.close() can hang on this install; exit hard once pytest has reported.
    if _app is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_status)
