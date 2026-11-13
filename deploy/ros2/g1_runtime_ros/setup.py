"""ament_python package g1_runtime_ros. The onboard packages (humanoid_arc_bme.onboard.runtime, .api, .nav, .deploy) are the
repository's own `humanoid_arc_bme/` directory, reached through the symlink next to this file, so ROS and plain-python runs use the same code.
Build: cd <repo> && colcon build --base-paths deploy/ros2 (see deploy/scripts/build.sh)."""

from glob import glob

from setuptools import setup

PKG = "g1_runtime_ros"

setup(
    name=PKG,
    version="0.2.0",
    packages=[PKG, "humanoid_arc_bme", "humanoid_arc_bme.onboard", "humanoid_arc_bme.onboard.runtime", "humanoid_arc_bme.onboard.runtime.io", "humanoid_arc_bme.onboard.runtime.web", "humanoid_arc_bme.onboard.api", "humanoid_arc_bme.onboard.nav", "humanoid_arc_bme.onboard.deploy"],
    package_data={"humanoid_arc_bme.onboard.runtime.web": ["static/*"], "humanoid_arc_bme.onboard.deploy": ["*.sh"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PKG]),
        ("share/" + PKG, ["package.xml"]),
        ("share/" + PKG + "/launch", glob("launch/*.launch.py")),
        ("share/" + PKG + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Czimber Mark",
    description="ROS 2 wrapper for the humanoid_arc_bme.onboard.runtime operator runtime and web UI",
    license="MIT",
    entry_points={"console_scripts": [
        "g1_runtime_node = g1_runtime_ros.node:main",
        "g1_runtime = humanoid_arc_bme.onboard.runtime.cli:main",
    ]},
)
