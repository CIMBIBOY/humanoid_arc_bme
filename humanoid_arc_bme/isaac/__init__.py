"""Isaac Sim backend of the onboard stack: the simulated G1 behind the same interfaces the robot uses.

Nothing here imports Isaac at package import time (`rates`, `mounts` and the DDS / deploy stand-in modules are plain Python);
the modules that need Kit import it inside their functions. Scenes arrive as a manifest (`scene.json`) and a USD path: this
package knows nothing about how they were made.
"""
