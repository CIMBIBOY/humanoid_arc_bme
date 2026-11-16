"""The rates the simulated robot runs at. Scene generators read PHYSICS_HZ from here instead of choosing their own.

PHYSICS_HZ and CONTROL_HZ are those of the locomotion policy in use. Today that is the STAND-IN policy (NVIDIA AGILE lower body,
trained in Isaac Lab at 200 Hz physics with decimation 4: IsaacLab `locomanipulation_g1_env_cfg.py`, `sim.dt = 1/200`,
`decimation = 4`). The GR00T/SONIC deploy is documented at about 50 Hz control (docs/plan/groot-behind-s1.md); the physics rate it
needs in simulation is UNVERIFIED until the deploy itself runs here.
"""

PHYSICS_HZ = 200          # from the stand-in policy's training config (see above)
CONTROL_HZ = 50           # policy rate = PHYSICS_HZ / 4
STACK_HZ = 20             # onboard stack tick (AutonomyLoop default, GrootPlannerController.send_hz)
assert PHYSICS_HZ % CONTROL_HZ == 0 and PHYSICS_HZ % STACK_HZ == 0
