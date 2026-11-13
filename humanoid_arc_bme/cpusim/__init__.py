"""CPU simulator for the onboard stack: a ray-cast RGB-D world, a kinematic robot and a closed-loop session.

Fast (no Isaac, no GPU) and deterministic, so the whole chain (text command -> perception -> task executor -> pipeline
runner -> GR00T controller -> arm kinematics) runs in unit tests and on a laptop. It is a KINEMATIC check of the logic and
the wiring: grasp physics (contact, friction) is checked in Isaac Lab with the pick-place task, not here.
"""
