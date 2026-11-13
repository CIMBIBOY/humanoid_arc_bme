"""Task layer: turns one typed operator command into base velocity and arm targets, tick by tick.

Stdlib + numpy. Nothing here actuates: the executor outputs a desired base command (the pipeline runner and S1 then
tighten it) and arm/hand joint targets (the controller then clamps and rate-limits them).
"""
