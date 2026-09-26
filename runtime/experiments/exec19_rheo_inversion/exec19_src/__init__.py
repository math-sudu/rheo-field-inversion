"""EXEC-19 rheology/softening field-inversion trial package.

Charter (PROTOCOL.md, frozen 2026-07-17): lambda-t viscous FE forward
(fedev_src condensed tier-C path, read-only import), in-package
viscous replay + forward-IFT Jacobian (AD-through-FE realization; the
fedev_src.diff tier-C boundary is NOT touched), Sulem lambda-frozen
two-zone closed-form anchors, two-layer identifiability protocol
(Fisher screen / profile confirmation), 9x32x6 synthetic recovery grid
with per-parameter release ladder, 935/915 field round.

Units convention of THIS package: MPa, metres, DAYS (viscosities in
MPa*day; doc272 GPa*h constants are converted at the scenario
boundary, 1 GPa*h = 1000/24 MPa*day).
"""
