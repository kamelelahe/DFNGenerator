from math import pi
import numpy as np


def Pwf_line_source_solution(inputs, t):
    """
    Calculate flowing bottom-hole pressure (Pwf) using line source solution.

    Parameters:
    Q: Flow rate (m^3/day)
    mu: Fluid viscosity (Pa.s)
    k: Permeability (mD)
    h: Reservoir thickness (m)
    re: External radius (m)
    rw: Wellbore radius (m)
    t: Time (days)
    pi: Initial reservoir pressure (bar)

    Returns:
    Pwf: Flowing bottom-hole pressure (bar)
    """
    # Extract inputs
    Q = inputs['Q']
    mu = inputs['mu']
    k = inputs['k'] * 9.869233e-16  # convert mD to m^2
    h = inputs['h']
    re = inputs['re']
    rw = inputs['rw']
    Pi = inputs['Pi'] * 1e5  # convert Bar tp Pa
    Ct = inputs['Ct']
    skin = inputs['skin']

    t = t * 24 * 3600  # convert day to seceont
    Q = Q / (3600 * 24)  # convert to per second
    # Hydraulic diffusivity
    HydraulicDiffusivity = k / (Ct * mu)
    t0 = 25 * rw ** 2 / HydraulicDiffusivity

    # dimesntionless time
    tD = t * (HydraulicDiffusivity / (rw ** 2))

    slope = (Q * mu) / (4 * pi * k * h)

    Pwf_end = Pi - slope * (np.log(tD) + 0.809 + 2 * skin)

    return Pwf_end / 1e5  # convert Pa to bar