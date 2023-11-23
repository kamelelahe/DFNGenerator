from math import pi
import numpy as np

def DimensionlessTime(inputs,time):
    tD=(inputs['k'] * 9.869233e-16 /(inputs['mu']*inputs['Ct']*(inputs['rw']**2)))*(time * 24 * 3600)

    return tD

def DimensionlessPressure(inputs,pressure):

    P_values = np.array(pressure)
    pD=(2*pi*inputs['k'] * 9.869233e-16  *inputs['h']/((inputs['Q'] / (3600 * 24))*inputs['mu']))*(inputs['Pi'] * 1e5-1e5*P_values)
    return pD