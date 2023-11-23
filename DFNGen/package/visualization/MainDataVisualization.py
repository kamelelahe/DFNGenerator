from LineSourceEq import Pwf_line_source_solution
from DimensionLess import DimensionlessTime, DimensionlessPressure
from VisualizedFromExcel import VisualizeFromExcel
from Visualize import Visualize
import numpy as np
outputDir = '04Result_SA'

#constant input parameters
inputs = {
    'h': 40,  # m
    're': 2000,  # m
    'Ct': 1e-8,  # 1/Pa
    'mu': 0.3774e-3,  # Pa.s (viscousity of water at 350K)
    'k':  2533.22 ,  # mD
    'Pi': 300,  # bar
    'rw': 0.15,  # m
    'Q': 2,  # m3/day
    'skin':0
}

simulationParams={
    'first_ts': 1e-3,
    'mult_ts': 4,
    'max_ts': 0.05,
    'tolerance_newton': 1e-4
}

simulation_time = 75

# usage:
grid_sizes = [20]
grid_refinements = [1]
dz_values = [5,2,1]

## Calculating analytical solution ##
t_min=((inputs['re']*inputs['re']*0.1)/(inputs['k']/(inputs['Ct']*inputs['mu'])))/(24*3600)#days
deltaT_analytical= simulationParams['max_ts']
t_values = np.arange(t_min+ deltaT_analytical, simulation_time, deltaT_analytical)
# Initialize an empty list to store Pwf values
Pwf_values = []
# Loop over t_values and calculate Pwf for each t
for t in t_values:
    Pwf = Pwf_line_source_solution(inputs,t)
    Pwf_values.append(Pwf)

## Conversion to DimensionLess Parameters ##
tD_analytical=DimensionlessTime(inputs, t_values)
pD_analytical=DimensionlessPressure(inputs, Pwf_values)

data = VisualizeFromExcel(excel_dir='Excels', labels=['grid size', 'grid refinement', 'dz'], output_dir=outputDir, inputs=inputs, grid_sizes=grid_sizes, grid_refinements=grid_refinements, dz_values=dz_values)
results = data.results

## visualization ##
v = Visualize(results=results, labels=['grid size', 'grid refinement', 'dz'], output_dir=outputDir, inputs=inputs)

v.plot_pD_tD(pD_analytical=pD_analytical, tD_analytical=tD_analytical, plotAnalytical=True, savePlot=False, showPlot=True)
v.plot_pD_tD_log(pD_analytical=pD_analytical, tD_analytical=tD_analytical, plotAnalytical=True,savePlot=False, showPlot=True)
v.plot_pressureVsTime(p_analytical=Pwf_values,t_analytical=t_values, plotAnalytical=True, savePlot=False, showPlot=True)
v.plot_pressureVsTime_semiLog(p_analytical=Pwf_values,t_analytical=t_values, plotAnalytical=True, savePlot=False, showPlot=True)
# ... (rest of your visualization calls here)