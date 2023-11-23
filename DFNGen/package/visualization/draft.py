import pandas as pd
import numpy as np
import os
from multiprocessing import freeze_support
from .model.ModelFracture import FractureModel
from .visualization.Visualize import Visualize
from darts.engines import redirect_darts_output
import pdb

# pdb.set_trace()


# constant input parameters
inputs = {
    'h': 1,  # m
    'Ct': 1e-8,  # 1/Pa
    'mu': 0.096e-3,  # Pa.s
    'k': 1,  # mD
    'phi': 0.1,
    'Pi': 400,  # bar
    'rw': 0.15,  # m
    'Q': 100,  # m3/day
    'skin': 0,
    'Cr': 1e-10,
    'prod_well_coords': [[500, 500, 0]]
}

simulationParams = {
    'first_ts': 1e-4,
    'mult_ts': 1.2,
    'max_ts': 0.001,
    'tolerance_newton': 1e-6
}
# File names and directories:
Simulation_name = '03MeshForSAOnLcMat'
DFN_name = '01FinalDatasetAfterMerging'
mesh_folder = '03MeshForSAOnLcMat'

size_report_step = 0.01  # Size of the reporting step (when output is writen to .vtk format)
num_report_steps = 3 # Number of reporting steps (see above)
start_time = 0  # Starting time of the simulation
end_time = size_report_step * num_report_steps  # End time of the simulation

bound_cond = 'wells_in_frac'
problem_type = 'fracture'  # or 'lineSource'
char_len_list = [16,8,4,2,1]
paramList = [16]  #
paramName = 'Lc_Mat'


# Create the ExcelWriter object outside of the loop
def main():
    DFN_directory = 'DFNs/' + DFN_name + '/Text'  # Assuming this is the directory containing all the DFN txt files
    # Get all the .txt files in the DFN_directory
    dfn_files = ['DFN001.txt']  # [f for f in os.listdir(DFN_directory) if f.endswith('.txt')]#['DFN005.txt']
    # DFN_textName = ['DFN001.txt']#dfn_file.split('.')[0]
    for dfn_file in dfn_files:
        results = {}
        for val in paramList:
            outputDir = 'DFNs/' + str(DFN_name) + '/results/' + str(Simulation_name)
            # Create directories if they don't exist
            os.makedirs(outputDir, exist_ok=True)
            DFN_textName = dfn_file.split('.')[0]  # Get the base name of the file without extension
            with pd.ExcelWriter(f'{outputDir}/{Simulation_name + str(DFN_textName)}.xlsx', engine='openpyxl') as writer:
                # Create a dictionary to store the results for each grid size

                for char_len in char_len_list:

                    output_dir_mesh = 'DFNs/' + str(
                        DFN_name) + '/Mesh/' + mesh_folder + '/' + DFN_textName + '/Processsed' + str(char_len)
                    print('---', output_dir_mesh, '----')
                    mesh_name = DFN_textName + '_raw_lc_' + str(char_len) + '.msh'
                    # Define output directory
                    mesh_path = output_dir_mesh + '/' + mesh_name
                    # Check if the mesh file exists
                    if not os.path.exists(mesh_path):
                        print(f"Mesh file {mesh_name} not found in {output_dir_mesh}. Skipping...")
                        continue
                    m = FractureModel(inputs=inputs, simulationParams=simulationParams,
                                      bound_cond=bound_cond, mesh_path=mesh_path)

                    # redirect output
                    redirect_darts_output(file_name=str(Simulation_name) + '.log')

                    print(
                        '============================================================================================')
                    print('++++++++++++++++++++++++++' + str(char_len) + '+++++++++++++++++++++++++++++++++++++')
                    m.init()

                    # Prepare property_array and cell_property
                    tot_unknws = m.reservoir.unstr_discr.fracture_cell_count + m.reservoir.unstr_discr.matrix_cell_count + len(
                        m.reservoir.wells) * 2
                    tot_properties = 2
                    pressure_field = m.physics.engine.X[:-1:2]
                    saturation_field = m.physics.engine.X[1::2]
                    property_array = np.empty((tot_unknws, tot_properties))
                    property_array[:, 0] = pressure_field
                    property_array[:, 1] = saturation_field

                    # Get initial pressures at the boundaries
                    initial_boundary_pressures = np.array(m.physics.engine.X)[:-1:2][
                        m.reservoir.bottom_boundary_cells.tolist() + m.reservoir.top_boundary_cells.tolist()]

                    for ith_step in range(num_report_steps):

                        # Create the model with the current grid size and refinement
                        print('================= Unstructured === Step :' + str(
                            ith_step + 1) + '======================================')
                        m.run_python(size_report_step)

                        # Store the results for the current grid size and refinement
                        SimName = Simulation_name + str(val) + 'char_len' + str(char_len)
                        data = pd.DataFrame.from_dict(m.physics.engine.time_data)
                        results[(char_len, val, DFN_textName)] = data

                        # Use the writer to write to a different sheet
                        sheet_name = 'char_len' + str(char_len) + str(val)
                        data.to_excel(writer, sheet_name=sheet_name)

                        # Prepare property_array and cell_property
                        pressure_field = m.physics.engine.X[:-1:2]
                        saturation_field = m.physics.engine.X[1::2]
                        property_array = np.empty((tot_unknws, tot_properties))
                        property_array[:, 0] = pressure_field
                        property_array[:, 1] = saturation_field

                        # Write to VTK
                        outputDir_VTK = outputDir + '/vtk_data'
                        if not os.path.exists(outputDir_VTK):
                            os.makedirs(outputDir_VTK)
                        fileName = DFN_textName + str(val) + str(char_len)
                        m.reservoir.unstr_discr.write_to_vtk(outputDir_VTK, property_array, m.physics.vars, ith_step,
                                                             fileName)

                        # Print timers and statistics for the run
                        m.print_timers()
                        m.print_stat()
                        # Check pressure drop at the boundaries
                        current_boundary_pressures = np.array(m.physics.engine.X)[:-1:2][
                            m.reservoir.bottom_boundary_cells.tolist() + m.reservoir.top_boundary_cells.tolist()]
                        pressure_drops = initial_boundary_pressures - current_boundary_pressures

                        # If pressure drop in any boundary cell exceeds a threshold (e.g., 1 bar), stop the simulation
                        # if any(pressure_drop > 1 for pressure_drop in pressure_drops):
                        #    print("Stopping simulation due to pressure drop at boundary.")
                        #    break
                        # generating visualized outputs
                        # v = Visualize(results=results,paramList=paramList,paramName=str(paramName), labels=['CharacteristicLength'], output_dir=outputDir, inputs=inputs)
                        # frameName = DFN_textName+str(val)+str(char_len) + str(ith_step)
                        # v.showVtk(fileName=frameName)
                        # v.imageVtk(fileName=frameName,OuutputName=frameName+'Grid',output_dir=outputDir, showImage=True, gridding=True)
                        # v.imageVtk(fileName=frameName,OuutputName=frameName+'3D',output_dir=outputDir, showImage=True, gridding=False)
                        # v.image2DVtk(fileName=frameName,OuutputName=frameName,output_dir=outputDir, showImage=True, gridding=False)
                    # v.makeGif(input_name=DFN_textName+str(val)+str(char_len), num_report_steps=num_report_steps, output_file=SimName +DFN_textName+ '.gif', duration=2)

        ## visualization ##
        v = Visualize(results=results, paramList=paramList, paramName=str(paramName),
                      labels=['Lc_frac', str(paramName), DFN_textName], output_dir=outputDir,
                      inputs=inputs)
        v.plot_pressureVsTime(fileName='P_T' + DFN_textName, plotAnalytical=False, savePlot=True, showPlot=False)
        v.plot_pressureVsTime_semiLog(fileName='P_T_log' + DFN_textName, plotAnalytical=False, savePlot=True,
                                      showPlot=False)


if __name__ == "__main__":
    freeze_support()
    main()
