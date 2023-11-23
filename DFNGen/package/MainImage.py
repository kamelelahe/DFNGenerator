import os
from multiprocessing import freeze_support
from .model.ModelLineSource  import LineSourceModel
from .visualization.Visualize import Visualize
from darts.engines import redirect_darts_output
import pdb

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
    'prod_well_coords': [[125, 250, 0]]
}

simulationParams = {
    'first_ts': 1e-4,
    'mult_ts': 1.2,
    'max_ts': 0.001,
    'tolerance_newton': 1e-6
}
# pdb.set_trace()
# File names and directories:
Simulation_name = '11TestForSymmetry'
DFN_name = '03FullDomain'

size_report_step = 0.05 # Size of the reporting step (when output is writen to .vtk format)
num_report_steps =5 # Number of reporting steps (see above)
start_time = 0  # Starting time of the simulation
end_time = size_report_step * num_report_steps  # End time of the simulation

bound_cond= 'wells_in_frac'
problem_type= 'fracture' #or 'lineSource'
char_len_list = [8]
paramList = ['full']
paramName='h'

# Create the ExcelWriter object outside of the loop
def main():
    # File names and directories:
    DFN_directory = 'DFNs/' + DFN_name + '/Text'  # Assuming this is the directory containing all the DFN txt files
    # Get all the .txt files in the DFN_directory
    dfn_files = ['DFN002.txt']#[f for f in os.listdir(DFN_directory) if f.endswith('.txt')]#['DFN005.txt']#
    for dfn_file in dfn_files:
        # Create a dictionary to store the results for each grid size
        results = {}
        for char_len in char_len_list:
            for val in paramList:
                DFN_textName = dfn_file.split('.')[0]  # Get the base name of the file without extension
                # Define output directory
                outputDir ='DFNs/' + str(DFN_name)+'/results/' + str(Simulation_name)
                # Create directories if they don't exist
                print('++++++++++++++++++++++++++' + str(char_len) + '+++++++++++++++++++++++++++++++++++++')
                for ith_step in range(num_report_steps):
                    #ith_step=ith_step+25
                    # Create the model with the current grid size and refinement
                    print('================= Unstructured === Step :'+ str(ith_step+1) + '======================================')
                    # Store the results for the current grid size and refinement
                    SimName= Simulation_name+str(val)+'char_len'+str(char_len)
                    # Write to VTK
                    #generating visualized output
                    v = Visualize(results=results,paramList=paramList,paramName=str(paramName), labels=['CharacteristicLength'], output_dir=outputDir, inputs=inputs)
                    frameName = DFN_textName+str(val)+str(char_len) + str(ith_step)
                    print(frameName)
                    #v.showVtk(fileName=frameName)
                    v.imageVtk(fileName=frameName,OuutputName=frameName+'Grid',output_dir=outputDir, showImage=True, gridding=True)
                    v.imageVtk(fileName=frameName,OuutputName=frameName+'3D',output_dir=outputDir, showImage=True, gridding=False)
                    v.image2DVtk(fileName=frameName,OuutputName=frameName,output_dir=outputDir, showImage=True, gridding=False)
                v.makeGif(input_name=DFN_textName+str(val)+str(char_len), num_report_steps=num_report_steps, output_file=SimName +DFN_textName+ '.gif', duration=2)


if __name__ == "__main__":
    freeze_support()
    main()
