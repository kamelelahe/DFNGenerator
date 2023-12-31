import numpy as np
from multiprocessing import freeze_support
from .preprocessing_code import frac_preprocessing
from.mesh_raw_fractures import mesh_raw_fractures
import os

def main():
    char_lens=[8]
    # File names and directories:
    DFN_name = '2.7.Test'
    Mesh_folder_name='05TestModified'


    DFN_directory = 'DFNs/'+DFN_name+'/Text'  # Assuming this is the directory containing all the DFN txt files
    preprocessing = False

    # Get all the .txt files in the DFN_directory
    dfn_files = ['DFN005.txt'] #[f for f in os.listdir(DFN_directory) if f.endswith('.txt')]

    for dfn_file in dfn_files:
        DFN_textName = dfn_file.split('.')[0]  # Get the base name of the file without extension

        # Define output directory
        for char_len in char_lens:

            output_dir = 'DFNs/' +DFN_name+'/'+Mesh_folder_name+'/'+ str(DFN_textName)+'/Processsed'+str(char_len)
            os.makedirs(output_dir, exist_ok=True)
            filename_base = DFN_textName
            frac_data_raw = np.genfromtxt(os.path.join(DFN_directory, dfn_file))



            # Input parameters
            decimals = 7  # in order to remove duplicates we need to have fixed number of decimals
            mesh_clean = True  # need gmsh installed and callable from command line in order to mesh!!!
            mesh_raw = True  # need gmsh installed and callable from command line in order to mesh!!!
            num_partition_x = 4  # number of partitions for parallel implementation of intersection finding algorithm
            num_partition_y = 4  # " ... "
            char_len_boundary=2*char_len
            char_len_matrix=2*char_len
            char_len_frac=char_len


            if preprocessing:
                # Input parameters for cleaning procedure
                angle_tol_straighten = 1e-7  # tolerance for straightening fracture segments [degrees]
                merge_threshold = 0.1 # tolerance for merging nodes in algebraic constraint, values on interval [0.5, 0.86] [-]

                #angle_tol_remove_segm = np.arctan(0.35) * 180 / np.pi   # tolerance for removing accute intersections, values on interval [15, 25] [degrees]
                angle_tol_remove_segm = 0  # This will ensure no segments are removed based on angle tolerance.
                angle_tol_straighten = 180  # This will ensure no edges are straightened.
                # frac_data_raw = np.genfromtxt(os.path.join('DFNs/' + str(DFN_name)+'/Text', 'DFN001.txt'))
                straighten_after_cln = False
                small_angle_iter = 0


                frac_preprocessing(frac_data_raw,char_len_cleaning=char_len, char_len_frac=char_len_frac,char_len_mat=char_len_matrix,
                                   output_dir=output_dir, filename_base=filename_base, merge_threshold=merge_threshold,
                                   height_res=1, angle_tol_small_intersect=angle_tol_remove_segm, apertures_raw=None, box_data=None, margin=25,
                                   mesh_clean=mesh_clean, mesh_raw=mesh_raw, angle_tol_straighten=angle_tol_straighten, straighten_after_cln=False,
                                   decimals=decimals,tolerance_zero=1e-10, tolerance_intersect=1e-10, calc_intersections_before=True,
                                   calc_intersections_after=True,num_partition_x=num_partition_x, num_partition_y=num_partition_y,
                                   partition_fractures_in_segms=True, matrix_perm=1, correct_aperture=True,
                                   small_angle_iter=0, char_len_mult=5, char_len_boundary=char_len_boundary, main_algo_iters=1)

            else:

                mesh_raw_fractures(frac_data_raw, char_len_frac=char_len_frac,char_len=char_len_matrix, output_dir=output_dir, filename_base=filename_base, height_res=1, apertures_raw=None,
                                   box_data=None, margin=25, mesh_raw=mesh_raw, decimals=decimals, tolerance_zero=1e-10, tolerance_intersect=1e-10,
                                   calc_intersections_before=True, num_partition_x=num_partition_x, num_partition_y=num_partition_y,
                                   partition_fractures_in_segms=True, matrix_perm=1, char_len_mult=10, char_len_boundary=char_len_boundary, main_algo_iters=1)

if __name__ == "__main__":
    freeze_support()
    main()
