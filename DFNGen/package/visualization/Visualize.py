from PIL import Image as PILImage
import os
import matplotlib.pyplot as plt
import numpy as np
from ..equations.DimensionLess import DimensionlessPressure, DimensionlessTime
import matplotlib.cm as cm
import pyvista as pv
from IPython.display import Image

class Visualize:

    def __init__(self, results, inputs, labels, output_dir, paramList, paramName):
        self.output_dir = output_dir
        self.results = results
        self.inputs = inputs
        self.labels = labels
        self.paramList = paramList
        self.paramName = paramName
        # Create a colormap
        self.colors = cm.rainbow(np.linspace(0, 1, len(self.results)))
        self.colorsAnalytical = cm.rainbow(np.linspace(0, 1, len(self.paramList)))

    def plot_pressureVsTime(self, p_analytical=[], t_analytical=[], plotAnalytical=True, showPlot=True,
                            savePlot=False, fileName='P_T'):
        # Plot the results
        fig, ax = plt.subplots(figsize=(11, 8))
        # Loop over the results
        for (color, (keys, df)) in zip(self.colors, self.results.items()):
            # Plot data
            label = ', '.join(f'{label}: {value}' for label, value in zip(self.labels, keys))
            if 'time' in df.columns:
                ax.plot(df['time'][3:], df['PRD : BHP (bar)'][3:], marker='o', markersize=2, linestyle='-', label=label,
                        color=color)

        if plotAnalytical:
            for (color, (key, value)) in zip(self.colorsAnalytical, p_analytical.items()):
                label = str(self.paramName) + ' : ' + str(key)
                ax.plot(t_analytical, value, label=label, color=color)

        ax.tick_params(labelsize=14)
        ax.set_xlabel('Time (day)', fontsize=14)
        ax.set_ylabel('BHP (bar)', fontsize=14)

        # Place the legend outside of the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid()
        plt.tight_layout()  # Adjust the layout

        if savePlot:
            plt.savefig(f'{self.output_dir}/{fileName}.png', format='png', dpi=300)
        if showPlot:
            plt.show()

    def plot_pressureVsTime_semiLog(self, p_analytical=[], t_analytical=[], plotAnalytical=True, showPlot=True,
                                    savePlot=False, fileName='P_logT'):
        # Plot the results
        fig, ax = plt.subplots(figsize=(11, 8))

        # Loop over the results
        for (color, (keys, df)) in zip(self.colors, self.results.items()):
            # Plot data
            label = ', '.join(f'{label}: {value}' for label, value in zip(self.labels, keys))
            if 'time' in df.columns:
                ax.plot(df['time'][3:], df['PRD : BHP (bar)'][3:], marker='o', markersize=2, linestyle='-', label=label,
                        color=color)

        if plotAnalytical:
            for (color, (key, value)) in zip(self.colorsAnalytical, p_analytical.items()):
                label = str(self.paramName) + ' : ' + str(key)
                ax.plot(t_analytical, value, label=label, color=color)

        ax.tick_params(labelsize=14)
        ax.set_xlabel('Time (day)', fontsize=14)
        ax.set_ylabel('BHP (bar)', fontsize=14)
        ax.set_xscale('log')

        # Place the legend outside of the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid()
        plt.tight_layout()  # Adjust the layout
        if savePlot:
            plt.savefig(f'{self.output_dir}/{fileName}.png', format='png', dpi=300)
        if showPlot:
            plt.show()

    def plot_pD_tD(self, tD_analytical={}, pD_analytical={}, plotAnalytical=True, showPlot=True, savePlot=False,
                   fileName='PD_TD'):
        # Plot the results
        fig, ax = plt.subplots(figsize=(11, 8))
        # Loop over the results
        for (color, (keys, df)) in zip(self.colors, self.results.items()):
            # Plot data
            if 'mu' in self.labels:
                mu_index = self.labels.index('mu')
                self.inputs['mu'] = keys[mu_index]
            if 'Q' in self.labels:
                q_index = self.labels.index('Q')
                self.inputs['Q'] = keys[q_index]
            if 'rw' in self.labels:
                rw_index = self.labels.index('rw')
                self.inputs['rw'] = keys[rw_index]
            if 'h' in self.labels:
                h_index = self.labels.index('h')
                self.inputs['h'] = keys[h_index]
            if 'k' in self.labels:
                k_index = self.labels.index('k')
                self.inputs['k'] = keys[k_index]
            if 'Ct' in self.labels:
                Ct_index = self.labels.index('Ct')
                self.inputs['Ct'] = keys[Ct_index]

            if  'time' in df.columns:
                tD_simulation = DimensionlessTime(self.inputs, df['time'][3:])
                pD_simulation = DimensionlessPressure(self.inputs, df['PRD : BHP (bar)'][3:])
                # Create a label from the keys
                print("self.labels:", self.labels)
                print("keys:", keys)
                label = ', '.join(f'{label}: {value}' for label, value in zip(self.labels, keys))
                ax.plot(tD_simulation, pD_simulation, marker='o', markersize=2, linestyle='-', label=label, color=color)

        if plotAnalytical:
            for (color, key) in zip(self.colorsAnalytical, pD_analytical.keys()):
                label = str(self.paramName) + ' : ' + str(key)
                ax.plot(tD_analytical[key], pD_analytical[key], label=label, color=color)

        ax.tick_params(labelsize=14)
        ax.set_xlabel('TD', fontsize=14)
        ax.set_ylabel('PD', fontsize=14)
        # Place the legend outside of the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.tight_layout()  # Adjust the layout
        if savePlot:
            plt.savefig(f'{self.output_dir}/{fileName}.png', format='png', dpi=300)
        if showPlot:
            plt.show()

    def plot_pD_tD_log(self, tD_analytical=[], pD_analytical={}, plotAnalytical=True, showPlot=True, savePlot=False,
                       fileName='PD_TD_log'):
        # Plot the results
        fig, ax = plt.subplots(figsize=(11, 8))

        # Loop over the results
        for (color, (keys, df)) in zip(self.colors, self.results.items()):
            # Plot data
            if 'mu' in self.labels:
                mu_index = self.labels.index('mu')
                self.inputs['mu'] = keys[mu_index]
            if 'Q' in self.labels:
                q_index = self.labels.index('Q')
                self.inputs['Q'] = keys[q_index]
            if 'rw' in self.labels:
                rw_index = self.labels.index('rw')
                self.inputs['rw'] = keys[rw_index]
            if 'h' in self.labels:
                h_index = self.labels.index('h')
                self.inputs['h'] = keys[h_index]
            if 'k' in self.labels:
                k_index = self.labels.index('k')
                self.inputs['k'] = keys[k_index]
            if 'Ct' in self.labels:
                Ct_index = self.labels.index('Ct')
                self.inputs['Ct'] = keys[Ct_index]
            if 'time' in df.columns:
                tD_simulation = DimensionlessTime(self.inputs, df['time'][3:])
                pD_simulation = DimensionlessPressure(self.inputs, df['PRD : BHP (bar)'][3:])
                # Create a label from the keys
                label = ', '.join(f'{label}: {value}' for label, value in zip(self.labels, keys))
                ax.plot(tD_simulation, pD_simulation, marker='o', markersize=2, linestyle='-', label=label, color=color)

        if plotAnalytical:
            for (color, key) in zip(self.colorsAnalytical, pD_analytical.keys()):
                label = str(self.paramName) + ' : ' + str(key)
                ax.plot(tD_analytical[key], pD_analytical[key], label=label, color=color)

        ax.tick_params(labelsize=14)
        ax.set_xlabel('TD', fontsize=14)
        ax.set_ylabel('PD', fontsize=14)
        ax.set_xscale('log')
        # Place the legend outside of the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.tight_layout()  # Adjust the layout
        if savePlot:
            plt.savefig(f'{self.output_dir}/{fileName}.png', format='png', dpi=300)
        if showPlot:
            plt.show()

    def showVtk(self, fileName):
        # get vts data
        mesh = pv.read(f'{self.output_dir}\\vtk_data\\{fileName}.vtk')

        # define plotter
        plotter = pv.Plotter()

        # set temperature as active scalar
        mesh.set_active_scalars('pressure')

        # add threshold levels
        thresT = mesh.threshold([0, self.inputs['Pi']], invert=False)

        # add outline of mesh
        outline = mesh.outline()

        # add elements to plotter
        plotter.set_background('#52576c')
        plotter.add_mesh(outline, color='k')
        plotter.add_mesh(thresT, cmap='coolwarm', show_edges=True,
                         scalar_bar_args={'title': 'pressure', 'fmt': '%.4f'})

        _ = plotter.add_axes(line_width=5, labels_off=False)

        plotter.show()

    def imageVtk(self, fileName, OuutputName, output_dir, showImage=True, gridding=True):
        # Specify the file
        i = 0  # adjust this according to the specific solution file you want

        # get vts data
        mesh = pv.read(f'{self.output_dir}\\vtk_data\\{fileName}.vtk')

        # set temperature as active scalar
        mesh.set_active_scalars('pressure')

        # add threshold levels
        thresT = mesh.threshold([0, self.inputs['Pi']+50], invert=False)

        # add outline of mesh
        outline = mesh.outline()

        # define plotter
        plotter = pv.Plotter(off_screen=True)

        # add elements to plotter
        plotter.set_background('#52576c')
        plotter.add_mesh(outline, color='k')
        if gridding:
            plotter.add_mesh(thresT, cmap='coolwarm', show_edges=True,
                             scalar_bar_args={'title': 'pressure', 'fmt': '%.4f'})
        else:
            plotter.add_mesh(thresT, cmap='coolwarm', show_edges=False,
                             scalar_bar_args={'title': 'pressure', 'fmt': '%.4f'})

        _ = plotter.add_axes(line_width=5, labels_off=False)
        ImageOutputDir = f'{output_dir}/VTK_Images'
        # Create directories if they don't exist
        os.makedirs(ImageOutputDir, exist_ok=True)

        # Render and create a screenshot
        plotter.screenshot(f'{ImageOutputDir}/{OuutputName}.png')

        if showImage:
            # Display the image
            Image(filename=f'{output_dir}/VTK_Images/{OuutputName}.png')

    def image2DVtk(self, fileName, OuutputName, output_dir, showImage=True, gridding=True):
        # get vts data
        mesh = pv.read(f'{self.output_dir}\\vtk_data\\{fileName}.vtk')

        # set temperature as active scalar
        mesh.set_active_scalars('pressure')

        # add threshold levels
        thresT = mesh.threshold([0, self.inputs['Pi']+50], invert=False)

        # add outline of mesh
        outline = mesh.outline()

        # define plotter
        plotter = pv.Plotter(off_screen=True)
        # add elements to plotter
        plotter.set_background('#52576c')
        plotter.add_mesh(outline, color='k')
        if gridding:
            plotter.add_mesh(thresT, cmap='coolwarm', show_edges=True,
                             scalar_bar_args={'title': 'pressure', 'fmt': '%.4f'})
        else:
            plotter.add_mesh(thresT, cmap='coolwarm', show_edges=False,
                             scalar_bar_args={'title': 'pressure', 'fmt': '%.4f'})

        # Set the camera position, focal point, and view up direction
        camera_position = (mesh.center[0], mesh.center[1], mesh.bounds[5] + 10)
        focal_point = mesh.center
        view_up = (0, 1, 0)  # Along Y-axis

        plotter.camera.position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.view_up = view_up

        plotter.camera.parallel_projection = True
        plotter.camera_set = True
        plotter.reset_camera()

        _ = plotter.add_axes(line_width=5, labels_off=False)
        ImageOutputDir = f'{output_dir}/VTK_Images'
        # Create directories if they don't exist
        os.makedirs(ImageOutputDir, exist_ok=True)

        # Render and create a screenshot
        plotter.screenshot(f'{ImageOutputDir}/{OuutputName}.png')

        if showImage:
            # Display the image
            Image(filename=f'{output_dir}/VTK_Images/{OuutputName}.png')
    def makeGif(self,input_name, num_report_steps, output_file, duration):
        images = []
        outputDir_vtk = f'{self.output_dir}/VTK_Images'
        for ith_step in range(num_report_steps):
            filename = os.path.join(outputDir_vtk,input_name + str(ith_step) + '.png')
            img = PILImage.open(filename)
            images.append(img)
        images[0].save(f'{self.output_dir}/{output_file}', save_all=True, append_images=images[1:],
                        duration=duration , loop=0)
