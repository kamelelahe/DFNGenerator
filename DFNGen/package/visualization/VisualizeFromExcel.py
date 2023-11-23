import os
import pandas as pd

class VisualizeFromExcel:
    def __init__(self, excel_dir, labels, output_dir, inputs, grid_sizes, grid_refinements, dz_values):
        self.excel_dir = excel_dir
        self.labels = labels
        self.output_dir = output_dir
        self.inputs = inputs
        self.grid_sizes = grid_sizes
        self.grid_refinements = grid_refinements
        self.dz_values = dz_values
        self.results = self._load_results()

    def _load_results(self):
        results = {}
        for file in os.listdir(self.excel_dir):
            if file.endswith(".xlsx"):
                for sheet_name in pd.ExcelFile(os.path.join(self.excel_dir, file)).sheet_names:
                    if 'GS' in sheet_name and 'GR' in sheet_name and 'Dz' in sheet_name:
                        keys = tuple(map(float, sheet_name.replace('GS', '').replace('GR', ',').replace('Dz', ',').split(',')))
                        if keys[0] in self.grid_sizes and keys[1] in self.grid_refinements and keys[2] in self.dz_values:
                            data = pd.read_excel(os.path.join(self.excel_dir, file), sheet_name=sheet_name)
                            results[keys] = data
        return results
