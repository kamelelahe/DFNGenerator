from .ConstantMuPropertyContainer import MyPropertyContainer
from .reservoir import UnstructReservoir
from darts.physics.geothermal.physics import  Geothermal
from darts.models.darts_model import DartsModel
from darts.physics.geothermal.property_container import PropertyContainer

class FractureModel(DartsModel):

    def __init__(self, inputs, simulationParams,mesh_path,bound_cond ='const_pres_rate',  n_points=128):
        """in init part, we initialize reservoir parameters. static properties, grid properties, location of wells, and petropysical proprties are add
        this stage"""
        # Call base class constructor
        super().__init__()

        self.bound_cond =bound_cond
        self.physics_type = 'geothermal'

        # Extract inputs
        self.flow_rate = inputs['Q']
        permeability = inputs['k']
        porosity=inputs['phi']
        reservoir_height = inputs['h']
        reservoir_radius = inputs['re']

        self.well_bore_radius = inputs['rw']
        self.initial_pressure = inputs['Pi']
        #compressibility = inputs['Ct']
        self.skin = inputs['skin']

        # Some permeability input data for the simulation
        permx = permeability  # Matrix permeability in the x-direction [mD]
        permy = permeability  # Matrix permeability in the y-direction [mD]
        permz = permeability  # Matrix permeability in the z-direction [mD]
        poro = porosity  # Matrix porosity [-]
        frac_aper = 1e-3 # Aperture of fracture cells (but also takes a list of apertures for each segment) [m]

        inj_well_coords = [[925, 960, 25]]
        prod_well_coords = inputs['prod_well_coords']

        msh_file_path = mesh_path

        # Instance of unstructured reservoir class from reservoir.py file.
        # When calling this class constructor, the def __init__(self, arg**) is executed which created the instance of
        # the class and constructs the object. In the process, the mesh is loaded, mesh information is calculated and
        # the discretization is executed. Besides that, also the boundary conditions of the simulations are
        # defined in this class --> in this case constant pressure/rate at the left (x==x_min) and right (x==x_max) side
        self.reservoir = UnstructReservoir(permx=permx, permy=permy, permz=permz, frac_aper=frac_aper,
                                           mesh_file=msh_file_path, poro=poro, bound_cond=self.bound_cond,
                                           physics_type=self.physics_type, inj_well_coords=inj_well_coords,
                                           prod_well_coords=prod_well_coords)


        self.property_container= MyPropertyContainer(inputs=inputs)
        #self.property_container= PropertyContainer()
        # create pre-defined physics for geothermal
        self.cell_property = ['pressure', 'enthalpy']
        self.physics = Geothermal(self.timer, property_container=self.property_container,
                                  n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=10000, cache=False)
        # Fill some additional values for geothermal runs:
        self.reservoir.hcap.fill(2200)
        self.reservoir.conduction.fill(181.44)

        self.physics.init_physics()

        # Linear solver
        # Set timestep parameters
        self.params.first_ts = simulationParams['first_ts']
        self.params.mult_ts = simulationParams['mult_ts']
        self.params.max_ts = simulationParams['max_ts']
        self.params.tolerance_newton = simulationParams['tolerance_newton']

        # Stop the timer for initialization
        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        # Set uniform initial conditions for pressure and temperature
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=self.initial_pressure,
                                                    uniform_temperature=350)

    def set_boundary_conditions(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name or 'INJ' in w.name:
                # Add controls for injection well:
                # Specify both pressure and temperature (since it's upstream for injection well)
                w.control = self.physics.new_bhp_water_inj(375, 308.15)
                # w.control = self.physics.new_rate_water_inj(4800, 308.15)

            else:
                # Specify bhp for particular production well:
                #w.control = self.physics.new_bhp_prod(self.flow_rate)
                w.control = self.physics.new_rate_water_prod(self.flow_rate)

        return 0
