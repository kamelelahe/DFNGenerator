import math


class apertureCorrelation:
    def __init__(self, fractures, rockType ,S_Hmax, S_hmin, S_v,stressAzimuth):

        self.stressAzimuth=stressAzimuth
        # Extracting rock properties from the dictionary based on rockType
        self.E_matrix = rockType["E"]
        self.nu_matrix = rockType["nu"]
        self.fractures = fractures
        self.compute_kn(self.fractures)
        #calculating the stress

        for frac in self.fractures:
            sigma_n = self.stress_decomposition(S_Hmax, S_hmin, S_v, stressAzimuth, frac['theta'])
            diff=sigma_n/frac['kn']
            newAperture=frac['fracture aperture']-diff
            if newAperture<0:
                newAperture=0
            frac['correctedAperture'+str(self.stressAzimuth)]=newAperture



    def compute_kn(self, fractures):
        """Compute the normal stiffness or spring constant."""
        E_fracture = 0.1 * self.E_matrix
        nu_fracture = 0.4 * self.nu_matrix
        for frac in fractures:
            frac['kn'] = E_fracture * (1 - nu_fracture) / (frac['fracture aperture'] * (1 + nu_fracture) * (1 - 2 * nu_fracture))

    def stress_decomposition(self, S_Hmax, S_hmin, S_v, stress_azimuth, fracture_orientation):
        # Calculating the difference between stress azimuth and fracture orientation
        delta_theta = abs(stress_azimuth - fracture_orientation)
        # Calculating the normal stress (σn) on the fracture
        sigma_n = S_Hmax * math.sin(math.radians(delta_theta)) ** 2 + \
                  S_hmin * math.cos(math.radians(delta_theta)) ** 2

        return sigma_n #to convert to Pa

    def computeDeformation(self,fractures,sigma_n):
        for frac in fractures:
            diff=sigma_n/frac['kn']
            newAperture=frac['fracture aperture']-diff
            if newAperture<0:
                newAperture=0
            frac['correctedAperture'+str(self.stressAzimuth)]=newAperture


