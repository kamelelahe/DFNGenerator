from .NewDFNGen.PDFs import PDFs
from .NewDFNGen.DFNGenerator import generateConjugateFractures
import numpy as np
from multiprocessing import freeze_support
# Initialize parameters for each distribution

DFN_name = '18Test-'
# Define output directory
outputDir = 'DFNs/' + str(DFN_name)


rockTypes={
    'carbonate':{
        "E": 19.9e9,  # Young's modulus in Pa
        "nu": 	0.373,  # Poisson ratio
        "rho":2700,  # density kg/m3 ref=https://www.hindawi.com/journals/mpe/2016/5967159/
    },
    'sandstone': {
        "E": 25e9,  # Young's modulus in Pa
        "nu": 0.25,  # Poisson ratio
        "rho": 2400,  # density kg/m3
    }
}# ref=https://www.hindawi.com/journals/mpe/2016/5967159/

set1={
    'I':0.05,
    'fractureLengthPDF': 'Negative power-law',
    'fractureLengthPDFParams': {"alpha": 2.5, "Lmin": 2},
    'spatialDisturbutionPDF':"Uniform",
    'spatialDisturbutionPDFParams': {"Lmin": 2, "Lmax":1000},
    'orientationDisturbutonPDF':"Von-Mises",
    'orientationDisturbutonPDFParams': {"kappa":30 , "loc":np.radians(0)},#direction with North
}

set2={
    'I':0.05,
    'fractureLengthPDF': 'Negative power-law',
    'fractureLengthPDFParams': {"alpha": 2.5, "Lmin": 2},
    'spatialDisturbutionPDF': "Log-Normal",
    'spatialDisturbutionPDFParams': {"mu": np.log(250), "sigma": 0.5, "Lmax":1000},
    'orientationDisturbutonPDF':"Von-Mises",
    'orientationDisturbutonPDFParams': {"kappa":28 , "loc":np.radians(60)},#direction with North
}

set3={
    'I':0.01,
    'fractureLengthPDF': 'Negative power-law',
    'fractureLengthPDFParams': {"alpha": 3, "Lmin": 4},
    'spatialDisturbutionPDF':"Uniform",
    'spatialDisturbutionPDFParams': {"Lmin": 2, "Lmax":1000},
    'orientationDisturbutonPDF':"Von-Mises",
    'orientationDisturbutonPDFParams': {"kappa":25 , "loc":np.radians(30)},#direction with North
}
stressAzimuth = [i for i in np.arange(0, 180, 22.5)]
depth=2000
domainLengthX=1000
domainLengthY=1000





def main():
    generateConjugateFractures(domainLengthX, domainLengthY, [set1,set2,set3],outputDir, stressAzimuth=None, rockType=rockTypes['sandstone'], depth=depth)

if __name__ == "__main__":
    freeze_support()
    main()