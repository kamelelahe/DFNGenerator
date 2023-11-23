from .PDFs import PDFs
import random
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import math
import os
from .apertureCalculation import aperture
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

class generateConjugateFractures:

    def __init__(self, domainLengthX, domainLengthY, Iy,thetaY, Ix,thetaX, fractureLengthPDF, fractureLengthPDFParams,spatialDisturbutionPDF, spatialDisturbutionPDFParams,outputDir, stressAzimuth =None,savePic=True):
        """
        Parameters:
            -domainLengthX= length of domain in x direction
            -domainLengthY= length of domain in y direction
            - Iy= fracture intensity in vertical direction
            - fractureLengthPDF= choose between: Fixed, Uniform, Log-Normal, Negative power-law, Negative exponential
            - fractureLengthPDFParams a dictionary including different items based on the PDF:
                +"Fixed": fractureLengthPDFParams{"fixed_value": }
                +"Uniform": fractureLengthPDFParams{"Lmin":
                                                    "Lmax":}
                +"Log-Normal" : fractureLengthPDFParams{"sigma":}
                +"Negative power-law": fractureLengthPDFParams{"alpha":
                                                              "Lmin":}
                +"Negative exponential": fractureLengthPDFParams{"lambda":}

        """
        ##  initialization of disturbution functions
        self.nameFractureLengthPDF=fractureLengthPDF
        self.fractureLengthPDFParams=fractureLengthPDFParams
        self.namespatialDisturbutionPDF=spatialDisturbutionPDF
        self.spatialDisturbutionPDFParams=spatialDisturbutionPDFParams
        self.spatialDisturbutionPDF =PDFs(self.namespatialDisturbutionPDF, self.spatialDisturbutionPDFParams)
        self.spatialDisturbutionPDFMode=self.spatialDisturbutionPDF.compute_mode()
        self.xmax=domainLengthX
        self.ymax=domainLengthY

        ## Section A: generate vertical fractures
        # Step A1: Fracture Characterization
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++ Section A: generate vertical fractures +++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('==========Step A1: Fracture Characterization============')
        self.thetaY= thetaY
        thetaYforGeneration=90-thetaY
        self.rawVerticalFractures= self.generateFractures(Iy,thetaYforGeneration,fractureLengthPDFParams)
        self.sortFractures(self.rawVerticalFractures)
        self.computeSpacing(self.rawVerticalFractures)


        #step A2: placing the fractures
        print('==========Step A2: placing the fractures============')
        self.processedVerticalFractures=self.place_fractures(self.rawVerticalFractures, thetaYforGeneration)

        ## Section B: generate conjugate fractures
        # Step B1: Fracture Characterization
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++ Section B: generate conjugate fractures +++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('==========Step B1: Fracture Characterization============')
        self.thetaX=thetaX
        thetaXforGeneration=90-thetaX
        self.rawConjugateFractures= self.generateFractures(Ix,thetaXforGeneration,fractureLengthPDFParams)
        self.sortFractures(self.rawConjugateFractures)
        self.computeAperture_SubLinearLengthApertureScaling(self.rawConjugateFractures)
        self.computeSpacing(self.rawConjugateFractures)

        #step B2: placing the fractures
        print('==========Step B2: placing the fractures============')
        self.outputDir= outputDir
        self.processedConjugateFractures=self.place_fractures(self.rawConjugateFractures, thetaXforGeneration)

        ############calculating the aperture
        self.stressAzimuth=stressAzimuth
        """
        ##section C-0
        # this section will deal with post processing in two parts :
            #1- merging fracture
            #2- removing small segments after intersection of fractures
        # By now, we decided to do this before meshing, with the help of the code provided
        # by Stephan de Hoop
        # the article can be found on: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR030743
        #
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++ Section C-0: Post processing conjugate fractures +++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('==========Step 1: merging closely spaced fractures============')
        for i in range(len(self.processedVerticalFractures)):
            prevEndAd = False
            PrevStartAd = False
            for j in range(len(self.processedConjugateFractures)):
                fracY = self.processedVerticalFractures[i]
                fracX = self.processedConjugateFractures[j]
                fracHNew, fracVNew,isAdjustedFracYStart, isAdjustedFracYEnd =merge_fractures(fracX,fracY,PrevStartAd,prevEndAd )
                self.processedConjugateFractures[j] = fracHNew
                self.processedVerticalFractures[i]=fracVNew
                currentEndAd = max(isAdjustedFracYEnd, prevEndAd)
                currentStartAd = max(isAdjustedFracYStart, PrevStartAd)
                prevEndAd = currentEndAd
                PrevStartAd = currentStartAd
                if currentStartAd and currentEndAd:
                    break
        """
        ## section C
        # this section deals with calculation realted to influence of stress field on aperture

        density=2200 #for sedimentary rock
        h=1000
        S_v= density*9.81*h

        # as we move to more tehnically active region, the diffrence between min and max horizontal stress becomes more pronounced
        rockType="limestone"
        S_Hmax=1.5*S_v
        S_hmin=0.9*S_v

        for azimuth in stressAzimuth:
            aperture(self.processedVerticalFractures, rockType ,S_Hmax, S_hmin, S_v,azimuth, self.thetaY)
            aperture(self.processedConjugateFractures, rockType ,S_Hmax, S_hmin, S_v,azimuth, self.thetaX)

        self.plot_aperture_changes( self.processedVerticalFractures, self.processedConjugateFractures,stressAzimuth,self.thetaY,self.thetaX ,'apertureChange')
        self.plot_stereographic([self.thetaX,self.thetaY],['X', 'Y'], stressAzimuth,'stratigraphy')
        ##section C
        # this section will deal with post processing in two parts :
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++ Section C: Generating the outputs +++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('==========Step 1: text file for input properties============')
        self.generate_input_properties_file('inputProperties')
        print('==========Step 2: generating the photo============')
        if savePic:
         self.plot_fractures(self.processedVerticalFractures,self.processedConjugateFractures,name= 'DFNPic',stressAzimuth= stressAzimuth)
        print('==========Step 3: text file for fracture coordinates============')
        self.generateTextFileForFractureCoordinates( 'fractureCoordinates',self.processedVerticalFractures,self.processedConjugateFractures)

        print('==========Step 4: text file for fracture apertures============')
        self.generateTextFileForFractureApertures('aperture', self.processedVerticalFractures,self.processedConjugateFractures)

        print('==========Step 5: text file for output properties============')
        self.gnerateOutputFileForOutputProperties('outputProperties', self.processedVerticalFractures,self.processedConjugateFractures)





    def placeLongestFracture(self,longestFractrue,theta):
        # Seed Selection and placing the longest fracture
        referenceWithinDomain= False
        number=0
        print('fracture length',longestFractrue['fracture length'])
        while not referenceWithinDomain:
            seed_x = random.uniform(0, self.xmax)
            seed_y = random.uniform(0, self.ymax)

            (new_x_start, new_y_start), (new_x_end, new_y_end) = self.fractureCordinate(longestFractrue, theta, seed_x,seed_y)
            if self.is_within_domain(new_x_start, new_y_start) and self.is_within_domain(new_x_end, new_y_end):
                referenceWithinDomain = True
                addedFracture = {
                    'number': number,
                    'x_start': new_x_start,
                    'y_start': new_y_start,
                    'x_end': new_x_end,
                    'y_end': new_y_end,
                    'fracture length': longestFractrue['fracture length'],
                    'fracture spacing': longestFractrue['fracture spacing'],
                }
        return addedFracture,seed_x,seed_y

    def place_fractures(self,fractures,theta):
        # Step A2: Fracture Placement
        processedFractures=[]
        print('--- placing the longest fracture----')
        addedFracture,seed_x,seed_y =self.placeLongestFracture(fractures[0], theta)
        processedFractures.append(addedFracture)
        print('--the longest fracture added--')
        numberOfTries=0
        number = 0
        # Introducing New Fractures
        print('--- placing rest of fractures ---')
        for fracture in fractures[1:]:
            isNewFractureAdded=False
            while not isNewFractureAdded:
                referenceWithinDomain = False
                while not referenceWithinDomain:
                    distance =  (self.spatialDisturbutionPDF.get_value() - self.spatialDisturbutionPDFMode )* random.choice([-1, 1]) # Randomness for + and -
                    if theta<90:
                        new_x_mid = seed_x + np.random.randint(0, self.ymax) * random.choice([-1, 1])
                        new_y_mid = seed_y + distance
                    else:
                        new_x_mid = seed_x + distance
                        new_y_mid = seed_y + np.random.randint(0, self.ymax)* random.choice([-1, 1])
                    (new_x_start, new_y_start), (new_x_end, new_y_end) = self.fractureCordinate(fracture, theta, new_x_mid,new_y_mid)
                    #add check proximity here
                    if self.is_within_domain(new_x_start, new_y_start) and self.is_within_domain(new_x_end, new_y_end):
                        referenceWithinDomain = True
                # Check proximity
                too_close = False
                for existing_frac in processedFractures:
                    existing_coords = ((existing_frac['x_start'], existing_frac['y_start']),
                                       (existing_frac['x_end'], existing_frac['y_end']))
                    new_coords = ((new_x_start, new_y_start), (new_x_end, new_y_end))
                    if segment_to_segment_distance(new_coords, existing_coords) < existing_frac['fracture spacing']:
                        too_close = True
                        numberOfTries+=1
                        print( "fracture number",str(fracture['number']),'with length of ', str(fracture['fracture length']),' has been relocated. Iy= ', str(self.computeIntensity(processedFractures)),"has been reached",str(numberOfTries), "times")
                        break
                if not too_close:
                    isNewFractureAdded= True
                    number += 1
                    addedFracture = {
                        'number': number,
                        'x_start': new_x_start,
                        'y_start': new_y_start,
                        'x_end': new_x_end,
                        'y_end': new_y_end,
                        'fracture length': fracture['fracture length'],
                        'fracture spacing': fracture['fracture spacing']
                    }
                    processedFractures.append(addedFracture)
            if numberOfTries > 10000:
                print("Max retries reached for vertical fractures, fracture generation for y stopped. Intensity is= ",
                      str(self.computeIntensity(processedFractures)))
                break

        self.computeAperture_SubLinearLengthApertureScaling(processedFractures)
        return processedFractures

    def generateFractures(self,fractureIntensity,theta,fractureLengthPDFParams):
        """
        Parameters:
        - fractureIntensity (float): Desired fracture intensity for the network.

        Returns:
        -a list of dictionaries with fractures properties sorted by fracture length and each dictionary including 'fracture length'
        """
        if theta>90:
            theta=180-theta

        print('theta=',theta)
        if theta >= 45:
            theta=90-theta
            thetaRadian = np.radians(theta)
            fractureLengthPDFParams["Lmax"]= self.ymax / math.cos(thetaRadian)
        else:
            thetaRadian = np.radians(theta)
            fractureLengthPDFParams["Lmax"]= self.xmax / math.cos(thetaRadian)

        print('fractureLengthPDFParams["Lmax"]=',fractureLengthPDFParams["Lmax"])


        fractureLengthPDF = PDFs(self.nameFractureLengthPDF, fractureLengthPDFParams)
        fractures=[]
        newFrac={}
        newFrac['fracture length']=fractureLengthPDF.get_value()
        fractures.append(newFrac)
        while self.computeIntensity(fractures) < fractureIntensity:
            newFrac = {}
            newFrac['fracture length'] = fractureLengthPDF.get_value()
            fractures.append(newFrac)

        return fractures

    def computeIntensity(self, fractures):
        total_length = sum([fracture['fracture length'] for fracture in fractures])
        area = self.xmax * self.ymax
        intensity = total_length / area
        return intensity

    def sortFractures(self, fractures):
        # Sort the fractures based on 'fracture length' in descending order
        fractures.sort(key=lambda x: x['fracture length'], reverse=True)
        # Assign a 'number' to each fracture starting from 0 for the longest
        for i, fracture in enumerate(fractures, start=0):
            fracture['number'] = i

    def computeAperture_SubLinearLengthApertureScaling(self, fractures):
        for fracture in fractures:
            length = fracture['fracture length']
            aperture = 1e-4 * (length ** 0.5)
            fracture['fracture aperture'] = aperture

    def computeSpacing(self,fractures):
        # Placeholder for computing the spacing for each fracture.
        for fracture in fractures:
            #I onl implemented power law here
            fracture['fracture spacing'] = 0.2 * fracture['fracture length'] ** 0.5

    def fractureCordinate(self,fracture,theta, midX, midY):
        half_length = fracture['fracture length'] / 2  # the longest fracture
        new_x_start = midX - half_length * np.cos(np.radians(theta))
        new_y_start = midY - half_length * np.sin(np.radians(theta))
        new_x_end = midX + half_length * np.cos(np.radians(theta))
        new_y_end = midY + half_length * np.sin(np.radians(theta))
        return (new_x_start,new_y_start), (new_x_end,new_y_end)

    def is_within_domain(self, x, y):
        return 0 <= x <= self.xmax and 0 <= y <= self.ymax

    def plot_fractures(self, y_dir_fractures, x_dir_fractures, name, stressAzimuth, number=0):
        """Plot the fractures, north direction, and stress directions and save the output to a PDF."""

        figDir = self.outputDir + '/pics'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"

        plt.figure(figsize=(12, 12))

        # Plot fractures
        for fracture in y_dir_fractures:
            x_start, y_start = fracture['x_start'], fracture['y_start']
            x_end, y_end = fracture['x_end'], fracture['y_end']
            plt.plot([x_start, x_end], [y_start, y_end], color='red', linewidth=fracture['fracture aperture']*1000)
        for fracture in x_dir_fractures:
            x_start, y_start = fracture['x_start'], fracture['y_start']
            x_end, y_end = fracture['x_end'], fracture['y_end']
            plt.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=fracture['fracture aperture']*1000)

        # Plot north direction outside the plot
        plt.annotate('N', xy=(1.02, 1.00), xycoords='axes fraction', fontsize=20, ha='center', va='center')
        #plt.annotate('', xy=(1.02, 1.02), xycoords='axes fraction', fontsize=20, ha='center', va='center')
        plt.annotate('', xy=(1.02, 0.98), xytext=(1.02, 0.9), xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center', va='center')
        plotAzimuth=False
        if plotAzimuth:
            # Plot stress azimuth direction outside the plot
            stressAzimuth=90-stressAzimuth
            stress_azimuth_rad = np.radians(stressAzimuth)
            delta_x = 0.08 * np.cos(stress_azimuth_rad)
            delta_y = 0.08 * np.sin(stress_azimuth_rad)
            plt.annotate(r'$S_{Hmax}}$', xy=(1.04, 0.79), xycoords='axes fraction', fontsize=10, ha='center', va='center')
            plt.annotate('', xy=(1.02 + delta_x, 0.81 + delta_y), xytext=(1.02, 0.81),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(facecolor='green', shrink=0.05), ha='center', va='center')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('DFN Visualization')
        plt.grid(True)
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)

        # Save the plot to a PDF
        plt.savefig(figDirFile, format='png', dpi=300)

    def generate_input_properties_file(self, name, number=0):
        """
        This function generates the input properties file.
        """
        InputPropertiesDir = self.outputDir + '/'+str(name)
        os.makedirs(InputPropertiesDir, exist_ok=True)
        InputPropertiesFile = f"{InputPropertiesDir}\\{name}{number + 1:03}.txt"
        with open(InputPropertiesFile, 'w') as f:
            f.write("xmax: " + str(self.xmax) + '\n')
            f.write("ymax: " + str(self.ymax) + '\n')
            f.write("thetaY: " + str(self.thetaY) + '\n')
            f.write("thetaX: " + str(self.thetaX) + '\n')

            f.write("nameFractureLengthPDF: " + str(self.nameFractureLengthPDF) + '\n')
            if isinstance(self.fractureLengthPDFParams, dict):
                # Convert dictionary to a pretty-printed JSON string for readability
                import json
                f.write(
                    "fractureLengthPDFParams: " + json.dumps(self.fractureLengthPDFParams, indent=4) + '\n')
            else:
                f.write("fractureLengthPDFParams: " + str(self.fractureLengthPDFParams) + '\n')


            f.write("namespatialDisturbutionPDF: " + str(self.namespatialDisturbutionPDF) + '\n')

            if isinstance(self.spatialDisturbutionPDFParams, dict):
                # Convert dictionary to a pretty-printed JSON string for readability
                import json
                f.write(
                    "spatialDisturbutionPDFParams: " + json.dumps(self.spatialDisturbutionPDFParams, indent=4) + '\n')
            else:
                f.write("spatialDisturbutionPDFParams: " + str(self.spatialDisturbutionPDFParams) + '\n')
            f.write("spatialDisturbutionPDFMode: " + str(self.spatialDisturbutionPDFMode) + '\n')

    def generateTextFileForFractureCoordinates(self, name,fracY , fracX,number=0):
        """
        This function generates the input properties file.
        """
        coordinatesOutputDir = self.outputDir + '/'+str(name)
        os.makedirs(coordinatesOutputDir, exist_ok=True)
        InputPropertiesFile = f"{coordinatesOutputDir}\\{name}{number + 1:03}.txt"
        with open(InputPropertiesFile, 'w') as fileID:
            for frac in fracY:
                fileID.write(f"{frac['x_start']} {frac['y_start']} {frac['x_end']} {frac['y_end']}\n")
            for frac in fracX:
                fileID.write(f"{frac['x_start']} {frac['y_start']} {frac['x_end']} {frac['y_end']}\n")

    def generateTextFileForFractureApertures(self, name, fracY , fracX,number=0):
        """
        This function generates the input properties file.
        """
        apertureOutputDir = self.outputDir + '/'+str(name)
        os.makedirs(apertureOutputDir, exist_ok=True)
        InputPropertiesFile = f"{apertureOutputDir}\\{name}{number + 1:03}.txt"
        self.computeAperture_SubLinearLengthApertureScaling(fracY)
        self.computeAperture_SubLinearLengthApertureScaling(fracX)

        with open(InputPropertiesFile, 'w') as fileID:
            for frac in fracY:
                fileID.write(f"{frac['fracture aperture']}\n")
            for frac in fracX:
                fileID.write(f"{frac['fracture aperture']}\n")

    def distanceOfWellFromClosetFracture(self, fracY, fracX, wellLocation):
        min_distance = float('inf')
        for frac in fracY + fracX:
            x_start, y_start = frac['x_start'], frac['y_start']
            x_end, y_end = frac['x_end'], frac['y_end']
            distance = point_to_segment_distance(np.array(wellLocation), np.array([x_start, y_start]),
                                                 np.array([x_end, y_end]))
            min_distance = min(min_distance, distance)
        return min_distance

    def gnerateOutputFileForOutputProperties(self,name,fracY , fracX,number=0):

        totalCountFracY = len(fracY)
        totalCountFracX = len(fracX)
        totalCountFrac = len(fracY) + len(fracX)
        totalIntersection = 0
        for fracV in fracY:
            for fracH in fracX:
                line1 = ((fracH['x_start'], fracH['y_start']), (fracH['x_end'], fracH['y_end']))
                line2 = ((fracV['x_start'], fracV['y_start']), (fracV['x_end'], fracV['y_end']))
                intersect= line_intersection(line1, line2)
                if intersect:
                    totalIntersection += 1
        areaRock=self.xmax*self.ymax
        connectivity = totalIntersection / (areaRock * totalCountFrac)

        wellLocation = (self.xmax / 2, self.ymax / 2)
        minDistanceFromWell = self.distanceOfWellFromClosetFracture(fracY, fracX, wellLocation)


        outputFileForOutputPropertiesDir = self.outputDir + '/'+str(name)
        os.makedirs(outputFileForOutputPropertiesDir, exist_ok=True)
        InputPropertiesFile = f"{outputFileForOutputPropertiesDir}\\{name}{number + 1:03}.txt"
        self.computeAperture_SubLinearLengthApertureScaling(fracY)
        self.computeAperture_SubLinearLengthApertureScaling(fracX)

        Ix=self.computeIntensity(fracX)
        Iy=self.computeIntensity(fracY)
        totalIntensity=Ix+Iy

        # Initialize variables
        lengthsX = [frac['fracture length'] for frac in fracX]
        lengthsY = [frac['fracture length'] for frac in fracY]
        aperturesX = [frac['fracture aperture'] for frac in fracX]
        aperturesY = [frac['fracture aperture'] for frac in fracY]

        # Calculate statistics for lengths and apertures in X direction
        minLx = min(lengthsX)
        maxLx = max(lengthsX)
        avgLx = sum(lengthsX) / len(lengthsX)
        minApertureX = min(aperturesX)
        maxApertureX = max(aperturesX)
        avgApertureX = sum(aperturesX) / len(aperturesX)

        # Calculate statistics for lengths and apertures in Y direction
        minLy = min(lengthsY)
        maxLy = max(lengthsY)
        avgLy = sum(lengthsY) / len(lengthsY)
        minApertureY = min(aperturesY)
        maxApertureY = max(aperturesY)
        avgApertureY = sum(aperturesY) / len(aperturesY)

        # Calculate average length and aperture for both directions
        averageLengthTotal = (sum(lengthsX) + sum(lengthsY)) / (len(lengthsX) + len(lengthsY))
        averageApertureTotal = (sum(aperturesX) + sum(aperturesY)) / (len(aperturesX) + len(aperturesY))

        with open(InputPropertiesFile, 'w') as fileID:
            fileID.write(f"totalCountFrac= {totalCountFracX}\n"
                         f"totalCountFrac= {totalCountFracY}\n"
                         f"totalCountFrac= {totalCountFrac}\n"
                         f"connectivity= {connectivity}\n"
                         f"wellLocation= {wellLocation}\n"
                         f"minDistanceFromWell= {minDistanceFromWell}\n"
                         f"Ix= {Ix}\n"
                         f"Iy= {Iy}\n"
                         f"totalIntensity= {totalIntensity}\n"
                         f"minLx= {minLx}\n"
                         f"maxLx= {maxLx}\n"
                         f"avgLx= {avgLx}\n"
                         f"minLy= {minLy}\n"
                         f"maxLy= {maxLy}\n"
                         f"avgLy= {avgLy}\n"
                         f"averageLengthTotal= {averageLengthTotal}\n"
                         f"minApertureX= {minApertureX}\n"
                         f"averageLength= {maxApertureX}\n"
                         f"maxApertureX= {avgApertureX}\n"
                         f"minApertureY= {minApertureY}\n"
                         f"maxApertureY= {maxApertureY}\n"
                         f"avgApertureY= {avgApertureY}\n"
                         f"averageApertureTotal= {averageApertureTotal}\n")

    def plot_aperture_changes(self, vertical_fractures, conjugate_fractures, stressAzimuth, thetaY,thetaX, name, number=0):
        figDir = self.outputDir + '/stressDependancy'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"
        # Store the percent differences for each azimuth
        percent_diffs_x = []
        percent_diffs_y = []

        for azimuth in stressAzimuth:
            # Retrieve the average aperture before deformation
            avg_aperture_before_x = np.mean([frac['fracture aperture'] for frac in vertical_fractures])
            avg_aperture_before_y = np.mean([frac['fracture aperture'] for frac in conjugate_fractures])

            # Retrieve the average aperture after deformation for the current azimuth
            avg_aperture_after_x = np.mean([frac['correctedAperture' + str(azimuth)] for frac in vertical_fractures])
            avg_aperture_after_y = np.mean([frac['correctedAperture' + str(azimuth)] for frac in conjugate_fractures])

            # Calculate the percent differences and store
            percent_diffs_x.append(100 * (avg_aperture_before_x - avg_aperture_after_x) / avg_aperture_before_x)
            percent_diffs_y.append(100 * (avg_aperture_before_y - avg_aperture_after_y) / avg_aperture_before_y)

        # Plotting
        fig, ax = plt.subplots()

        # Set of bar positions for each azimuth
        bar_positions_x = np.arange(len(stressAzimuth)) * 2 - 0.2
        bar_positions_y = np.arange(len(stressAzimuth)) * 2 + 0.2

        ax.bar(bar_positions_x, percent_diffs_x, width=0.4, label='x, orientation= '+str(thetaX), color='blue')
        ax.bar(bar_positions_y, percent_diffs_y, width=0.4, label='y, orientation= '+str(thetaY), color='orange')

        # Setting the title, labels, and legend
        ax.set_title('Deformed Fracture Aperture (%)')
        ax.set_xticks(np.arange(len(stressAzimuth)) * 2)
        ax.set_xticklabels(['azimuth=' + str(a) for a in stressAzimuth])
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figDirFile, format='png', dpi=300)

    def plot_stereographic(self,fracture_orientations,fracture_labels, shmax_azimuths,name,number=0):
        figDir = self.outputDir + '/stressDependancy'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # Plotting the circle
        circle = plt.Circle((0, 0), 1, color='green', fill=False)
        ax.add_artist(circle)

        # Plotting and labeling fractures
        for orientation, label in zip(fracture_orientations, fracture_labels):
            x = [np.sin(np.radians(orientation)), -np.sin(np.radians(orientation))]
            y = [np.cos(np.radians(orientation)), -np.cos(np.radians(orientation))]
            ax.plot(x, y)
            ax.text(x[1] * 1.1, y[1] * 1.1, label, ha="center", va="center")

        # Plotting the S_hmax azimuths
        # Plotting the S_hmax azimuths
        fixed_arrow_length = 0.2  # Adjust this to your desired arrow length
        distance_from_center = 1.5  # Adjust this to change arrow's starting distance from circle center

        # Plotting the S_hmax azimuths
        fixed_arrow_length = 0.2  # Adjust this to your desired arrow length
        distance_from_center = 1.2  # Adjust this to change arrow's starting distance from circle center
        # Plotting the S_hmax azimuths
        cmap = get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=360)
        for azimuth in shmax_azimuths:
            color = cmap(norm(azimuth))
            x = np.sin(np.radians(azimuth))
            y = np.cos(np.radians(azimuth))

            # Starting points for arrow
            start_x = x * distance_from_center
            start_y = y * distance_from_center

            # Calculate the x and y components of the arrow length based on the azimuth
            delta_x = fixed_arrow_length * np.sin(np.radians(azimuth))
            delta_y = fixed_arrow_length * np.cos(np.radians(azimuth))

            # Ending points for arrow
            end_x = start_x + delta_x
            end_y = start_y + delta_y

            # textPosition
            xText = start_x + delta_x*2
            yText = start_y + delta_y*2

            ax.arrow(start_x, start_y, delta_x, delta_y, head_width=0.05, head_length=0.1, fc=color, ec=color)
            ax.text(xText, yText, str(azimuth), ha="center", va="center")
            x = np.sin(np.radians(azimuth+180))
            y = np.cos(np.radians(azimuth+180))

            # Starting points for arrow
            start_x = x * distance_from_center
            start_y = y * distance_from_center

            # Calculate the x and y components of the arrow length based on the azimuth
            delta_x = fixed_arrow_length * np.sin(np.radians(azimuth+180))
            delta_y = fixed_arrow_length * np.cos(np.radians(azimuth+180))

            # Ending points for arrow
            end_x = start_x + delta_x
            end_y = start_y + delta_y
            ax.arrow(start_x, start_y, delta_x, delta_y, head_width=0.05, head_length=0.1, fc=color, ec=color)

        ax.set_aspect('equal', 'box')
        ax.axis('off')
        plt.savefig(figDirFile, format='png', dpi=300)


def point_to_segment_distance( p, a, b):
    """Compute the distance from point p to segment [a, b]."""
    if np.all(a == b):
        return np.linalg.norm(p - a)
    v = b - a
    w = p - a
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - a)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - b)

    b = c1 / c2
    pb = a + b * v
    return np.linalg.norm(p - pb)
def segment_to_segment_distance(s1, s2):
    """Compute the distance between two segments s1 and s2."""
    s1_start, s1_end = s1
    s2_start, s2_end = s2

    distances = [
        point_to_segment_distance(np.array(s1_start), np.array(s2_start), np.array(s2_end)),
        point_to_segment_distance(np.array(s1_end), np.array(s2_start), np.array(s2_end)),
        point_to_segment_distance(np.array(s2_start), np.array(s1_start), np.array(s1_end)),
        point_to_segment_distance(np.array(s2_end), np.array(s1_start), np.array(s1_end))
    ]
    return min(distances)

def is_point_on_line_segment(point, line1):
    (x, y) = point
    ((x1, y1), (x2, y2)) = line1

    # Check if point is out of bounds
    if (x < min(x1, x2)) or (x > max(x1, x2)) or (y < min(y1, y2)) or (y > max(y1, y2)):
        return False

    # Handle vertical line case
    if x1 == x2:
        return x == x1

    # Calculate slope of line segment
    m = (y2 - y1) / (x2 - x1)

    # Check if y-coordinate of point matches the line equation
    return abs(y - (m * (x - x1) + y1)) < 1e-9
def line_intersection(line1, line2):
    ((x1, y1), (x2, y2)) = line1
    ((x3, y3), (x4, y4)) = line2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return False, None, None  # Lines are parallel and don't intersect

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    # Check if the intersection point lies within the bounds of both line segments
    if (min(x1, x2) <= x <= max(x1, x2) and
        min(y1, y2) <= y <= max(y1, y2) and
        min(x3, x4) <= x <= max(x3, x4) and
        min(y3, y4) <= y <= max(y3, y4)):
        return True, x, y
    else:
        return False, x, y
# Function to calculate distance between two points
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5
def point_to_line_distance(point, line):
    x0, y0 = point
    x1, y1, x2, y2 = line

    # If the line segment is vertical
    if x1 == x2:
        # If the point's y-coordinate is between the y-coordinates of the line segment's endpoints
        if min(y1, y2) <= y0 <= max(y1, y2):
            distance = abs(x0 - x1)
        else:
            dist_to_start = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            dist_to_end = math.sqrt((x0 - x2)**2 + (y0 - y2)**2)
            distance = min(dist_to_start, dist_to_end)
        slope = "undefined"
    else:
        # Calculate the slope of the line
        m = (y2 - y1) / (x2 - x1)
        # Convert the line to the form ax + by + c = 0
        a = m
        b = -1
        c = y1 - m * x1
        # Calculate the perpendicular distance
        distance = abs(a * x0 + b * y0 + c) / math.sqrt(a**2 + b**2)
        slope = m

        # Check if the perpendicular from the point to the line falls outside the segment
        dot1 = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
        dot2 = (x0 - x2) * (x1 - x2) + (y0 - y2) * (y1 - y2)
        if dot1 * dot2 > 0:
            dist_to_start = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            dist_to_end = math.sqrt((x0 - x2)**2 + (y0 - y2)**2)
            distance = min(dist_to_start, dist_to_end)

    return distance




def merge_fractures(fracX, fracY, PrevStartAd, prevEndAd):
    isAdjustedFracYStart, isAdjustedFracYEnd = PrevStartAd, prevEndAd
    line1 = ((fracX['x_start'], fracX['y_start']), (fracX['x_end'], fracX['y_end']))
    line2 = ((fracY['x_start'], fracY['y_start']), (fracY['x_end'], fracY['y_end']))

    intersect, x, y = line_intersection(line1, line2)

    if intersect:
        return fracX, fracY, isAdjustedFracYStart, isAdjustedFracYEnd
    line1 = (np.array([fracX['x_start'], fracX['y_start']]), np.array([fracX['x_end'], fracX['y_end']]))
    line2 = (np.array([fracY['x_start'], fracY['y_start']]), np.array([fracY['x_end'], fracY['y_end']]))

    # Find the longer fracture
    lenX = fracX['fracture length']
    lenY = fracY['fracture length']
    threshold=5 * (fracX['fracture spacing'] if lenX > lenY else fracY['fracture spacing']*5)
    if segment_to_segment_distance(line1, line2) > threshold:
        return fracX, fracY, isAdjustedFracYStart, isAdjustedFracYEnd

    def adjust_point(frac, point_key, x, y):
        frac['x_'+point_key], frac['y_'+point_key] = x, y

    # Scenario A: The intersection point lies on fracX
    if is_point_on_line_segment((x, y), line1):
        dist_start = point_to_segment_distance(line2[0], line1[0], line1[1])
        dist_end = point_to_segment_distance(line2[1], line1[0], line1[1])
        if dist_start< threshold or dist_end<threshold:
            if dist_start < dist_end:
                if not isAdjustedFracYStart:
                    adjust_point(fracY, 'start', x, y)
                    isAdjustedFracYStart = True
            else:
                if not isAdjustedFracYEnd:
                    adjust_point(fracY, 'end', x, y)
                    isAdjustedFracYEnd = True

    # Scenario B: The intersection point lies on fracY
    elif is_point_on_line_segment((x, y), line2):
        dist_start = point_to_segment_distance(line1[0], line2[0], line2[1])
        dist_end = point_to_segment_distance(line1[1], line2[0], line2[1])
        if dist_start < threshold or dist_end < threshold:
            if dist_start < dist_end:
                adjust_point(fracX, 'start', x, y)
            else:
                adjust_point(fracX, 'end', x, y)

    # Scenario C: The intersection point doesn’t lie on fracX or fracY
    else:
        # Adjust endpoint of fracY closest to the intersection point
        dist_start = point_to_segment_distance(line2[0], line1[0], line1[1])
        dist_end = point_to_segment_distance(line2[1], line1[0], line1[1])
        if dist_start < threshold or dist_end < threshold:
            if dist_start < dist_end:
                if not isAdjustedFracYStart:
                    adjust_point(fracY, 'start', x, y)
                    isAdjustedFracYStart = True
            else:
                if not isAdjustedFracYEnd:
                    adjust_point(fracY, 'end', x, y)
                    isAdjustedFracYEnd = True

        # Adjust endpoint of fracX closest to the intersection point
        dist_start = point_to_segment_distance(line1[0], line2[0], line2[1])
        dist_end = point_to_segment_distance(line1[1], line2[0], line2[1])
        if dist_start < threshold or dist_end < threshold:
            if dist_start < dist_end:
                adjust_point(fracX, 'start', x, y)
            else:
                adjust_point(fracX, 'end', x, y)
    return fracX, fracY, isAdjustedFracYStart, isAdjustedFracYEnd