# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:40:52 2021

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script analyses the processed data, extracting and assessing the relevant
    tibial load metrics.
    
"""

# %% Import packages

import opensim as osim
import os
import pandas as pd
import glob
import numpy as np
from scipy import integrate
# from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# %% Define functions

#Convert opensim array to python list
#For use in other functions
def osimToList(array):
    temp = []
    for i in range(array.getSize()):
        temp.append(array.get(i))

    return temp

#Convert opensim storage object to dataframe
def stoToDF(fileName, sampleInterval = None, filterFreq = None):
    """
    
    Parameters
    ----------
    fileName: filename of opensim .mot or .sto file

    Returns
    -------
    df: pandas data frame
    
    """
    
    #Load file
    osimSto = osim.Storage(fileName)
    
    #Resample
    if sampleInterval != None:
        osimSto.resampleLinear(sampleInterval)
    
    #Filter
    if filterFreq != None:
        osimSto.lowpassFIR(4, int(filterFreq))
    
    #Get column labels
    labels = osimToList(osimSto.getColumnLabels())    
    
    #Set time values
    time = osim.ArrayDouble()
    osimSto.getTimeColumn(time)
    time = osimToList(time)    
    
    #Allocate space for data
    data = []
    #Loop through columns and get data
    for i in range(osimSto.getSize()):
        temp = osimToList(osimSto.getStateVector(i).getData())
        temp.insert(0, time[i])
        data.append(temp)
        
    #Convert to dataframe
    df = pd.DataFrame(data, columns=labels)
    
    return df

# %% Set-up

#Create main directory variable
mainDir = os.getcwd()

#Create path to data storage folder
os.chdir('..\\Data')
dataDir = os.getcwd()

#Get directory list
dList = os.listdir()
#Just keep directories with 'RBDS' for subject list
subList = []
for item in dList:
    if os.path.isdir(item):
        #Check for subject directory and append if appropriate
        if item.startswith('RBDS'):
            subList.append(item)
            
#Get participant info
df_subInfo = pd.read_csv('ParticipantInfo\\RBDSinfo.csv')

#Set sample interval
sampleInterval = 1 / 150 #150 Hz
            
#Settings for number of gait cycles
#See later note
nGC = 30

#### TODO: other setup parameters?

# %% Extract participant data

##### TODO: space to store data...

#Loop through subject list
for ii in range(len(subList)):
    
    #Navigate to participant directory
    os.chdir(subList[ii])
    
    #Identify static trial name
    staticFile = []
    for file in glob.glob('*static.c3d'):
        staticFile.append(file)
    
    #Get subject mass based on static file name and participant info database
    #Need check in place as files are labelled differently for certain participants
    if ii >= 21:
        newStaticFile = staticFile[0].split('0')[0]+'00'+staticFile[0].split('0')[1]
        mass = df_subInfo.loc[df_subInfo['FileName'] == newStaticFile,['Mass']].values[0][0]
    else:
        mass = df_subInfo.loc[df_subInfo['FileName'] == staticFile[0],['Mass']].values[0][0]
    
    #Identify dynamic trial names
    dynamicFiles = []
    for file in glob.glob('*run*.c3d'):
        dynamicFiles.append(file)
        
    #Loop through trials
    for tt in range(len(dynamicFiles)):
        
        #Calculate tibial force across the running trial
        #Note, this focuses on right leg forces
        
        #Import the GRF data
        df_grf = stoToDF(dynamicFiles[tt].split('.')[0]+'_grf.mot')
        
        #Import the ankle and knee joint centre position data
        df_ajc = stoToDF(dynamicFiles[tt].split('.')[0]+'_AJC_pos.sto')
        df_kjc = stoToDF(dynamicFiles[tt].split('.')[0]+'_KJC_pos.sto')
        
        #Import the inverse dynamics data
        #Filter included
        df_id = stoToDF(dynamicFiles[tt].split('.')[0]+'_id.sto',
                        filterFreq = 10) #10Hz filter
        
        #Extract time for position data
        timePos = df_id['time'].to_numpy()
        timePos[0] = 0 #correct for negative time value that seems to come up
        
        #Extract time for GRF data
        timeGrf = df_grf['time'].to_numpy()
        
        #Downsample GRF to position data
        grfVars = ['ground_force_r_vx', 'ground_force_r_vy', 'ground_force_r_vz']
        grf = np.zeros([len(timePos),3])
        for gg in range(len(grfVars)):
            cubicF = interp1d(timeGrf, df_grf[grfVars[gg]].to_numpy(),
                              kind = 'cubic', fill_value = 'extrapolate')
            grf[:,gg] = cubicF(np.linspace(timePos[0], timePos[-1], len(timePos)))
            
        #Downsample COP to position data
        copVars = ['ground_force_r_px', 'ground_force_r_py', 'ground_force_r_pz']
        cop = np.zeros([len(timePos),3])
        for gg in range(len(grfVars)):
            cubicF = interp1d(timeGrf, df_grf[copVars[gg]].to_numpy(),
                              kind = 'cubic', fill_value = 'extrapolate')
            cop[:,gg] = cubicF(np.linspace(timePos[0], timePos[-1], len(timePos)))
        
        #Extract the times for stance phases based on GRF data
        #Based on info in Fukuchi et al. paper there should be at least 30 stance
        #phases available for every run, so we'll use that value
        stanceInd = np.zeros([nGC,2],int)
        #stanceTime = np.zeros([nGC,2])
        startSearch = np.where(grf[:,1] < 100)[0][0] #start search at first non force values
        #Loop through desired number of gait cycles
        for gg in range(nGC):
            #Identify index of next foot on
            currFootOn = np.where(grf[startSearch:-1,1] > 100)[0][0] + startSearch
            #Identify index of next foot off
            currFootOff = np.where(grf[currFootOn:-1,1] < 100)[0][0] + currFootOn - 1
            #Append to stance index array
            stanceInd[gg,:] = [currFootOn,currFootOff]
            #Update search variable
            startSearch = currFootOff + 1
        
        #Extract the data
        #Ankle position
        ajc = np.transpose(np.array([df_ajc['state_0'].to_numpy(),
                                     df_ajc['state_1'].to_numpy(),
                                     df_ajc['state_2'].to_numpy()]))
        #Knee position
        kjc = np.transpose(np.array([df_kjc['state_0'].to_numpy(),
                                     df_kjc['state_1'].to_numpy(),
                                     df_kjc['state_2'].to_numpy()]))
        #Ankle moment
        ankleMom = df_id['ankle_angle_r_moment'].to_numpy() * -1
        
        #### This seems to just reverse what is done...unnecessary...!
        
        # #Set GRF vector points taking into consideration COP
        # grfMag = grf + cop    
        
        # #Calculate the GRF vector considering COP position
        # grfVec = grfMag - cop
        
        #Calculate tibial vector based on knee and ankle joint centre
        tibVec = kjc - ajc
        
        #Calculate tibial force
        #It's easier to loop this through step by step with some of the functions
        Ft = np.zeros(len(timePos))
        for vv in range(len(timePos)):
            
            #Calculate vector angle between GRF and tibia vector
            #Check for zero force data
            if np.isclose(grf[vv,1],0):
                #Set vector angle to zero as it will error anyway
                vecAng == 0
            else:
                vecAng = np.arccos(np.dot((grf[vv] / np.linalg.norm(grf[vv])),
                                          (tibVec[vv] / np.linalg.norm(tibVec[vv]))))
                
            #Calculate current Fm
            #Check for negative ankle moment and set to zero if so
            if ankleMom[vv] < 0:
                Fm = 0                
            else:
                Fm = ankleMom[vv] / 0.05
            
            #Calculate Ft as per Matijevich et al. (2020)
            Ft[vv] = np.dot(np.linalg.norm(grf[vv]), np.cos(vecAng)) + Fm
                
        #Normalise to body mass (in N)
        Ft_norm = Ft / (mass * 9.80665)
        
        #Extract tibial bone load summary variables
        #These are tibial compression force peak (i.e. BW) and tibial compression
        #force impulse (i.e. BW per second)
        peakTibForce = np.zeros([nGC])
        impulseTibForce = np.zeros([nGC])
        #Loop through stance phases
        for gg in range(nGC):
            #Extract current tibial force from stance phase
            currFt = Ft_norm[stanceInd[gg,0]:stanceInd[gg,1]]
            #Extract current time for integral calculation
            currT = timePos[stanceInd[gg,0]:stanceInd[gg,1]]
            #Calculate peak force and allocate
            peakTibForce[gg] = np.max(currFt)
            #Calculate time integral for impulse and store
            impulseTibForce[gg] = integrate.cumtrapz(currFt, currT)[-1]
                    
        #Import tibial sensor body velocity data
        df_sensors = stoToDF(dynamicFiles[tt].split('.')[0]+'_BodyKinematics_vel_bodyLocal.sto')
        
        #Extract the relevant data
        ##### TODO: factor in multiple sensors
        timeSensor = df_sensors['time'].to_numpy()
        timeSensor[0] = 0 #correct for negative time value that seems to come up
        
        #Distal medial tibia
        distMedSensor_vel = np.transpose(np.array([df_sensors['distalMedialTibia_sensor_r_X'].to_numpy(),
                                                   df_sensors['distalMedialTibia_sensor_r_Y'].to_numpy(),
                                                   df_sensors['distalMedialTibia_sensor_r_Z'].to_numpy()]))
        
        #Integrate velocity to acceleration
        distMedSensor_acc = np.transpose(np.array([np.diff(distMedSensor_vel[:,0], axis = 0) / np.diff(timeSensor),
                                                   np.diff(distMedSensor_vel[:,1], axis = 0) / np.diff(timeSensor),
                                                   np.diff(distMedSensor_vel[:,2], axis = 0) / np.diff(timeSensor)]))
        
        #Add zero at the start of acceleration variables to align with other data
        distMedSensor_acc = np.concatenate((np.zeros([1,3]),distMedSensor_acc))
        
        #Add varying degrees of Gaussian noise to accelerometer signals
        #Approach here is based on Rapp et al. (2021) whereby they add 15% of
        #the standard deviation of the signal in the simulated random noise.
        #Here we add varying degrees of this formula, adding 5%, 10%, 15%, 20% 
        #and 25%
        
        
        
        
        #Convert to 'g' units
        distMedSensor_g = distMedSensor_acc / 9.80665


# %% Adding noise to signal

# Rapp et al. adds Gaussian noise with a standard deviation that was 15% of the
# standard deviation of the signal

#Create a sine wave signal
t = np.arange(0,10,0.1)
amplitude = np.sin(t)
plt.plot(t, amplitude)

#Work out standard deviation of sin wave
sinSD = np.std(amplitude)
sinSD_15per = 0.15 * sinSD

#Add the noise to the signal (0 mean, SD)
amplitudeNoisy = amplitude + np.random.normal(0, sinSD_15per, amplitude.shape)
plt.plot(t, amplitudeNoisy)

# %%
