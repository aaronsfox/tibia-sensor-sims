# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:06:05 2020

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This function works through the Fukuchi et al. (2017) running C3D data and
    processes the marker/GRF data using Opensim to extract joint kinematics and
    kinetics using inverse kinematics and dynamics, and then residual reduction
    algorithm to optimise match between experimental GRF and kinematics.
    
    Specific notes on analysis processes are outlined in the comments.
    
"""

# %% Import packages

import opensim as osim
import os
import pandas as pd
import glob
import btk
import shutil
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

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
            
#Set generic opensim parameters
#Note that scaling paths are relative to each participants directory

#Set generic model path
genModelPath = '..\\..\\OpensimModel'

#Set generic model file
genModelFileName = genModelPath+'\\LaiArnold2017_refined_Fukuchi2017-running.osim'

#Add model geometry to search paths
osim.ModelVisualizer.addDirToGeometrySearchPaths(genModelPath+'\\Geometry')

#Create the measurement set object for later use
measurementSetObject = osim.OpenSimObject.makeObjectFromFile('..\\OpensimModel\\scaleMeasurementSet_Fukuchi2017-running.xml')
measurementSet = osim.MeasurementSet.safeDownCast(measurementSetObject)

#Set the scale task set
scaleTaskSet = osim.IKTaskSet('..\\OpensimModel\\scaleTasks_Fukuchi2017-running.xml')

#Set scale order
scaleOrder = osim.ArrayStr()
scaleOrder.set(0,'measurements')

#Set IK task set
ikTaskSet = osim.IKTaskSet('..\\OpensimModel\\ikTasks_Fukuchi2017-running.xml')

#Set up a list of markers once flattened to keep
#This mainly gets rid of the random ASY markers
markersFlatList = ['L.ASIS_1','L.ASIS_2','L.ASIS_3',
                   'L.Heel.Bottom_1','L.Heel.Bottom_2','L.Heel.Bottom_3',
                   'L.Heel.Lateral_1','L.Heel.Lateral_2','L.Heel.Lateral_3',
                   'L.Heel.Top_1','L.Heel.Top_2','L.Heel.Top_3',
                   'L.Iliac.Crest_1','L.Iliac.Crest_2','L.Iliac.Crest_3',
                   'L.MT1_1','L.MT1_2','L.MT1_3',
                   'L.MT5_1','L.MT5_2','L.MT5_3',
                   'L.PSIS_1','L.PSIS_2','L.PSIS_3',
                   'L.Shank.Bottom.Lateral_1','L.Shank.Bottom.Lateral_2','L.Shank.Bottom.Lateral_3',
                   'L.Shank.Bottom.Medial_1','L.Shank.Bottom.Medial_2','L.Shank.Bottom.Medial_3',
                   'L.Shank.Top.Lateral_1','L.Shank.Top.Lateral_2','L.Shank.Top.Lateral_3',
                   'L.Shank.Top.Medial_1','L.Shank.Top.Medial_2','L.Shank.Top.Medial_3',
                   'L.Thigh.Bottom.Lateral_1','L.Thigh.Bottom.Lateral_2','L.Thigh.Bottom.Lateral_3',
                   'L.Thigh.Bottom.Medial_1','L.Thigh.Bottom.Medial_2','L.Thigh.Bottom.Medial_3',
                   'L.Thigh.Top.Lateral_1','L.Thigh.Top.Lateral_2','L.Thigh.Top.Lateral_3',
                   'L.Thigh.Top.Medial_1','L.Thigh.Top.Medial_2','L.Thigh.Top.Medial_3',
                   'R.Heel.Bottom_1','R.Heel.Bottom_2','R.Heel.Bottom_3',
                   'R.Heel.Lateral_1','R.Heel.Lateral_2','R.Heel.Lateral_3',
                   'R.Heel.Top_1','R.Heel.Top_2','R.Heel.Top_3',
                   'R.Iliac.Crest_1','R.Iliac.Crest_2','R.Iliac.Crest_3',
                   'R.MT1_1','R.MT1_2','R.MT1_3',
                   'R.MT5_1','R.MT5_2','R.MT5_3',
                   'R.PSIS_1','R.PSIS_2','R.PSIS_3',
                   'R.Shank.Bottom.Lateral_1','R.Shank.Bottom.Lateral_2','R.Shank.Bottom.Lateral_3',
                   'R.Shank.Bottom.Medial_1','R.Shank.Bottom.Medial_2','R.Shank.Bottom.Medial_3',
                   'R.Shank.Top.Lateral_1','R.Shank.Top.Lateral_2','R.Shank.Top.Lateral_3',
                   'R.Shank.Top.Medial_1','R.Shank.Top.Medial_2','R.Shank.Top.Medial_3',
                   'R.Thigh.Bottom.Lateral_1','R.Thigh.Bottom.Lateral_2','R.Thigh.Bottom.Lateral_3',
                   'R.Thigh.Bottom.Medial_1','R.Thigh.Bottom.Medial_2','R.Thigh.Bottom.Medial_3',
                   'R.Thigh.Top.Lateral_1','R.Thigh.Top.Lateral_2','R.Thigh.Top.Lateral_3',
                   'R.Thigh.Top.Medial_1','R.Thigh.Top.Medial_2','R.Thigh.Top.Medial_3']

#Set sensor parameters
#Currently based on APDM Opal sensors
#43.7 x 39.7 x 13.7 mm (LxWxH)
#TODO: Need to confirm that depth, width and height match inertia axes
sensorMass = 0.025 #25 grams
sensorLength = 0.0437
sensorWidth = 0.0397
sensorDepth = 0.0137

#Calculate inertia tensor for sensor body
Ixx = (1/12)*sensorMass*(sensorLength**2 + sensorDepth**2)
Iyy = (1/12)*sensorMass*(sensorWidth**2 + sensorDepth**2)
Izz = (1/12)*sensorMass*(sensorWidth**2 + sensorLength**2)

#Set bodies for analysis later on
bodyStr = osim.ArrayStr()
bodyStr.append('tibia_r')
bodyStr.append('distalMedialTibia_sensor_r')

# %% Loop through subjects and process data

#Set-up lists to store opensim objects in
#This seems necessary to doubling up on objects and crashing Python
scaleTool = []
ikTool = []
idTool = []
anTool = []
bkTool = []
pkTool = []

#Loop through subjects
#Set starting subject index to account for starting at an index > 0 in lists
startInd = 0
for ii in range(0,len(subList)):
    
    #Navigate to subject directory
    os.chdir(subList[ii])
    
    # %% Model scaling
    
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
    
    #Convert c3d markers to trc file
    
    #Set-up data adapters
    c3d = osim.C3DFileAdapter()
    trc = osim.TRCFileAdapter()
    
    #Get markers
    static = c3d.read(staticFile[0])
    markersStatic = c3d.getMarkersTable(static)
    
    #Write static data to file
    trc.write(markersStatic, staticFile[0].split('.')[0]+'.trc')
    
    #Set variable for static trial trc
    staticTrial_trc = staticFile[0].split('.')[0]+'.trc'
    
    #Set scale time range
    scaleTimeRange = osim.ArrayDouble()
    scaleTimeRange.set(0,osim.Storage(staticTrial_trc).getFirstTime())
    scaleTimeRange.set(1,osim.Storage(staticTrial_trc).getFirstTime()+0.5)
    
    #Set-up scale tool    
    scaleTool.append(osim.ScaleTool())
    
    #Set mass in scale tool
    scaleTool[ii-startInd].setSubjectMass(mass)
    
    #Set generic model file
    scaleTool[ii-startInd].getGenericModelMaker().setModelFileName(genModelFileName)
    
    #Set the measurement set
    scaleTool[ii-startInd].getModelScaler().setMeasurementSet(measurementSet)
    
    #Set scale tasks
    for k in range(0,scaleTaskSet.getSize()-1):
        scaleTool[ii-startInd].getMarkerPlacer().getIKTaskSet().adoptAndAppend(scaleTaskSet.get(k))
    
    #Set marker file
    scaleTool[ii-startInd].getMarkerPlacer().setMarkerFileName(staticTrial_trc)
    scaleTool[ii-startInd].getModelScaler().setMarkerFileName(staticTrial_trc)
    
    #Set options
    scaleTool[ii-startInd].getModelScaler().setPreserveMassDist(True)
    scaleTool[ii-startInd].getModelScaler().setScalingOrder(scaleOrder)
    
    #Set time ranges
    scaleTool[ii-startInd].getMarkerPlacer().setTimeRange(scaleTimeRange)
    scaleTool[ii-startInd].getModelScaler().setTimeRange(scaleTimeRange)
    
    #Set output files
    scaleTool[ii-startInd].getModelScaler().setOutputModelFileName(subList[ii]+'_scaledModel.osim')
    scaleTool[ii-startInd].getModelScaler().setOutputScaleFileName(subList[ii]+'_scaleSet.xml')
    
    #Set marker adjuster parameters
    scaleTool[ii-startInd].getMarkerPlacer().setOutputMotionFileName(staticTrial_trc.split('.')[0]+'.mot')
    scaleTool[ii-startInd].getMarkerPlacer().setOutputModelFileName(subList[ii]+'_scaledModelAdjusted.osim')
    
    #Print and run scale tool
    scaleTool[ii-startInd].printToXML(subList[ii]+'_scaleSetup.xml')
    scaleTool[ii-startInd].run()
    
    #Add the 'sensor' markers back into the adjusted model from the scaled one
    #Load the models
    scaledModel = osim.Model(subList[ii]+'_scaledModel.osim')
    adjustedModel = osim.Model(subList[ii]+'_scaledModelAdjusted.osim')
    #Copy the markers across
    adjustedModel.getMarkerSet().adoptAndAppend(scaledModel.getMarkerSet().get('R.Tibia.Distal.Medial'))
    #Finalise connections
    adjustedModel.finalizeConnections()
    #Re-print model
    adjustedModel.printToXML(subList[ii]+'_scaledModelAdjusted.osim')
    
    #Create a model with 'sensor' bodies added
    
    ##### TODO: loop for multiple sensors
    
    #Create the distal medial tibia sensor body
    distalMedialSensorBody = osim.Body()
    distalMedialSensorBody.setName('distalMedialTibia_sensor_r')
    distalMedialSensorBody.setMass(sensorMass)
    distalMedialSensorBody.setMassCenter(osim.Vec3(0,0,0))
    distalMedialSensorBody.setInertia(osim.Inertia(Ixx,Iyy,Izz,0,0,0))
    #TODO: check order these should be in
    distalMedialSensorBody.attachGeometry(osim.Brick(osim.Vec3(sensorWidth/2, sensorLength/2, sensorDepth/2)))
    distalMedialSensorBody.get_attached_geometry(0).setColor(osim.Vec3(0/255,102/255,255/255)) #blue
    adjustedModel.addBody(distalMedialSensorBody)
    
    #Join the distal medial tibia sensor to the model
    #Get the marker location for the parent location
    locationInParent = adjustedModel.getMarkerSet().get('R.Tibia.Distal.Medial').get_location()
    orientationInParent = osim.Vec3(0,0,0) #Align to same axis as tibia
    #Set child details
    locationInChild = osim.Vec3(0,0,0)
    orientationInChild = osim.Vec3(0,0,0)
    #Create the joint
    distalMedialSensorJoint = osim.WeldJoint('distalMedialTibia_joint_r', #joint name
                                             adjustedModel.getBodySet().get('tibia_r'), #parent body
                                             locationInParent, orientationInParent,
                                             distalMedialSensorBody, #new body
                                             locationInChild, orientationInChild)
    
    #Add joint to the model
    adjustedModel.addJoint(distalMedialSensorJoint)
    
    #Edit color
    adjustedModel.getBodySet().get('distalMedialTibia_sensor_r').get_attached_geometry(0).getColor().get(2)
    
    #Finalise connections
    adjustedModel.finalizeConnections()
    
    # #Initialize the system (checks model consistency).
    # adjustedModel.initSystem()
    
    #Print model to new file
    adjustedModel.printToXML(subList[ii]+'_scaledModelAdjusted_sensors.osim')
    
    #Print confirmation
    print('Scaling complete for '+subList[ii])
    
    # %% Inverse kinematics and dynamics
    
    ##### NOTE: IK commented out as this analysis is already done...
    ##### Additions required for creating force files
        
    #Identify dynamic trial names
    dynamicFiles = []
    for file in glob.glob('*run*.c3d'):
        dynamicFiles.append(file)
        
    # #Create subject storage lists within tools current subject
    # ikTool.append([])
    idTool.append([])
    anTool.append([])
    bkTool.append([])
    pkTool.append([])
    
    #Loop through trials
    for tt in range(len(dynamicFiles)):
        
        #Convert c3d to TRC file
        
        #Set-up data adapters
        c3d = osim.C3DFileAdapter()
        trc = osim.TRCFileAdapter()
        
        #Get markers
        dynamic = c3d.read(dynamicFiles[tt])
        markersDynamic = c3d.getMarkersTable(dynamic)
        
        #Write data to file (unfiltered)
        trc.write(markersDynamic, dynamicFiles[tt].split('.')[0]+'.trc')
        
        #Filter marker data at 10 Hz like in paper
                
        #Define filter (10Hz low-pass 4th Order Butterworth)
        fs = float(markersDynamic.getTableMetaDataAsString('DataRate'))
        fc = 10.0
        w = fc / (fs / 2.0)
        order = 2 #4th order with bi-directional filter
        b,a = butter(order, w, btype = 'low', analog = False)
        
        #Flatten table to filter
        markersFlat = markersDynamic.flatten()
        
        #Get time in array format
        t = np.array(markersFlat.getIndependentColumn())
        
        #There are special cases where a marker might not be present (doesn't
        #happen very often), but need to check to avoid errors
        #Set blank list to store present markers (see comment below)
        markersKeptList = list()
        #Check and append markers present to list
        for cc in range(len(markersFlatList)):
            if markersFlatList[cc] in list(markersFlat.getColumnLabels()):
                markersKeptList.append(markersFlatList[cc])
        
        #Create numpy array to store filtered data in
        filtData = np.empty((markersFlat.getNumRows(),
                              len(markersKeptList)))
                        
        #Loop through column labels
        for cc in range(len(markersKeptList)):
            
            #Get the data in an array format
            datArr = markersFlat.getDependentColumn(markersKeptList[cc]).to_numpy()
            
            #Interpolate across nans if present
            if np.sum(np.isnan(datArr)) > 0:
            
                #Remove nans from data array and corresponding times
                yi = datArr[~np.isnan(datArr)]
                xi = t[~np.isnan(datArr)]
                
                #Interpolate data to fill gaps
                cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
                dVals = cubicF(np.linspace(t[0], t[-1], len(t)))
                
            else:
                
                #Set data values to the original array
                dVals = datArr
            
            #Filter data
            dFilt = filtfilt(b, a, dVals)
            
            #Set filtered data back in data array
            filtData[:,cc] = dFilt
            
        #Create blank time series table to fill
        filtMarkers = osim.TimeSeriesTable()
        
        #Fill row vector with data and append to the timeseries table
        #Loop through rows
        for rr in range(len(t)):
            #Create row vector from current row of array
            row = osim.RowVector.createFromMat(filtData[rr,:])
            #Append row to table
            filtMarkers.appendRow(rr,row)
            #Set time value
            filtMarkers.setIndependentValueAtIndex(rr,t[rr])
        
        #Set column labels
        filtMarkers.setColumnLabels(markersKeptList)
        
        #Pack table back to Vec3
        filtTRC = filtMarkers.packVec3()
        
        #Write meta data
        filtTRC.addTableMetaDataString('DataRate',str(fs)) #Data rate
        filtTRC.addTableMetaDataString('CameraRate',str(fs)) #Camera rate
        filtTRC.addTableMetaDataString('NumFrames',str(len(t))) #Number frames
        filtTRC.addTableMetaDataString('NumMarkers',str(len(markersKeptList))) #Number markers
        filtTRC.addTableMetaDataString('Units',markersDynamic.getTableMetaDataString('Units')) #Units
        
        #Write data to file (unfiltered)
        trc.write(filtTRC, dynamicFiles[tt].split('.')[0]+'_filtered.trc')
        
        #Set variable for dynamic trial trc
        dynamicTrial_trc = dynamicFiles[tt].split('.')[0]+'_filtered.trc'
        
        #Convert c3d force data to file
        
        #Load in the c3d data via btk
        dynamicC3D = btk.btkAcquisitionFileReader()
        dynamicC3D.SetFilename(dynamicFiles[tt])
        dynamicC3D.Update()
        dynamicAcq = dynamicC3D.GetOutput()
        
        #Extract the force platforms data
        forcePlatforms = btk.btkForcePlatformsExtractor()
        forcePlatforms.SetInput(dynamicAcq)
        forcePlatforms.Update()
        
        #Get the wrenchs for position and force data
        groundReactionWrenches = btk.btkGroundReactionWrenchFilter()
        groundReactionWrenches.SetInput(forcePlatforms.GetOutput())
        groundReactionWrenches.Update()
        
        #Grab the data from the singular force platform
        grf = groundReactionWrenches.GetOutput().GetItem(0).GetForce().GetValues()
        grm = groundReactionWrenches.GetOutput().GetItem(0).GetMoment().GetValues()
        cop = groundReactionWrenches.GetOutput().GetItem(0).GetPosition().GetValues()
        
        #Convert mm units to m
        grm = grm / 1000
        cop = cop / 1000
        
        #Define filter for forces data 
        #Replicates the Fukuchi et al. (2017) processing
        #(10Hz low-pass 4th Order Butterworth)
        fs = 300 #force sampling rate
        fc = 10.0
        w = fc / (fs / 2.0)
        order = 2 #4th order with bi-directional filter
        b,a = butter(order, w, btype = 'low', analog = False)
        
        #Create time in array format
        tf = np.linspace(0,len(grf)/fs - (1/fs), num = len(grf))
        
        #Apply lowpass filter to force data
        grfFilt = np.zeros((np.size(grf,0), np.size(grf,1)))
        for jj in range(0,np.size(grf,1)):
            
            #Interpolate across nans if present
            if np.sum(np.isnan(grf[:,jj])) > 0:
                
                #Extract data into own array
                datArr = grf[:,jj]
                
                #Remove nans from data array and corresponding times
                yi = datArr[~np.isnan(datArr)]
                xi = tf[~np.isnan(datArr)]
                
                #Interpolate data to fill gaps
                cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
                dVals = cubicF(np.linspace(tf[0], tf[-1], len(tf)))
                
                #Replace original data with interpolated
                grf[:,jj] = dVals
            
            #Filter data
            grfFilt[:,jj] = filtfilt(b, a, grf[:,jj])
            
        #Apply lowpass filter to moment data
        grmFilt = np.zeros((np.size(grm,0), np.size(grm,1)))
        for jj in range(0,np.size(grm,1)):
            
            #Interpolate across nans if present
            if np.sum(np.isnan(grm[:,jj])) > 0:
                
                #Extract data into own array
                datArr = grm[:,jj]
                
                #Remove nans from data array and corresponding times
                yi = datArr[~np.isnan(datArr)]
                xi = tf[~np.isnan(datArr)]
                
                #Interpolate data to fill gaps
                cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
                dVals = cubicF(np.linspace(tf[0], tf[-1], len(tf)))
                
                #Replace original data with interpolated
                grm[:,jj] = dVals
            
            #Filter data
            grmFilt[:,jj] = filtfilt(b, a, grm[:,jj])
        
        ##Apply lowpass filter to cop data
        #No filter, but interpolate cop data if there are nan's
        copFilt = np.zeros((np.size(cop,0), np.size(cop,1)))
        for jj in range(0,np.size(cop,1)):
            
            #Interpolate across nans if present
            if np.sum(np.isnan(cop[:,jj])) > 0:
                
                #Extract data into own array
                datArr = cop[:,jj]
                
                #Remove nans from data array and corresponding times
                yi = datArr[~np.isnan(datArr)]
                xi = tf[~np.isnan(datArr)]
                
                #Interpolate data to fill gaps
                cubicF = interp1d(xi, yi, kind = 'cubic', fill_value = 'extrapolate')
                dVals = cubicF(np.linspace(tf[0], tf[-1], len(tf)))
                
                #Replace original data with interpolated
                copFilt[:,jj] = dVals
                
            else:
            
                #Keep data
                copFilt[:,jj] = cop[:,jj] #filtfilt(b, a, cop[:,jj])
        
        #Get boolean of force threshold
        offForce = grfFilt[:,1] < 100 #20 ### higher threshold needed for CoP noise around foot strike
        
        #Set off data to zero
        grfFilt[offForce, :] = 0
        grmFilt[offForce, :] = 0
        copFilt[offForce, :] = 0
        
        #Extract data into matrix
        forcesMat = np.zeros((len(grf), 3 * 3 * 1)) #1 for number of plates
        colNo = 0
        forcesMat[:,colNo:colNo+3] = grfFilt
        forcesMat[:,colNo+3:colNo+3+3] = copFilt
        forcesMat[:,colNo+6:colNo+6+3] = grmFilt
        
        #Cast array to dataframe
        #Set force labels
        forceLabels = ['Fx','Fy','Fz','Px','Py','Pz','Mx','My','Mz']
        forceNames = list()
        for pp in range(0,len(forceLabels)):
            forceNames.append(forceLabels[pp]+str(1))
        #Place in dataframe
        df_forces = pd.DataFrame(forcesMat, columns = forceNames)
        
        #Extract the forces relating to each foot
        
        #Get all the transitions from True to False in offForce, and other direction too
        footOn = np.where(offForce & np.roll(~offForce,-1) == True)[0] + 1
        footOff = np.where(~offForce & np.roll(offForce,-1) == True)[0] + 1
        
        #Remove last foot on if it equates to length of grf (an error)
        if footOn[-1] == len(grf):
            footOn = footOn[0:-1]
        
        #Sort out the first foot on index if the first foot off is first
        if footOff[0] < footOn[0]:
            footOn = np.append(0,footOn)
            
        #Sort out the last foot on to off
        if footOff[-1] < footOn[-1]:
            footOff = np.append(footOff,len(offForce)-1)
            
        #Loop through foot on events and identify which MT1 marker is closer to COP
        #After identifying this, append the data to relevant variable
        
        #Set empty arrays for variables
        #Right forces
        ground_force_r_vx = np.zeros(len(offForce))
        ground_force_r_vy = np.zeros(len(offForce))
        ground_force_r_vz = np.zeros(len(offForce))
        #Left forces
        ground_force_l_vx = np.zeros(len(offForce))
        ground_force_l_vy = np.zeros(len(offForce))
        ground_force_l_vz = np.zeros(len(offForce))
        #Right torques
        ground_torque_r_x = np.zeros(len(offForce))
        ground_torque_r_y = np.zeros(len(offForce))
        ground_torque_r_z = np.zeros(len(offForce))
        #Left torques
        ground_torque_l_x = np.zeros(len(offForce))
        ground_torque_l_y = np.zeros(len(offForce))
        ground_torque_l_z = np.zeros(len(offForce))
        #Right position
        ground_force_r_px = np.zeros(len(offForce))
        ground_force_r_py = np.zeros(len(offForce))
        ground_force_r_pz = np.zeros(len(offForce))
        #Left position
        ground_force_l_px = np.zeros(len(offForce))
        ground_force_l_py = np.zeros(len(offForce))
        ground_force_l_pz = np.zeros(len(offForce))
        
        #Loop through force events
        for pp in range(len(footOn)):
            
            #Set the index for the current force event
            #This goes to the middle of the current foot on to foot off
            forceInd = int(footOn[pp] + ((footOff[pp] - footOn[pp]) / 2))
            
            #Get the time of the current foot on event
            currTime = tf[forceInd]
            
            #Get the COP at the current foot on event
            currCOP = np.array([df_forces.iloc[forceInd]['Px1'],
                                df_forces.iloc[forceInd]['Py1'],
                                df_forces.iloc[forceInd]['Pz1']])
            
            #Get the current MT5 positions from marker data
            mkrInd = int(np.where(t > currTime)[0][0] - 1)
            currRMT1 = np.array([filtTRC.getDependentColumn('R.MT1').getElt(0,mkrInd).get(0)/1000,
                                 filtTRC.getDependentColumn('R.MT1').getElt(0,mkrInd).get(1)/1000,
                                 filtTRC.getDependentColumn('R.MT1').getElt(0,mkrInd).get(2)/1000])
            currLMT1 = np.array([filtTRC.getDependentColumn('L.MT1').getElt(0,mkrInd).get(0)/1000,
                                 filtTRC.getDependentColumn('L.MT1').getElt(0,mkrInd).get(1)/1000,
                                 filtTRC.getDependentColumn('L.MT1').getElt(0,mkrInd).get(2)/1000])
            
            #Calculate 3D distance between COP and markers
            distRMT1 = (((currRMT1[0]-currCOP[0])**2)+
                        ((currRMT1[1]-currCOP[1])**2)+
                        ((currRMT1[2]-currCOP[2])**2))**(1/2)
            distLMT1 = (((currLMT1[0]-currCOP[0])**2)+
                        ((currLMT1[1]-currCOP[1])**2)+
                        ((currLMT1[2]-currCOP[2])**2))**(1/2)
            
            #Allocate closest marker
            if distRMT1 < distLMT1:
                #Append force data to relevant variable
                ground_force_r_vx[footOn[pp]:footOff[pp]] = df_forces['Fx1'].values[footOn[pp]:footOff[pp]]
                ground_force_r_vy[footOn[pp]:footOff[pp]] = df_forces['Fy1'].values[footOn[pp]:footOff[pp]]
                ground_force_r_vz[footOn[pp]:footOff[pp]] = df_forces['Fz1'].values[footOn[pp]:footOff[pp]]
                ground_torque_r_x[footOn[pp]:footOff[pp]] = df_forces['Mx1'].values[footOn[pp]:footOff[pp]]
                ground_torque_r_y[footOn[pp]:footOff[pp]] = df_forces['My1'].values[footOn[pp]:footOff[pp]]
                ground_torque_r_z[footOn[pp]:footOff[pp]] = df_forces['Mz1'].values[footOn[pp]:footOff[pp]]
                ground_force_r_px[footOn[pp]:footOff[pp]] = df_forces['Px1'].values[footOn[pp]:footOff[pp]]
                ground_force_r_py[footOn[pp]:footOff[pp]] = df_forces['Py1'].values[footOn[pp]:footOff[pp]]
                ground_force_r_pz[footOn[pp]:footOff[pp]] = df_forces['Pz1'].values[footOn[pp]:footOff[pp]]
            elif distRMT1 > distLMT1:
                #Append force data to relevant variable
                ground_force_l_vx[footOn[pp]:footOff[pp]] = df_forces['Fx1'].values[footOn[pp]:footOff[pp]]
                ground_force_l_vy[footOn[pp]:footOff[pp]] = df_forces['Fy1'].values[footOn[pp]:footOff[pp]]
                ground_force_l_vz[footOn[pp]:footOff[pp]] = df_forces['Fz1'].values[footOn[pp]:footOff[pp]]
                ground_torque_l_x[footOn[pp]:footOff[pp]] = df_forces['Mx1'].values[footOn[pp]:footOff[pp]]
                ground_torque_l_y[footOn[pp]:footOff[pp]] = df_forces['My1'].values[footOn[pp]:footOff[pp]]
                ground_torque_l_z[footOn[pp]:footOff[pp]] = df_forces['Mz1'].values[footOn[pp]:footOff[pp]]
                ground_force_l_px[footOn[pp]:footOff[pp]] = df_forces['Px1'].values[footOn[pp]:footOff[pp]]
                ground_force_l_py[footOn[pp]:footOff[pp]] = df_forces['Py1'].values[footOn[pp]:footOff[pp]]
                ground_force_l_pz[footOn[pp]:footOff[pp]] = df_forces['Pz1'].values[footOn[pp]:footOff[pp]]
                
        #Export forces to .mot file
        #Set labels for file
        motLabels = ['time',
                     'ground_force_r_vx', 'ground_force_r_vy', 'ground_force_r_vz',
                     'ground_force_r_px', 'ground_force_r_py', 'ground_force_r_pz',
                     'ground_torque_r_x', 'ground_torque_r_y', 'ground_torque_r_z',
                     'ground_force_l_vx', 'ground_force_l_vy', 'ground_force_l_vz',
                     'ground_force_l_px', 'ground_force_l_py', 'ground_force_l_pz',
                     'ground_torque_l_x', 'ground_torque_l_y', 'ground_torque_l_z']
        
        #Create storage object
        forcesStorage = osim.Storage()
        
        #Set storage file labels
        colLabels = osim.ArrayStr()
        for cc in range(0,len(motLabels)):
            colLabels.append(motLabels[cc])
        forcesStorage.setColumnLabels(colLabels)
        
        #Create data array
        forceData = np.transpose(np.array([ground_force_r_vx, ground_force_r_vy, ground_force_r_vz,
                                           ground_force_r_px, ground_force_r_py, ground_force_r_pz,
                                           ground_torque_r_x, ground_torque_r_y, ground_torque_r_z,
                                           ground_force_l_vx, ground_force_l_vy, ground_force_l_vz,
                                           ground_force_l_px, ground_force_l_py, ground_force_l_pz,
                                           ground_torque_l_x, ground_torque_l_y, ground_torque_l_z]))
        
        #Append data to storage object
        nrow, ncol = forceData.shape
        
        #Add data
        for qq in range(nrow):
            row = osim.ArrayDouble()
            for jj in range(ncol):
                row.append(forceData[qq,jj])
            #Add data to storage
            forcesStorage.append(tf[qq], row)
            
        #Set name for storage object
        forcesStorage.setName(dynamicFiles[tt].split('.')[0]+'_grf')
        
        #Print to file
        forcesStorage.printResult(forcesStorage, dynamicFiles[tt].split('.')[0]+'_grf',
                                  os.getcwd(), 1/fs, '.mot')
        
        #Create external forces .xml file
        forceXML = osim.ExternalLoads()
        
        #Create and append the right GRF external force
        rightGRF = osim.ExternalForce()
        rightGRF.setName('RightGRF')
        rightGRF.setAppliedToBodyName('calcn_r')
        rightGRF.setForceExpressedInBodyName('ground')
        rightGRF.setPointExpressedInBodyName('ground')
        rightGRF.setForceIdentifier('ground_force_r_v')
        rightGRF.setPointIdentifier('ground_force_r_p')
        rightGRF.setTorqueIdentifier('ground_torque_r_')
        forceXML.cloneAndAppend(rightGRF)
        
        #Create and append the left GRF external force
        leftGRF = osim.ExternalForce()
        leftGRF.setName('LeftGRF')
        leftGRF.setAppliedToBodyName('calcn_l')
        leftGRF.setForceExpressedInBodyName('ground')
        leftGRF.setPointExpressedInBodyName('ground')
        leftGRF.setForceIdentifier('ground_force_l_v')
        leftGRF.setPointIdentifier('ground_force_l_p')
        leftGRF.setTorqueIdentifier('ground_torque_l_')
        forceXML.cloneAndAppend(leftGRF)
        
        #Set GRF datafile
        forceXML.setDataFileName(dynamicFiles[tt].split('.')[0]+'_grf'+'.mot')
        
        #Set filtering for kinematics
        forceXML.setLowpassCutoffFrequencyForLoadKinematics(10)
        
        #Write to file
        forceXML.printToXML(dynamicFiles[tt].split('.')[0]+'_grf'+'.xml')
        
        #Get start and end times for use in analyses
        startTime = osim.Storage(dynamicTrial_trc).getFirstTime()
        endTime = osim.Storage(dynamicTrial_trc).getLastTime()
        
        # #Initialise IK tool
        # ikTool[ii-startInd].append(osim.InverseKinematicsTool())
        
        # #Set the model
        # ikTool[ii-startInd][tt].set_model_file(subList[ii]+'_scaledModelAdjusted.osim')
        
        # #Set task set
        # ikTool[ii-startInd][tt].set_IKTaskSet(ikTaskSet)
        
        # #Set marker file
        # ikTool[ii-startInd][tt].set_marker_file(dynamicTrial_trc)
        
        # #Set times
        # ikTool[ii-startInd][tt].setStartTime(startTime)
        # ikTool[ii-startInd][tt].setEndTime(endTime)
        
        # # #Set error reporting to false
        # # ikTool[ii-startInd][tt].set_report_errors(False)
        
        # #Set output filename
        # ikTool[ii-startInd][tt].set_output_motion_file(dynamicFiles[tt].split('.')[0]+'_ik.mot')
        
        # #Print and run tool
        # ikTool[ii-startInd][tt].printToXML(dynamicFiles[tt].split('.')[0]+'_setupIK.xml')
        # ikTool[ii-startInd][tt].run()
        
        # #Rename marker errors file
        # shutil.move('_ik_marker_errors.sto',dynamicFiles[tt].split('.')[0]+'_ik_marker_errors.sto')
        
        # #Print confirmation
        # print('IK complete for '+dynamicFiles[tt].split('.')[0])
        
        #Initialise ID tool
        idTool[ii-startInd].append(osim.InverseDynamicsTool())
        
        #Set the model
        idTool[ii-startInd][tt].setModelFileName(subList[ii]+'_scaledModelAdjusted_sensors.osim')
        
        #Set times
        idTool[ii-startInd][tt].setStartTime(startTime)
        idTool[ii-startInd][tt].setEndTime(endTime)
        
        #Set external loads file
        idTool[ii-startInd][tt].setExternalLoadsFileName(dynamicFiles[tt].split('.')[0]+'_grf'+'.xml')
        
        #Set kinematics
        idTool[ii-startInd][tt].setCoordinatesFileName(dynamicFiles[tt].split('.')[0]+'_ik.mot')
        
        #Set lowpass filter frequency for kinematics (10Hz)
        idTool[ii-startInd][tt].setLowpassCutoffFrequency(10)
        
        #Set output filename
        idTool[ii-startInd][tt].setOutputGenForceFileName(dynamicFiles[tt].split('.')[0]+'_id.sto')
        
        #Print and run tool
        idTool[ii-startInd][tt].printToXML(dynamicFiles[tt].split('.')[0]+'_setupID.xml')
        idTool[ii-startInd][tt].run()
        
        ##### NOTE: filter ID results given lack of CoP noise
        
        #Print confirmation
        print('ID complete for '+dynamicFiles[tt].split('.')[0])
        
        #Run analysis to extract body kinematics
        
        #Initialise analysis tool
        anTool[ii-startInd].append(osim.AnalyzeTool())
        
        #Set the model
        anTool[ii-startInd][tt].setModelFilename(subList[ii]+'_scaledModelAdjusted_sensors.osim')
        
        #Set times
        anTool[ii-startInd][tt].setStartTime(startTime)
        anTool[ii-startInd][tt].setFinalTime(endTime)
        
        #Set external loads file
        anTool[ii-startInd][tt].setExternalLoadsFileName(dynamicFiles[tt].split('.')[0]+'_grf'+'.xml')
        
        #Set kinematics
        anTool[ii-startInd][tt].setCoordinatesFileName(dynamicFiles[tt].split('.')[0]+'_ik.mot')
        
        #Set lowpass filter frequency for kinematics (10Hz)
        anTool[ii-startInd][tt].setLowpassCutoffFrequency(10)
        
        #Create body kinematics analysis
        bkTool[ii-startInd].append(osim.BodyKinematics())
        
        #Set times
        bkTool[ii-startInd][tt].setStartTime(startTime)
        bkTool[ii-startInd][tt].setEndTime(endTime)
        
        #Set bodies for analysis
        bkTool[ii-startInd][tt].setBodiesToRecord(bodyStr)
        
        #Set to report in body frame
        bkTool[ii-startInd][tt].setExpressResultsInLocalFrame(True)
        
        #Append body kinematics to analyse tool
        anTool[ii-startInd][tt].getAnalysisSet().cloneAndAppend(bkTool[ii-startInd][tt])
        
        #Create point kinematics analysis
        pkTool[ii-startInd].append(osim.PointKinematics())
        
        #Set times
        pkTool[ii-startInd][tt].setStartTime(startTime)
        pkTool[ii-startInd][tt].setEndTime(endTime)
        
        #Set knee joint centre point details and append to analysis tool
        pkTool[ii-startInd][tt].setBody(adjustedModel.getBodySet().get('tibia_r'))
        pkTool[ii-startInd][tt].setRelativeToBody(adjustedModel.getGround())
        pkTool[ii-startInd][tt].setPoint(osim.Vec3(0,0,0))
        pkTool[ii-startInd][tt].setPointName('KJC')
        pkTool[ii-startInd][tt].setName('KJC_PointKinematics')
        anTool[ii-startInd][tt].getAnalysisSet().cloneAndAppend(pkTool[ii-startInd][tt])
        
        #Set ankle joint centre point details and append to analysis tool
        pkTool[ii-startInd][tt].setBody(adjustedModel.getBodySet().get('talus_r'))
        pkTool[ii-startInd][tt].setRelativeToBody(adjustedModel.getGround())
        pkTool[ii-startInd][tt].setPoint(osim.Vec3(0,0,0))
        pkTool[ii-startInd][tt].setPointName('AJC')
        pkTool[ii-startInd][tt].setName('AJC_PointKinematics')
        anTool[ii-startInd][tt].getAnalysisSet().cloneAndAppend(pkTool[ii-startInd][tt])
        
        #Print and run tool
        anTool[ii-startInd][tt].printToXML(dynamicFiles[tt].split('.')[0]+'_setupAnalyze.xml')
        #Analyze tool needs to be re-imported to run for some reason...
        osim.AnalyzeTool(dynamicFiles[tt].split('.')[0]+'_setupAnalyze.xml').run()
        
        #Rename body kinematics outputs
        shutil.move('_BodyKinematics_vel_bodyLocal.sto',
                    dynamicFiles[tt].split('.')[0]+'_BodyKinematics_vel_bodyLocal.sto')
        shutil.move('_BodyKinematics_pos_global.sto',
                    dynamicFiles[tt].split('.')[0]+'_BodyKinematics_pos_bodyLocal.sto')
        shutil.move('_BodyKinematics_acc_bodyLocal.sto',
                    dynamicFiles[tt].split('.')[0]+'_BodyKinematics_acc_bodyLocal.sto')
        
        #Rename point kinematic outputs
        shutil.move('_AJC_PointKinematics_AJC_acc.sto',
                    dynamicFiles[tt].split('.')[0]+'_AJC_acc.sto')
        shutil.move('_AJC_PointKinematics_AJC_pos.sto',
                    dynamicFiles[tt].split('.')[0]+'_AJC_pos.sto')
        shutil.move('_AJC_PointKinematics_AJC_vel.sto',
                    dynamicFiles[tt].split('.')[0]+'_AJC_vel.sto')
        shutil.move('_KJC_PointKinematics_KJC_acc.sto',
                    dynamicFiles[tt].split('.')[0]+'_KJC_acc.sto')
        shutil.move('_KJC_PointKinematics_KJC_pos.sto',
                    dynamicFiles[tt].split('.')[0]+'_KJC_pos.sto')
        shutil.move('_KJC_PointKinematics_KJC_vel.sto',
                    dynamicFiles[tt].split('.')[0]+'_KJC_vel.sto')
        
        #Print confirmation
        print('Body & point kinematics complete for '+dynamicFiles[tt].split('.')[0])
    
    #Print confirmation for subject
    print('IK & ID & Analyze complete for '+subList[ii])
    
    #Shift back up to data directory
    os.chdir(dataDir)
    
# %% Finish up

#Navigate back to main directory
os.chdir(mainDir)
    
# %% ----- End of processFukuchi2017-Running.py -----