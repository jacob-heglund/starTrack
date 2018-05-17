# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:50:36 2018

@author: Jacob
"""

from satSimFunc import *

#TODO: find range of "safe" starting thetas and omegas that don't break the simulation

# =============================================================================
# function description:
# generates a set of n swaths of the sky
# =============================================================================
# inputs: 
# n - number of simulations to run
# dt - timestep in the dynamics sim
# tFinal - ending time of each sim
# thetaPlot - generates a plot of the angular position after each sim
# =============================================================================
# outputs: 

def generateData(thetaInit, omegaInit, alphaInit, visualizeSky, trial):
    dt = .5
    numDataPoints = int(tFinal / dt)
    
    # save directory
    directory = r"C:/dev/starTrack/data/trial" + str(trial) + "/"            
    
    thetaREC, omegaREC, alphaREC,tREC = satSim(thetaInit, omegaInit, alphaInit, tFinal, dt, trial, visualizeSky, directory)
    
    thetaREC = np.transpose(thetaREC)
    omegaREC = np.transpose(omegaREC)
    alphaREC = np.transpose(alphaREC)
    tREC = np.reshape(tREC, (numDataPoints,1))
    
    csvData = np.hstack((tREC, thetaREC, omegaREC, alphaREC))
    fileNameCSV = "rotational_dynamics.csv"
    savePathCSV = directory + fileNameCSV
    np.savetxt(savePathCSV, csvData, fmt='%.5f', delimiter=',', comments='',header= "t,thetaX,thetaY,thetaZ,omegaX,omegaY,omegaZ,alphaX,alphaY,alphaZ")


np.random.seed(1)

for trial in range(1,10):
    print('trial:' + str(trial))
    tFinal = 10.0
    
    thetaInitX = np.random.uniform(low = -0.785398163, high = 0.785398163)
    thetaInitY = np.random.uniform(low = -0.785398163, high = 0.785398163)
    thetaInitZ = 0.0
    thetaInit = np.transpose(np.array((thetaInitX, thetaInitY, thetaInitZ)))
    
    # 1 deg/s
    omegaInitX = np.random.uniform(low = -0.01745329252, high = 0.01745329252)
    omegaInitY = np.random.uniform(low = -0.01745329252, high = 0.01745329252)
    omegaInitZ = 0
    omegaInit = np.transpose(np.array((omegaInitX, omegaInitY, omegaInitZ)))
    '''
    thetaInit = np.transpose(np.array((-0.785398163, -0.785398163, 0.0)))
    omegaInit = np.transpose(np.array((-0.01745329252, -0.01745329252, 0.0)))
    '''
    # this should be zero
    alphaInit = np.transpose(np.array((0, 0, 0)))
    
    visualizeSky = True
    
    generateData(thetaInit, omegaInit, alphaInit, visualizeSky, trial)


#speak = wincl.Dispatch("SAPI.SpVoice")
#speak.Speak("Program Complete")













