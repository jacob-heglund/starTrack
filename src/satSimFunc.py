import numpy as np
import matplotlib.pyplot as plt
import PIL as pil
import win32com.client as wincl
import winsound
from os import makedirs

# =============================================================================
# function description:
# function satSim takes initial conditions and returns the angular position 
# in degrees as Euler angles as a function of time
# =============================================================================
# inputs: (thetaInit, omegaInit, alphaInit, simTime, dt, plot)
#       thetaInit - initial angular position about principle body axes, 3x1 array
#       omegaInit - initial angular velocity about principle body axes, 3x1 array
#       alphaInit - initial angular acceleration about principle body axes, 3x1 array
#       tFinal    - end time of the simulation
#       dt        - timestep between evaluations of the simulation
#       plot      - outputs a plot of the three angular positions as a function of time
#                   to the terminal
# =============================================================================
# outputs: (thetaREC, omegaREC, tREC)
#       thetaREC - theta at each timestep in radians
#       omegaREC - omega at each timestep in radians
#       tREC     - time at each timestep
# =============================================================================
# notes: 
# using the X-Y-Z Euler Angle convention:
# theta[0] = angle rotated about the body X axis in radians
# theta[1] = angle rotated about the body Y axis in radians
# theta[2] = angle rotated about the body Z axis in radians

# and the reference frame has Euler angles
# thetaRef[0] = thetaRef[1] = thetaRef[2] = 0
# =============================================================================
# example:
# thetaInit = np.transpose(np.array((2, 1, 1)))
# omegaInit = np.transpose(np.array((0, 5, 0)))
# alphaInit = np.transpose(np.array((0, 0, 0)))
# tFinal = 10
# dt = .1
# (theta, omega, time) = satSim(thetaInit, omegaInit, alphaInit, tFinal, dt, False)
# =============================================================================

def satSim(thetaInit, omegaInit, alphaInit, tFinal, dt, trial, visualizeSky, directory):
    # star tracker parameters
    FPS = 2
    FOV = 20 # degrees in each direction
    numPics = FPS * tFinal
    
    # dynamics parameters
    numTimeSteps = int(tFinal / dt)
    tInit = 0
    thetaREC = np.zeros((3, numTimeSteps))
    omegaREC = np.zeros((3, numTimeSteps))
    alphaREC = np.zeros((3, numTimeSteps))
    tREC =  np.zeros((1,numTimeSteps))
    
    # initial pose of the satellite
    satPoseInit = np.eye(4)

    # create sub-directories to save images
    subDirFOV = r"satFOV/"
    makedirs(directory + subDirFOV)
    subDirFOVsky = r"satFOVsky/"
    makedirs(directory + subDirFOVsky)
    
    # simulate the dynamics of the satellite while getting images from the tracker
    for i in range(0, numTimeSteps):
        tCurr = tInit + dt*i
        
        # get angular position and velocity as a function of time
        thetaCurr = thetaInit + omegaInit*tCurr + .5*alphaInit*tCurr**2
        omegaCurr = omegaInit + alphaInit*tCurr
        alphaCurr = alphaInit
        
        # find current satellite pose in space
        satPoseCurr = rotationEuler(satPoseInit, thetaCurr[0], thetaCurr[1], thetaCurr[2])
        
        # get the view of the current timestep of the satellite
        satImageFOV, satImageFOVsky = pose2view(satPoseCurr, visualizeSky)
                     
        # save the image with a unique filename
        fileNameFOV = "t" + str(tCurr) + ".jpg"
        fileNameFOVsky = "t" + str(tCurr) + ".jpg"
        
        savePathFOV = directory + subDirFOV + fileNameFOV
        savePathFOVsky = directory + subDirFOVsky + fileNameFOVsky
        
        satImageFOV.save(savePathFOV)
        satImageFOVsky.save(savePathFOVsky)

        printStr = "files saved for t = " + str(tCurr)
        print(printStr)
            
        thetaREC[:, i] = thetaCurr
        omegaREC[:, i] = omegaCurr
        alphaREC[:, i] = alphaCurr
        tREC[:, i] = tCurr
    return thetaREC, omegaREC, alphaREC,tREC

# =============================================================================
# function description:
# function takes a 4x4 numpy array and rotates using a homogenous transformation
# matrix using the XYZ Euler angle convention
# =============================================================================
    # inputs: (satPoseInit, thetaX, thetaY, thetaZ)
# satPoseInit - a 4x4 numpy array representing the initial pose of the satellite
# thetaX - angle rotated about the world frame X axis
# thetaY - angle rotated about the world frame Y axis
# thetaZ - angle rotated about the world frame Z axis      
# =============================================================================
# outputs: satPoseFinal
# satPoseFinal - a 4x4 numpy array representing the initial pose of the satellite

def rotationEuler(satPoseInit, thetaX, thetaY, thetaZ):
    Rx = np.array(([1, 0, 0], 
                   [0, np.cos(thetaX), -np.sin(thetaX)], 
                   [0, np.sin(thetaX), np.cos(thetaX)]))
    
    Ry = np.array(([np.cos(thetaY), 0, np.sin(thetaY)], 
                    [0, 1, 0],
                    [-np.sin(thetaY), 0, np.cos(thetaY)]))
            
    Rz = np.array(([np.cos(thetaZ), -np.sin(thetaZ), 0],
                    [np.sin(thetaZ), np.cos(thetaZ), 0],
                    [0, 0, 1]))
    
    # the transform matrix from all three rotations is
    # (the @ sign is numpy's implementation of matrix multiplication)
    T = Rx @ Ry @ Rz
    p = np.array(([0], [0], [0]))    
    T = np.append(T, p, axis = 1)
    aug = np.array(([0, 0, 0, 1]))
    T = np.vstack((T, aug))
    
    satPoseFinal = np.matmul(T, satPoseInit)
    return satPoseFinal

#==============================================================================
# function description: finds the pixel at which the star tracker is pointing 
# for given pose of the satellite
# We have chosen the first column of the satellite to represent the 
# direction in which the star tracker is pointing.

# First, find the the right ascension and declination using spherical coordinates,
# then convert to image coordinates using the inverse flat square (plate carree) 
# projection which maps from a rectangular image to a sphere
    
# Using the physics convention, we have the set (r, theta, phi), where r is 
# chosen to be a unit distance, theta is the polar angle, and phi is the azimuthal angle.
# =============================================================================
# inputs: satPose
# satPose - 4x4 matrix that describes the pose of the satellite in space
# =============================================================================
# outputs: the view of the satellite as an image

def pose2view(satPose, visualizeSky):
    trackerVec = np.reshape(satPose[:, 0], (4,1))
    x = trackerVec[0]
    y = trackerVec[1]
    z = trackerVec[2]
    
    theta = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))
    phi = np.arctan2(y, x)

    declin = np.deg2rad(90) - theta
    rightAsc = phi
    
    declinDeg = np.rad2deg(declin)
    rightAscDeg = np.rad2deg(rightAsc)
    
    # TOD: fix going off the edge 
    
    # TOD: fix weird image processing problems (white dots where they shouldn't be
    # when the satellite rotates about the x axis)
    # HAC: satellite x axis rotation is 0 for all time, so the star
    # tracker image is always an unrotated square
    
    thetaX = 0
    image = satFOV(rightAscDeg, declinDeg, thetaX, visualizeSky)
    
    return image

#==============================================================================
# function description: 
# transform the coordinate system so the top left pixel is the origin
# positive x moves right along the image, and positive y moves down the image
# get the pixel values from the continuous right ascension and declination
# using the floor function
# =============================================================================
# inputs: rightAsc, dec
# =============================================================================
# outputs: p1pix - the pixel coordinate in the image (origin of this coordinate
# system is in the upper left of the image)
    
def celest2pixel(rightAsc, declin):
    #size of the image in pixels
    xSize = 16384
    ySize = 8192
    # TOD: deal with going past the limits of (-180, 180) in the x direction and 
    #(-90, 90) in the y direction

    # the homogenous transform between the two coordinate systems
    q1 = xSize/2
    q2 = ySize/2
    q3 = 0
    T = np.array(([1, 0, 0, q1], [0, -1, 0, q2], [0, 0, -1, q3], [0, 0, 0, 1]))
    
    # the position in the celestial coordinate system, augmented with a 1 
    # to allow us to use homoegenous transformation matrix on the point
    p0 = np.array(([rightAsc, declin, 0, 1]))
    pixelConversion = np.array(([xSize/360, ySize/180, 0, 1]))
    
    p0pix = np.multiply(p0, pixelConversion)
    p0pix = np.reshape(p0pix, (4,1))

    p1pix = np.dot(T, p0pix)
    p1pix = np.floor(p1pix)

    p1pix = p1pix[0:2]
    #xPix = p1pix[0]
    #yPix = p1pix[1]
    return p1pix

#==============================================================================
# function description: find the FOV of the star tracker
# =============================================================================
# inputs: rightAsc, declin, thetaX (in degrees)
# =============================================================================
# outputs: imageSatView, imageSatViewSky

def satFOV(rightAsc, declin, thetaZ, visualizeSky):
    # several sources suggest this is a reasonable value for star tracker FOV
    # read more in fovInfo.txt
    xFOV = 20 # deg
    yFOV = 20 # deg
    
    # the satellite FOV is a square defined by 4 bounds
    leftBound = rightAsc - xFOV/2
    rightBound = rightAsc + xFOV/2
    bottomBound = declin - yFOV/2
    topBound = declin + yFOV/2
    
    # pixel density in the image is 45.51 pixels / 1 degree 
    # therefore, over 20 degrees of FOV, we have 
    # (45.51 pixels / 1 deg) * 20 deg = 910.2 pixels
    # sample 920 points evenly along this interval so we don't miss any pixels
    numPoints = 920
    
    # divide by 2 since this takes forever otherwise
    #numPoints = int(920/2)
    xBound = np.linspace(leftBound, rightBound, numPoints)
    yBound = np.linspace(topBound, bottomBound, numPoints)
    
    # this will be filled with points sampled evenly from the satellite's FOV
    # with sampledPts[0, i] = rightAsc, sampledPts[1, i] = declin
    sampledPts = np.zeros((3, numPoints**2))
    idx = 0
    
    for i in range(numPoints):
        for j in range(numPoints):
            point = np.array(([xBound[j], yBound[i]]))
            sampledPts[0:2, idx] = point 
            idx = idx + 1
    
    # rotate these points to find their corresponding rightAsc, declin as
    # the rotated satellite would see them
    
    # Euler rotation matrix - rotates about the axis coming out of the
    # rightAsc-declin plane, so equivalent to a positive rotation about the z-axis
    Rz = np.array(([np.cos(np.deg2rad(thetaZ)), -np.sin(np.deg2rad(thetaZ)), 0],
                    [np.sin(np.deg2rad(thetaZ)), np.cos(np.deg2rad(thetaZ)), 0],
                    [0, 0, 1]))
    
    sampledPtsRot = np.dot(Rz, sampledPts)
    
    #pixelCoords = np.zeros((2, numPoints**2))
    imageSkymap = pil.Image.open('TychoSkymap.tif')
    rgbSkymap = imageSkymap.convert('RGB')   

    xSizeFOV = numPoints
    ySizeFOV = numPoints
    
    xSizeSky = 1
    ySizeSky = 1
    
    if visualizeSky:
        xSizeSky = 16384
        ySizeSky = 8192
    
    
    rgbInit = (125, 125, 125)
    
    # create a new image of the satellite's FOV
    imageSatView = pil.Image.new('RGB', (xSizeFOV, ySizeFOV), (rgbInit))
    pixelsSatView = imageSatView.load()    
    
    # create a new image of the satellite's FOV in the context of the sky
    imageSatViewSky = pil.Image.new('RGB', (xSizeSky, ySizeSky), (rgbInit))
    if visualizeSky:
        pixelsSatViewSky = imageSatViewSky.load()    
    
    
    for i in range(numPoints**2):
        # get the pixel coordinates from the rotated celestial coordinates
        pixelCoordsSkymap = np.reshape(celest2pixel(sampledPtsRot[0, i], sampledPtsRot[1, i]), (2))
        r, g, b = rgbSkymap.getpixel((pixelCoordsSkymap[0], pixelCoordsSkymap[1]))
        
        xSatView = (i % xSizeFOV)
        ySatView = int(np.floor(i / ySizeFOV))
        pixelsSatView[xSatView, ySatView] = (r,g,b)
                
        # create the image of the FOV in in the context of the whole sky (big image, really slow)
        if visualizeSky:
            pixelsSatViewSky[pixelCoordsSkymap[0], pixelCoordsSkymap[1]] = (r, g, b)
        
        
    return imageSatView, imageSatViewSky




























