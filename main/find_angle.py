import os, sys
import h5py
import cv2
import numpy
import gc
import mpi4py
import matplotlib.pyplot as plt
from scipy import ndimage
from mpi4py import MPI
from time import time

sys.path.append(os.path.abspath('../myFunctions'))
import fileIO
import imageProcess
import myCythonFunc
import dataViewer
import misc
import tracking

inputFile = r'Z:\Geeta-Share\cube assembly\Attachment Angle\20160614\20160614-019\019.avi'
outputFile = r'Z:\Geeta-Share\cube assembly\Attachment Angle\20160614\20160614-019\019.h5'
inputDir = r'Z:\Geeta-Share\cube assembly\Attachment Angle\20160614\20160614-019'
outputDir = r'Z:\Geeta-Share\cube assembly\Attachment Angle\20160614\20160614-019\output'
pixInNM = 1.090185
fps = 25
microscope = 'JOEL2010' #'JOEL2010','T12'
camera = 'One-view' #'Orius', 'One-view'
owner = 'Shu Fen'
zfillVal = 6
fontScale = 1
structure = [[1,1,1],[1,1,1],[1,1,1]]


#######################################################################
# DATA PROCESSSING
# 1. READ THE INPUT FILES AND STORE THEM FRAME-WISE IN H5 FILE
# 2. PERFORM BACKGROUND SUBTRACTION (IF REQUIRED)
#######################################################################
#########
# PART 1
#########

#fp = fileIO.createH5(outputFile)
#[gImgRawStack,row,col,numFrames] = fileIO.readAVI(inputFile)
#frameList = range(1,numFrames+1)
#for frame in frameList:
    #fileIO.writeH5Dataset(fp,'/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal),gImgRawStack[:,:,frame-1])
        
#fp.attrs['inputFile'] = inputFile
#fp.attrs['outputFile'] = outputFile
#fp.attrs['inputDir'] = inputDir
#fp.attrs['outputDir'] = outputDir
#fp.attrs['pixInNM'] = pixInNM
#fp.attrs['pixInAngstrom'] = pixInNM*10
#fp.attrs['fps'] = fps
#fp.attrs['microscope'] = microscope
#fp.attrs['camera'] = camera
#fp.attrs['owner'] = owner
#fp.attrs['row'] = row
#fp.attrs['col'] = col
#fp.attrs['numFrames'] = numFrames
#fp.attrs['frameList'] = range(1,numFrames+1)
#fp.attrs['zfillVal'] = zfillVal
    
#fileIO.mkdirs(outputDir)
#fileIO.saveImageSequence(gImgRawStack,outputDir+'/dataProcessing/gImgRawStack')
    
#del gImgRawStack
#fp.flush(), fp.close()
#gc.collect()



#######################################################################
# FINDING OUT RELATIVE DISTANCE AND ANGLE BETWEEN PARTICLES
##############################################################################

#print "Finding the relative distance"
#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#time = []
#for frame in frameList:
    #time.append(1.0*frame/fps)
#time = numpy.asarray(time)
#slopes1 = time.copy(); slopes1[:] = numpy.NaN
#slopes2 = time.copy(); slopes2[:] = numpy.NaN
#intersection_angles = time.copy(); intersection_angles[:] = numpy.NaN

#for frame in frameList:
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #helper = imageProcess.FindAngleHelper(gImgRaw)
    #helper.connect()
    #plt.show()
    #slopes1[frame-1] = helper.first_slope
    #slopes2[frame-1] = helper.second_slope
    #intersection_angles[frame-1] = helper.intersection_angle

    #print frame, helper.first_slope, helper.second_slope, helper.intersection_angle
#numpy.savetxt(outputDir+'/angle.dat', numpy.column_stack([time, slopes1, slopes2, intersection_angles]),fmt='%.6f')

    
#######################################################################
# Plotting Graph Between Time and Relative Distance\Angles
#######################################################################

#def remove_nan_for_plot(x,y):
    #not_nan = ~numpy.isnan(y)
    #return x[not_nan], y[not_nan]


#def plot_line(x, y, xlabel, ylabel, figsize=[4,2.5], 
					#xlimits=None, ylimits=None, savefile=None):

    #x, y = remove_nan_for_plot(x, y)
    #plt.figure(figsize=figsize)
    #plt.plot(x, y, '-o', color='steelblue', lw=1, mfc='none', mec='orangered', ms=2)
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #if xlimits is not None:
        #plt.xlim(xlimits)
    #if ylimits is not None:
        #plt.ylim(ylimits)
    #plt.tight_layout()
    #if savefile is not None:
        #plt.savefig(savefile, dpi=300)
    #plt.show()

#txtfile = numpy.loadtxt(outputDir+'/angle.dat')
#time = txtfile[:,0]
#slope_difference = txtfile[:,3]
#plot_line(x=time, y=slope_difference,xlabel='time (seconds)', ylabel='slope_difference (degrees)', 
            #xlimits=[0,2], ylimits=[-50, 10], 
            #savefile=outputDir+'/slope_difference.png')
