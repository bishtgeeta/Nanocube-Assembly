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

inputFile = r'Z:\Geeta-Share\cubes assembly\20160614-019-output\20160614-019.avi'
outputFile = r'Z:\Geeta-Share\cubes assembly\20160614-019-output\20160614-019.h5'
inputDir = r'Z:\Geeta-Share\cubes assembly\20160614-019-output'
outputDir = r'Z:\Geeta-Share\cubes assembly\20160614-019-output\output'
pixInNM = 1/0.9172754
fps = 25
microscope = 'JOEL2010' #'JOEL2010','T12'
camera = 'One-view' #'Orius', 'One-view'
owner = 'Shu Fen'
zfillVal = 6
fontScale = 1
structure = [[1,1,1],[1,1,1],[1,1,1]]

#######################################################################
# INITIALIZATION FOR THE MPI ENVIRONMENT
#######################################################################
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#######################################################################

if (rank==0):
    tic = time()
#######################################################################
# DATA PROCESSSING
# 1. READ THE INPUT FILES AND STORE THEM FRAME-WISE IN H5 FILE
# 2. PERFORM BACKGROUND SUBTRACTION (IF REQUIRED)
#######################################################################
#########
# PART 1
#########
#if (rank==0):
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
#comm.Barrier()

#########
# PART 2
#########
#if (rank==0):
    #print "Inverting the image and performing background subtraction"
#invertFlag=True
#bgSubFlag= True; bgSubSigmaTHT=2; radiusTHT=15

#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#for frame in procFrameList[rank]:
    #gImgProc = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #if (invertFlag==True):
        #gImgProc = imageProcess.invertImage(gImgProc)
    #if (bgSubFlag==True):
        #gImgProc = imageProcess.subtractBackground(gImgProc, sigma=bgSubSigmaTHT, radius=radiusTHT)
    #cv2.imwrite(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',gImgProc)

#comm.Barrier()
    
#if (rank==0):
    #for frame in frameList:
        #gImgProc = cv2.imread(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',0)
        #fileIO.writeH5Dataset(fp,'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal),gImgProc)
        
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################


#######################################################################
# IMAGE SEGMENTATION
#######################################################################
#if (rank==0):
    #print "Performing segmentation for all the frames"
    
#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#areaRange = numpy.array([400,4000], dtype='float64')

#for frame in procFrameList[rank]:
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #gImgNorm = imageProcess.normalize(gImgRaw,min=0,max=230)
    #gImgProc = fp['/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)].value
    #bImg = gImgProc>=myCythonFunc.threshold_kapur(gImgProc.flatten())
    #bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange)
    #bImg = imageProcess.binary_erosion(bImg, iterations=1)
    #bImgBdry = imageProcess.normalize(imageProcess.boundary(bImg))
    #finalImage = numpy.column_stack((numpy.maximum(gImgNorm,bImgBdry), gImgNorm))
    #cv2.imwrite(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png', finalImage)
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################


#######################################################################
# CREATE BINARY IMAGES INTO HDF5 FILE
#######################################################################
#if (rank==0):
    #print "Creating binary images and writing into h5 file"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#for frame in procFrameList[rank]:
    #bImg = cv2.imread(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png',0)[0:row,0:col]
    #bImg = bImg==255
    #bImg = imageProcess.fillHoles(bImg)
    #bImg = imageProcess.binary_opening(bImg, iterations=1)
    #numpy.save(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy', bImg)
   
#comm.barrier()
#if (rank==0):
    #for frame in frameList:
        #bImg = numpy.load(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        #fileIO.writeH5Dataset(fp,'/segmentation/bImgStack/'+str(frame).zfill(zfillVal),bImg)
        #fileIO.delete(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################

#######################################################################
# LABELLING PARTICLES
#######################################################################
#centerDispRange = [40,40]
#perAreaChangeRange = [20,20]
#missFramesTh = 10
    
#if (rank==0):
    #print "Labelling segmented particles"
    #fp = h5py.File(outputFile, 'r+')
    #[row,col,numFrames,frameList] = misc.getVitals(fp)
    #maxID, occurenceFrameList = tracking.labelParticles(fp, centerDispRange=centerDispRange, perAreaChangeRange=perAreaChangeRange, missFramesTh=missFramesTh, structure=structure)
    #fp.attrs['particleList'] = range(1,maxID+1)
    #numpy.savetxt(outputDir+'/frameOccurenceList.dat',numpy.column_stack((fp.attrs['particleList'],occurenceFrameList)),fmt='%d')
    #fp.flush(), fp.close()
#comm.Barrier()

#if (rank==0):
    #print "Generating images with labelled particles"
#fp = h5py.File(outputFile, 'r')
#tracking.generateLabelImages(fp,outputDir+'/segmentation/tracking')
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################


#######################################################################
# REMOVING UNWANTED PARTICLES
#######################################################################
#keepList = [1,2,3,5]
#removeList = []

#if (rank==0):
	#print "Removing unwanted particles"

#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#particleList = fp.attrs['particleList']
#if not removeList:
	#removeList = [s for s in particleList if s not in keepList]
#tracking.removeParticles(fp,removeList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# GLOBAL RELABELING OF PARTICLES
#######################################################################
#correctionList = [[3, 1]]

#if (rank==0):
	#print "Global relabeling of  particles"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.globalRelabelParticles(fp,correctionList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# RELABEL PARTICLES IN THE ORDER OF OCCURENCE
#######################################################################
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.relabelParticles(fp,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################

#######################################################################
# GENERATING IMAGES WITH LABELLED PARTICLES
#######################################################################
#if (rank==0):
    #print "Generating images with labelled particles"
#fp = h5py.File(outputFile, 'r')
#tracking.generateLabelImages(fp,outputDir+'/segmentation/tracking',fontScale,size,rank)
#fp.flush(), fp.close()
#comm.Barrier()
######################################################################


#######################################################################
# FINDING OUT THE MEASURES FOR TRACKED PARTICLES
#######################################################################
#if (rank==0):
    #print "Finding measures for tracked particles"

#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#particleList = fp.attrs['particleList']
#zfillVal = fp.attrs['zfillVal']
#procFrameList = numpy.array_split(frameList,size)
#fps = fp.attrs['fps']
#pixInNM = fp.attrs['pixInNM']

#outFile = open(str(rank)+'.dat','wb')

#area=True
#perimeter=True
#circularity=False
#pixelList=False
#bdryPixelList=False
#centroid=True
#intensityList=False
#sumIntensity=False
#effRadius=False
#radius=False
#circumRadius=False
#inRadius=False
#radiusOFgyration=False
#orientation=True

#for frame in procFrameList[rank]:
    #labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #outFile.write("%f " %(1.0*frame/fps))
    #for particle in particleList:
        #bImg = labelImg==particle
        #if (bImg.max() == True):
            #label, numLabel, dictionary = imageProcess.regionProps(bImg, gImgRaw, structure=structure, centroid=centroid, area=area, perimeter=perimeter,orientation=orientation)
            #outFile.write("%f %f %f %f %f " %(dictionary['centroid'][0][1]*pixInNM, (row-dictionary['centroid'][0][0])*pixInNM, dictionary['area'][0]*pixInNM*pixInNM, dictionary['perimeter'][0]*pixInNM, dictionary['orientation'][0]))
        #else:
            #outFile.write("nan nan nan nan nan ")
    #outFile.write("\n")
#outFile.close()
#fp.flush(), fp.close()
#comm.Barrier()

#if (rank==0):
    #for r in range(size):
        #if (r==0):
            #measures = numpy.loadtxt(str(r)+'.dat')
        #else:
            #measures = numpy.row_stack((measures,numpy.loadtxt(str(r)+'.dat')))
        #fileIO.delete(str(r)+'.dat')
    #measures = measures[numpy.argsort(measures[:,0])]
    #numpy.savetxt(outputDir+'/imgDataNM.dat', measures, fmt='%.6f')
#######################################################################


#######################################################################
# FINDING OUT RELATIVE DISTANCE AND ANGLE BETWEEN PARTICLES
#######################################################################
#def get_sign_of_angle(centroid1, centroid2, intersection_point):
    #line1 = centroid2 - centroid1
    #line2 = intersection_point - centroid1
    #return numpy.sign(numpy.cross(line1, line2))
    
#def get_value_of_angle(first_slope, second_slope):
    #diff = first_slope - second_slope
    #if diff < 0:
        #return 180 + diff
    #else:
        #return diff

#if (rank==0):
    #print "Finding the relative distance"
    #txtfile = numpy.loadtxt(outputDir+'/imgDataNM.dat')
    #time = txtfile[:,0]
    #x1 = txtfile[:,1]
    #y1 = txtfile[:,2]
    #x2 = txtfile[:,6]
    #y2 = txtfile[:,7]
    #relative_distance = numpy.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    #slopes1 = numpy.empty(x1.shape)
    #slopes1[:] = numpy.NaN
    #slopes2 = numpy.empty(x1.shape)
    #slopes2[:] = numpy.NaN
    #intersection_angles = numpy.empty(x1.shape)
    #intersection_angles[:] = numpy.NaN
    #fp = h5py.File(outputFile, 'r')
    #for frame in range(1,57):
        #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
        #helper = imageProcess.FindAngleHelper(gImgRaw)
        #helper.connect()
        #plt.show()
        #slopes1[frame-1] = helper.first_slope
        #slopes2[frame-1] = helper.second_slope
        #intersection_angles[frame-1] = helper.intersection_angle

        #print frame, helper.first_slope, helper.second_slope, helper.intersection_angle
    #numpy.savetxt(outputDir+'/relative_distance.dat', numpy.column_stack([time, x1, y1, x2, y2, relative_distance, slopes1, slopes2, intersection_angles]),fmt='%.6f')
    
#######################################################################
# Plotting Graph Between Time and Relative Distance
#######################################################################

def remove_nan_for_plot(x,y):
    not_nan = ~numpy.isnan(y)
    return x[not_nan], y[not_nan]


def plot_line(x, y, xlabel, ylabel, figsize=[6,3.5], 
					xlimits=None, ylimits=None, savefile=None):

    x, y = remove_nan_for_plot(x, y)
    plt.figure(figsize=figsize)
    plt.plot(x, y, '-o', color='steelblue', lw=2, mfc='none', mec='orangered', ms=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlimits is not None:
        plt.xlim(xlimits)
    if ylimits is not None:
        plt.ylim(ylimits)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile, dpi=300)
    plt.show()

if (rank == 0):
    txtfile = numpy.loadtxt(outputDir+'/relative_distance.dat')
    time = txtfile[:,0]
    relative_distance = txtfile[:,5]
    slope_difference = txtfile[:,8]
    plot_line(x=time, y=relative_distance, xlimits=[0, 0.5], ylimits=[70, 85], 
				xlabel='time (seconds)', ylabel='relative distance (nm)', 
				savefile=outputDir+'/nanocube_relative_distance.png')
    plot_line(x=time, y=slope_difference, xlimits=[0,1.5], ylimits=[-60, 10], 
				xlabel='time (seconds)', ylabel='slope_difference (degrees)', 
				savefile=outputDir+'/nanocube_slope_difference.png')
