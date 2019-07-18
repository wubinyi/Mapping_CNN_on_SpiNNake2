import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
s2sFolderPath = os.path.join(projectFolder, "spiNNaker2Simulator")
if s2sFolderPath not in sys.path:
	sys.path.insert(0, s2sFolderPath)
distriFolderPath = os.path.join(projectFolder, "distributor")
if distriFolderPath not in sys.path:
	sys.path.insert(0, distriFolderPath)
icproValidatePath = os.path.join(projectFolder, "icproValidate")
if icproValidatePath not in sys.path:
	sys.path.insert(0, icproValidatePath)
from distributionGeneral import DistributionGeneralClass
from spiNNaker2TriBlockSimulator import SpiNNaker2TriBlock
from spiNNakerSimulatorGeneral import *
from dataGenerator import convDataSplitterForTf, fcDataSplitterForTf
import numpy as np
import tensorflow as tf


#============================================================================================================
# 									convolution splitting scheme validation
#============================================================================================================
def convSplitSchemeValidation():
	# S1: Get splitting result
	modelLayersSplitInfo, modelLayersParameter = DistributionGeneralClass.vgg(sramBlocks=1, decreaseSize=False, 
		forSpiNNaker2=False, logEn=False, printFlag=True)
	# S2: Validate each layer
	layerNames = list(modelLayersSplitInfo.keys())
	for layerName in layerNames:
		if "CONV_" not in layerName:
			continue
		validateResult = singleConvValidate(modelLayersSplitInfo[layerName], modelLayersParameter[layerName])
		if validateResult:
			print("{} Validation: {}".format(layerName, "PASS"))
		else:
			print("{} Validation: {}".format(layerName, "FAILED"))

def singleConvValidate(layerSplitInfo, layerTypeParameter):
	# Determine if width/height is splitted into parts with different size
	layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	preDefineTf = inActiSplitInfo[0][2] == 0 and inActiSplitInfo[1][2] == 0
	# S1: Get splitting data according splitting scheme
	inActiBlocks, inActiDimBlocks, weightBlocks, weightDimBlocks, outActiBlocks, inActi, weight, outActi = \
		convDataSplitterForTf(layerSplitInfo, layerTypeParameter)
	# S2: Create tensorflow session and place holder
	# If all pieces having the same size, then create session only one time 
	# 	for saving program execution time.
	# If all pieces having different size, then create the session during running
	# 	for fitting the size, however, the speed will be very slow, as creating tf session
	# 	is a heavy task.
	if preDefineTf:
		X = tf.placeholder(tf.float32, inActiDimBlocks[0])
		W = tf.placeholder(tf.float32, weightDimBlocks[0])
		Y = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding='VALID')
		sess = tf.Session()
	# S3: Get outActi based on each part of inActi and weight
	# 	  Compare with splitting data
	outActiIndex = 0
	for weightIndex in range(len(weightBlocks)):
		for inActiIndex in range(len(inActiBlocks)):
			if preDefineTf:
				inActiFloat = np.array(inActiBlocks[inActiIndex], dtype=np.float32)
				weightFloat = np.array(weightBlocks[weightIndex], dtype=np.float32)
				outActiFloat = sess.run(Y, feed_dict={X: inActiFloat, W: weightFloat})
				refOutActi = np.array(outActiFloat, dtype=np.uint32)
			else:
				refOutActi = getConvResult(inActiDimBlocks[inActiIndex], weightDimBlocks[weightIndex], 
					inActiBlocks[inActiIndex], weightBlocks[weightIndex])
			if not (np.array_equal(refOutActi, outActiBlocks[outActiIndex])):
				print("Please decrease the value of validation data to avoid overflow!!!")
				return False
			outActiIndex += 1
	return True

def getConvResult(inActiDim, filterDim, inActi, weight):
	inActiFloat = np.array(inActi, dtype=np.float32)
	weightFloat = np.array(weight, dtype=np.float32)
	# Generate output-activation
	X = tf.placeholder(tf.float32, inActiDim)
	W = tf.placeholder(tf.float32, filterDim)
	Y = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding='VALID')
	sess = tf.Session()
	outActiFloat = sess.run(Y, feed_dict={X: inActiFloat, W: weightFloat})
	outActi = np.array(outActiFloat, dtype=np.uint32)
	return outActi



#============================================================================================================
# 									fully-connected splitting scheme validation
#============================================================================================================
def fcSplitSchemeValidation():
	# S1: Get splitting result
	modelLayersSplitInfo, modelLayersParameter = DistributionGeneralClass.vgg(sramBlocks=1, decreaseSize=False, 
		forSpiNNaker2=False, logEn=False, printFlag=True)
	# S2: Validate each layer
	layerNames = list(modelLayersSplitInfo.keys())
	for layerName in layerNames:
		if "FC_" not in layerName:
			continue
		validateResult = singleFcValidate(modelLayersSplitInfo[layerName], modelLayersParameter[layerName])
		if validateResult:
			print("{} Validation: {}".format(layerName, "PASS"))
		else:
			print("{} Validation: {}".format(layerName, "FAILED"))

def singleFcValidate(layerSplitInfo, layerTypeParameter):
	# S1: Get splitting data according splitting scheme
	inActiBlocks, inActiDimBlocks, weightBlocks, weightDimBlocks, outActiBlocks, outActiDimBlocks = \
		fcDataSplitterForTf(layerSplitInfo, layerTypeParameter)
	numOfInActiBlock = len(inActiBlocks)
	numOfWeightBlock = len(weightBlocks)
	# S2: Create tensorflow session and place holder
	# Determine if width/height is splitted into parts with different size
	layerType, inActiSplitInfo, weightSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	weightWidthSplitInfo = weightSplitInfo[0]
	weightWidthDouble = weightWidthSplitInfo[2] != 0
	weightHeightSplitInfo = weightSplitInfo[1]
	weightHeightDouble = weightHeightSplitInfo[2] != 0
	# sess1 sess2
	# sess3 sess4
	X1 = tf.placeholder(tf.float32, [1,weightHeightSplitInfo[0]])
	W1 = tf.placeholder(tf.float32, [weightHeightSplitInfo[0],weightWidthSplitInfo[0]])
	Y1 = tf.matmul(X1, W1)
	sess1 = tf.Session()
	if weightWidthDouble:
		X2 = tf.placeholder(tf.float32, [1,weightHeightSplitInfo[0]])
		W2 = tf.placeholder(tf.float32, [weightHeightSplitInfo[0],weightWidthSplitInfo[2]])
		Y2 = tf.matmul(X2, W2)
		sess2 = tf.Session()
	if weightHeightDouble:
		X3 = tf.placeholder(tf.float32, [1,weightHeightSplitInfo[2]])
		W3 = tf.placeholder(tf.float32, [weightHeightSplitInfo[2],weightWidthSplitInfo[0]])
		Y3 = tf.matmul(X3, W3)
		sess3 = tf.Session()
	if weightWidthDouble and weightHeightDouble:
		X4 = tf.placeholder(tf.float32, [1,weightHeightSplitInfo[2]])
		W4 = tf.placeholder(tf.float32, [weightHeightSplitInfo[2],weightWidthSplitInfo[2]])
		Y4 = tf.matmul(X4, W4)
		sess4 = tf.Session()
	# S3: Get outActi based on each part of inActi and weight
	# 	  Compare with splitting data
	for weightColumnIndex in range(numOfWeightBlock//numOfInActiBlock):
		outActiDim = outActiDimBlocks[weightColumnIndex]
		outActiRef = np.zeros((outActiDim[0], outActiDim[1]), dtype=np.uint32)
		for inActiIndex in range(numOfInActiBlock):
			weightIndex = weightColumnIndex * numOfInActiBlock + inActiIndex
			# 
			inActiFloat = np.array(inActiBlocks[inActiIndex], dtype=np.float32)
			weightFloat = np.array(weightBlocks[weightIndex], dtype=np.float32)
			# 
			weightDim = weightDimBlocks[weightIndex]
			weightHeight = weightDim[0]
			weightWidth = weightDim[1]
			if weightHeight == weightHeightSplitInfo[0] and weightWidth == weightWidthSplitInfo[0]:
				outActiFloat = sess1.run(Y1, feed_dict={X1: inActiFloat, W1: weightFloat})
			elif weightHeight == weightHeightSplitInfo[0] and weightWidth == weightWidthSplitInfo[2]:
				outActiFloat = sess2.run(Y2, feed_dict={X2: inActiFloat, W2: weightFloat})
			elif weightHeight == weightHeightSplitInfo[2] and weightWidth == weightWidthSplitInfo[0]:
				outActiFloat = sess3.run(Y3, feed_dict={X3: inActiFloat, W3: weightFloat})
			elif weightHeight == weightHeightSplitInfo[2] and weightWidth == weightWidthSplitInfo[2]:
				outActiFloat = sess4.run(Y4, feed_dict={X4: inActiFloat, W4: weightFloat})
			else:
				assert(False), "No tensorflow session available"
			outActiTemp = np.array(outActiFloat, dtype=np.uint32)
			# 
			outActiRef = np.add(outActiRef, outActiTemp)
		#
		if not (np.array_equal(outActiRef, outActiBlocks[weightColumnIndex])):
			print("Please decrease the value of validation data to avoid overflow!!!")
			return False
	return True	


if __name__ == "__main__":
	# convSplitSchemeValidation()
	fcSplitSchemeValidation()