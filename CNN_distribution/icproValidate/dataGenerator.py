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
parserSplitterPath = os.path.join(projectFolder, "parserSplitter")
if parserSplitterPath not in sys.path:
	sys.path.insert(0, parserSplitterPath)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(0, os.path.dirname(os.getcwd()))
import tensorflow as tf
import numpy as np
import math
from nnGeneral import BasicOperation, PE_SRAM_LIMIT
from spiNNaker2General import SpiNNakerBasic
from enum import Enum, auto


# CONV_1   | [[226, 1, 0, 0], [18, 14, 0, 0], 3]  | [[3, 3, 3, [4, 16, 0, 0]], 1]    | [[224, 1, 0, 0], [16, 14, 0, 0], [4, 16, 0, 0]] | 
# CONV_3   | [[114, 1, 0, 0], [10, 14, 0, 0], 64] | [[3, 3, 64, [4, 32, 0, 0]], 1]   | [[112, 1, 0, 0], [8, 14, 0, 0], [4, 32, 0, 0]]  | 
# CONV_4   | [[114, 1, 0, 0], [4, 56, 0, 0], 128] | [[3, 3, 128, [4, 32, 0, 0]], 1]  | [[112, 1, 0, 0], [2, 56, 0, 0], [4, 32, 0, 0]]  |  
# CONV_6   | [[58, 1, 0, 0], [10, 7, 0, 0], 128]  | [[3, 3, 128, [4, 64, 0, 0]], 1]  | [[56, 1, 0, 0], [8, 7, 0, 0], [4, 64, 0, 0]]  	 | 
# CONV_7   | [[58, 1, 0, 0], [5, 18, 4, 1], 256]  | [[3, 3, 256, [4, 64, 0, 0]], 1]  | [[56, 1, 0, 0], [3, 18, 2, 1], [4, 64, 0, 0]]   |  
# CONV_8   | [[58, 1, 0, 0], [4, 28, 0, 0], 256]  | [[3, 3, 256, [4, 64, 0, 0]], 1]  | [[56, 1, 0, 0], [2, 28, 0, 0], [4, 64, 0, 0]]   |  
# CONV_10  | [[30, 1, 0, 0], [9, 4, 0, 0], 256]   | [[3, 3, 256, [4, 128, 0, 0]], 1] | [[28, 1, 0, 0], [7, 4, 0, 0], [4, 128, 0, 0]]   | 
# CONV_11  | [[16, 2, 0, 0], [9, 4, 0, 0], 512]   | [[3, 3, 512, [4, 128, 0, 0]], 1] | [[14, 2, 0, 0], [7, 4, 0, 0], [4, 128, 0, 0]]   |  
# CONV_12  | [[30, 1, 0, 0], [4, 14, 0, 0], 512]  | [[3, 3, 512, [4, 128, 0, 0]], 1] | [[28, 1, 0, 0], [2, 14, 0, 0], [4, 128, 0, 0]]  |  
# CONV_14  | [[16, 1, 0, 0], [9, 2, 0, 0], 512]   | [[3, 3, 512, [4, 128, 0, 0]], 1] | [[14, 1, 0, 0], [7, 2, 0, 0], [4, 128, 0, 0]]   | 
# CONV_15  | [[16, 1, 0, 0], [9, 2, 0, 0], 512]   | [[3, 3, 512, [4, 128, 0, 0]], 1] | [[14, 1, 0, 0], [7, 2, 0, 0], [4, 128, 0, 0]]   | 
# CONV_16  | [[16, 1, 0, 0], [4, 7, 0, 0], 512]   | [[3, 3, 512, [4, 128, 0, 0]], 1] | [[14, 1, 0, 0], [2, 7, 0, 0], [4, 128, 0, 0]]   | 

# FC_18    | [[186, 113, 185, 22], 1]             | [[512, 8, 0, 0], [186, 113, 185, 22]] | [[512, 8, 0, 0], 1] |  
# FCE_18   | [[512, 8, 0, 0], 135]                |                  [0]                  | [[512, 8, 0, 0], 1] |  
# FC_19    | [[373, 4, 372, 7], 1]                |   [[256, 4, 0, 0], [373, 4, 372, 7]]  | [[256, 4, 0, 0], 1] | 
# FCE_19   | [[256, 4, 0, 0], 11]                 |                  [0]                  | [[256, 4, 0, 0], 1] | 

# CONV: tensorflow dimension
CONV_ACTI_BATCH_AXIS = 0
CONV_ACTI_HEIGHT_AXIS = 1
CONV_ACTI_WIDTH_AXIS = 2
CONV_ACTI_CHANNEL_AXIS = 3
CONV_WEIGHT_HEIGHT_AXIS = 0
CONV_WEIGHT_WIDTH_AXIS = 1
CONV_WEIGHT_IN_CHANNEL_AXIS = 2
CONV_WEIGHT_OUT_CHANNEL_AXIS = 3
# FC: tensorflow dimension
FC_ACTI_BATCH_HEIGHT_AXIS = 0
FC_ACTI_WIDTH_AXIS = 1
FC_WEIGHT_HEIGHT_AXIS = 0
FC_WEIGHT_WIDTH_AXIS = 1
# 
MIN_NUM = 0
MAX_NUM = 20
# 
COLUMN_ALIGN = 16
ROW_ALIGN = 4
OUT_ALIGN = 4
SRAM_BW_BYTES = 16
# 
INACTI_FILENAME = "inActi.blk"
WEIGHT_FILENAME = "weight.blk"
OUTACTI_FILENAME = "outActi.blk"
# 
SRAM_DATA_ADDR = 0x8100
# 
class DataType(Enum):
	UINT8 = auto()
	UINT32 = auto()

def customPrint(content):
	print(content)

def customAssert(condition, content):
	assert(condition), "{}".format(content)

#============================================================================================================
# 						input-activation, wegiht and output-activation generator for CONV
# CONV:
# 	Argus: 		Dimension of input-activation: 	[width, height, inchannel]
# 				Dimension of weight:			[width, height, inchannel, outchannel]
# 	Returns: 	input-activation:	[batch, width, height, inchannel]
# 				weight:				[width, height, inchannel, outchannel]
# 				output-activation:	[batch, width, height, outchannel]
#============================================================================================================
def convDataGenerator(inActiDim, filterDim):
	inActiWidth, inActiHeight, inActiInChannel = inActiDim[0], inActiDim[1], inActiDim[2]
	filterWidth, filterHeight, filterInChannel, filterOutChannel = filterDim[0], filterDim[1], filterDim[2], filterDim[3]
	assert(inActiInChannel==filterInChannel), \
		"Input-Channel of input[{}] and weight[{}] ummatch".format(inActiInChannel, filterInChannel)
	# Generate random input-activation and weight
	inActi = np.random.randint(low=MIN_NUM, high=MAX_NUM, size=(1,inActiHeight,inActiWidth,inActiInChannel), dtype=np.uint8)
	inActiFloat = np.array(inActi, dtype=np.float32)
	weight = np.random.randint(low=MIN_NUM, high=MAX_NUM, size=(filterHeight,filterWidth,filterInChannel,filterOutChannel), dtype=np.uint8)
	weightFloat = np.array(weight, dtype=np.float32)
	# Generate output-activation
	X = tf.placeholder(tf.float32, [1,inActiHeight,inActiWidth,inActiInChannel])
	W = tf.placeholder(tf.float32, [filterHeight,filterWidth,filterInChannel,filterOutChannel])
	Y = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding='VALID')
	sess = tf.Session()
	outActiFloat = sess.run(Y, feed_dict={X: inActiFloat, W: weightFloat})
	outActi = np.array(outActiFloat, dtype=np.uint32)
	return convActiSwapaxes(inActi), convWeightSwapaxes(weight), convActiSwapaxes(outActi)

def convActiSwapaxes(acti):
	return np.swapaxes(acti, CONV_ACTI_HEIGHT_AXIS, CONV_ACTI_WIDTH_AXIS)

def convWeightSwapaxes(weight):
	return np.swapaxes(weight, CONV_WEIGHT_HEIGHT_AXIS, CONV_WEIGHT_WIDTH_AXIS)

def convGetOutActiDim(inActiDim, filterDim, stride=1):
	inActiWidth, inActiHeight, inActiInChannel = inActiDim[0], inActiDim[1], inActiDim[2]
	filterWidth, filterHeight, filterInChannel, filterOutChannel = filterDim[0], filterDim[1], filterDim[2], filterDim[3]
	assert(inActiInChannel==filterInChannel), \
		"Input-Channel of input[{}] and weight[{}] ummatch".format(inActiInChannel, filterInChannel)
	outActiWidth = math.floor((inActiWidth - filterWidth) / stride) + 1
	outActiHeight = math.floor((inActiHeight - filterHeight) / stride) + 1
	outActiOutChannel = filterOutChannel
	return [outActiWidth, outActiHeight, outActiOutChannel]

#============================================================================================================
# 			Seperate input-activation, weight and output-activation into blocks for tensorflow
# CONV:
# 	Argus:		Layer splitting information
# 				Layer parameter
# 	Returns:	input-activation blocks
# 				weight blocks
# 				output-activation blocks
#============================================================================================================
def convDataSplitterForTf(layerSplitInfo, layerTypeParameter):
	layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	weightSplitInfo = weightStrideSplitInfo[0]
	layerType, layerParameter = layerTypeParameter
	(inActiDim, weightDim, stride, outActiDim), poolStride = layerParameter
	customAssert(weightDim[0]==weightDim[1], "Limitation of filter width and height")
	overlapLen = weightDim[0] - stride
	# TODO: check requiredPEs and parts
	# TODO: check layerType
	# Get CONV data: 
	# [batch, width, height, inchannel], [width, height, inchannel, outchannel], [batch, width, height, outchannel]
	inActi, weight, outActi = convDataGenerator(inActiDim, weightDim)
	inActi = convActiSwapaxes(inActi)
	weight = convWeightSwapaxes(weight)
	outActi = convActiSwapaxes(outActi)
	# Split input-activation
	inActiBlocks = []
	inActiDimBlocks = []
	inActiWidthSplitInfo = inActiSplitInfo[0]
	inActiHeightSplitInfo = inActiSplitInfo[1]
	inActiChannelSplitInfo = inActiSplitInfo[2]
	inActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(inActiWidthSplitInfo)
	inActiHeightParts = BasicOperation.getTotalPartsFromSplitInfo(inActiHeightSplitInfo)
	inActiChannelParts = BasicOperation.getTotalPartsFromSplitInfo(inActiChannelSplitInfo)
	for channelIndex in range(inActiChannelParts):
		channelBase, channelEnd = convWidthHeightChannelBase(inActiChannelSplitInfo, channelIndex)
		for heightIndex in range(inActiHeightParts):
			heightBase, heightEnd = convWidthHeightChannelBase(inActiHeightSplitInfo, heightIndex, overlapLen=overlapLen)
			for widthIndex in range(inActiWidthParts):
				widthBase, widthEnd = convWidthHeightChannelBase(inActiWidthSplitInfo, widthIndex, overlapLen=overlapLen)
				inActiBlock = inActi[:, heightBase:heightEnd, widthBase:widthEnd, channelBase:channelEnd]
				inActiBlocks.append(inActiBlock)
				inActiDimBlocks.append([1, heightEnd-heightBase, widthEnd-widthBase, channelEnd-channelBase])
	# Split weight
	weightBlocks = []
	weightDimBlocks = []
	weightInChannelSplitInfo = weightSplitInfo[2]
	weightOutChannelSplitInfo = weightSplitInfo[3]
	weightInChannelParts = BasicOperation.getTotalPartsFromSplitInfo(weightInChannelSplitInfo)
	weightOutChannelParts = BasicOperation.getTotalPartsFromSplitInfo(weightOutChannelSplitInfo)
	for outChannelIndex in range(weightOutChannelParts):
		outChannelBase, outChannelEnd = convWidthHeightChannelBase(weightOutChannelSplitInfo, outChannelIndex)
		for inChannelIndex in range(weightInChannelParts):
			inChannelBase, inChannelEnd = convWidthHeightChannelBase(weightInChannelSplitInfo, inChannelIndex)
			weightBlock = weight[:, :, inChannelBase:inChannelEnd, outChannelBase:outChannelEnd]
			weightBlocks.append(weightBlock)
			weightDimBlocks.append([weightDim[1], weightDim[0], inChannelEnd-inChannelBase, outChannelEnd-outChannelBase])
	# Split output-activation
	outActiBlocks = []
	outActiWidthSplitInfo = outActiSplitInfo[0]
	outActiHeightSplitInfo = outActiSplitInfo[1]
	outActiChannelSplitInfo = outActiSplitInfo[2]
	outActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(outActiWidthSplitInfo)
	outActiHeightParts = BasicOperation.getTotalPartsFromSplitInfo(outActiHeightSplitInfo)
	outActiChannelParts = BasicOperation.getTotalPartsFromSplitInfo(outActiChannelSplitInfo)
	for channelIndex in range(outActiChannelParts):
		channelBase, channelEnd = convWidthHeightChannelBase(outActiChannelSplitInfo, channelIndex)
		for heightIndex in range(outActiHeightParts):
			heightBase, heightEnd = convWidthHeightChannelBase(outActiHeightSplitInfo, heightIndex)
			for widthIndex in range(outActiWidthParts):
				widthBase, widthEnd = convWidthHeightChannelBase(outActiWidthSplitInfo, widthIndex)
				outActiBlock = outActi[:, heightBase:heightEnd, widthBase:widthEnd, channelBase:channelEnd]
				outActiBlocks.append(outActiBlock)
	# # Not check the data size, otherwise need to realize another alignment function
	# # Get size of input-activation, weight and output-activation of largest part
	# inActiBlockAlignSize = convActiDataAlign(inActiBlocks[0]).size
	# weightBlockAlignSize = convWeightAlign(weightBlocks[0]).size
	# outActiBlockAlignSize = convActiDataAlign(outActiBlocks[0], inputActiFlag=False).size
	# totalSize = inActiBlockAlignSize + weightBlockAlignSize + outActiBlockAlignSize
	# customAssert(totalSize <= PE_SRAM_LIMIT, "{}={}+{}+{}->SRAM overflow!!! \n->{} \n->{}".format(totalSize, \
	# 	inActiBlockAlignSize, weightBlockAlignSize, outActiBlockAlignSize, layerSplitInfo, layerTypeParameter))
	return inActiBlocks, inActiDimBlocks, weightBlocks, weightDimBlocks, outActiBlocks, inActi, weight, outActi

def convSpittedDataDim(layerSplitInfo, layerTypeParameter):
	layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	weightSplitInfo = weightStrideSplitInfo[0]
	layerType, layerParameter = layerTypeParameter
	if len(layerParameter) == 2:
		(inActiDim, weightDim, stride, outActiDim), poolStride = layerParameter
	else:
		(inActiDim, weightDim, stride, outActiDim), poolStride, _ = layerParameter
	customAssert(weightDim[0]==weightDim[1], "Limitation of filter width and height")
	overlapLen = weightDim[0] - stride
	# TODO: check requiredPEs and parts
	# TODO: check layerType
	# Get CONV data: 
	# [batch, width, height, inchannel], [width, height, inchannel, outchannel], [batch, width, height, outchannel]
	# Split input-activation
	inActiDimBlocks = []
	inActiWidthSplitInfo = inActiSplitInfo[0]
	inActiHeightSplitInfo = inActiSplitInfo[1]
	inActiChannelSplitInfo = inActiSplitInfo[2]
	inActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(inActiWidthSplitInfo)
	inActiHeightParts = BasicOperation.getTotalPartsFromSplitInfo(inActiHeightSplitInfo)
	inActiChannelParts = BasicOperation.getTotalPartsFromSplitInfo(inActiChannelSplitInfo)
	for channelIndex in range(inActiChannelParts):
		channelBase, channelEnd = convWidthHeightChannelBase(inActiChannelSplitInfo, channelIndex)
		for heightIndex in range(inActiHeightParts):
			heightBase, heightEnd = convWidthHeightChannelBase(inActiHeightSplitInfo, heightIndex, overlapLen=overlapLen)
			for widthIndex in range(inActiWidthParts):
				widthBase, widthEnd = convWidthHeightChannelBase(inActiWidthSplitInfo, widthIndex, overlapLen=overlapLen)
				inActiDimBlocks.append([widthEnd-widthBase, heightEnd-heightBase, channelEnd-channelBase])
	# Split weight
	weightDimBlocks = []
	weightInChannelSplitInfo = weightSplitInfo[2]
	weightOutChannelSplitInfo = weightSplitInfo[3]
	weightInChannelParts = BasicOperation.getTotalPartsFromSplitInfo(weightInChannelSplitInfo)
	weightOutChannelParts = BasicOperation.getTotalPartsFromSplitInfo(weightOutChannelSplitInfo)
	for outChannelIndex in range(weightOutChannelParts):
		outChannelBase, outChannelEnd = convWidthHeightChannelBase(weightOutChannelSplitInfo, outChannelIndex)
		for inChannelIndex in range(weightInChannelParts):
			inChannelBase, inChannelEnd = convWidthHeightChannelBase(weightInChannelSplitInfo, inChannelIndex)
			weightDimBlocks.append([weightDim[0], weightDim[1], inChannelEnd-inChannelBase, outChannelEnd-outChannelBase])
	# Split output-activation
	outActiDimBlocks = []
	outActiWidthSplitInfo = outActiSplitInfo[0]
	outActiHeightSplitInfo = outActiSplitInfo[1]
	outActiChannelSplitInfo = outActiSplitInfo[2]
	outActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(outActiWidthSplitInfo)
	outActiHeightParts = BasicOperation.getTotalPartsFromSplitInfo(outActiHeightSplitInfo)
	outActiChannelParts = BasicOperation.getTotalPartsFromSplitInfo(outActiChannelSplitInfo)
	for channelIndex in range(outActiChannelParts):
		channelBase, channelEnd = convWidthHeightChannelBase(outActiChannelSplitInfo, channelIndex)
		for heightIndex in range(outActiHeightParts):
			heightBase, heightEnd = convWidthHeightChannelBase(outActiHeightSplitInfo, heightIndex)
			for widthIndex in range(outActiWidthParts):
				widthBase, widthEnd = convWidthHeightChannelBase(outActiWidthSplitInfo, widthIndex)
				outActiDimBlocks.append([widthEnd-widthBase, heightEnd-heightBase, channelEnd-channelBase])
	return inActiDimBlocks, weightDimBlocks, outActiDimBlocks
#============================================================================================================
# 						Seperate input-activation, weight and output-activation into blocks 
# CONV:
# 	Argus:		Layer splitting information
# 				Layer parameter
# 	Returns:	input-activation blocks
# 				weight blocks
# 				output-activation blocks
#============================================================================================================
def convDataSplitter(layerSplitInfo, layerTypeParameter, baseAddress=SRAM_DATA_ADDR):
	layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	weightSplitInfo = weightStrideSplitInfo[0]
	layerType, layerParameter = layerTypeParameter
	(inActiDim, weightDim, stride, outActiDim), poolStride = layerParameter
	customAssert(weightDim[0]==weightDim[1], "Limitation of filter width and height")
	overlapLen = weightDim[0] - stride
	# TODO: check requiredPEs and parts
	# TODO: check layerType
	# Get CONV data: 
	# [batch, width, height, inchannel], [width, height, inchannel, outchannel], [batch, width, height, outchannel]
	inActi, weight, outActi = convDataGenerator(inActiDim, weightDim)
	# # Debug
	# inActiDim = inActi.shape
	# customPrint("inActi: ")
	# for heightIndex in range(inActiDim[2]):
	# 	customPrint("{}".format(inActi[0, :, heightIndex, 0]))
	# Split input-activation
	inActiAlignBlocks = []
	inActiWidthSplitInfo = inActiSplitInfo[0]
	inActiHeightSplitInfo = inActiSplitInfo[1]
	inActiChannelSplitInfo = inActiSplitInfo[2]
	inActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(inActiWidthSplitInfo)
	inActiHeightParts = BasicOperation.getTotalPartsFromSplitInfo(inActiHeightSplitInfo)
	inActiChannelParts = BasicOperation.getTotalPartsFromSplitInfo(inActiChannelSplitInfo)
	for channelIndex in range(inActiChannelParts):
		channelBase, channelEnd = convWidthHeightChannelBase(inActiChannelSplitInfo, channelIndex)
		for heightIndex in range(inActiHeightParts):
			heightBase, heightEnd = convWidthHeightChannelBase(inActiHeightSplitInfo, heightIndex, overlapLen=overlapLen)
			for widthIndex in range(inActiWidthParts):
				widthBase, widthEnd = convWidthHeightChannelBase(inActiWidthSplitInfo, widthIndex, overlapLen=overlapLen)
				inActiBlock = inActi[:, widthBase:widthEnd, heightBase:heightEnd, channelBase:channelEnd]
				inActiBlockAlign = convActiDataAlign(inActiBlock)
				inActiAlignBlocks.append(inActiBlockAlign)
	# Split weight
	weightAlignBlocks = []
	weightInChannelSplitInfo = weightSplitInfo[2]
	weightOutChannelSplitInfo = weightSplitInfo[3]
	weightInChannelParts = BasicOperation.getTotalPartsFromSplitInfo(weightInChannelSplitInfo)
	weightOutChannelParts = BasicOperation.getTotalPartsFromSplitInfo(weightOutChannelSplitInfo)
	for outChannelIndex in range(weightOutChannelParts):
		outChannelBase, outChannelEnd = convWidthHeightChannelBase(weightOutChannelSplitInfo, outChannelIndex)
		for inChannelIndex in range(weightInChannelParts):
			inChannelBase, inChannelEnd = convWidthHeightChannelBase(weightInChannelSplitInfo, inChannelIndex)
			weightBlock = weight[:, :, inChannelBase:inChannelEnd, outChannelBase:outChannelEnd]
			weightBlockAlign = convWeightAlign(weightBlock)
			weightAlignBlocks.append(weightBlockAlign)
	# Split output-activation
	outActiAlignBlocks = []
	outActiWidthSplitInfo = outActiSplitInfo[0]
	outActiHeightSplitInfo = outActiSplitInfo[1]
	outActiChannelSplitInfo = outActiSplitInfo[2]
	outActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(outActiWidthSplitInfo)
	outActiHeightParts = BasicOperation.getTotalPartsFromSplitInfo(outActiHeightSplitInfo)
	outActiChannelParts = BasicOperation.getTotalPartsFromSplitInfo(outActiChannelSplitInfo)
	for channelIndex in range(outActiChannelParts):
		channelBase, channelEnd = convWidthHeightChannelBase(outActiChannelSplitInfo, channelIndex)
		for heightIndex in range(outActiHeightParts):
			heightBase, heightEnd = convWidthHeightChannelBase(outActiHeightSplitInfo, heightIndex)
			for widthIndex in range(outActiWidthParts):
				widthBase, widthEnd = convWidthHeightChannelBase(outActiWidthSplitInfo, widthIndex)
				outActiBlock = outActi[:, widthBase:widthEnd, heightBase:heightEnd, channelBase:channelEnd]
				outActiBlockAlign = convActiDataAlign(outActiBlock, inputActiFlag=False)
				outActiAlignBlocks.append(outActiBlockAlign)
	# Get size of input-activation, weight and output-activation of largest part
	# !!! Because of address align to 16 bytes, it can cause SRAM overflow problem
	inActiBaseAddr = addressAlign(baseAddress)
	print("inActiBaseAddr:  {}".format(hex(inActiBaseAddr)))
	weightBaseAddr = addressAlign(inActiBaseAddr + inActiAlignBlocks[0].size)
	print("weightBaseAddr:  {}".format(hex(weightBaseAddr)))
	outActiBaseAddr = addressAlign(weightBaseAddr + weightAlignBlocks[0].size)
	print("outActiBaseAddr: {}".format(hex(outActiBaseAddr)))
	finalAddr = outActiBaseAddr + outActiAlignBlocks[0].size
	customAssert(finalAddr-baseAddress < PE_SRAM_LIMIT, "SRAM overflow!!!")
	return inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr

def addressAlign(address):
	addressAlign = address + (SRAM_BW_BYTES - address) % SRAM_BW_BYTES
	return addressAlign

def convWidthHeightChannelBase(splitInfo, partIndex, overlapLen=0):
	# Base
	base = 0
	for index in range(partIndex):
		base += SpiNNakerBasic.getPartLengthFromSplitInfo(splitInfo, index)
	totalOverlapLen = overlapLen * partIndex
	base -= totalOverlapLen
	# Length
	length =  SpiNNakerBasic.getPartLengthFromSplitInfo(splitInfo, partIndex)
	# End
	end = base + length
	return base, end

def convActiDataAlign(acti, inputActiFlag=True):
	# swap axis: [batch, width, height, inchannel] -> [batch, height, width, inchannel]
	acti = np.swapaxes(acti, CONV_ACTI_HEIGHT_AXIS, CONV_ACTI_WIDTH_AXIS)
	# move axis: [batch, height, width, inchannel] -> [batch, inchannel, height, width]
	acti = np.moveaxis(acti, CONV_ACTI_CHANNEL_AXIS, 1)
	# Generate zeros
	actiDim = acti.shape
	if inputActiFlag:
		widthZeros = (COLUMN_ALIGN - (actiDim[3] % COLUMN_ALIGN)) % COLUMN_ALIGN
		zeros = np.zeros((actiDim[0], actiDim[1], actiDim[2], widthZeros), dtype=np.uint8)
	else:
		widthZeros = (OUT_ALIGN - (actiDim[3] % OUT_ALIGN)) % OUT_ALIGN
		zeros = np.zeros((actiDim[0], actiDim[1], actiDim[2], widthZeros), dtype=np.uint32)
	# Concatenate zeros to acti
	actiZeros = np.concatenate((acti, zeros), axis=-1)
	actiZerosDim = actiZeros.shape
	actiZerosWidth = actiZerosDim[3]
	# Flatten concatenated acti
	actiZeros = np.ravel(actiZeros)
	# 22.01.2019 -> As MLA will read next 16 bytes, when inActiWidth < 16, which may be not exist
	# See "mlaValidate(inWidth=4, inHeight=4, inChannel=3, outChannel=5)" and print inActi and data to operBBuffer
	# if inputActiFlag and actiZerosWidth < 32:
	# if inputActiFlag:
	# 	actiZeros = np.append(actiZeros, np.zeros((16, 1)))
	return actiZeros

def convWeightAlign(weight):
	# swap axis: [width, height, inchannel, outchannel] -> [height, width, inchannel, outchannel]
	weight = np.swapaxes(weight, CONV_WEIGHT_HEIGHT_AXIS, CONV_WEIGHT_WIDTH_AXIS)
	# move axis: [height, width, inchannel, outchannel] -> [inchannel, height, width, outchannel]
	weight = np.moveaxis(weight, CONV_WEIGHT_IN_CHANNEL_AXIS, 0)
	# Generate zeros
	weightDim = weight.shape
	outChannelZeros = (ROW_ALIGN - (weightDim[3] % ROW_ALIGN)) % ROW_ALIGN
	zeros = np.zeros((weightDim[0], weightDim[1], weightDim[2], outChannelZeros), dtype=np.uint8)
	# Concatenate zeros to acti
	weightZeros = np.concatenate((weight, zeros), axis=-1)
	# Reshape to [inchannel, height, width, outchannel/4, 4]
	weightZerosDim = weightZeros.shape
	weightZeros = np.reshape(weightZeros, 
		(weightZerosDim[0], weightZerosDim[1], weightZerosDim[2], weightZerosDim[3]//ROW_ALIGN, ROW_ALIGN))
	# Move axis: [inchannel, height, width, outchannel/4, 4] -> [outchannel/4, inchannel, height, width, 4]
	weightZeros = np.moveaxis(weightZeros, CONV_WEIGHT_OUT_CHANNEL_AXIS, 0)
	# Flatten concatenated acti
	weightZeros = np.ravel(weightZeros)
	if weightZeros.size % 16 != 0:
		weightZeros = np.append(weightZeros, np.zeros((16-(weightZeros.size % 16), 1)))
	return weightZeros

#============================================================================================================
# 						input-activation, wegiht and output-activation generator for FC
# FC:
# 	Argus: 		Dimension of input-activation: 	[width, height_batch]
# 				Dimension of weight:			[width, height]
# 	Returns: 	input-activation:	[width, height_batch]
# 				weight:				[width, height]
# 				output-activation:	[width, height_batch]
#============================================================================================================
def fcDataGenerator(inActiDim, weightDim):
	inActiWidth, inActiHeightBatch = inActiDim[0], inActiDim[1]
	weightWidth, weightHeight = weightDim[0], weightDim[1]
	assert(inActiWidth==weightHeight), \
		"Width of input[{}] and height of weight[{}] ummatch".format(inActiWidth, weightHeight)
	assert(inActiHeightBatch==1), "Height of input[{}] muss be 1".format(inActiHeightBatch)
	# Generate random input-activation and weight
	inActi = np.random.randint(low=MIN_NUM, high=MAX_NUM, size=(inActiHeightBatch,inActiWidth), dtype=np.uint8)
	inActiFloat = np.array(inActi, dtype=np.float32)
	# customPrint("inActi: \n{}".format(inActi))
	# customPrint("inActi shape: \n{}".format(inActi.shape))
	weight = np.random.randint(low=MIN_NUM, high=MAX_NUM, size=(weightHeight,weightWidth), dtype=np.uint8)
	weightFloat = np.array(weight, dtype=np.float32)
	# customPrint("weight: \n{}".format(weight))
	# customPrint("weight shape: \n{}".format(weight.shape))
	# Generate output-activation
	X = tf.placeholder(tf.float32, [inActiHeightBatch,inActiWidth])
	W = tf.placeholder(tf.float32, [weightHeight,weightWidth])
	Y = tf.matmul(X, W)
	sess = tf.Session()
	outActiFloat = sess.run(Y, feed_dict={X: inActiFloat, W: weightFloat})
	outActi = np.array(outActiFloat, dtype=np.uint32)
	# customPrint("outActi: \n{}".format(outActi))
	# customPrint("outActi shape: \n{}".format(outActi.shape))
	return fcActiSwapaxes(inActi), fcWeightSwapaxes(weight), fcActiSwapaxes(outActi)

def fcActiSwapaxes(acti):
	return np.swapaxes(acti, FC_ACTI_BATCH_HEIGHT_AXIS, FC_ACTI_WIDTH_AXIS)

def fcWeightSwapaxes(weight):
	return np.swapaxes(weight, FC_WEIGHT_HEIGHT_AXIS, FC_WEIGHT_WIDTH_AXIS)

def fcGetOutActiDim(inActiDim, weightDim):
	inActiWidth, inActiHeightBatch = inActiDim[0], inActiDim[1]
	weightWidth, weightHeight = weightDim[0], weightDim[1]
	assert(inActiWidth==weightHeight), \
		"Width of input[{}] and height of weight[{}] ummatch".format(inActiWidth, weightHeight)
	assert(inActiHeightBatch==1), "Height of input[{}] muss be 1".format(inActiHeightBatch)
	outActiWidth = weightWidth
	outActiHeightBatch = inActiHeightBatch
	return [outActiWidth, outActiHeightBatch]

def fcDataSplitter(layerSplitInfo, layerTypeParameter, baseAddress=SRAM_DATA_ADDR):
	'''
	For ICPRO
	'''
	layerType, inActiSplitInfo, weightSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	customAssert(inActiSplitInfo[1]==1, "Limitation of filter width and height")
	layerType, layerParameter = layerTypeParameter
	inNeuron, outNeuron = layerParameter
	# TODO: check requiredPEs and parts
	# TODO: check layerType
	# Get CONV data: 
	# [width, heightBatch], [width, height], [width, heightBatch]
	inActi, weight, outActi = fcDataGenerator([inNeuron, 1], [outNeuron, inNeuron])
	# # Debug
	# inActiDim = inActi.shape
	# customPrint("inActi: ")
	# for heightIndex in range(inActiDim[2]):
	# 	customPrint("{}".format(inActi[0, :, heightIndex, 0]))
	# Split input-activation
	inActiAlignBlocks = []
	inActiWidthSplitInfo = inActiSplitInfo[0]
	inActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(inActiWidthSplitInfo)
	for widthIndex in range(inActiWidthParts): 
		widthBase, widthEnd = fcWidthHeightChannelBase(inActiWidthSplitInfo, widthIndex)
		inActiBlock = inActi[widthBase:widthEnd, :]
		inActiBlockAlign = fcActiAlign(inActiBlock)
		inActiAlignBlocks.append(inActiBlockAlign)
	# Split weight
	weightAlignBlocks = []
	weightWidthSplitInfo = weightSplitInfo[0]
	weightHeightSplitInfo = weightSplitInfo[1]
	weightWidthParts = BasicOperation.getTotalPartsFromSplitInfo(weightWidthSplitInfo)
	weightHeightParts = BasicOperation.getTotalPartsFromSplitInfo(weightHeightSplitInfo)
	for widthIndex in range(weightWidthParts):
		widthBase, widthEnd = fcWidthHeightChannelBase(weightWidthSplitInfo, widthIndex)
		for heightIndex in range(weightHeightParts):
			heightBase, heightEnd = fcWidthHeightChannelBase(weightHeightSplitInfo, heightIndex)
			weightBlock = weight[widthBase:widthEnd, heightBase:heightEnd]
			weightBlockAlign = fcWeightAlign(weightBlock)
			weightAlignBlocks.append(weightBlockAlign)
	# Split output-activation
	outActiAlignBlocks = []
	outActiWidthSplitInfo = outActiSplitInfo[0]
	outActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(outActiWidthSplitInfo)
	for widthIndex in range(outActiWidthParts):
		widthBase, widthEnd = fcWidthHeightChannelBase(outActiWidthSplitInfo, widthIndex)
		outActiBlock = outActi[widthBase:widthEnd, :]
		outActiBlockAlign = fcActiAlign(outActiBlock)
		outActiAlignBlocks.append(outActiBlockAlign)
	# Get size of input-activation, weight and output-activation of largest part
	# !!! Because of address align to 16 bytes, it can cause SRAM overflow problem
	inActiBaseAddr = addressAlign(baseAddress)
	print("inActiBaseAddr:  {}".format(hex(inActiBaseAddr)))
	weightBaseAddr = addressAlign(inActiBaseAddr + inActiAlignBlocks[0].size)
	print("weightBaseAddr:  {}".format(hex(weightBaseAddr)))
	outActiBaseAddr = addressAlign(weightBaseAddr + weightAlignBlocks[0].size)
	print("outActiBaseAddr: {}".format(hex(outActiBaseAddr)))
	finalAddr = outActiBaseAddr + outActiAlignBlocks[0].size
	customAssert(finalAddr-baseAddress < PE_SRAM_LIMIT, "SRAM overflow!!!")
	return inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr

def fcWidthHeightChannelBase(splitInfo, partIndex):
	return convWidthHeightChannelBase(splitInfo, partIndex, overlapLen=0)

def fcWeightAlign(weight):
	# [width, heightBatch] -> [heightBatch, width]
	weight = np.swapaxes(weight, FC_WEIGHT_HEIGHT_AXIS, FC_WEIGHT_WIDTH_AXIS)
	# Generate zeros
	weightDim = weight.shape
	widthZeros = (COLUMN_ALIGN - (weightDim[1] % COLUMN_ALIGN)) % COLUMN_ALIGN
	zeros = np.zeros((weightDim[0], widthZeros), dtype=np.uint8)
	# Concatenate zeros to weight
	weightZeros = np.concatenate((weight, zeros), axis=-1)
	# Flatten concatenated weight
	weightZeros = np.ravel(weightZeros)
	return weightZeros

def fcActiAlign(acti, inputActiFlag=True):
	if inputActiFlag:
		# [width, heightBatch]
		actiDim = acti.shape
		# Generate zeros
		heightBatchZeros = (ROW_ALIGN - (actiDim[1] % ROW_ALIGN)) % ROW_ALIGN
		zeros = np.zeros((actiDim[0], heightBatchZeros), dtype=np.uint8)
		# Concatenate zeros to acti
		actiZeros = np.concatenate((acti, zeros), axis=-1)
		# Reshape to [width, heightBatch/4, 4]
		actiZerosDim = actiZeros.shape
		actiZeros = np.reshape(actiZeros, (actiZerosDim[0], actiZerosDim[1]//ROW_ALIGN, ROW_ALIGN))
		# Move axis: [width, heightBatch/4, 4] -> [heightBatch/4, width, 4]
		actiZeros = np.moveaxis(actiZeros, 1, 0)
		# Flatten concatenated acti
		actiZeros = np.ravel(actiZeros)
		# Align to 128-bit
		if actiZeros.size % 16 != 0:
			actiZeros = np.append(actiZeros, np.zeros((16-(actiZeros.size%16), 1)))
		return actiZeros
	else:
		# [width, heightBatch] -> [heightBatch, width]
		acti = np.swapaxes(acti, FC_ACTI_BATCH_HEIGHT_AXIS, FC_ACTI_WIDTH_AXIS)
		actiDim = acti.shape
		# Generate zeros
		widthZeros = (OUT_ALIGN - (actiDim[1] % OUT_ALIGN)) % OUT_ALIGN
		zeros = np.zeros((actiDim[0], widthZeros), dtype=np.uint32)
		# Concatenate zeros to acti
		actiZeros = np.concatenate((acti, zeros), axis=-1)
		# Flatten concatenated acti
		actiZeros = np.ravel(actiZeros)
		return actiZeros

def fcSpittedDataDim(layerSplitInfo, widthHeightOrder=True):
	layerType, inActiSplitInfo, weightSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	customAssert(inActiSplitInfo[1]==1, "Limitation of inActi height/batch")
	# TODO: check requiredPEs and parts
	# TODO: check layerType
	# [width, heightBatch], [width, height], [width, heightBatch]
	# 
	inActiDimBlocks = []
	inActiWidthSplitInfo = inActiSplitInfo[0]
	inActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(inActiWidthSplitInfo)
	for widthIndex in range(inActiWidthParts): 
		widthBase, widthEnd = fcWidthHeightChannelBase(inActiWidthSplitInfo, widthIndex)
		inActiDimBlocks.append([widthEnd-widthBase, 1])
	# Split weight
	weightDimBlocks = []
	weightWidthSplitInfo = weightSplitInfo[0]
	weightHeightSplitInfo = weightSplitInfo[1]
	weightWidthParts = BasicOperation.getTotalPartsFromSplitInfo(weightWidthSplitInfo)
	weightHeightParts = BasicOperation.getTotalPartsFromSplitInfo(weightHeightSplitInfo)
	if widthHeightOrder:
		for widthIndex in range(weightWidthParts):
			widthBase, widthEnd = fcWidthHeightChannelBase(weightWidthSplitInfo, widthIndex)
			for heightIndex in range(weightHeightParts):
				heightBase, heightEnd = fcWidthHeightChannelBase(weightHeightSplitInfo, heightIndex)
				weightDimBlocks.append([widthEnd-widthBase, heightEnd-heightBase])
	else:
		for heightIndex in range(weightHeightParts):
			heightBase, heightEnd = fcWidthHeightChannelBase(weightHeightSplitInfo, heightIndex)
			for widthIndex in range(weightWidthParts):
				widthBase, widthEnd = fcWidthHeightChannelBase(weightWidthSplitInfo, widthIndex)
				weightDimBlocks.append([widthEnd-widthBase, heightEnd-heightBase])		
	# Split output-activation
	outActiDimBlocks = []
	outActiWidthSplitInfo = outActiSplitInfo[0]
	outActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(outActiWidthSplitInfo)
	for widthIndex in range(outActiWidthParts):
		widthBase, widthEnd = fcWidthHeightChannelBase(outActiWidthSplitInfo, widthIndex)
		outActiDimBlocks.append([widthEnd-widthBase, 1])
	return inActiDimBlocks, weightDimBlocks, outActiDimBlocks

def fcDataSplitterForTf(layerSplitInfo, layerTypeParameter):
	'''
	For ICPRO
	'''
	layerType, inActiSplitInfo, weightSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
	customAssert(inActiSplitInfo[1]==1, "Limitation of inActi height/batch")
	layerType, layerParameter = layerTypeParameter
	inNeuron, outNeuron = layerParameter
	# TODO: check requiredPEs and parts
	# TODO: check layerType
	# Get CONV data: 
	# [width, heightBatch], [width, height], [width, heightBatch]
	inActi, weight, outActi = fcDataGenerator([inNeuron, 1], [outNeuron, inNeuron])
	# [heightBatch, width], [height, width], [width, heightBatch]
	inActi = fcActiSwapaxes(inActi)
	weight = fcWeightSwapaxes(weight)
	outActi = fcActiSwapaxes(outActi)
	# # Debug
	# inActiDim = inActi.shape
	# customPrint("inActi: ")
	# for heightIndex in range(inActiDim[2]):
	# 	customPrint("{}".format(inActi[0, :, heightIndex, 0]))
	# Split input-activation
	inActiBlocks = []
	inActiDimBlocks = []
	inActiWidthSplitInfo = inActiSplitInfo[0]
	inActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(inActiWidthSplitInfo)
	for widthIndex in range(inActiWidthParts): 
		widthBase, widthEnd = fcWidthHeightChannelBase(inActiWidthSplitInfo, widthIndex)
		inActiBlock = inActi[:, widthBase:widthEnd]
		inActiBlocks.append(inActiBlock)
		inActiDimBlocks.append([1, widthEnd-widthBase])
	# Split weight
	weightBlocks = []
	weightDimBlocks = []
	weightWidthSplitInfo = weightSplitInfo[0]
	weightHeightSplitInfo = weightSplitInfo[1]
	weightWidthParts = BasicOperation.getTotalPartsFromSplitInfo(weightWidthSplitInfo)
	weightHeightParts = BasicOperation.getTotalPartsFromSplitInfo(weightHeightSplitInfo)
	for widthIndex in range(weightWidthParts):
		widthBase, widthEnd = fcWidthHeightChannelBase(weightWidthSplitInfo, widthIndex)
		for heightIndex in range(weightHeightParts):
			heightBase, heightEnd = fcWidthHeightChannelBase(weightHeightSplitInfo, heightIndex)
			weightBlock = weight[heightBase:heightEnd, widthBase:widthEnd]
			weightBlocks.append(weightBlock)
			weightDimBlocks.append([heightEnd-heightBase, widthEnd-widthBase])
	# Split output-activation
	outActiBlocks = []
	outActiDimBlocks = []
	outActiWidthSplitInfo = outActiSplitInfo[0]
	outActiWidthParts = BasicOperation.getTotalPartsFromSplitInfo(outActiWidthSplitInfo)
	for widthIndex in range(outActiWidthParts):
		widthBase, widthEnd = fcWidthHeightChannelBase(outActiWidthSplitInfo, widthIndex)
		outActiBlock = outActi[:, widthBase:widthEnd]
		outActiBlocks.append(outActiBlock)
		outActiDimBlocks.append([1, widthEnd-widthBase])
	return inActiBlocks, inActiDimBlocks, weightBlocks, weightDimBlocks, outActiBlocks, outActiDimBlocks

#============================================================================================================
# 				Write input-activation, weight and output-activation alignment blocks into files
#============================================================================================================
def writeIntoFile(fileName, dataBlocks, dataType=DataType.UINT8):
	# addressAlign = address + (SRAM_BW_BYTES - address) % SRAM_BW_BYTES
	# customAssert(addressAlign==address, "Address [{}] doesn't align.".format(address))
	with open(fileName, "w") as file:
		for block in dataBlocks:
			file.write("@" + "\n")
			counter = 0
			for element in block:
				# Write data
				if DataType.UINT8 == dataType:
					file.write(format(element, "02X") + " ")
					counter += 1
				elif DataType.UINT32 == dataType:
					for byteIndex in range(4):
						file.write(format((element >> 8*byteIndex) & 0xFF, "02X") + " ")
					counter += 4
				else:
					customAssert(False, "Unsupport data type: {}".format(dataType))
				# Each 16 Bytes as one line 
				if SRAM_BW_BYTES == counter:
					counter = 0
					file.write("\n")
			if counter != 0:
				counter = 0
				file.write("\n")

#============================================================================================================
# 				Write input-activation, weight and output-activation alignment blocks into files
# CONV:
# 	Argus:		Layer splitting information
# 				input-activation
# 				weight
# 				output-activation
# 	Returns:	input-activation blocks
# 				weight blocks
# 				output-activation blocks
#============================================================================================================
def vggConv10():
	layerSplitInfo = (0, [[30, 1, 0, 0], [9, 4, 0, 0], 256], [[3, 3, 256, [4, 128, 0, 0]], 1], [[28, 1, 0, 0], [7, 4, 0, 0], [4, 128, 0, 0]], 32256, 512)
	layerTypeParameter = (0, (([30,30,256],[3,3,256,512],1,[28,28,512]), 1))
	inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
	writeIntoFile(INACTI_FILENAME, inActiAlignBlocks, dataType=DataType.UINT8)
	writeIntoFile(WEIGHT_FILENAME, weightAlignBlocks, dataType=DataType.UINT8)
	writeIntoFile(OUTACTI_FILENAME, outActiAlignBlocks, dataType=DataType.UINT32)

def vggFc19():
	layerSplitInfo = (0, [[373, 4, 372, 7], 1], [[256, 4, 0, 0], [373, 4, 372, 7]], [[256, 4, 0, 0], 1], 5968, 44)
	layerTypeParameter = (0, (4096, 1024))
	inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr = fcDataSplitter(layerSplitInfo, layerTypeParameter)




if __name__ == "__main__":
	# # CONV_10:
	# # Input: 	[30,30,256] -> [[30, 1, 0, 0], [9, 4, 0, 0], 256]   
	# # Weight: 	[3,3,256,512] -> [[3, 3, 256, [4, 128, 0, 0]], 1]
	# # Output: 	[28,28,512] -> [[28, 1, 0, 0], [7, 4, 0, 0], [4, 128, 0, 0]] 
	# layerSplitInfo = (0, [[30, 1, 0, 0], [9, 4, 0, 0], 256], [[3, 3, 256, [4, 128, 0, 0]], 1], [[28, 1, 0, 0], [7, 4, 0, 0], [4, 128, 0, 0]], 32256, 512)
	# layerTypeParameter = (0, (([30,30,256],[3,3,256,512],1,[28,28,512]), 1))
	# inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
	# writeIntoFile(INACTI_FILENAME, inActiBaseAddr, inActiAlignBlocks, dataType=DataType.UINT8)
	# writeIntoFile(WEIGHT_FILENAME, weightBaseAddr, weightAlignBlocks, dataType=DataType.UINT8)
	# writeIntoFile(OUTACTI_FILENAME, outActiBaseAddr, outActiAlignBlocks, dataType=DataType.UINT32)
	# # Debug
	# customPrint("Parts of inActiBlocks: {}".format(len(inActiBlocks)))
	# for index in range(len(inActiBlocks)):
	# 	# customPrint("inActi subarray: \n{}".format(inActiBlocks[index]))
	# 	inActiSub = inActiBlocks[index]
	# 	inActiSubDim = inActiSub.shape
	# 	customPrint("parts: {}, width {}".format(index, inActiSubDim))
	# 	# # for channelIndex in range(inActiSubDim[3]):
	# 	# for heightIndex in range(inActiSubDim[2]):
	# 	# 	# [batch, inchannel, height, width]
	# 	# 	customPrint("{}".format(inActiSub[0, 0, heightIndex, :]))
	# 	customPrint("{}".format(inActiSub))
	# vggConv10()
	vggFc19()