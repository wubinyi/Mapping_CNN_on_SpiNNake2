import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
s2sFolderPath = os.path.join(projectFolder, "spiNNaker2Simulator")
if s2sFolderPath not in sys.path:
	sys.path.insert(0, s2sFolderPath)
parserSplitterPath = os.path.join(projectFolder, "parserSplitter")
if parserSplitterPath not in sys.path:
	sys.path.insert(0, parserSplitterPath)

from enum import Enum, auto
from nnGeneral import *
import math
from spiNNakerSimulatorGeneral import NUM_OF_PES_IN_ALL_DOUBLOCK

class PoolType(Enum):
	MAXPOOL = auto()
	AVGPOOL = auto()


MAXPOOL_TOTAL_STEP = 3

class PoolLayerMapper(object):
	'''
	For Max pooling, use ARM Core to accelerate pooling layer
	For Average pooling, firstly use MAC array, them use ARM Core (For ARM core maybe use shift operation)
	'''
	def __init__(self, layerName, layerParameter, prelayerSplitInfo=None, 
		limitedInActiEffeUtil=0.1, limitedSramUtilization=0.9, scale=0.7, forSpiNNaker2=False):
		'''
		Args:
			layerParameter: tuple, 
				e.g. ("input activation DIM"(list), "pooling filter DIM"(list), stride(int), 
						poolType(Enum), "output activation DIM"(list))
				"input activation DIM"  = [width, height, in_channel]
				"poolingfilter DIM" 	= [width, height]
				"output activation DIM" = [width, height, out_channel]
				poolType: enumeration of PoolType
			layerName: String for layer name, e.g. "POOL_x", where x is layer number.
		'''
		self.mapperReset(layerName, layerParameter, limitedInActiEffeUtil, 
			limitedSramUtilization, scale, prelayerSplitInfo)
		self.forSpiNNaker2 = forSpiNNaker2

	def mapperReset(self, layerName, layerParameter, limitedInActiEffeUtil, 
		limitedSramUtilization, scale, prelayerSplitInfo=None):
		'''
		Reset PoolLayerMapper variables
		'''
		self.layerName = layerName
		self.layerParameter = layerParameter
		self.prelayerSplitInfo = prelayerSplitInfo
		_, poolFilterDim, stride, poolType, _ = self.layerParameter
		if min(poolFilterDim) > stride:
			self.limitedInActiEffeUtil = limitedInActiEffeUtil
		else:
			self.limitedInActiEffeUtil = limitedInActiEffeUtil
		self.limitedSramUtilization = limitedSramUtilization
		self.scale = scale
		self.poolSplitInfo = {}
		self.theoreticalComputationClock = [0]*8
		self.actualComputationClock = [0]*8
		self.layerMapInfo = {}
		self.layerNumberStr = layerName[len(LayerType.POOL.name+"_"):]
		self.splitStep = 1
		self.poolType = poolType

	def getSplitInfo(self):
		return self.poolSplitInfo

	def getClocks(self):
		return max(self.actualComputationClock)

	def map(self):
		inActivationDim, _, _, poolType, outActivationDim = self.layerParameter
		assert(inActivationDim[2] == outActivationDim[2]), \
			"input channel-[] and output channel should equal".format(inActivationDim[2], outActivationDim[2])	
		if self.prelayerSplitInfo == None:	
			if PoolType.MAXPOOL == poolType:
				self.maxPoolMap()
			elif PoolType.AVGPOOL == poolType:
				BasicOperation.customPrintT("---"*10)
				BasicOperation.customPrintT("---"*4+" NOT DEFINED "+"---"*4)
				BasicOperation.customPrintT("---"*10)
		else:
			if PoolType.MAXPOOL == poolType:
				self.setToPrelayerSplitScheme()
			elif PoolType.AVGPOOL == poolType:
				self.setToPrelayerSplitScheme()
		return self.layerMapInfo

	# ==================================================================================================
	# 							(MAX-POOL) Using Splitting scheme from pre-layer
	# MAX-POOL + NON-OVERLAP
	# ==================================================================================================
	def setToPrelayerSplitScheme(self):
		assert(self.prelayerSplitInfo != None), "prelayerSplitInfo should not be None."
		# Extract layer parameter
		inActivationDim, poolFilterDim, stride, _, outActivationDim = self.layerParameter
		filterWidth = poolFilterDim[0]
		filterHeight = poolFilterDim[1]
		# assert(filterWidth == stride), "Filter Width: Only support NON-OVERLAP Pooling"
		# assert(filterHeight == stride), "Filter Height: Only support NON-OVERLAP Pooling"
		# Extract pre-layer splitting scheme
		inWidthSplitInfo, inHeightSplitInfo, inChannelSplitInfo = self.preLayerSplitInfoReSplit()
		assert(BasicOperation.splitInfoToTotalLength(inWidthSplitInfo) == inActivationDim[0]), \
			"width not equal."
		assert(BasicOperation.splitInfoToTotalLength(inHeightSplitInfo) == inActivationDim[1]), \
			"height not equal."
		# Generate splitting info
		self.poolSplitInfo[POOL_IN_WIDTH] = inWidthSplitInfo
		self.poolSplitInfo[POOL_IN_HEIGHT] = inHeightSplitInfo
		self.poolSplitInfo[POOL_CHANNEL] = inChannelSplitInfo
		self.poolSplitInfo[POOL_FILTER_WIDTH] = filterWidth
		self.poolSplitInfo[POOL_FILTER_HEIGHT] = filterHeight
		self.poolSplitInfo[POOL_FILTER_STRIDE] = stride
		self.poolSplitInfo[POOL_OUT_WIDTH] = PoolLayerMapper.poolSplitInfoInToOut(inWidthSplitInfo, filterWidth, stride)
		self.poolSplitInfo[POOL_OUT_HEIGHT] = PoolLayerMapper.poolSplitInfoInToOut(inHeightSplitInfo, filterHeight, stride)
		BasicOperation.customPrintF("self.poolSplitInfo[POOL_IN_WIDTH]: {}".format(self.poolSplitInfo[POOL_IN_WIDTH]))
		BasicOperation.customPrintF("self.poolSplitInfo[POOL_IN_HEIGHT]: {}".format(self.poolSplitInfo[POOL_IN_HEIGHT]))
		BasicOperation.customPrintF("self.poolSplitInfo[POOL_CHANNEL]: {}".format(self.poolSplitInfo[POOL_CHANNEL]))
		BasicOperation.customPrintF("self.poolSplitInfo[POOL_OUT_WIDTH]: {}".format(self.poolSplitInfo[POOL_OUT_WIDTH]))
		BasicOperation.customPrintF("self.poolSplitInfo[POOL_OUT_HEIGHT]: {}".format(self.poolSplitInfo[POOL_OUT_HEIGHT]))
		self.clockComputation()
		# PE allocation infomation
		outWidthSplitInfo = self.poolSplitInfo[POOL_OUT_WIDTH]
		outHeightSplitInfo = self.poolSplitInfo[POOL_OUT_HEIGHT]
		nameList = [LayerType.POOLTLF.name, LayerType.POOLTLB.name,
					LayerType.POOLTRF.name, LayerType.POOLTRB.name,
					LayerType.POOLBLF.name, LayerType.POOLBLB.name,
					LayerType.POOLBRF.name, LayerType.POOLBRB.name]
		for heightInfoIndex in range(2):
			for widthInfoIndex in range(2):
				for channelInfoIndex in range(2):	
					clockIndex = channelInfoIndex + widthInfoIndex * 2 + heightInfoIndex * 4
					self.maxpoolNonoverlapInsertSplitResult(nameList[clockIndex]+"_"+self.layerNumberStr, 
							inWidthSplitInfo[widthInfoIndex*2:widthInfoIndex*2+2], 
							inHeightSplitInfo[heightInfoIndex*2:heightInfoIndex*2+2],							
							inChannelSplitInfo[channelInfoIndex*2:channelInfoIndex*2+2],
							outWidthSplitInfo[widthInfoIndex*2:widthInfoIndex*2+2], 
							outHeightSplitInfo[heightInfoIndex*2:heightInfoIndex*2+2], 
							clockIndex)

	def preLayerSplitInfoReSplit(self):
		_, poolFilterDim, stride, _, _ = self.layerParameter
		preOutWidthSplitInfo = self.prelayerSplitInfo[QUAN_WIDTH]
		preOutHeightSplitInfo = self.prelayerSplitInfo[QUAN_HEIGHT]
		preOutChannelSplitInfo = self.prelayerSplitInfo[QUAN_CHANNEL]
		if preOutWidthSplitInfo[0] < stride:
			preOutWidth = preOutWidthSplitInfo[0] * preOutWidthSplitInfo[1] + preOutWidthSplitInfo[2] * preOutWidthSplitInfo[3]
			partsOfInWidth = math.ceil(preOutHeight / stride)
			inWidthSplitInfo = BasicOperation.oneDimSplitting(preOutWidth, partsOfInWidth) 
		else:
			inWidthSplitInfo = preOutWidthSplitInfo
		if preOutHeightSplitInfo[0] < stride:
			preOutHeight = preOutHeightSplitInfo[0] * preOutHeightSplitInfo[1] + preOutHeightSplitInfo[2] * preOutHeightSplitInfo[3]
			partsOfInHeight = math.ceil(preOutHeight / stride)
			inHeightSplitInfo = BasicOperation.oneDimSplitting(preOutHeight, partsOfInHeight)
		else:
			inHeightSplitInfo = preOutHeightSplitInfo
		inChannelSplitInfo = preOutChannelSplitInfo
		return inWidthSplitInfo, inHeightSplitInfo, inChannelSplitInfo

	def maxpoolNonoverlapInsertSplitResult(self, layerName, inWidthInfo, inHeightInfo, inChannelInfo, 
		outWidthInfo, outHeightInfo, clockIndex):
		# Extract parameter
		inWidth = inWidthInfo[0]
		partsOfInWidth = inWidthInfo[1]
		inHeight = inHeightInfo[0]
		partsOfInHeight = inHeightInfo[1]
		inChannel = inChannelInfo[0]
		partsOfInChannel = inChannelInfo[1]
		outWidth = outWidthInfo[0]
		partsOfOutWidth = outWidthInfo[1]
		outHeight = outHeightInfo[0]
		partsOfOutHeight = outHeightInfo[1]
		if partsOfInWidth == 0 or partsOfInHeight == 0 or partsOfInChannel == 0 or partsOfOutWidth == 0 or partsOfOutHeight == 0:
			return
		# Create PE allocation information
		peAllocationInfo = {}
		peAllocationInfo[PEs] = partsOfOutWidth * partsOfOutHeight * partsOfInChannel
		peAllocationInfo[INACTI] =  inWidth * inHeight * inChannel
		peAllocationInfo[INACTIALIG] = PoolLayerMapper.poolSizeAlignCompute(inWidth, inHeight, inChannel)
		peAllocationInfo[WEIGHT] = 0
		peAllocationInfo[WEIGHTALIG] = 0
		peAllocationInfo[OUTACTI] = outWidth * outHeight * inChannel
		peAllocationInfo[OUTACTIALIG] = PoolLayerMapper.poolSizeAlignCompute(outWidth, outHeight, inChannel)
		# insert mapping information
		self.insertPoolLayerMap(layerName, peAllocationInfo, 1.0, clockIndex)

	@staticmethod
	def poolSplitInfoInToOut(lengthList, filterLenght, stride):
		outList = [0] * 4
		outList[0] = PoolLayerMapper.poolInToOut(lengthList[0], filterLenght, stride)
		outList[1] = lengthList[1]
		outList[2] = PoolLayerMapper.poolInToOut(lengthList[2], filterLenght, stride)
		outList[3] = lengthList[3]
		return outList

	@staticmethod
	def poolInToOut(length, filterLenght, stride):
		return int((length - filterLenght) / stride) + 1

	# ==================================================================================================
	# 									(MAX-POOL) Step 1: No Splitting
	# ==================================================================================================
	def maxPoolMap(self):
		# Extract layer parameter
		inActivationDim, poolFilterDim, stride, _, outActivationDim = self.layerParameter
		filterWidth = poolFilterDim[0]
		filterHeight = poolFilterDim[1]
		# Get pooling layer split info and update computation clock
		self.poolSplitInfo[POOL_IN_WIDTH] = inActivationDim[0]
		self.poolSplitInfo[POOL_IN_HEIGHT] = inActivationDim[1]
		self.poolSplitInfo[POOL_CHANNEL] = inActivationDim[2]
		self.poolSplitInfo[POOL_FILTER_WIDTH] = filterWidth
		self.poolSplitInfo[POOL_FILTER_HEIGHT] = filterHeight
		self.poolSplitInfo[POOL_FILTER_STRIDE] = stride
		self.poolSplitInfo[POOL_OUT_WIDTH] = outActivationDim[0]
		self.poolSplitInfo[POOL_OUT_HEIGHT] = outActivationDim[1]
		self.clockComputation()
		# Get input-activation size
		inActivationSize = BasicOperation.listInnerProduct(inActivationDim)
		inActivationSizeAlign = PoolLayerMapper.poolSizeAlignCompute(\
			inActivationDim[0], inActivationDim[1], inActivationDim[2])
		# Get filter size
		poolFilterSize = 0
		poolFilterSizeAlign = 0
		# Get output-activation size
		outActivationSize = BasicOperation.listInnerProduct(outActivationDim)
		outActivationSizeAlign = PoolLayerMapper.poolSizeAlignCompute(\
			outActivationDim[0], outActivationDim[1], outActivationDim[2])
		# PE allocation infomation
		peAllocationInfo = {}
		peAllocationInfo[PEs] = 1
		peAllocationInfo[INACTI] = inActivationSize
		peAllocationInfo[INACTIALIG] = inActivationSizeAlign
		peAllocationInfo[WEIGHT] = poolFilterSize
		peAllocationInfo[WEIGHTALIG] = poolFilterSizeAlign
		peAllocationInfo[OUTACTI] = outActivationSize
		peAllocationInfo[OUTACTIALIG] = outActivationSizeAlign
		# Deteermine if SRAM limitation is meet
		if inActivationSizeAlign + poolFilterSizeAlign + outActivationSizeAlign <= PE_SRAM_LIMIT:
			self.insertPoolLayerMap(self.layerName, peAllocationInfo)
		else:
			self.insertPoolLayerMap(self.layerName, peAllocationInfo, invalid=True)
			self.maxPoolSplit(peAllocationInfo)

	# ==================================================================================================
	# 					(MAX-POOL) Step 2: Splitting width, height and channel together
	# ==================================================================================================
	def maxPoolSplit(self, prePeAllocationInfo):
		'''
		For max-pooling, there is no alignment problem, as it is accelerated by ARM Core.
		However, because the input comes from previous layer, in order to reduce data rearrangment (transformation),
			when splitting the channel dimension, it is recommend to align to MAC_ARRAY_ROW (=4)
		Moreover I still declare an alignment function for future extension.
		'''
		_, poolFilterDim, stride, _, _ = self.layerParameter
		# Get optimum split scheme
		outWidthSplitInfo, outHeightSplitInfo, outChannelSplitInfo, inActiUtil = self.maxPoolFindSplitScheme(prePeAllocationInfo)
		if outWidthSplitInfo == None:
			outWidthSplitInfo, outHeightSplitInfo, outChannelSplitInfo, inActiUtil = self.maxPoolFindSplitScheme(prePeAllocationInfo, relaxRestrict=True)
		# Insert split information
		self.poolSplitInfo[POOL_OUT_WIDTH] = outWidthSplitInfo
		self.poolSplitInfo[POOL_OUT_HEIGHT] = outHeightSplitInfo
		self.poolSplitInfo[POOL_CHANNEL] = outChannelSplitInfo
		inWidthSplitInfo = PoolLayerMapper.poolSplitInfoOutToIn(outWidthSplitInfo, stride, poolFilterDim[0])
		inHeightSplitInfo = PoolLayerMapper.poolSplitInfoOutToIn(outHeightSplitInfo, stride, poolFilterDim[1])
		self.poolSplitInfo[POOL_IN_WIDTH] = inWidthSplitInfo
		self.poolSplitInfo[POOL_IN_HEIGHT] = inHeightSplitInfo
		self.clockComputation()
		BasicOperation.customPrintF(">>----"*10)
		BasicOperation.customPrintF("--> outWidthSplitInfo: {}".format(outWidthSplitInfo))
		BasicOperation.customPrintF("--> outHeightSplitInfo: {}".format(outHeightSplitInfo))
		BasicOperation.customPrintF("--> outChannelSplitInfo: {}".format(outChannelSplitInfo))
		BasicOperation.customPrintF("----<<"*10)
		nameList = [LayerType.POOLTLF.name, LayerType.POOLTLB.name,
					LayerType.POOLTRF.name, LayerType.POOLTRB.name,
					LayerType.POOLBLF.name, LayerType.POOLBLB.name,
					LayerType.POOLBRF.name, LayerType.POOLBRB.name]
		for heightInfoIndex in range(2):
			for widthInfoIndex in range(2):
				for channelInfoIndex in range(2):	
					clockIndex = channelInfoIndex + widthInfoIndex * 2 + heightInfoIndex * 4
					self.maxpoolInsertSplitResult(nameList[clockIndex]+"_"+self.layerNumberStr, 
							prePeAllocationInfo, 
							outWidthSplitInfo[widthInfoIndex*2:widthInfoIndex*2+2], 
							outHeightSplitInfo[heightInfoIndex*2:heightInfoIndex*2+2], 
							outChannelSplitInfo[channelInfoIndex*2:channelInfoIndex*2+2], 
							inActiUtil, 
							clockIndex)

	def maxpoolInsertSplitResult(self, layerName, prePeAllocationInfo, widthInfo, heightInfo, 
		channelInfo, inActivationUtilization, clockIndex):
		# Extract parameter
		_, poolFilterDim, stride, _, _ = self.layerParameter
		outWidth = widthInfo[0]
		partsOfOutWidth = widthInfo[1]
		outHeight = heightInfo[0]
		partsOfOutHeight = heightInfo[1]
		inChannel = channelInfo[0]
		partsOfInChannel = channelInfo[1]
		if partsOfOutWidth == 0 or partsOfOutHeight == 0 or partsOfInChannel == 0:
			return
		inWidth = PoolLayerMapper.poolOutToIn(outWidth, stride, poolFilterDim[0])
		inHeight = PoolLayerMapper.poolOutToIn(outHeight, stride, poolFilterDim[1])
		# Create PE allocation information
		peAllocationInfo = {}
		peAllocationInfo[PEs] = prePeAllocationInfo[PEs] * partsOfOutWidth * partsOfOutHeight * partsOfInChannel
		peAllocationInfo[INACTI] =  inWidth * inHeight * inChannel
		peAllocationInfo[INACTIALIG] = PoolLayerMapper.poolSizeAlignCompute(inWidth, inHeight, inChannel)
		peAllocationInfo[WEIGHT] = prePeAllocationInfo[WEIGHT]
		peAllocationInfo[WEIGHTALIG] = prePeAllocationInfo[WEIGHTALIG]
		peAllocationInfo[OUTACTI] = outWidth * outHeight * inChannel
		peAllocationInfo[OUTACTIALIG] = PoolLayerMapper.poolSizeAlignCompute(outWidth, outHeight, inChannel)
		# insert mapping information
		self.insertPoolLayerMap(layerName, peAllocationInfo, inActivationUtilization, clockIndex)

	def maxPoolFindSplitScheme(self, prePeAllocationInfo, relaxRestrict=False):
		'''
		The splitting scheme based on the output activation.
		For the "inActiUtil", it could not be always equal to 1. Especially when stride < pool_width/pool_height.
		'''
		inActivationDim, poolFilterDim, stride, _, outActivationDim = self.layerParameter
		nonoverlapInActiSize = BasicOperation.listInnerProduct(inActivationDim[:2])
		outActiWidth = outActivationDim[0]
		outActiHeight = outActivationDim[1]
		outActiChannel = outActivationDim[2]
		poolFilterWidth = poolFilterDim[0]
		poolFilterHeight = poolFilterDim[1]
		preInActivationSizeAlign = prePeAllocationInfo[INACTIALIG]
		preOutActivationSizeAlign = prePeAllocationInfo[OUTACTIALIG]
		# Get minimum splitting parts
		minimumParts = math.ceil((preInActivationSizeAlign + preOutActivationSizeAlign) / PE_SRAM_LIMIT)
		# Get maximum parts by splitting channel/width/height
		maxPartsByWidthHeight = math.ceil(minimumParts / self.scale)
		maxPartsByWidth = maxPartsByWidthHeight if maxPartsByWidthHeight < outActiWidth else outActiWidth
		maxPartsByHeight = maxPartsByWidthHeight if maxPartsByWidthHeight < outActiHeight else outActiHeight
		maxPartsByChannel = math.ceil(outActiChannel)
		channelAlign = BasicOperation.ceilOf(outActiChannel, MAC_ARRAY_ROW)
		# print("----> maxPartsByChannel: {}".format(maxPartsByChannel))
		# print("----> maxPartsByWidth: {}".format(maxPartsByWidth))
		# print("----> maxPartsByHeight: {}".format(maxPartsByHeight))
		# Begin searching the optimum splitting scheme
		partsByWidth = maxPartsByWidth
		partsByHeight = maxPartsByHeight
		partsByChannel = maxPartsByChannel
		widthSplitInfo = None
		heightSplitInfo = None
		channelSplitInfo = None
		inActiUtil = 0
		increasedSize = 10 * nonoverlapInActiSize
		for partsByChannelTemp in range(maxPartsByChannel, 0, -1):
			# maxPartsByWidth = math.ceil(minimumParts / partsByChannelTemp / self.scale)
			# maxPartsByWidth = maxPartsByWidth if maxPartsByWidth < outActiWidth else outActiWidth
			# maxPartsByWidth = maxPartsByWidth if maxPartsByWidth > 0 else 1
			for partsByWidthTemp in range(maxPartsByWidth, 0, -1):
				# maxPartsByHeight = math.ceil(minimumParts / partsByChannelTemp / partsByWidthTemp / self.scale)
				# maxPartsByHeight = maxPartsByHeight if maxPartsByHeight < outActiWidth else outActiWidth	
				# maxPartsByHeight = maxPartsByHeight if maxPartsByHeight > 0 else 1			
				for partsByHeightTemp in range(maxPartsByHeight, 0, -1):
				# for partsByHeightTemp in range(1, maxPartsByHeight+1):
					# Special for spinnaker
					if self.forSpiNNaker2:
						if partsByChannelTemp * partsByWidthTemp * partsByHeightTemp < NUM_OF_PES_IN_ALL_DOUBLOCK:
							continue
					# Splitting channel, width and height
					widthSplitInfoTemp = BasicOperation.oneDimSplitting(outActiWidth, partsByWidthTemp)
					heightSplitInfoTemp = BasicOperation.oneDimSplitting(outActiHeight, partsByHeightTemp)
					channelSplitInfoTemp = BasicOperation.oneDimSplitting(outActiChannel, partsByChannelTemp)
					# For spiNNaker, each part should be the same
					if self.forSpiNNaker2:
						if widthSplitInfoTemp[2] != 0:
							continue
						if not relaxRestrict:
							if heightSplitInfoTemp[2] != 0:
								continue
						if channelSplitInfoTemp[2] != 0:
							continue
					# # Determine if the channel alignment is meet
					# splitChannelAlign = BasicOperation.listCeilOf(channelSplitInfoTemp, MAC_ARRAY_ROW)
					# if splitChannelAlign != channelAlign:
					# 	channelSplitInfoTemp = BasicOperation.oneDimSplittingWithAlign(\
					# 						outActiChannel, partsByChannelTemp, MAC_ARRAY_ROW)
					# 	if channelSplitInfoTemp == None:
					# 		continue
					# Determine if meet the limitedInActiEffeUtil (Determined by width/height splitting info)
					overlapWidthTemp = PoolLayerMapper.poolOutToInOverlapDim(widthSplitInfoTemp, stride, poolFilterWidth)
					overlapHeightTemp = PoolLayerMapper.poolOutToInOverlapDim(heightSplitInfoTemp, stride, poolFilterHeight)
					overlapSizeTemp = overlapWidthTemp * overlapHeightTemp
					inActiUtilTemp =  nonoverlapInActiSize / overlapSizeTemp
					assert(inActiUtilTemp <= 1.0), "inActiUtilTemp-[{}] should not larger than 1".format(inActiUtilTemp)
					if inActiUtilTemp < self.limitedInActiEffeUtil:
						continue
					increasedSizeTemp = overlapSizeTemp - nonoverlapInActiSize
					# Determine if the largest parts meet the SRAM limitation
					outActiSizeOfLargestPartAlignTemp = PoolLayerMapper.poolSizeAlignCompute(\
						widthSplitInfoTemp[0], heightSplitInfoTemp[0], channelSplitInfoTemp[0])
					inActiDimOfLargestPartTemp = PoolLayerMapper.poolActivationOutToIn(\
						[widthSplitInfoTemp[0], heightSplitInfoTemp[0]], stride, poolFilterDim)
					inActiSizeOfLargestPartAlignTemp = PoolLayerMapper.poolSizeAlignCompute(\
						inActiDimOfLargestPartTemp[0], inActiDimOfLargestPartTemp[1], channelSplitInfoTemp[0])
					largestPartSizeAlignTemp = inActiSizeOfLargestPartAlignTemp + outActiSizeOfLargestPartAlignTemp
					if largestPartSizeAlignTemp > PE_SRAM_LIMIT:
						continue
					# if finding meet scheme, choose as few parts as possbble
					# Moreover, teng to selecte splitting width as few as possible
					if self.forSpiNNaker2:
						partsFlag = partsByWidthTemp * partsByHeightTemp * partsByChannelTemp > partsByWidth * partsByHeight * partsByChannel
						if increasedSizeTemp < increasedSize or (increasedSizeTemp == increasedSize and partsFlag):
							increasedSize = increasedSizeTemp
							partsByChannel = partsByChannelTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = widthSplitInfoTemp
							heightSplitInfo = heightSplitInfoTemp
							channelSplitInfo = channelSplitInfoTemp
							inActiUtil = inActiUtilTemp
					else:
						partsFlag = partsByWidthTemp * partsByHeightTemp * partsByChannelTemp <= partsByWidth * partsByHeight * partsByChannel
						if increasedSizeTemp < increasedSize or (increasedSizeTemp == increasedSize and partsFlag):
							increasedSize = increasedSizeTemp
							partsByChannel = partsByChannelTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = widthSplitInfoTemp
							heightSplitInfo = heightSplitInfoTemp
							channelSplitInfo = channelSplitInfoTemp
							inActiUtil = inActiUtilTemp
		return widthSplitInfo, heightSplitInfo, channelSplitInfo, inActiUtil

	@staticmethod
	def poolOutToInOverlapDim(outActivationDimSplitInfo, stride, correspondingFilterDimSize):
		'''
		Compute the input-activation dimension size (width/height) according filter-dimension, 
			stride and output-activation splitting info.
		Note that the dimension size contains the overlap-part.

		Args:
			outActivationDimSplitInfo: list, [largeXxx, partsOfLargeXxx, smallXxx, partsOfSmallXxx]
			stride: int
			correspondingFilterDimSize: int, corresponding dimension size of filter

		Returns:
			the dimension size (width/height) of input-activation, int
		'''
		return ((outActivationDimSplitInfo[0] - 1) * stride + correspondingFilterDimSize) * outActivationDimSplitInfo[1] + \
			 	((outActivationDimSplitInfo[2] - 1) * stride + correspondingFilterDimSize) * outActivationDimSplitInfo[3]

	@staticmethod
	def poolActivationOutToIn(outActivation2D, stride, poolFilter2D):
		'''
		Compute the input-activation dimension size according filter-dimension, stride and output-activation dimension.
		output-activation -> input-activation: [width, height]

		Args:
			outActivation2D: list, [width, height]
			stride: int
			poolFilter2D: list, [width, height]

		Returns:
			the dimension size of input-activation [width, height]
		'''
		return [(x-1)*stride+y for x, y in zip(outActivation2D, poolFilter2D)]

	@staticmethod
	def poolSplitInfoOutToIn(splitInfo, stride, correspondingFilterDimSize):
		'''
		Compute the input-activation splitting info according filter, stride and output-activation splitting info..

		Args:
			outActivationDimSplitInfo: list, [largeXxx, partsOfLargeXxx, smallXxx, partsOfSmallXxx]
			stride: int
			correspondingFilterDimSize: int, corresponding dimension size of filter

		Returns:
			the splitting info of input-activation, list, [largeXxx, partsOfLargeXxx, smallXxx, partsOfSmallXxx]
		'''		
		outSplitInfo = splitInfo.copy()
		outSplitInfo[0] = PoolLayerMapper.poolOutToIn(splitInfo[0], stride, correspondingFilterDimSize)
		outSplitInfo[2] = PoolLayerMapper.poolOutToIn(splitInfo[2], stride, correspondingFilterDimSize)
		return outSplitInfo

	@staticmethod
	def poolOutToIn(outActivationDimSize, stride, correspondingFilterDimSize):
		'''
		width/height of output -> width/height of input
		'''
		if outActivationDimSize == 0:
			return 0
		return (outActivationDimSize - 1) * stride + correspondingFilterDimSize

	@staticmethod
	def poolSizeAlignCompute(width, height, channel):
		'''
		compute the alignment size of activation
		'''
		return width * height * channel

	# ==================================================================================================
	# 								Insert layer Mapping information
	# ==================================================================================================
	def insertPoolLayerMap(self, layerName, peAllocationInfo, inActivationUtilization=1.0, 
			clockIndex=0, invalid=False):
		# Extract pe allocation infomation
		numOfPEs = peAllocationInfo[PEs]
		inActivationSizeOnEachPe = peAllocationInfo[INACTI]
		inActivationSizeOnEachPeAlign = peAllocationInfo[INACTIALIG]
		filterSizeOnEachPe = peAllocationInfo[WEIGHT] 
		filterSizeOnEachPeAlign = peAllocationInfo[WEIGHTALIG]
		outActivationSizeOnEachPe = peAllocationInfo[OUTACTI]
		outActivationSizeOnEachPeAlign = peAllocationInfo[OUTACTIALIG]
		# Get sram utilization
		sramUtilization = (inActivationSizeOnEachPeAlign + filterSizeOnEachPeAlign + outActivationSizeOnEachPeAlign) / \
							PE_SRAM_LIMIT
		# invalid allocation, it means the layer will be remap again
		if invalid:
			layerName = layerName + "*"
		# Get effective utilization of SRAM for input/output-activation and filter
		inActivationPercentage = inActivationSizeOnEachPe / inActivationSizeOnEachPeAlign
		if filterSizeOnEachPeAlign:
			filterPercentage = filterSizeOnEachPe/filterSizeOnEachPeAlign
		else:
			filterPercentage = 0
		outActivationPercentage = outActivationSizeOnEachPe / outActivationSizeOnEachPeAlign
		try:
			speedRatio = self.theoreticalComputationClock[clockIndex] / self.actualComputationClock[clockIndex]
		except Exception as e:
			print(e)
			BasicOperation.customPrintT("clockIndex: {}".format(clockIndex))
		# Insert the allocation information to layer mapping information
		self.layerMapInfo[layerName] = [numOfPEs, \
			"{} [{}({:.2%})]".format(inActivationSizeOnEachPeAlign, \
							inActivationSizeOnEachPe, \
							inActivationPercentage), \
			"{} [{}({:.2%})]".format(filterSizeOnEachPeAlign, \
									filterSizeOnEachPe, \
									filterPercentage), \
			"{} [{}({:.2%})]".format(outActivationSizeOnEachPeAlign,  \
										outActivationSizeOnEachPe, \
										outActivationPercentage), \
			"{} [{}({:.2f})]".format(self.actualComputationClock[clockIndex],  \
										self.theoreticalComputationClock[clockIndex], \
										speedRatio),
			"{:.2f}".format(inActivationUtilization),
			"{:.2f}".format(sramUtilization)]
	# ==================================================================================================
	# 										Computation clocks
	# ==================================================================================================
	def clockComputation(self):
		'''
		The computation clock of each PE is determined by the splittedInActivation and splittedFilter
		As there are 2 kinds of widthOfInActivation, 2 kinds of heightOfInActivation and 2 kinds of 
			inChannelOfInActivation, therefore, there are total 8 kinds of computation clocks.
		We expected their difference of computation amount between each other will not be large.

		(WHI - WHi) - (wHI - wHi) - (WhI - Whi) - (whI - whi)
		Height - Width - inChannel
		'''
		_, poolFilterDim, stride, _, _ = self.layerParameter
		inHeights = [0, 0]
		inWidths = [0, 0]
		inChannels = [0, 0]
		if isinstance(self.poolSplitInfo[POOL_IN_HEIGHT], list):
			inHeights[0] = self.poolSplitInfo[POOL_IN_HEIGHT][0]
			inHeights[1] = self.poolSplitInfo[POOL_IN_HEIGHT][2]
		else:
			inHeights[0] = self.poolSplitInfo[POOL_IN_HEIGHT]
			inHeights[1] = 0
		if isinstance(self.poolSplitInfo[POOL_IN_WIDTH], list):
			inWidths[0] = self.poolSplitInfo[POOL_IN_WIDTH][0]
			inWidths[1] = self.poolSplitInfo[POOL_IN_WIDTH][2]
		else:
			inWidths[0] = self.poolSplitInfo[POOL_IN_WIDTH]
			inWidths[1] = 0			
		if isinstance(self.poolSplitInfo[POOL_CHANNEL], list):
			inChannels[0] = self.poolSplitInfo[POOL_CHANNEL][0]
			inChannels[1] = self.poolSplitInfo[POOL_CHANNEL][2]
		else:
			inChannels[0] = self.poolSplitInfo[POOL_CHANNEL]
			inChannels[1] = 0
		filterWidth = poolFilterDim[0]
		filterHeight = poolFilterDim[1]
		index = 0
		for inHeight in inHeights:
			for inWidth in inWidths:
				for inChannel in inChannels:
					self.theoreticalComputationClock[index] = PoolLayerMapper.poolTheoreticalClocks(\
						inWidth, inHeight, inChannel, filterWidth, filterHeight, stride)
					self.actualComputationClock[index] = PoolLayerMapper.poolActualClocks(\
						inWidth, inHeight, inChannel, filterWidth, filterHeight, stride)
					index = index + 1

	@staticmethod
	def poolTheoreticalClocks(inWidth, inHeight, inChannel, filterWidth, filterHeight, stride):
		oneIterationClocks = filterWidth * filterHeight - 1
		iterations = (math.floor((inWidth - filterWidth) / stride) + 1) * (math.floor((inHeight - filterHeight) / stride) + 1)
		return iterations * oneIterationClocks * inChannel

	@staticmethod
	def poolActualClocks(inWidth, inHeight, inChannel, filterWidth, filterHeight, stride):
		oneIterationClocks = filterWidth * filterHeight - 1
		iterations = (math.floor((inWidth - filterWidth) / stride) + 1) * (math.floor((inHeight - filterHeight) / stride) + 1)
		return iterations * oneIterationClocks * inChannel