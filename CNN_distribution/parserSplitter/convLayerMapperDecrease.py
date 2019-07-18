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

from nnGeneral import *
from convLayerMapper import *
from spiNNaker2General import *


class ConvLayerMapperDecrease(ConvLayerMapper):
	'''
	Distribute dense layer into SpiNNaker2. All input/output-activation and filters need to place in local SRAM.
	As each convolutional layer can be very large, which need to be splitted into small part to meet the SRAM limitation.
	The splitting strategy is listed below:
		Step 1. Try to place all input/output-activation and filter into SRAM. 
			f not meet SRAM limitation, turn to step 2.
		Step 2. Splitting output-channel of filters (which splits filters and output-activation): 
			In this step, the size of filters and output-activation in SRAM will decrease.
			===== Note that in this step keep MAC array utilization as highest =====
			output-channel affect the utilization of ROWs of MAC array.
			If not meet SRAM limitation, turn to step 3.
		Step 3. Splitting the width and height of input- and output-activation based on output-activation.
			In this step, the size of input- and output-activation in SRAM will decrease.
			===== Note that in this step keep MAC array utilization as highest =====
			width of activation will affect the utilization of COLUMNs of MAC array.
			===== Note that in this step keep memory utilization as least to limitedInActiEffeUtil =====
			Splitting the input-activation will increase the input-activation size, as each part will overlap
			with each other.
			If not meet SRAM limitation, turn to step 4.
		Step 4. Splitting the input-channel.
			In this step, the size of input-activation and filter in SRAM will decrease.
			However, when splitting input-channel, we need to sum over the output-activation, which can only be accelerated
			by ARM Core.
	Note that width/height of filters will not be splitted, as it will seriously increase the computation amount of ARM core,
	because there are a lot of results will be sum over.
	'''
	def __init__(self, layerName, layerParameter, limitedInActiEffeUtil=0.1, limitedSramUtilization=0.9, scale=0.7, 
		sramAvailable=PE_SRAM_LIMIT):
		'''
		Args:
			layerParameter: tuple, 
				e.g. ("input activation DIM"(list), "filter DIM"(list), stride(int), "output activation DIM"(list))
				"input activation DIM"  = [width, height, in_channel]
				"filter DIM" 		    = [width, height, in_channel, out_channel]
				"output activation DIM" = [width, height, out_channel]
			layerName: String for layer name, e.g. "CONV_x", where x is layer number.
		'''
		ConvLayerMapper.__init__(self, layerName, layerParameter, limitedInActiEffeUtil, 
			limitedSramUtilization, scale, sramAvailable)

	# ==================================================================================================
	# 					Step 3: width and height of activation (based on output-activation)
	# ==================================================================================================
	def findOutputActivationSplitScheme(self, prePeAllocationInfo, outChannelSplittingResult, sufficientPeFlag):
		'''
		This function is only need for number_of_filter_on_each_pe <= 4.
		Splitting width and height of activation will only decrease the size of input/output-activation on each PE,
			while size of filters stays unchanged.
		'''
		# Extract layer infomation and pre-allocation information
		inActivationDim, filterDim, stride, outActivationDim = self.layerParameter
		inChannels = inActivationDim[2]
		filterWidth = filterDim[0]
		filterHeight = filterDim[1]
		outActivationHeight = outActivationDim[1]
		preInActivationSizeAlign = prePeAllocationInfo[INACTIALIG]
		preFilterSizeAlign = prePeAllocationInfo[WEIGHTALIG]
		preOutActivationSizeAlign = prePeAllocationInfo[OUTACTIALIG]
		# Get output channel splitting infomation
		filtersOnPe = outChannelSplittingResult[0]
		assert(filtersOnPe <= MAC_ARRAY_ROW), "filtersOnPe-[{}] should not larger than 4".format(filtersOnPe)
		numOfPes = outChannelSplittingResult[1]
		assert(prePeAllocationInfo[PEs] == numOfPes), \
			"prePeAllocationInfo[PEs]-[{}] should equals outChannelSplittingResult[1]-[{}]".\
			format(prePeAllocationInfo[PEs], outChannelSplittingResult[1])		
		# Split the output activation along the width and height
		# Obtain width splitting infomation
		macUtilizateAlongColumn, partsByWidth, partsByHeight, widthSplitInfo, meetFlag = \
			self.convOutActivationWidthHeightSplitting(prePeAllocationInfo, outChannelSplittingResult)
		if self.inActivationUtilization == 0 or meetFlag == False:
			macUtilizateAlongColumn, partsByWidth, partsByHeight, widthSplitInfo, meetFlag = \
				self.convOutActivationWidthHeightSplitting(prePeAllocationInfo, outChannelSplittingResult, relaxRestrict=True)		
		assert(macUtilizateAlongColumn == self.maxMacUtilizationAlongColumn), \
			"{} with size {}: macUtilizateAlongColumn-[{}] should equal maxMacUtilizationAlongColumn.".\
			format(self.layerName, self.sramAvailable, macUtilizateAlongColumn)
		if sufficientPeFlag:
			BasicOperation.customPrintF("sufficientSublayerSizes: partsByWidth {}".format(partsByWidth))
			BasicOperation.customPrintF("sufficientSublayerSizes: partsByHeight {}".format(partsByHeight))
		else:
			BasicOperation.customPrintF("insufficientSublayerSizes: partsByWidth {}".format(partsByWidth))
			BasicOperation.customPrintF("insufficientSublayerSizes: partsByHeight {}".format(partsByHeight))
		BasicOperation.customPrintF("------<<"*10+"\n")
		# Obtain height splitting infomation
		# heightSplitInfo = BasicOperation.oneDimSplitting(outActivationHeight, partsByHeight)
		heightSplitInfo = self.splittingWithPoolStride(outActivationHeight, partsByHeight)
		return widthSplitInfo, heightSplitInfo

	def convOutActivationWidthHeightSplitting(self, prePeAllocationInfo, outChannelSplittingResult, relaxRestrict=False):
		# Extract layer infomation and pre-allocation information
		inActivationDim, filterDim, stride, outActivationDim = self.layerParameter
		inActivationWidth, inActivationHeight = inActivationDim[0], inActivationDim[1]
		filterWidth, filterHeight, inChannels, outChannels = filterDim[0], filterDim[1], filterDim[2], filterDim[3]
		outActivationWidth, outActivationHeight = outActivationDim[0], outActivationDim[1]
		assert(inActivationDim[2]==inChannels), "inChannel unmatch"
		assert(outActivationDim[2]==outChannels), "outChannel unmatch"
		preInActivationSizeAlign = prePeAllocationInfo[INACTIALIG]
		preFilterSizeAlign = prePeAllocationInfo[WEIGHTALIG]
		preOutActivationSizeAlign = prePeAllocationInfo[OUTACTIALIG]
		numOfFilter = outChannelSplittingResult[0]
		overlapLen = filterWidth - stride
		# Get minimum splitted parts
		minSplittedParts = math.ceil((preOutActivationSizeAlign + preInActivationSizeAlign) / (self.sramAvailable - preFilterSizeAlign))
		minSplittedPartsCalib = math.ceil(minSplittedParts / self.limitedInActiEffeUtil)
		BasicOperation.customPrintF("minSplittedParts: {}".format(minSplittedParts))
		BasicOperation.customPrintF("minSplittedPartsCalib: {}".format(minSplittedPartsCalib))
		# Get input-activation size with alignment before splitting
		inActivationSizeAlignWithoutSplitting = ConvLayerMapper.convInActivationSizeAlign(inActivationDim[0], 
																							inActivationDim[1],
																							inActivationDim[2],
																							filterDim[0])
		outActivationSizeAlignWithoutSplitting = ConvLayerMapper.convOutActivationSizeAlign(outActivationWidth,
																							outActivationHeight,
																							outChannels)
		# Obtain the optimum partsByWidth and partsByHeight
		macUtilizateAlongColumn = 0
		partsByWidth = 0
		partsByHeight = 0
		widthSplitInfo = None
		meetFlag = False
		inActiUtil = 0
		inWidthUtil = 0
		maxPartsByWidth = math.ceil(outActivationWidth / (MAC_ARRAY_COLUMN*self.scale)) + 1
		inActiIncreasedSize = inActivationSizeAlignWithoutSplitting
		increasedSize = inActivationSizeAlignWithoutSplitting + outActivationSizeAlignWithoutSplitting
		for partsByWidthTemp in range(maxPartsByWidth-1, 0, -1):
		# for partsByWidthTemp in range(1, maxPartsByWidth):
			# maxPartsByHeight = math.ceil(minSplittedPartsCalib / partsByWidthTemp) + 1
			for partsByHeightTemp in range(1, outActivationHeight+1):
				# Obtain output-activation width/height splitting information
				# outActivationWidthSplitInfoTemp = BasicOperation.oneDimSplitting(outActivationWidth, partsByWidthTemp)
				outActivationWidthSplitInfoTemp = self.splittingWithPoolStride(outActivationWidth, partsByWidthTemp)
				if outActivationWidthSplitInfoTemp == None:
					continue
				widthOfLargeWidthTemp = outActivationWidthSplitInfoTemp[0]
				partsOfLargeWidthTemp = outActivationWidthSplitInfoTemp[1]
				widthOfSmallWidthTemp = outActivationWidthSplitInfoTemp[2]
				partsOfSmallWidthTemp = outActivationWidthSplitInfoTemp[3]
				# outActivationHeightSplitInfoTemp = BasicOperation.oneDimSplitting(outActivationHeight, partsByHeightTemp)
				outActivationHeightSplitInfoTemp = self.splittingWithPoolStride(outActivationHeight, partsByHeightTemp)
				if outActivationHeightSplitInfoTemp == None:
					continue
				heightOfLargeHeightTemp = outActivationHeightSplitInfoTemp[0]
				partsOfLargeHeightTemp = outActivationHeightSplitInfoTemp[1]
				heightOfSmallHeightTemp = outActivationHeightSplitInfoTemp[2]
				partsOfSmallHeightTemp = outActivationHeightSplitInfoTemp[3]
				# Avoid this case: VGG-CONV_16
				# Old Scheme: 1 x 7 - inputHeight:[4,7,0,0]; New Scheme: 1 x 3 - inputHeight:[8,1,6,2]
				# Because the 4 PEs in new scheme are run in different speed, for data reuse, the faster PE need to wait
				# 	for slower PE, which makes the actual acceleration is slower than old scheme.
				# How to avoid: avoiding different blocks/task in 4 PEs
				# if (partsOfLargeHeight + partsOfSmallHeight) == 1 and self.layerName == "CONV_16":
				# 	BasicOperation.customPrintT("{}-{}-{}-{}".format(partsOfLargeHeightTemp, partsOfSmallHeightTemp, partsOfLargeWidthTemp, partsOfSmallWidthTemp))
				hasDiffFlag =  self.hasDifferentTasksInQpe(partsOfLargeHeightTemp, partsOfSmallHeightTemp, \
												partsOfLargeWidthTemp, partsOfSmallWidthTemp)
				if hasDiffFlag:
					continue
				# Judge if output-activation splitting scheme fit for pooling layer
				if self.poolStrideNotMeet(widthOfLargeWidthTemp, relaxRestrict):
					continue
				if self.poolStrideNotMeet(widthOfSmallWidthTemp, relaxRestrict):
					continue
				if self.poolStrideNotMeet(heightOfLargeHeightTemp, relaxRestrict):
					continue
				if self.poolStrideNotMeet(heightOfSmallHeightTemp, relaxRestrict):
					continue
				# Obtain MAC array utilication rate and do judgement
				macUtilizateAlongColumnTemp = ConvLayerMapper.getMacUtilAlongWidth(widthOfLargeWidthTemp) * \
												partsOfLargeWidthTemp + \
												ConvLayerMapper.getMacUtilAlongOutChannel(widthOfSmallWidthTemp) * \
												partsOfSmallWidthTemp
				if macUtilizateAlongColumnTemp != self.maxMacUtilizationAlongColumn:
					continue
				# Obtain memory utilization rate and do judgement
				# Obtain width alignment utilization
				# input-activation overlap Size
				inWidthOfLargeWidthPartAlign = ConvLayerMapper.convInActivationWidthAlign( \
					ConvLayerMapper.outToIn(widthOfLargeWidthTemp, stride, filterWidth), filterWidth)
				inWidthOfSmallWidthPartAlign = ConvLayerMapper.convInActivationWidthAlign( \
					ConvLayerMapper.outToIn(widthOfSmallWidthTemp, stride, filterWidth), filterWidth)
				inWidthOverlap = inWidthOfLargeWidthPartAlign * partsOfLargeWidthTemp + \
								inWidthOfSmallWidthPartAlign * partsOfSmallWidthTemp		
				inHeightOverlap = ConvLayerMapper.totalHeightInActivationOverlap(outActivationHeight = outActivationHeight, 
																partsByHeightOutActivation = partsByHeightTemp, 
																filterHeight = filterHeight, 
																stride = stride)
				inActivationSizeAlignSplitting = inWidthOverlap * inHeightOverlap * inChannels
				inActiUtilTemp = inActivationSizeAlignWithoutSplitting / inActivationSizeAlignSplitting
				if inActiUtilTemp < self.limitedInActiEffeUtil:
					continue
				# inWidthUtilTemp: Deprecated after change convInActivationWidthAlign() 2019.1.14
				inWidthUtilTemp = ConvLayerMapper.convInActivationWidthAlign(inActivationDim[0], filterDim[0]) / inWidthOverlap 
				inActiIncreasedSizeTemp = inActivationSizeAlignSplitting - inActivationSizeAlignWithoutSplitting
				# output-activation increased size caused by splitting
				outActiWidthInc = ConvLayerMapper.convOutActivationWidthAlign(widthOfLargeWidthTemp) * partsOfLargeWidthTemp + \
					ConvLayerMapper.convOutActivationWidthAlign(widthOfSmallWidthTemp) * partsOfSmallWidthTemp
				outActiHeightInc = BasicOperation.splitInfoIntegration(outActivationHeightSplitInfoTemp)
				outActiIncreasedSizeTemp = outActiWidthInc * outActiHeightInc * outChannels - outActivationSizeAlignWithoutSplitting
				# For fc layer, wenn splitting the weights, it need to determine if the largest part size > sramAvailable, 
				# however, it is not necessary for convolutional layer
				# Because when splitting the output-activation, we cannot guarantee that we can achieve <= sramAvailable.
				# If achieving <= sramAvailable, "meetFlag" will set to be true, at the same time, we put effort to find
				# partsByWidth * partsByHeight as small as possible. When "meetFlag" is false, we are going to split 
				# output-activation more parts.
				inActivationLargePartWidth = ConvLayerMapper.outToIn(widthOfLargeWidthTemp, stride, filterWidth)
				inActivationLargePartHeight = ConvLayerMapper.outToIn(heightOfLargeHeightTemp, stride, filterHeight)
				inActivationLargePartSizeAlign =  ConvLayerMapper.convInActivationSizeAlign(inActivationLargePartWidth, 
																							inActivationLargePartHeight,
																							inChannels,
																							filterWidth)
				# As MLA only support stride = 1, when calculating the size, need to seem stride always be 1.
				# outActivationLargePartSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(widthOfLargeWidthTemp,
				# 																				heightOfLargeHeightTemp,
				# 																				numOfFilter)
				outActiWidthS1 = ConvLayerMapper.convInToOut(inActivationLargePartWidth, filterWidth)
				outActiHeightS1 = ConvLayerMapper.convInToOut(inActivationLargePartHeight, filterHeight)
				outActivationLargePartSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(outActiWidthS1,
																								outActiHeightS1,
																								numOfFilter)	
				largestPartSize = inActivationLargePartSizeAlign + preFilterSizeAlign + outActivationLargePartSizeAlign			
				if outActivationLargePartSizeAlign >= self.sramAvailable:
					continue
				if largestPartSize <= self.sramAvailable:
					if meetFlag == False:
						partsByWidth = maxPartsByWidth
						partsByHeight = outActivationHeight
						inActiIncreasedSize = inActivationSizeAlignWithoutSplitting
						increasedSize = inActivationSizeAlignWithoutSplitting + outActivationSizeAlignWithoutSplitting
					meetFlag = True
				else:
					if meetFlag:
						continue
				# According to meetFlag, two different scheme are used.
				increasedSizeTemp = inActiIncreasedSizeTemp+outActiIncreasedSizeTemp
				# if self.layerName == "CONV_12" and partsByWidthTemp==2:
				# 	BasicOperation.customPrintT("-----poolStride: {}".format(self.poolStride))
				# 	BasicOperation.customPrintT("meetFlag: {}".format(meetFlag))
				# 	BasicOperation.customPrintT("largestPartSize: {}".format(largestPartSize))
				# 	BasicOperation.customPrintT("sramAvailable: {}".format(self.sramAvailable))
				# 	BasicOperation.customPrintT("parts: {}, {}:{}".format(partsByWidthTemp*partsByHeightTemp, partsByWidthTemp, partsByHeightTemp))
				# 	BasicOperation.customPrintT("outActivationSizeAlignWithoutSplitting: {}".format(outActivationSizeAlignWithoutSplitting))
				# 	BasicOperation.customPrintT("{} = {} + {}".format(inActiIncreasedSizeTemp+outActiIncreasedSizeTemp, inActiIncreasedSizeTemp, outActiIncreasedSizeTemp))
				# 	BasicOperation.customPrintT("increasedSize: {}".format(increasedSize))
				# 	BasicOperation.customPause()	
				if meetFlag:
					# Find as high input-activation width utilization as possbile
					# Find as few part as possible
					# if inActiIncreasedSizeTemp < inActiIncreasedSize:
					if increasedSizeTemp < increasedSize:
						macUtilizateAlongColumn = macUtilizateAlongColumnTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = outActivationWidthSplitInfoTemp
						inActiUtil = inActiUtilTemp
						inWidthUtil = inWidthUtilTemp
						inActiIncreasedSize = inActiIncreasedSizeTemp
						increasedSize = increasedSizeTemp
						# if self.layerName == "CONV_12" and meetFlag:
						# 	BasicOperation.customPrintT("=====poolStride: {}".format(self.poolStride))
						# 	BasicOperation.customPrintT("parts: {}, {}:{}".format(partsByWidthTemp*partsByHeightTemp, partsByWidthTemp, partsByHeightTemp))
						# 	BasicOperation.customPrintT("outActivationSizeAlignWithoutSplitting: {}".format(outActivationSizeAlignWithoutSplitting))
						# 	BasicOperation.customPrintT("{} = {} + {}".format(inActiIncreasedSizeTemp+outActiIncreasedSizeTemp, inActiIncreasedSizeTemp, outActiIncreasedSizeTemp))
						# 	BasicOperation.customPrintT("increasedSize: {}".format(increasedSize))
						# 	BasicOperation.customPause()					
					# elif inActiIncreasedSizeTemp == inActiIncreasedSize:
					elif increasedSizeTemp == increasedSize:
						if partsByWidthTemp * partsByHeightTemp <= partsByWidth * partsByHeight:
							macUtilizateAlongColumn = macUtilizateAlongColumnTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = outActivationWidthSplitInfoTemp
							inActiUtil = inActiUtilTemp
							inWidthUtil = inWidthUtilTemp
							inActiIncreasedSize = inActiIncreasedSizeTemp
							increasedSize = increasedSizeTemp
							# if self.layerName == "CONV_12" and meetFlag:
							# 			BasicOperation.customPrintT("poolStride: {}".format(self.poolStride))
							# 			BasicOperation.customPrintT("parts: {}, {}:{}".format(partsByWidthTemp*partsByHeightTemp, partsByWidthTemp, partsByHeightTemp))
							# 			BasicOperation.customPrintT("outActivationSizeAlignWithoutSplitting: {}".format(outActivationSizeAlignWithoutSplitting))
							# 			BasicOperation.customPrintT("{} = {} + {}".format(inActiIncreasedSizeTemp+outActiIncreasedSizeTemp, inActiIncreasedSizeTemp, outActiIncreasedSizeTemp))
							# 			BasicOperation.customPrintT("increasedSize: {}".format(increasedSize))
							# 			BasicOperation.customPause()	
				else:
					# limitation of each PE's SRAM utilization
					# This part is same as part of self.inChannelSplittingMap()
					ratio = (inActivationLargePartSizeAlign + preFilterSizeAlign) / \
							(self.sramAvailable - outActivationLargePartSizeAlign)
					if ratio > inChannels:
						continue
					# This part is abanded after using dynamic sram allocation
					# maxInChannelsAfterSplitting = math.floor(inChannels / ratio)
					# partsSplittingInChannel = math.ceil(inChannels / maxInChannelsAfterSplitting) 
					# inChannelSplitInfo = BasicOperation.oneDimSplitting(inChannels, partsSplittingInChannel)
					# largestPartSize = (inActivationLargePartSizeAlign + preFilterSizeAlign) / inChannels * \
					# 	inChannelSplitInfo[0] + outActivationLargePartSizeAlign
					# if (largestPartSize/self.sramAvailable) < self.limitedSramUtilization:
					# 	continue
					# Update split information
					# if inActiIncreasedSizeTemp < inActiIncreasedSize:
					if increasedSizeTemp < increasedSize:
						macUtilizateAlongColumn = macUtilizateAlongColumnTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = outActivationWidthSplitInfoTemp
						inActiUtil = inActiUtilTemp
						inWidthUtil = inWidthUtilTemp
						inActiIncreasedSize = inActiIncreasedSizeTemp
						increasedSize = increasedSizeTemp
					# elif inActiIncreasedSizeTemp == inActiIncreasedSize:
					elif increasedSizeTemp == increasedSize:
						if partsByWidthTemp * partsByHeightTemp >= partsByWidth * partsByHeight:
							macUtilizateAlongColumn = macUtilizateAlongColumnTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = outActivationWidthSplitInfoTemp
							inActiUtil = inActiUtilTemp
							inWidthUtil = inWidthUtilTemp
							inActiIncreasedSize = inActiIncreasedSizeTemp
							increasedSize = increasedSizeTemp
		# assert(inActiUtil != 0), "inActiUtil should not be 0! for {}".format(self.layerName)
		if inActiUtil == 0:
			BasicOperation.customPrintT("--->---"*8)
			BasicOperation.customPrintT("inActiUtil should not be 0! for {}".format(self.layerName))
			BasicOperation.customPrintT("---<---"*8)
		self.inActivationUtilization = inActiUtil
		return macUtilizateAlongColumn, partsByWidth, partsByHeight, widthSplitInfo, meetFlag

	def splittingWithPoolStride(self, length, parts):
		assert(length%self.poolStride==0), "Unequally division"
		scaleLength = math.floor(length / self.poolStride)
		if scaleLength < parts:
			return None
		splitInfo = BasicOperation.oneDimSplitting(scaleLength, parts)
		splitInfo[0] = splitInfo[0] * self.poolStride
		splitInfo[2] = splitInfo[2] * self.poolStride
		return splitInfo

	def hasDifferentTasksInQpe(self, partsOfLargeHeight, partsOfSmallHeight, partsOfLargeWidth, partsOfSmallWidth):
		taskOneDiff = partsOfLargeHeight * partsOfLargeWidth % PES_ON_QPE != 0
		taskTwoDiff = partsOfLargeHeight * partsOfSmallWidth % PES_ON_QPE != 0
		taskTwoEmpty = partsOfLargeHeight * partsOfSmallWidth == 0
		taskThreeDiff = partsOfSmallHeight * partsOfLargeWidth % PES_ON_QPE != 0
		taskThreeEmpty = partsOfSmallHeight * partsOfLargeWidth == 0
		taskFourDiff = partsOfSmallHeight * partsOfSmallWidth % PES_ON_QPE != 0
		taskFourEmpty = partsOfSmallHeight * partsOfSmallWidth == 0
		# if (partsOfLargeHeight + partsOfSmallHeight) == 1 and self.layerName == "CONV_16":
		# 	BasicOperation.customPrintT("{}-{}-{}-{}".format(partsOfLargeHeight, partsOfSmallHeight, partsOfLargeWidth, partsOfSmallWidth))
		# 	BasicOperation.customPrintT("taskOneDiff: {}".format(taskOneDiff))
		# 	BasicOperation.customPrintT("taskTwoDiff: {}".format(taskTwoDiff))
		# 	BasicOperation.customPrintT("taskThreeDiff: {}".format(taskThreeDiff))
		# 	BasicOperation.customPrintT("taskFourDiff: {}".format(taskFourDiff))
		# 	BasicOperation.customPrintT("taskTwoEmpty: {}".format(taskTwoEmpty))
		# 	BasicOperation.customPrintT("taskThreeEmpty: {}".format(taskThreeEmpty))
		# 	BasicOperation.customPrintT("taskFourEmpty: {}".format(taskFourEmpty))
		# 	BasicOperation.customPause()
		if taskOneDiff:
			if taskTwoEmpty and taskThreeEmpty and taskFourEmpty:
				return False
			else:
				return True
		elif taskTwoDiff:
			if taskThreeEmpty and taskFourEmpty:
				return False
			else:
				return True
		elif taskThreeDiff:
			if taskFourEmpty:
				return False
			else:
				return True
		else:
			return False