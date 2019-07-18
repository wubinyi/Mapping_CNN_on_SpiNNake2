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


class ConvLayerMapper():
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
		self.mapperReset(layerName, layerParameter, limitedInActiEffeUtil, scale, limitedSramUtilization, sramAvailable)

	def mapperReset(self, layerName, layerParameter, limitedInActiEffeUtil, scale, limitedSramUtilization, sramAvailable):
		self.layerName = layerName
		self.layerNumberStr = layerName[len("CONV_"):]
		self.layerParameter, poolDimStride = layerParameter
		# This part is seems strange because resnet is different from VGG
		if isinstance(poolDimStride, tuple):
			self.poolDim, self.poolStride = poolDimStride
		else:
			self.poolDim = [poolDimStride, poolDimStride]
			self.poolStride = poolDimStride
		inActivationDim, filterDim, stride, outActivationDim = self.layerParameter
		assert(inActivationDim[2] == filterDim[2]), \
			"Input channels {}-{} should be equal.".format(inActivationDim[2], filterDim[2])
		assert(outActivationDim[2] == filterDim[3]), \
			"Output channels {}-{} should be equal.".format(inActivationDim[2], filterDim[2])
		self.layerMapInfo = {}
		self.maxMacUtilizationAlongRow = 0
		self.maxMacUtilizationAlongColumn = 0
		self.limitedInActiEffeUtil = limitedInActiEffeUtil
		self.scale = scale
		self.convSplitInfo = {}
		self.theoreticalComputationClock = [0] * 16
		self.actualComputationClock = [0] * 16
		self.convExtraLayerInfo = None
		self.limitedSramUtilization = limitedSramUtilization
		self.inActivationUtilization = 1.0
		self.sramAvailable = sramAvailable

	def getSplitInfo(self):
		return self.convSplitInfo

	def getClocks(self):
		return max(self.actualComputationClock)

	def getExtraLayer(self):
		return self.convExtraLayerInfo

	def isMeet(self):
		if CONV_IN_CHANNEL not in self.convSplitInfo:
			return False
		else:
			return not isinstance(self.convSplitInfo[CONV_IN_CHANNEL], list)

	# ==================================================================================================
	# 										Step 1: No Splitting
	# ==================================================================================================
	def map(self):
		'''
		Step 1: Try to place all input/output-activation and filter into SRAM.
		'''
		# Extract layer parameter
		inActivationDim, filterDim, stride, outActivationDim = self.layerParameter
		# Get convolutional layer split info and update computation clock
		self.convSplitInfo[CONV_IN_WIDTH] = inActivationDim[0]
		self.convSplitInfo[CONV_IN_HEIGHT] = inActivationDim[1]
		self.convSplitInfo[CONV_IN_CHANNEL] = inActivationDim[2]
		self.convSplitInfo[CONV_FILTER_WIDTH] = filterDim[0]
		self.convSplitInfo[CONV_FILTER_HEIGHT] = filterDim[1]
		self.convSplitInfo[CONV_FILTER_STRIDE] = stride
		self.convSplitInfo[CONV_OUT_WIDTH] = outActivationDim[0]
		self.convSplitInfo[CONV_OUT_HEIGHT] = outActivationDim[1]
		self.convSplitInfo[CONV_OUT_CHANNEL] = outActivationDim[2]
		self.clockComputation()
		# Get maximum MAC array utilication
		self.maxMacUtilizationAlongRow = ConvLayerMapper.getMacUtilAlongOutChannel(filterDim[3])
		self.maxMacUtilizationAlongColumn = ConvLayerMapper.getMacUtilAlongWidth(outActivationDim[0])
		# Get input-activation size
		inActivationSize = BasicOperation.listInnerProduct(inActivationDim)
		inActivationSizeAlign = ConvLayerMapper.convInActivationSizeAlign(inActivationDim[0], 
																			inActivationDim[1], 
																			inActivationDim[2], 
																			filterDim[0])
		# Get filter size
		filterSize = BasicOperation.listInnerProduct(filterDim)
		filterSizeAlign = ConvLayerMapper.convFilterSizeAlign(filterDim[0],
																filterDim[1],
																filterDim[2],
																filterDim[3])
		# Get output-activation size
		outActivationSize = BasicOperation.listInnerProduct(outActivationDim) * MAC_OUT_BYTES
		outActivationSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(outActivationDim[0], 
																				outActivationDim[1], 
																				outActivationDim[2])
		# merge PE allocation infomation
		peAllocationInfo = {}
		peAllocationInfo[PEs] = 1
		peAllocationInfo[INACTI] = inActivationSize
		peAllocationInfo[INACTIALIG] = inActivationSizeAlign
		peAllocationInfo[WEIGHT] = filterSize
		peAllocationInfo[WEIGHTALIG] = filterSizeAlign
		peAllocationInfo[OUTACTI] = outActivationSize
		peAllocationInfo[OUTACTIALIG] = outActivationSizeAlign
		# SRAM size limitation
		if inActivationSizeAlign + filterSizeAlign + outActivationSizeAlign > self.sramAvailable:
			self.insertConvLayerMap(self.layerName, peAllocationInfo, clockIndex=0, invalid=True)
			self.outputChannelSplittingMap(peAllocationInfo)
		else:
			self.insertConvLayerMap(self.layerName, peAllocationInfo, clockIndex=0)	
		# Get extra layer(needed when splitting input-channel)
		self.getConvExtraLayer()

		return self.layerMapInfo

	# ==================================================================================================
	# 									Step 2: Output Channels
	# ==================================================================================================
	def outputChannelSplittingMap(self, prePeAllocationInfo):
		'''
		Step 2. Splitting output-channel of filters
		'''
		numOfSplittedParts, meetFlag = self.findOutputChannelSplitScheme(prePeAllocationInfo)
		self.outputChannelSpliter(prePeAllocationInfo, numOfSplittedParts, meetFlag)

	def findOutputChannelSplitScheme(self, prePeAllocationInfo):
		'''
		Looking for a split scheme, how many parts will filters be splited.
		'''
		# Extract layer infomation and pre-allocation information
		_, filterDim, _, outActivationDim = self.layerParameter
		numOfFilter = filterDim[3]
		BasicOperation.customPrintF(">>------"*10)
		BasicOperation.customPrintF("layerName: {}".format(self.layerName))
		BasicOperation.customPrintF("outActiWidth: {}, outActiHeight: {}".format(outActivationDim[0], outActivationDim[1]))
		BasicOperation.customPrintF("Total filters: {}".format(numOfFilter))
		outActicationWidth = outActivationDim[0]
		outActivationHeight = outActivationDim[1]
		preInActivationSizeAlign = prePeAllocationInfo[INACTIALIG]
		# maxSplittedParts indicate that filters are splitted into the most parts 
		#	while keeping the MAC array utilization highest.
		# However, when numOfSplittedParts is smaller than maxSplittedParts, 
		# 	it cannot guarantee that it still achieves the highest MAC array utilization.
		# Therefore, we need to check the MAC array utilication when splitting filters.
		maxSplittedParts = math.ceil(numOfFilter/MAC_ARRAY_ROW)
		numOfSplittedParts = maxSplittedParts
		meetFlag = False
		# Hier must be range(1, maxSplittedParts+1) not range(maxSplittedParts, 0, -1)
		# Because for out_channel=21, we prefer[4,4,4,4,1] rather than [5,4,4,4]
		# Coorperate with the second assertion in self.outputChannelSpliter()
		for splittingPartsTemp in range(1, maxSplittedParts+1):
			# Determine if MAC array utilization is meet
			outChannelSplittingResult = BasicOperation.oneDimSplitting(numOfFilter, splittingPartsTemp)
			macUtilicationAlongRowTemp = outChannelSplittingResult[1] * \
											ConvLayerMapper.getMacUtilAlongOutChannel(outChannelSplittingResult[0]) + \
											outChannelSplittingResult[3] * \
											ConvLayerMapper.getMacUtilAlongOutChannel(outChannelSplittingResult[2])
			if macUtilicationAlongRowTemp == self.maxMacUtilizationAlongRow:
				# Determine if the largest part meet the SRAM limitation
				numOfFilterOnLargePart = outChannelSplittingResult[0]
				filterSizeAlign = ConvLayerMapper.convFilterSizeAlign(filterDim[0], 
																		filterDim[1], 
																		filterDim[2], 
																		numOfFilterOnLargePart)
				outActivationAlign = ConvLayerMapper.convOutActivationSizeAlign(outActicationWidth,
																				outActivationHeight,
																				numOfFilterOnLargePart)				
				largePartSizeAlign = preInActivationSizeAlign + filterSizeAlign + outActivationAlign
				if largePartSizeAlign <= self.sramAvailable:	
					numOfSplittedParts = splittingPartsTemp
					meetFlag = True
					break
		return numOfSplittedParts, meetFlag

	def outputChannelSpliter(self, prePeAllocationInfo, numOfSplittedParts, meetFlag):
		'''
		Splitting output channels will only decrease the size of filters and output-activation on each PE,
			while size of input-activation stays unchanged.
		'''
		# Extract layer infomation and pre-allocation information
		_, filterDim, _, outActivationDim = self.layerParameter
		numOfFilter = filterDim[3]
		sizeOfEachFilter = BasicOperation.listInnerProduct(filterDim[:3])
		outActicationWidth = outActivationDim[0]
		outActivationHeight = outActivationDim[1]
		preInActivationSize = prePeAllocationInfo[INACTI]
		preInActivationSizeAlign = prePeAllocationInfo[INACTIALIG]
		# Get splitting result
		# As filters may cannot be equally splitted, therefore the some PEs may contain less filters.
		# These PEs will be called insufficient-PEs, while the others call sufficient-PEs.
		# Sufficient-PE contains one more filter than insufficient-PE.
		outChannelSplittingResult = BasicOperation.oneDimSplitting(numOfFilter, numOfSplittedParts)
		self.convSplitInfo[CONV_OUT_CHANNEL] = outChannelSplittingResult
		self.clockComputation()
		filtersOnSuffPe = outChannelSplittingResult[0]
		numOfSuffPe = outChannelSplittingResult[1]
		filtersOnInsuPe = outChannelSplittingResult[2]
		numOfInsuPe = outChannelSplittingResult[3]		
		# Get sufficient-PEs information
			# Get filter size of each sufficient-PEs
		filterSizeOnSuffPe = sizeOfEachFilter * filtersOnSuffPe
		filterSizeOnSuffPeAlign = ConvLayerMapper.convFilterSizeAlign(filterDim[0],
																		filterDim[1],
																		filterDim[2],
																		filtersOnSuffPe)
			# Get output activation size of of each sufficient-PEs
		outActivationSizeOnSuffPe = outActicationWidth * outActivationHeight * filtersOnSuffPe * MAC_OUT_BYTES
		outActivationSizeOnSuffPeAlign = ConvLayerMapper.convOutActivationSizeAlign(outActicationWidth,
																					outActivationHeight,
																					filtersOnSuffPe)
			# merge sufficient PE allocation infomation
		suffPeAllocationInfo = {}
		suffPeAllocationInfo[PEs] = numOfSuffPe
		suffPeAllocationInfo[INACTI] = preInActivationSize
		suffPeAllocationInfo[INACTIALIG] = preInActivationSizeAlign
		suffPeAllocationInfo[WEIGHT] = filterSizeOnSuffPe
		suffPeAllocationInfo[WEIGHTALIG] = filterSizeOnSuffPeAlign
		suffPeAllocationInfo[OUTACTI] = outActivationSizeOnSuffPe
		suffPeAllocationInfo[OUTACTIALIG] = outActivationSizeOnSuffPeAlign
		BasicOperation.customPrintF("Sufficient PE allocation: {}: {}".format(PEs, numOfSuffPe))
			# Assertion for no SRAM limitation when meetFlag=True.
			# This assertion cannot be removed.
		if meetFlag:
			assert(preInActivationSizeAlign + filterSizeOnSuffPeAlign + outActivationSizeOnSuffPeAlign <= self.sramAvailable), \
					"There should be no memory limitation problem."
		# Getting insufficient-PEs information 
		insuPeAllocationInfo = None 
		if numOfInsuPe != 0:
			# Get filter size of each insufficient-PEs
			filterSizeOnInsuPe = sizeOfEachFilter * filtersOnInsuPe
			filterSizeOnInsuPeAlign = ConvLayerMapper.convFilterSizeAlign(filterDim[0],
																		filterDim[1],
																		filterDim[2],
																		filtersOnInsuPe)
			# This assertion cannot be removed. Cooperate with method self.findSplitScheme()
			assert(filterSizeOnSuffPeAlign == filterSizeOnInsuPeAlign), \
				"filterSizeOnSuffPeAlign[{}] should be equal to filterSizeOnInsuPeAlign[{}]".\
				format(filterSizeOnSuffPeAlign, filterSizeOnInsuPeAlign)
			# Get output activation size of of each insufficient-PEs
			outActivationSizeOnInsuPe = outActicationWidth * outActivationHeight * filtersOnInsuPe * MAC_OUT_BYTES
			outActivationSizeOnInsuPeAlign = ConvLayerMapper.convOutActivationSizeAlign(outActicationWidth,
																						outActivationHeight,
																						filtersOnInsuPe)
			# merge insufficient PE allocation infomation
			insuPeAllocationInfo = {}
			insuPeAllocationInfo[PEs] = numOfInsuPe
			insuPeAllocationInfo[INACTI] = preInActivationSize
			insuPeAllocationInfo[INACTIALIG] = preInActivationSizeAlign
			insuPeAllocationInfo[WEIGHT] = filterSizeOnInsuPe
			insuPeAllocationInfo[WEIGHTALIG] = filterSizeOnInsuPeAlign
			insuPeAllocationInfo[OUTACTI] = outActivationSizeOnInsuPe
			insuPeAllocationInfo[OUTACTIALIG] = outActivationSizeOnInsuPeAlign
			BasicOperation.customPrintF("Insufficient PE allocation: {}: {}".format(PEs, numOfInsuPe))
		# Determine if further splitting is needed 
		if not meetFlag:
			# sufficient-PEs allocation information
			self.insertConvLayerMap("CONVSU_"+self.layerNumberStr, suffPeAllocationInfo, clockIndex=0, invalid=True)
			self.inoutputActivationSplittingMap("CONVSU", suffPeAllocationInfo, outChannelSplittingResult[0:2], clockIndex=0)
			# insufficient PE allocation infomation
			if insuPeAllocationInfo != None:
				self.insertConvLayerMap("CONVIS_"+self.layerNumberStr, insuPeAllocationInfo, clockIndex=8, invalid=True)
				self.inoutputActivationSplittingMap("CONVIS", insuPeAllocationInfo, outChannelSplittingResult[2:4], clockIndex=8)
		else:
			widthSplitInfo = BasicOperation.oneDimSplitting(outActicationWidth, 1)
			heightSplitInfo = BasicOperation.oneDimSplitting(outActivationHeight, 1)
			widthInSplitInfo, heightInSplitInfo = self.outActivationSplitInfo2InActivation(widthSplitInfo, heightSplitInfo)
			self.convSplitInfo[CONV_OUT_WIDTH] = widthSplitInfo
			self.convSplitInfo[CONV_OUT_HEIGHT] = heightSplitInfo
			self.convSplitInfo[CONV_IN_WIDTH] = widthInSplitInfo
			self.convSplitInfo[CONV_IN_HEIGHT] = heightInSplitInfo
			# sufficient-PEs allocation information
			self.insertConvLayerMap("CONVSU_"+self.layerNumberStr, suffPeAllocationInfo, clockIndex=0)
			# insufficient PE allocation infomation
			if insuPeAllocationInfo != None:
				self.insertConvLayerMap("CONVIS_"+self.layerNumberStr, insuPeAllocationInfo, clockIndex=8)

	# ==================================================================================================
	# 					Step 3: width and height of activation (based on output-activation)
	# ==================================================================================================
	def inoutputActivationSplittingMap(self, namePrefix, prePeAllocationInfo, outChannelSplittingResult, clockIndex):
		'''
		Step 3. Splitting the width and height of input- and output-activation based on output-activation.
		This function is only need for number_of_filter_on_each_pe <= 4.
		Splitting width and height of activation will only decrease the size of input/output-activation on each PE,
			while size of filters stays unchanged.
		'''
		sufficientPeFlag = "CONVSU" in namePrefix
		# Find splitting scheme and extract width/height splitting infomation 
		widthSplitInfo, heightSplitInfo = self.findOutputActivationSplitScheme(prePeAllocationInfo, \
			outChannelSplittingResult, sufficientPeFlag)
		# Extract layer infomation and pre-allocation information
		preNumOfPe = prePeAllocationInfo[PEs]
		preFilterSize = prePeAllocationInfo[WEIGHT]
		preFilterSizeAlign = prePeAllocationInfo[WEIGHTALIG]
		# Get output channel splitting infomation
		filtersOnPe = outChannelSplittingResult[0]
		# 
		widthInSplitInfo, heightInSplitInfo = self.outActivationSplitInfo2InActivation(widthSplitInfo, heightSplitInfo)
		self.convSplitInfo[CONV_OUT_WIDTH] = widthSplitInfo
		self.convSplitInfo[CONV_OUT_HEIGHT] = heightSplitInfo
		self.convSplitInfo[CONV_IN_WIDTH] = widthInSplitInfo
		self.convSplitInfo[CONV_IN_HEIGHT] = heightInSplitInfo
		self.clockComputation()
			# Extract width splitting infomation
		widthOfLargeWidth = widthSplitInfo[0]
		partsOfLargeWidth = widthSplitInfo[1]
		widthOfSmallWidth = widthSplitInfo[2]
		partsOfSmallWidth = widthSplitInfo[3]
			# Extract height splitting infomation
		heightOfLargeHeight = heightSplitInfo[0]
		partsOfLargeHeight = heightSplitInfo[1]
		heightOfSmallHeight = heightSplitInfo[2]
		partsOfSmallHeight = heightSplitInfo[3]
		# Add PE allocation information
		# Temp result for calculating size of output activation 
		outActiTemp = filtersOnPe * MAC_OUT_BYTES
		# create PE allocation dictionary
		peAllocationInfo = {}
		peAllocationInfo[PEs] = 0
		peAllocationInfo[INACTI] = 0
		peAllocationInfo[INACTIALIG] = 0
		peAllocationInfo[WEIGHT] = preFilterSize
		peAllocationInfo[WEIGHTALIG] = preFilterSizeAlign
		peAllocationInfo[OUTACTI] = 0
		peAllocationInfo[OUTACTIALIG] = 0
		# Add CONVxxTL
		splittedInActivationSize, splittedInActivationSizeAlign = \
			self.computeSplittedInActivationSize(widthOfLargeWidth, heightOfLargeHeight)
		splittedOutActivationSize = widthOfLargeWidth * heightOfLargeHeight * outActiTemp
		# As MLA only support stride = 1, when calculating the size, need to seem stride always be 1.
		_, _, convStride, _ = self.layerParameter
		outActiWidthS1 = ConvLayerMapper.convOutToOut(widthOfLargeWidth, convStride)
		outActiHeightS1 = ConvLayerMapper.convOutToOut(heightOfLargeHeight, convStride)
		splittedOutActivationSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(outActiWidthS1, 
																					outActiHeightS1, 
																					filtersOnPe)
		peAllocationInfo[PEs] = preNumOfPe * partsOfLargeWidth * partsOfLargeHeight
		peAllocationInfo[INACTI] = splittedInActivationSize
		peAllocationInfo[INACTIALIG] = splittedInActivationSizeAlign
		peAllocationInfo[OUTACTI] = splittedOutActivationSize
		peAllocationInfo[OUTACTIALIG] = splittedOutActivationSizeAlign
		if splittedInActivationSizeAlign + preFilterSizeAlign + splittedOutActivationSizeAlign <= self.sramAvailable:
			self.insertConvLayerMap(namePrefix+"TL_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex)
		else:
			self.insertConvLayerMap(namePrefix+"TL_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex, invalid=True)
			self.inChannelSplittingMap(namePrefix+"TL", peAllocationInfo, clockIndex=clockIndex)
		# Add CONVxxTR
		if partsOfSmallWidth > 0:
			splittedInActivationSize, splittedInActivationSizeAlign = \
				self.computeSplittedInActivationSize(widthOfSmallWidth, heightOfLargeHeight)
			splittedOutActivationSize = widthOfSmallWidth * heightOfLargeHeight * outActiTemp
			splittedOutActivationSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(widthOfSmallWidth, 
																						heightOfLargeHeight, 
																						filtersOnPe)
			peAllocationInfo[PEs] = preNumOfPe * partsOfSmallWidth * partsOfLargeHeight
			peAllocationInfo[INACTI] = splittedInActivationSize
			peAllocationInfo[INACTIALIG] = splittedInActivationSizeAlign
			peAllocationInfo[OUTACTI] = splittedOutActivationSize
			peAllocationInfo[OUTACTIALIG] = splittedOutActivationSizeAlign
			if splittedInActivationSizeAlign + preFilterSizeAlign + splittedOutActivationSizeAlign <= self.sramAvailable:
				self.insertConvLayerMap(namePrefix+"TR_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex+2)		
			else:
				self.insertConvLayerMap(namePrefix+"TR_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex+2, invalid=True)
				self.inChannelSplittingMap(namePrefix+"TR", peAllocationInfo, clockIndex=clockIndex+2)				
		# Add CONVxxBL
		if partsOfSmallHeight > 0:
			splittedInActivationSize, splittedInActivationSizeAlign = \
				self.computeSplittedInActivationSize(widthOfLargeWidth, heightOfSmallHeight)
			splittedOutActivationSize = widthOfLargeWidth * heightOfSmallHeight * outActiTemp
			splittedOutActivationSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(widthOfLargeWidth, 
																						heightOfSmallHeight, 
																						filtersOnPe)
			peAllocationInfo[PEs] = preNumOfPe * partsOfLargeWidth * partsOfSmallHeight
			peAllocationInfo[INACTI] = splittedInActivationSize
			peAllocationInfo[INACTIALIG] = splittedInActivationSizeAlign
			peAllocationInfo[OUTACTI] = splittedOutActivationSize
			peAllocationInfo[OUTACTIALIG] = splittedOutActivationSizeAlign
			if splittedInActivationSizeAlign + preFilterSizeAlign + splittedOutActivationSizeAlign <= self.sramAvailable:
				self.insertConvLayerMap(namePrefix+"BL_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex+4)	
			else:
				self.insertConvLayerMap(namePrefix+"BL_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex+4, invalid=True)
				self.inChannelSplittingMap(namePrefix+"BL", peAllocationInfo, clockIndex=clockIndex+4)					
		# Add CONVxxBR
		if partsOfSmallWidth > 0 and partsOfSmallHeight > 0:
			splittedInActivationSize, splittedInActivationSizeAlign = \
				self.computeSplittedInActivationSize(widthOfSmallWidth, heightOfSmallHeight)
			splittedOutActivationSize = widthOfSmallWidth * heightOfSmallHeight * outActiTemp
			splittedOutActivationSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(widthOfSmallWidth, 
																						heightOfSmallHeight, 
																						filtersOnPe)
			peAllocationInfo[PEs] = preNumOfPe * partsOfSmallWidth * partsOfSmallHeight
			peAllocationInfo[INACTI] = splittedInActivationSize
			peAllocationInfo[INACTIALIG] = splittedInActivationSizeAlign
			peAllocationInfo[OUTACTI] = splittedOutActivationSize
			peAllocationInfo[OUTACTIALIG] = splittedOutActivationSizeAlign
			if splittedInActivationSizeAlign + preFilterSizeAlign + splittedOutActivationSizeAlign <= self.sramAvailable:
				self.insertConvLayerMap(namePrefix+"BR_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex+6)	
			else:
				self.insertConvLayerMap(namePrefix+"BR_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex+6, invalid=True)
				self.inChannelSplittingMap(namePrefix+"BR", peAllocationInfo, clockIndex=clockIndex+6)	

	def outActivationSplitInfo2InActivation(self, widthOfOutActivationSplitInfo, heightOfOutActivationSplitInfo):
		_, filterDim, stride, _ = self.layerParameter
		filterWidth = filterDim[0]
		filterHeight = filterDim[1]
		widthOfInActivationSplitInfo = widthOfOutActivationSplitInfo.copy()
		widthOfInActivationSplitInfo[0] = ConvLayerMapper.outToIn(widthOfInActivationSplitInfo[0], stride, filterWidth)
		widthOfInActivationSplitInfo[2] = ConvLayerMapper.outToIn(widthOfInActivationSplitInfo[2], stride, filterWidth)
		heightOfInActivationSplitInfo = heightOfOutActivationSplitInfo.copy()
		heightOfInActivationSplitInfo[0] = ConvLayerMapper.outToIn(heightOfInActivationSplitInfo[0], stride, filterHeight)
		heightOfInActivationSplitInfo[2] = ConvLayerMapper.outToIn(heightOfInActivationSplitInfo[2], stride, filterHeight)
		return widthOfInActivationSplitInfo, heightOfInActivationSplitInfo

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
		heightSplitInfo = BasicOperation.oneDimSplitting(outActivationHeight, partsByHeight)
		return widthSplitInfo, heightSplitInfo

	def convOutActivationWidthHeightSplitting(self, prePeAllocationInfo, outChannelSplittingResult, relaxRestrict=False):
		# Extract layer infomation and pre-allocation information
		inActivationDim, filterDim, stride, outActivationDim = self.layerParameter
		filterWidth, filterHeight, inChannels = filterDim[0], filterDim[1], filterDim[2]
		outActivationWidth, outActivationHeight = outActivationDim[0], outActivationDim[1]
		preInActivationSizeAlign = prePeAllocationInfo[INACTIALIG]
		preFilterSizeAlign = prePeAllocationInfo[WEIGHTALIG]
		preOutActivationSizeAlign = prePeAllocationInfo[OUTACTIALIG]
		numOfFilter = outChannelSplittingResult[0]
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
		# Obtain the optimum partsByWidth and partsByHeight
		macUtilizateAlongColumn = 0
		partsByWidth = 0
		partsByHeight = 0
		widthSplitInfo = None
		meetFlag = False
		inActiUtil = 0
		inWidthUtil = 0
		maxPartsByWidth = math.ceil(outActivationWidth / (MAC_ARRAY_COLUMN*self.scale)) + 1
		for partsByWidthTemp in range(maxPartsByWidth-1, 0, -1):
		# for partsByWidthTemp in range(1, maxPartsByWidth):
			# maxPartsByHeight = math.ceil(minSplittedPartsCalib / partsByWidthTemp) + 1
			for partsByHeightTemp in range(1, outActivationHeight+1):
				# Obtain output-activation width/height splitting information
				outActivationWidthSplitInfoTemp = BasicOperation.oneDimSplitting(outActivationWidth, partsByWidthTemp)
				widthOfLargeWidthTemp = outActivationWidthSplitInfoTemp[0]
				partsOfLargeWidthTemp = outActivationWidthSplitInfoTemp[1]
				widthOfSmallWidthTemp = outActivationWidthSplitInfoTemp[2]
				partsOfSmallWidthTemp = outActivationWidthSplitInfoTemp[3]
				outActivationHeightSplitInfoTemp = BasicOperation.oneDimSplitting(outActivationHeight, partsByHeightTemp)
				heightOfLargeHeightTemp = outActivationHeightSplitInfoTemp[0]
				heightOfSmallHeightTemp = outActivationHeightSplitInfoTemp[2]
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
				inWidthOfLargeWidthPartAlign = ConvLayerMapper.convInActivationWidthAlign( \
					ConvLayerMapper.outToIn(widthOfLargeWidthTemp, stride, filterWidth), filterWidth)
				inWidthOfSmallWidthPartAlign = ConvLayerMapper.convInActivationWidthAlign( \
					ConvLayerMapper.outToIn(widthOfSmallWidthTemp, stride, filterWidth), filterWidth)	
				inWidthOverlap = inWidthOfLargeWidthPartAlign * partsOfLargeWidthTemp + \
								inWidthOfSmallWidthPartAlign	* partsOfSmallWidthTemp		
				inHeightOverlap = ConvLayerMapper.totalHeightInActivationOverlap(outActivationHeight = outActivationHeight, 
																partsByHeightOutActivation = partsByHeightTemp, 
																filterHeight = filterHeight, 
																stride = stride)
				inActivationSizeAlignSplitting = inWidthOverlap * inHeightOverlap * inChannels
				inActiUtilTemp = inActivationSizeAlignWithoutSplitting / inActivationSizeAlignSplitting
				if inActiUtilTemp < self.limitedInActiEffeUtil:
					continue
				inWidthUtilTemp = ConvLayerMapper.convInActivationWidthAlign(inActivationDim[0], filterDim[0]) / inWidthOverlap 
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
					meetFlag = True
				else:
					if meetFlag:
						continue
				# According to meetFlag, two different scheme are used.
				if meetFlag:
					# Find as high input-activation width utilization as possbile
					# Find as few part as possible
					if inWidthUtilTemp > inWidthUtil:
						macUtilizateAlongColumn = macUtilizateAlongColumnTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = outActivationWidthSplitInfoTemp
						inActiUtil = inActiUtilTemp
						inWidthUtil = inWidthUtilTemp
					elif inWidthUtilTemp == inWidthUtil:
						if partsByWidthTemp * partsByHeightTemp <= partsByWidth * partsByHeight:
							macUtilizateAlongColumn = macUtilizateAlongColumnTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = outActivationWidthSplitInfoTemp
							inActiUtil = inActiUtilTemp
							inWidthUtil = inWidthUtilTemp
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
					if inWidthUtilTemp > inWidthUtil:
						macUtilizateAlongColumn = macUtilizateAlongColumnTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = outActivationWidthSplitInfoTemp
						inActiUtil = inActiUtilTemp
						inWidthUtil = inWidthUtilTemp
					elif inWidthUtilTemp == inWidthUtil:
						if partsByWidthTemp * partsByHeightTemp >= partsByWidth * partsByHeight:
							macUtilizateAlongColumn = macUtilizateAlongColumnTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = outActivationWidthSplitInfoTemp
							inActiUtil = inActiUtilTemp
							inWidthUtil = inWidthUtilTemp
		# assert(inActiUtil != 0), "inActiUtil should not be 0! for {}".format(self.layerName)
		if inActiUtil == 0:
			BasicOperation.customPrintT("--->---"*8)
			BasicOperation.customPrintT("inActiUtil should not be 0! for {}".format(self.layerName))
			BasicOperation.customPrintT("---<---"*8)
		self.inActivationUtilization = inActiUtil
		return macUtilizateAlongColumn, partsByWidth, partsByHeight, widthSplitInfo, meetFlag

	def poolStrideNotMeet(self, dimension, relaxRestrict):
		if self.poolDim[0] != self.poolStride:
			return False
		else:
			if relaxRestrict:
				if dimension % self.poolStride != 0 and dimension != 1:
					return True
				else:
					return False			
			else:
				if dimension % self.poolStride != 0:
					return True
				else:
					return False

	@staticmethod
	def totalHeightInActivationOverlap(outActivationHeight, partsByHeightOutActivation, filterHeight, stride):
		heightSplitInfo = BasicOperation.oneDimSplitting(outActivationHeight, partsByHeightOutActivation)
		heightOfLargeHeight = heightSplitInfo[0]
		partsOfLargeHeight = heightSplitInfo[1]
		heightOfSmallHeight = heightSplitInfo[2]
		partsOfSmallHeight = heightSplitInfo[3]
		return ((heightOfLargeHeight - 1) * stride + filterHeight) * partsOfLargeHeight + \
				((heightOfSmallHeight - 1) * stride + filterHeight) * partsOfSmallHeight

	@staticmethod
	def outToIn(outActivationDimSize, stride, correspondingFilterDimSize):
		'''
		outActivationDimSize and correspondingFilterDimSize should be size of width/height
		'''
		assert(outActivationDimSize >= 0), \
			"Dimension size [{}] of output-activation should be >= 0".format(outActivationDimSize)
		if outActivationDimSize == 0:
			return 0
		else:
			if correspondingFilterDimSize == 1:
				return (outActivationDimSize - 1) * stride + stride
			return (outActivationDimSize - 1) * stride + correspondingFilterDimSize

	def computeSplittedInActivationSize(self, outActivationWidth, outActivationHeight):
		inActivationDim, filterDim, stride, outActivationDim = self.layerParameter
		inChannels = inActivationDim[2]
		filterWidth = filterDim[0]
		filterHeight = filterDim[1]
		splittedInActivationWidth = ConvLayerMapper.outToIn(outActivationWidth, stride, filterWidth)
		splittedInActivationWidthAlign = ConvLayerMapper.convInActivationWidthAlign(splittedInActivationWidth, filterWidth)
		splittedInActivationHeight = ConvLayerMapper.outToIn(outActivationHeight, stride, filterHeight)
		splittedInActivationSize = splittedInActivationWidth * splittedInActivationHeight * inChannels
		splittedInActivationSizeAlign = splittedInActivationWidthAlign * splittedInActivationHeight * inChannels
		return splittedInActivationSize, splittedInActivationSizeAlign

	# ==================================================================================================
	# 							Step 4: input channel of input-activation/filter
	# ==================================================================================================
	def inChannelSplittingMap(self, namePrefix, prePeAllocationInfo, clockIndex):
		'''
		Splitting input channel will only decrease the size of input-activation and filters.`
		Size of output-activation stays unchanged.
		'''
		# Extract layer infomation and pre-allocation information
		preNumOfPe = prePeAllocationInfo[PEs]
		preInActivationSizeOnEachPe = prePeAllocationInfo[INACTI]
		preInActivationSizeOnEachPeAlign = prePeAllocationInfo[INACTIALIG]
		preFilterSizeOnEachPe = prePeAllocationInfo[WEIGHT] 
		preFilterSizeOnEachPeAlign = prePeAllocationInfo[WEIGHTALIG]
		preOutActivationSizeOnEachPe = prePeAllocationInfo[OUTACTI]
		preOutActivationSizeOnEachPeAlign = prePeAllocationInfo[OUTACTIALIG]
		inActivationDim, _, _, _ = self.layerParameter
		inChannels = inActivationDim[2]
		# Split input channels
		ratio = (preInActivationSizeOnEachPeAlign + preFilterSizeOnEachPeAlign) / \
				(self.sramAvailable - preOutActivationSizeOnEachPeAlign)
		maxInChannelsAfterSplitting = math.floor(inChannels / ratio)
		partsSplitting = math.ceil(inChannels / maxInChannelsAfterSplitting) 
		inChannelSplitInfo = BasicOperation.oneDimSplitting(inChannels, partsSplitting)
		self.convSplitInfo[CONV_IN_CHANNEL] = inChannelSplitInfo
		self.clockComputation()
		largeInChannels = inChannelSplitInfo[0]
		partsOfLargeInChannels = inChannelSplitInfo[1]
		smallInChannels = inChannelSplitInfo[2]
		partsOfSmallInChannels = inChannelSplitInfo[3]
		assert(largeInChannels <= maxInChannelsAfterSplitting), \
			"largeInChannels has too large input channels, largeInChannels: {}; maxInChannelsAfterSplitting: {}". \
			format(largeInChannels, maxInChannelsAfterSplitting)	
		# create PE allocation dictionary
		peAllocationInfo = {}
		peAllocationInfo[PEs] = 0
		peAllocationInfo[INACTI] = 0
		peAllocationInfo[INACTIALIG] = 0
		peAllocationInfo[WEIGHT] = 0
		peAllocationInfo[WEIGHTALIG] = 0
		peAllocationInfo[OUTACTI] = preOutActivationSizeOnEachPe
		peAllocationInfo[OUTACTIALIG] = preOutActivationSizeOnEachPeAlign
		# Add CONVxxxxP
		peAllocationInfo[PEs] = preNumOfPe*partsOfLargeInChannels
		peAllocationInfo[INACTI] = int(preInActivationSizeOnEachPe / inChannels * largeInChannels)
		peAllocationInfo[INACTIALIG] = int(preInActivationSizeOnEachPeAlign / inChannels * largeInChannels)
		peAllocationInfo[WEIGHT] = int(preFilterSizeOnEachPe / inChannels * largeInChannels)
		peAllocationInfo[WEIGHTALIG] = int(preFilterSizeOnEachPeAlign / inChannels * largeInChannels)
		self.insertConvLayerMap(namePrefix+"P_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex)
		# Add CONVxxxxM
		if partsOfSmallInChannels > 0:
			peAllocationInfo[PEs] = preNumOfPe*partsOfSmallInChannels
			peAllocationInfo[INACTI] = int(preInActivationSizeOnEachPe / inChannels * smallInChannels)
			peAllocationInfo[INACTIALIG] = int(preInActivationSizeOnEachPeAlign / inChannels * smallInChannels)
			peAllocationInfo[WEIGHT] = int(preFilterSizeOnEachPe / inChannels * smallInChannels)
			peAllocationInfo[WEIGHTALIG] = int(preFilterSizeOnEachPeAlign / inChannels * smallInChannels)
			self.insertConvLayerMap(namePrefix+"M_"+self.layerNumberStr, peAllocationInfo, clockIndex=clockIndex+1)

	# ==================================================================================================
	# 									Insert sum over layer
	# ==================================================================================================
	def getConvExtraLayer(self):
		inChannels = self.convSplitInfo[CONV_IN_CHANNEL]
		if not isinstance(inChannels, list):
			return
		if inChannels[1] + inChannels[3] == 1:
			return
		# Get out activation width
		outWidth = BasicOperation.splitInfoIntegration(self.convSplitInfo[CONV_OUT_WIDTH])
		# Get out activation height	
		outHeight = BasicOperation.splitInfoIntegration(self.convSplitInfo[CONV_OUT_HEIGHT])
		# Get out activation output-channel	
		outChannel = BasicOperation.splitInfoIntegration(self.convSplitInfo[CONV_OUT_CHANNEL])
		# Get input activation input-channel	
		inChannelSplitInfo = self.convSplitInfo[CONV_IN_CHANNEL]				
		partsOfInChannel = inChannelSplitInfo[1] + inChannelSplitInfo[3]
		# 
		inActivationDimOfExtraLayer = [self.convSplitInfo[CONV_OUT_WIDTH], self.convSplitInfo[CONV_OUT_HEIGHT], \
										self.convSplitInfo[CONV_OUT_CHANNEL], self.convSplitInfo[CONV_IN_CHANNEL]]
		weightDimOfExtraLayer = [0]
		outActivationDimOfExtraLayer = [self.convSplitInfo[CONV_OUT_WIDTH], self.convSplitInfo[CONV_OUT_HEIGHT], \
										self.convSplitInfo[CONV_OUT_CHANNEL]]
		computationClocks = outWidth * outHeight * outChannel * (partsOfInChannel - 1)
		layerName = "CONVE_"+self.layerNumberStr
		convExtraLayerParameter = (LayerType.CONVE, inActivationDimOfExtraLayer, \
			weightDimOfExtraLayer, outActivationDimOfExtraLayer, computationClocks)
		self.convExtraLayerInfo = (layerName, convExtraLayerParameter)

	# ==================================================================================================
	# 									Insert layer Mapping information
	# ==================================================================================================
	def insertConvLayerMap(self, layerName, peAllocationInfo, clockIndex=0, invalid=False):
		# Extract pe allocation infomation
		numOfPEs = peAllocationInfo[PEs]
		inActivationSizeOnEachPe = peAllocationInfo[INACTI]
		inActivationSizeOnEachPeAlign = peAllocationInfo[INACTIALIG]
		filterSizeOnEachPe = peAllocationInfo[WEIGHT] 
		filterSizeOnEachPeAlign = peAllocationInfo[WEIGHTALIG]
		outActivationSizeOnEachPe = peAllocationInfo[OUTACTI]
		outActivationSizeOnEachPeAlign = peAllocationInfo[OUTACTIALIG]
		convSize = inActivationSizeOnEachPe + filterSizeOnEachPe + outActivationSizeOnEachPe
		requireSramSize = inActivationSizeOnEachPeAlign + filterSizeOnEachPeAlign + outActivationSizeOnEachPeAlign
		# invalid allocation, it means the layer will be remap again
		if invalid:
			layerName = layerName + "*"
		else:
			assert(requireSramSize <= self.sramAvailable), \
				"{}: SRAM overflow".format(layerName)
		# Get utilization of SRAM
		sramUtilizationNoAlign = convSize / self.sramAvailable
		sramUtilization = requireSramSize / self.sramAvailable
		# Get effective utilization of SRAM for input/output-activation and filter
		inActivationPercentage = inActivationSizeOnEachPe / inActivationSizeOnEachPeAlign
		filterPercentage = filterSizeOnEachPe/filterSizeOnEachPeAlign
		outActivationPercentage = outActivationSizeOnEachPe / outActivationSizeOnEachPeAlign
		speedRatio = self.theoreticalComputationClock[clockIndex] / self.actualComputationClock[clockIndex]
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
			"{:.2f}".format(self.inActivationUtilization),
			"{:.2f}[{:.2f}]".format(sramUtilization, sramUtilizationNoAlign)]

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
		outChannel - Height - Width - inChannel
		'''
		outChannels = [0, 0]
		inHeights = [0, 0]
		inWidths = [0, 0]
		inChannels = [0, 0]
		if isinstance(self.convSplitInfo[CONV_OUT_CHANNEL], list):
			outChannels[0] = self.convSplitInfo[CONV_OUT_CHANNEL][0]
			outChannels[1] = self.convSplitInfo[CONV_OUT_CHANNEL][2]
		else:
			outChannels[0] = self.convSplitInfo[CONV_OUT_CHANNEL]
			outChannels[1] = 0	
		if isinstance(self.convSplitInfo[CONV_IN_HEIGHT], list):
			inHeights[0] = self.convSplitInfo[CONV_IN_HEIGHT][0]
			inHeights[1] = self.convSplitInfo[CONV_IN_HEIGHT][2]
		else:
			inHeights[0] = self.convSplitInfo[CONV_IN_HEIGHT]
			inHeights[1] = 0
		if isinstance(self.convSplitInfo[CONV_IN_WIDTH], list):
			inWidths[0] = self.convSplitInfo[CONV_IN_WIDTH][0]
			inWidths[1] = self.convSplitInfo[CONV_IN_WIDTH][2]
		else:
			inWidths[0] = self.convSplitInfo[CONV_IN_WIDTH]
			inWidths[1] = 0			
		if isinstance(self.convSplitInfo[CONV_IN_CHANNEL], list):
			inChannels[0] = self.convSplitInfo[CONV_IN_CHANNEL][0]
			inChannels[1] = self.convSplitInfo[CONV_IN_CHANNEL][2]
		else:
			inChannels[0] = self.convSplitInfo[CONV_IN_CHANNEL]
			inChannels[1] = 0
		filterWidth = self.convSplitInfo[CONV_FILTER_WIDTH]
		filterHeight = self.convSplitInfo[CONV_FILTER_HEIGHT]
		stride = self.convSplitInfo[CONV_FILTER_STRIDE]
		index = 0
		for outChannel in outChannels:
			for inHeight in inHeights:
				for inWidth in inWidths:
					for inChannel in inChannels:
						self.theoreticalComputationClock[index] = ConvLayerMapper.convTheoreticalClocks(\
							inWidth, inHeight, inChannel, filterWidth, filterHeight, stride, outChannel)
						self.actualComputationClock[index] = ConvLayerMapper.convActualClocks(\
							inWidth, inHeight, inChannel, filterWidth, filterHeight, stride, outChannel)
						index = index + 1

	@staticmethod
	def convTheoreticalClocks(inWidth, inHeight, inChannel, filterWidth, filterHeight, stride, outChannel):
		oneIterationClocks = filterWidth * filterHeight
		iterations = (math.floor((inWidth - filterWidth) / stride) + 1) * (math.floor((inHeight - filterHeight) / stride) + 1)
		return iterations * oneIterationClocks * inChannel * outChannel

	@staticmethod
	def convActualClocks(inWidth, inHeight, inChannel, filterWidth, filterHeight, stride, outChannel):
		scale = stride * stride # limitation of MLA
		oneIterationClocks = filterWidth * filterHeight
		iterations = math.ceil((math.floor((inWidth - filterWidth) / stride) + 1) / MAC_ARRAY_COLUMN) * \
					(math.floor((inHeight - filterHeight) / stride) + 1)
		return iterations * oneIterationClocks * inChannel * math.ceil(outChannel/MAC_ARRAY_ROW) * scale

	# ==================================================================================================
	# 										Common function
	#  	These functions could be upgraded when understanding the hardware deeper.
	# 	These functions are all about alignment problem
	# ==================================================================================================
	@staticmethod
	def getMacUtilAlongWidth(width):
		'''
		output activation width
		'''
		return BasicOperation.ceilOf(width, MAC_ARRAY_COLUMN)

	@staticmethod
	def getMacUtilAlongOutChannel(outChannel):
		'''
		output activation output-channel / filter output-channel
		'''
		return BasicOperation.ceilOf(outChannel, MAC_ARRAY_ROW)

	@staticmethod
	def convInActivationSizeAlign(width, height, inChannel, filterWidth):
		inActivationWidthAlign = ConvLayerMapper.convInActivationWidthAlign(width, filterWidth)
		return inActivationWidthAlign * height * inChannel

	@staticmethod
	def convInActivationWidthAlign(inActivationWidth, filterWidth=0):
		'''
		Argument "filterWidth" is unused
		'''
		# inActivationWidthAlign = 0
		# if inActivationWidth <= MAC_ARRAY_COLUMN:
		# 	inActivationWidthAlign = MAC_ARRAY_COLUMN
		# else:
		# 	inActivationWidthAlign = MAC_ARRAY_COLUMN + BasicOperation.ceilOf(inActivationWidth-MAC_ARRAY_COLUMN, SRAM_BYTES_ALIGN)
		# return inActivationWidthAlign
		return BasicOperation.ceilOf(inActivationWidth, MAC_ARRAY_COLUMN)
		# if inActivationWidth < MAC_ARRAY_COLUMN + filterWidth - 1:
		# 	inActivationWidthAlign = (MAC_ARRAY_COLUMN + filterWidth - 1)
		# else:
		# 	inActivationWidthAlign = inActivationWidth
		# return inActivationWidthAlign

	@staticmethod
	def convFilterSizeAlign(width, height, inChannel, outChannel):
		# return width * height * inChannel * ConvLayerMapper.convFilterOutChannelAlign(outChannel)
		filterSize = width * height * inChannel * ConvLayerMapper.convFilterOutChannelAlign(outChannel)
		return BasicOperation.ceilOf(filterSize, SRAM_BYTES_ALIGN)

	@staticmethod
	def convFilterOutChannelAlign(outChannel):
		return BasicOperation.ceilOf(outChannel, MAC_ARRAY_ROW)

	@staticmethod
	def convOutActivationSizeAlign(outWidth, outHeight, outChannel):
		return ConvLayerMapper.convOutActivationWidthAlign(outWidth) * outHeight * outChannel

	@staticmethod
	def convOutActivationWidthAlign(outWidth):
		return BasicOperation.ceilOf(outWidth*MAC_OUT_BYTES, MAC_ARRAY_COLUMN)

	@staticmethod
	def convInToOut(inWH, filterWH, stride=1):
		return (inWH - filterWH)//stride + 1

	@staticmethod
	def convOutToOut(outWH, stride):
		return (outWH - 1) * stride + 1