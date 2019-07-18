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
from spiNNaker2General import PES_ON_QPE
from spiNNakerSimulatorGeneral import NUM_OF_PES_IN_ALL_DOUBLOCK


class ConvLayerMapperForSpiNNaker(ConvLayerMapper):
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
	# 									Step 2: Output Channels
	# ==================================================================================================
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
				assert(False), "insufficient PE should not exsit"
				self.insertConvLayerMap("CONVIS_"+self.layerNumberStr, insuPeAllocationInfo, clockIndex=8, invalid=True)
				self.inoutputActivationSplittingMap("CONVIS", insuPeAllocationInfo, outChannelSplittingResult[2:4], clockIndex=8)
		else:
			if numOfSuffPe + numOfInsuPe >= 128:
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
			else:
				# Resplit the filter
				# numOfSuffPe = numOfSuffPe * (filtersOnSuffPe // 4)
				# filtersOnSuffPe = 4
				filtersOnSuffPe, numOfSuffPe = self.reDecomposeFilter(filtersOnSuffPe, numOfSuffPe)
				outChannelSplittingResult[0] = filtersOnSuffPe
				outChannelSplittingResult[1] = numOfSuffPe
				# 
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
				suffPeAllocationInfo[PEs] = numOfSuffPe
				suffPeAllocationInfo[WEIGHT] = filterSizeOnSuffPe
				suffPeAllocationInfo[WEIGHTALIG] = filterSizeOnSuffPeAlign
				suffPeAllocationInfo[OUTACTI] = outActivationSizeOnSuffPe
				suffPeAllocationInfo[OUTACTIALIG] = outActivationSizeOnSuffPeAlign
				# sufficient-PEs allocation information
				self.insertConvLayerMap("CONVSU_"+self.layerNumberStr, suffPeAllocationInfo, clockIndex=0, invalid=True)
				self.inoutputActivationSplittingMap("CONVSU", suffPeAllocationInfo, outChannelSplittingResult[0:2], clockIndex=0)
				# insufficient PE allocation infomation
				if insuPeAllocationInfo != None:
					assert(False), "insufficient PE should not exsit"
					self.insertConvLayerMap("CONVIS_"+self.layerNumberStr, insuPeAllocationInfo, clockIndex=8, invalid=True)
					self.inoutputActivationSplittingMap("CONVIS", insuPeAllocationInfo, outChannelSplittingResult[2:4], clockIndex=8)	

	def reDecomposeFilter(self, filtersOnSuffPeOrigin, numOfSuffPeOrigin):
		for tempNumofFilters in range(4, filtersOnSuffPeOrigin+1, 4):
			numOfSuffPe = int(numOfSuffPeOrigin * filtersOnSuffPeOrigin // tempNumofFilters)		
			filtersOnSuffPe = tempNumofFilters
			if numOfSuffPe <= 128:
				return 	filtersOnSuffPe, numOfSuffPe

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
		outChannels = filterDim[3]
		outActivationHeight = outActivationDim[1]
		preInActivationSizeAlign = prePeAllocationInfo[INACTIALIG]
		preFilterSizeAlign = prePeAllocationInfo[WEIGHTALIG]
		preOutActivationSizeAlign = prePeAllocationInfo[OUTACTIALIG]
		# Get output channel splitting infomation
		# filtersOnPe = outChannelSplittingResult[0]
		# assert(filtersOnPe <= MAC_ARRAY_ROW), "filtersOnPe-[{}] should not larger than 4".format(filtersOnPe)
		numOfPes = outChannelSplittingResult[1]
		assert(prePeAllocationInfo[PEs] == numOfPes), \
			"prePeAllocationInfo[PEs]-[{}] should equals outChannelSplittingResult[1]-[{}]".\
			format(prePeAllocationInfo[PEs], outChannelSplittingResult[1])		
		# Split the output activation along the width and height
		# Obtain width splitting infomation
		partsByWidth, partsByHeight, widthSplitInfo, heightSplitInfo, increasedSize, meetFlag, increasedFilterTimes = \
			self.convOutActivationWidthHeightSplitting(prePeAllocationInfo, outChannelSplittingResult)
		if self.inActivationUtilization == 0 or meetFlag == False:
			partsByWidth1, partsByHeight1, widthSplitInfo1, heightSplitInfo1, increasedSize1, meetFlag1, increasedFilterTimes1 = \
				self.convOutActivationWidthHeightSplitting(prePeAllocationInfo, outChannelSplittingResult, partsOfHeightStep=2)
			partsByWidth2, partsByHeight2, widthSplitInfo2, heightSplitInfo2, increasedSize2, meetFlag2, increasedFilterTimes2 = \
				self.convOutActivationWidthHeightSplitting(prePeAllocationInfo, outChannelSplittingResult, partsOfHeightStep=1)
			if meetFlag1:
				estClks1 = self.clockEstimate(increasedSize1, widthSplitInfo1, heightSplitInfo1, inChannels, 
					filterWidth, filterHeight, stride, outChannels)
			else:
				estClks1 = math.inf
			if meetFlag2:
				estClks2 = self.clockEstimate(increasedSize2, widthSplitInfo2, heightSplitInfo2, inChannels, 
					filterWidth, filterHeight, stride, outChannels)
			else:
				estClks2 = math.inf
			if meetFlag1 == False and meetFlag2 == False:
				assert(False), "{}-Could not find suitable splitting result".format(self.layerName)
			else:
				# Clock
				if estClks1 > estClks2:
					self.doubleFilterProcess(prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes2)
					return widthSplitInfo2, heightSplitInfo2
				elif estClks1 < estClks2:
					self.doubleFilterProcess(prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes1)
					return widthSplitInfo1, heightSplitInfo1
				else:
				# increased size
					if increasedSize1 > increasedSize2:
						self.doubleFilterProcess(prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes2)
						return widthSplitInfo2, heightSplitInfo2
					elif increasedSize1 < increasedSize2:
						self.doubleFilterProcess(prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes1)
						return widthSplitInfo1, heightSplitInfo1
					else:
				# Number of parts
						if self.selectFirstScheme(partsByWidth1, partsByHeight1, partsByWidth2, partsByHeight2, prePeAllocationInfo):
							self.doubleFilterProcess(prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes1)
							return widthSplitInfo1, heightSplitInfo1
						else:
							self.doubleFilterProcess(prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes2)
							return widthSplitInfo2, heightSplitInfo2
		self.doubleFilterProcess(prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes)
		return widthSplitInfo, heightSplitInfo

	def doubleFilterProcess(self, prePeAllocationInfo, outChannelSplittingResult, increasedFilterTimes):
		self.convSplitInfo[CONV_OUT_CHANNEL][0] = self.convSplitInfo[CONV_OUT_CHANNEL][0] * increasedFilterTimes
		self.convSplitInfo[CONV_OUT_CHANNEL][1] = self.convSplitInfo[CONV_OUT_CHANNEL][1] // increasedFilterTimes
		outChannelSplittingResult[0] = self.convSplitInfo[CONV_OUT_CHANNEL][0]
		outChannelSplittingResult[1] = self.convSplitInfo[CONV_OUT_CHANNEL][1]
		# 
		prePeAllocationInfo[PEs] = prePeAllocationInfo[PEs] // increasedFilterTimes
		prePeAllocationInfo[WEIGHT] = prePeAllocationInfo[WEIGHT] * increasedFilterTimes
		prePeAllocationInfo[WEIGHTALIG] = prePeAllocationInfo[WEIGHTALIG] * increasedFilterTimes
		prePeAllocationInfo[OUTACTI] = prePeAllocationInfo[OUTACTI] * increasedFilterTimes
		prePeAllocationInfo[OUTACTIALIG] = prePeAllocationInfo[OUTACTIALIG] * increasedFilterTimes

	def selectFirstScheme(self, partsByWidth1, partsByHeight1, partsByWidth2, partsByHeight2, prePeAllocationInfo):
		partsOfFilters = prePeAllocationInfo[PEs]
		partsOfInActi1 = partsByWidth1 * partsByHeight1
		partsOfInActi2 = partsByWidth2 * partsByHeight2
		partsOfScheme1 = partsOfInActi1 * partsOfFilters
		partsOfScheme2 = partsOfInActi2 * partsOfFilters
		if partsOfScheme1 >= 128 and partsOfScheme2 < 128:
			return True
		if partsOfScheme1 < 128 and partsOfScheme2 >= 128:
			return False
		partsByHeight1Align = int(math.pow(2, math.ceil(math.log2(partsByHeight1))))
		partsByHeight2Align = int(math.pow(2, math.ceil(math.log2(partsByHeight2))))
		utilizationRate1 = partsByHeight1 / partsByHeight1Align
		utilizationRate2 = partsByHeight2 / partsByHeight2Align
		if utilizationRate1 > utilizationRate2:
			return True
		elif utilizationRate2 > utilizationRate1:
			return False
		else:
			return True

	def clockEstimate(self, increasedSize, widthSplitInfo, heightSplitInfo, inChannel, 
		filterWidth, filterHeight, stride, outChannel):
		stride = 1 # limitation of MLA
		inWidth = self.outToIn(widthSplitInfo[0], stride, filterWidth)
		inHeight = self.outToIn(heightSplitInfo[0], stride, filterHeight)
		partsOfWidth = BasicOperation.getTotalPartsFromSplitInfo(widthSplitInfo)
		# print("layerName: {}".format(self.layerName))
		# print("partsOfWidth: {}".format(partsOfWidth))
		partsOfHeight = BasicOperation.getTotalPartsFromSplitInfo(heightSplitInfo)
		# print("partsOfHeight: {}".format(partsOfHeight))
		parts = partsOfWidth * partsOfHeight
		# print("{}-{}-{}-{}-{}-{}-{}".format(inWidth, inHeight, inChannel, filterWidth, filterHeight, stride, MAC_ARRAY_ROW))
		compClks = self.convActualClocks(inWidth, inHeight, inChannel, filterWidth, filterHeight, stride, MAC_ARRAY_ROW)
		# print("compClks: {}".format(compClks))
		totalParts = math.pow(2, math.ceil(math.log(parts, 2)))
		# print("round (128): {}".format(math.ceil((totalParts * outChannel // MAC_ARRAY_ROW) / 128)))
		compClks = compClks * math.ceil((totalParts * outChannel // MAC_ARRAY_ROW) / 128)
		# print("compClks: {}".format(compClks))
		loadClks = increasedSize * 2 / MAC_ARRAY_COLUMN / 4
		# print("loadClks: {}".format(loadClks))
		# input("")
		return compClks + loadClks

	def convOutActivationWidthHeightSplitting(self, prePeAllocationInfo, outChannelSplittingResult, relaxRestrict=False, partsOfHeightStep=4):
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
		preNumOfPes = outChannelSplittingResult[1]
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
		heightSplitInfo = None
		meetFlag = False
		inActiUtil = 0
		maxPartsByWidth = math.ceil(outActivationWidth / (MAC_ARRAY_COLUMN*self.scale)) + 1
		inActiIncreasedSize = inActivationSizeAlignWithoutSplitting
		increasedSize = inActivationSizeAlignWithoutSplitting + outActivationSizeAlignWithoutSplitting
		increasedFilterTimes = 1
		# Step 1:
		for partsByWidthTemp in range(maxPartsByWidth-1, 0, -1):
		# for partsByWidthTemp in range(1, maxPartsByWidth):
			# maxPartsByHeight = math.ceil(minSplittedPartsCalib / partsByWidthTemp) + 1
			for partsByHeightTemp in range(partsOfHeightStep, outActivationHeight+1, partsOfHeightStep):
				# Speical for spiNNaker
				if preNumOfPes * partsByWidthTemp * partsByHeightTemp < NUM_OF_PES_IN_ALL_DOUBLOCK:
					continue
				# S1: Obtain output-activation width/height splitting information
				outActivationWidthSplitInfoTemp = self.splittingWithPoolStride(outActivationWidth, partsByWidthTemp)
					# Stop this iteration when no splitting result
				if outActivationWidthSplitInfoTemp == None:
					continue
				widthOfLargeWidthTemp = outActivationWidthSplitInfoTemp[0]
				partsOfLargeWidthTemp = outActivationWidthSplitInfoTemp[1]
				widthOfSmallWidthTemp = outActivationWidthSplitInfoTemp[2]
				partsOfSmallWidthTemp = outActivationWidthSplitInfoTemp[3]
					# For spiNNaker2, only one width dimension can exsit
				if partsOfSmallWidthTemp != 0:
					continue				
				outActivationHeightSplitInfoTemp = self.splittingWithPoolStride(outActivationHeight, partsByHeightTemp)
					# Stop this iteration when no splitting result
				if outActivationHeightSplitInfoTemp == None:
					continue
				heightOfLargeHeightTemp = outActivationHeightSplitInfoTemp[0]
				partsOfLargeHeightTemp = outActivationHeightSplitInfoTemp[1]
				heightOfSmallHeightTemp = outActivationHeightSplitInfoTemp[2]
				partsOfSmallHeightTemp = outActivationHeightSplitInfoTemp[3]
					# For spiNNaker2, only one width dimension can exsit
				if partsOfSmallHeightTemp != 0:
					continue
				# S2: Judge if the total parts meet the rules for SpiNNaker2
				if 4 == partsOfHeightStep:
					if not (partsOfLargeWidthTemp*partsOfLargeHeightTemp in [4,8,16,32,64,128]):
						continue
				# S3: Avoid this case: VGG-CONV_16
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
				# Deprecated???? S4: Judge if output-activation splitting scheme fit for pooling layer
				if self.poolStrideNotMeet(widthOfLargeWidthTemp, relaxRestrict):
					continue
				if self.poolStrideNotMeet(widthOfSmallWidthTemp, relaxRestrict):
					continue
				if self.poolStrideNotMeet(heightOfLargeHeightTemp, relaxRestrict):
					continue
				if self.poolStrideNotMeet(heightOfSmallHeightTemp, relaxRestrict):
					continue
				# S5: Obtain MAC array utilication rate and do judgement
				macUtilizateAlongColumnTemp = ConvLayerMapper.getMacUtilAlongWidth(widthOfLargeWidthTemp) * \
												partsOfLargeWidthTemp + \
												ConvLayerMapper.getMacUtilAlongOutChannel(widthOfSmallWidthTemp) * \
												partsOfSmallWidthTemp
				if macUtilizateAlongColumnTemp != self.maxMacUtilizationAlongColumn:
					continue
				# Deprecated???? S6: Obtain memory utilization rate and do judgement
				# Get size of input-activation after splitting (including overlap size)
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
				# S7: Get increase size of input/output-activation
					# input-activation increased size caused by splitting
				inActiIncreasedSizeTemp = inActivationSizeAlignSplitting - inActivationSizeAlignWithoutSplitting
					# output-activation increased size caused by splitting
				outActiWidthInc = ConvLayerMapper.convOutActivationWidthAlign(widthOfLargeWidthTemp) * partsOfLargeWidthTemp + \
					ConvLayerMapper.convOutActivationWidthAlign(widthOfSmallWidthTemp) * partsOfSmallWidthTemp
				outActiHeightInc = BasicOperation.splitInfoIntegration(outActivationHeightSplitInfoTemp)
				outActiIncreasedSizeTemp = outActiWidthInc * outActiHeightInc * outChannels - outActivationSizeAlignWithoutSplitting
				increasedSizeTemp = inActiIncreasedSizeTemp+outActiIncreasedSizeTemp
				# S8: For fc layer, wenn splitting the weights, it need to determine if the largest part size > sramAvailable, 
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
				outActiWidthS1 = ConvLayerMapper.convInToOut(inActivationLargePartWidth, filterWidth)
				outActiHeightS1 = ConvLayerMapper.convInToOut(inActivationLargePartHeight, filterHeight)
				outActivationLargePartSizeAlign = ConvLayerMapper.convOutActivationSizeAlign(outActiWidthS1,
																								outActiHeightS1,
																								numOfFilter)				
				largestPartSize = inActivationLargePartSizeAlign + preFilterSizeAlign * 2 + outActivationLargePartSizeAlign			
				if outActivationLargePartSizeAlign > self.sramAvailable:
					continue
				# S9: Determine if the splitting result meet the SRAM limitation
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
				# S10: According to meetFlag, two different update scheme are used.
				if meetFlag:
					# Find as high input-activation width utilization as possbile
					# Find as few part as possible
					if increasedSizeTemp < increasedSize:
						macUtilizateAlongColumn = macUtilizateAlongColumnTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = outActivationWidthSplitInfoTemp
						heightSplitInfo = outActivationHeightSplitInfoTemp
						inActiUtil = inActiUtilTemp
						inActiIncreasedSize = inActiIncreasedSizeTemp
						increasedSize = increasedSizeTemp
						# 
						increasedFilterTimes = self.getIncreasedFilterTimes(inActivationLargePartSizeAlign, preFilterSizeAlign, 
							outActivationLargePartSizeAlign, preNumOfPes, partsByWidthTemp, partsByHeightTemp)
					elif increasedSizeTemp == increasedSize:
						if partsByWidthTemp * partsByHeightTemp <= partsByWidth * partsByHeight:
							macUtilizateAlongColumn = macUtilizateAlongColumnTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = outActivationWidthSplitInfoTemp
							heightSplitInfo = outActivationHeightSplitInfoTemp
							inActiUtil = inActiUtilTemp
							inActiIncreasedSize = inActiIncreasedSizeTemp
							increasedSize = increasedSizeTemp
							# 
							increasedFilterTimes = self.getIncreasedFilterTimes(inActivationLargePartSizeAlign, preFilterSizeAlign, 
								outActivationLargePartSizeAlign, preNumOfPes, partsByWidthTemp, partsByHeightTemp)
				else:
					# limitation of each PE's SRAM utilization
					# This part is same as part of self.inChannelSplittingMap()
					ratio = (inActivationLargePartSizeAlign + preFilterSizeAlign) / \
							(self.sramAvailable - outActivationLargePartSizeAlign)
					if ratio > inChannels:
						continue
					if increasedSizeTemp < increasedSize:
						macUtilizateAlongColumn = macUtilizateAlongColumnTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = outActivationWidthSplitInfoTemp
						heightSplitInfo = outActivationHeightSplitInfoTemp
						inActiUtil = inActiUtilTemp
						inActiIncreasedSize = inActiIncreasedSizeTemp
						increasedSize = increasedSizeTemp
					elif increasedSizeTemp == increasedSize:
						if partsByWidthTemp * partsByHeightTemp >= partsByWidth * partsByHeight:
							macUtilizateAlongColumn = macUtilizateAlongColumnTemp
							partsByWidth = partsByWidthTemp
							partsByHeight = partsByHeightTemp
							widthSplitInfo = outActivationWidthSplitInfoTemp
							heightSplitInfo = outActivationHeightSplitInfoTemp
							inActiUtil = inActiUtilTemp
							inActiIncreasedSize = inActiIncreasedSizeTemp
							increasedSize = increasedSizeTemp
		# Step 2: when searching stage is finish, check if the meet-splitting scheme has been found
		if inActiUtil == 0:
			BasicOperation.customPrintT("--->---"*8)
			BasicOperation.customPrintT("inActiUtil should not be 0! for {}".format(self.layerName))
			BasicOperation.customPrintT("---<---"*8)
		self.inActivationUtilization = inActiUtil
		return partsByWidth, partsByHeight, widthSplitInfo, heightSplitInfo, increasedSize, meetFlag, increasedFilterTimes

	def getIncreasedFilterTimes(self, inActivationLargePartSizeAlign, preFilterSizeAlign, outActivationLargePartSizeAlign, 
		preNumOfPes, partsByWidthTemp, partsByHeightTemp):
		increasedFilterTimes = 1
		maxTimes = math.floor((self.sramAvailable - inActivationLargePartSizeAlign) / (preFilterSizeAlign * 2 + outActivationLargePartSizeAlign))
		# if self.layerName == "CONV_SC_14":
		# 	print("\n--------------------------------------------------------")
		# 	print("inActivationLargePartSizeAlign: {}".format(inActivationLargePartSizeAlign))
		# 	print("preFilterSizeAlign: {}".format(preFilterSizeAlign))
		# 	print("outActivationLargePartSizeAlign: {}".format(outActivationLargePartSizeAlign))
		# 	input("maxTimes: {}".format(maxTimes))
		for times in range(maxTimes, 1, -1):
			increasedSize = inActivationLargePartSizeAlign + preFilterSizeAlign * 2 * times + outActivationLargePartSizeAlign * times
			# if self.layerName == "CONV_SC_14":
			# 	input("Times[{}] ---> {}-{}".format(times, increasedSize, self.sramAvailable))
			if increasedSize <= self.sramAvailable:
				if preNumOfPes % times == 0 and int(preNumOfPes/times) >= 32 and int(preNumOfPes/times) * partsByWidthTemp * partsByHeightTemp >= 128:
					increasedFilterTimes = times
					break
		return increasedFilterTimes

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