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


FC_MAX_M = 200
FC_MAX_WEIGHT_WIDTH = 16 * FC_MAX_M
FC_MAX_K = 64
FC_MAX_WEIGHT_HEIGHT = 4 * FC_MAX_K

class FcLayerMapper():
	'''
	In this class, dense also refers to fully-connected layer
	Distribute fully-connected layer into SpiNNaker2. All input/output-activation and filters need to place in local SRAM.
	As each fully-connected layer can be very large, which need to be splitted into small part to meet the SRAM limitation.
	The splitting strategy is listed below:
	Step 1: No splitting: try to place all input, weight and output into one PE.
		If SRAM is not large enough, turn to step 2.
		Corresponding method: self.map()
	Step 2. Weight splitting
		Corresponding method: self.weightSplittingMap()
	'''
	def __init__(self, layerName, layerParameter, limitedSramUtilization=0.1, scale=0.7, sramAvailable=PE_SRAM_LIMIT, 
		forSpiNNaker2=False):
		'''
		Args:
			layerParameter: tuple, e.g. (inNeuron, outNeuron), where inNeuron and outNeuron are int.
			layerName: String for layer name, e.g. "FC_x", where x is layer number.
			scale： Number, which must be positive and cannot be larger than 1, 
					used for seaching optimum splitting scheme. 
					The smaller the number, the larger the calculation for iterate all possible splitting scheme.
					Recommand to keep default value. 
					This default is only reduced if a suitable splitting scheme cannot be found.
		'''
		self.mapperReset(layerName, layerParameter, limitedSramUtilization, scale, sramAvailable, forSpiNNaker2)

	def mapperReset(self, layerName, layerParameter, limitedSramUtilization, scale, sramAvailable, forSpiNNaker2):
		self.layerName = layerName
		self.layerParameter = layerParameter
		self.layerMapInfo = {}
		self.layerNumberStr = self.layerName[len("FC_"):]
		self.scale = scale
		assert(self.scale <= 1), "scale cannot be large than 1"
		self.maxMacUtilizationAlongColumn = 0
		self.denseSplitInfo = {}
		self.theoreticalComputationClock = [0] * 4
		self.actualComputationClock = [0] * 4
		self.denseExtraLayerInfo = None
		self.limitedSramUtilization = limitedSramUtilization
		self.sramAvailable = sramAvailable
		self.meetFlag = False
		self.forSpiNNaker2 = forSpiNNaker2

	def getSplitInfo(self):
		return self.denseSplitInfo

	def getClocks(self):
		return max(self.actualComputationClock)

	def getExtraLayer(self):
		return self.denseExtraLayerInfo

	def isMeet(self):
		return self.meetFlag

	# ==================================================================================================
	# 										Step 1: No Splitting
	# ==================================================================================================
	def map(self):
		'''
		Step 1: No splitting: try to place all input, weight and output into one PE.

		Returns:
			Dictionary contains the PE allocation information
			{"FCxxx":[number_of_PE(int), input_each_PE(string), weight_each_PE(string), output_each_PE(string)],
			 "FCxxx":[number_of_PE(int), input_each_PE(string), weight_each_PE(string), output_each_PE(string)]
			 ...}
		'''
		inNeuron, outNeuron = self.layerParameter
		# Get fully-connected layer split info and update computation clock
		self.denseSplitInfo[FC_IN] = inNeuron
		self.denseSplitInfo[FC_OUT] = outNeuron
		self.clockComputation()
		# Get maxMacUtilizationAlongColumn
		self.maxMacUtilizationAlongColumn = self.getMacUtilization(outNeuron)
		BasicOperation.customPrintF(">>------"*10)
		BasicOperation.customPrintF("maxMacUtilizationAlongColumn: {}".format(self.maxMacUtilizationAlongColumn))
		# Total needed memory size for weight and input/output activation.
		# This is the minimum value. After spliting, it can be larger, because of alignment.
		# Get input activation size
		inActivationSize = inNeuron
		inActivationSizeAlign = FcLayerMapper.inActivationAlign(inNeuron)
		# Get output activation size
		outActivationSize = outNeuron * MAC_OUT_BYTES
		outActivationSizeAlign = FcLayerMapper.outActivationAlign(outNeuron)
		# Get weight size
		weightSize = inNeuron * outNeuron
		weightSizeAlign = FcLayerMapper.weightAlign(inNeuron, outNeuron)
		# Get allocation information
		peAllocationInfo = {}
		peAllocationInfo[PEs] = 1
		peAllocationInfo[INACTI] = inActivationSize
		peAllocationInfo[INACTIALIG] = inActivationSizeAlign
		peAllocationInfo[WEIGHT] = weightSize
		peAllocationInfo[WEIGHTALIG] = weightSizeAlign
		peAllocationInfo[OUTACTI] = outActivationSize
		peAllocationInfo[OUTACTIALIG] = outActivationSizeAlign
		# SRAM limitation: check if all data can put into one PE's SRAM
		ratio = math.ceil((inActivationSizeAlign + outActivationSizeAlign + weightSizeAlign) / self.sramAvailable)
		# If weight+outActivation small enough, just place all of them in one PE
		if ratio == 1:
			self.meetFlag = True
			self.inserDenseLayerMap(self.layerName, peAllocationInfo)
		# If weight+outActivation too large, split them and place them into different PEs
		else:
			self.meetFlag = False
			self.inserDenseLayerMap(self.layerName, peAllocationInfo, invalid=True)
			ratio = (inActivationSizeAlign + outActivationSizeAlign + weightSizeAlign) / self.sramAvailable
			self.weightSplittingMap(ratio)
		# Get extra layer(needed when splitting input-channel)
		self.getDenseExtraLayer()

		return self.layerMapInfo

	# ==================================================================================================
	# 									Step 2: width and height of weight
	# ==================================================================================================
	def weightSplittingMap(self, ratio):
		'''
		Called by self.map(). Find the optimun splitting scheme and calculate the needed memory information for
			each PE

		Args:
			ratio: the ratio between needed memory size for input+weight+output and limitation of SRAM.
					This ratio comes from the Step 1.
					Give information to Step 2 that the minimum parts it needs to split the weight
		'''
		inNeuron, outNeuron = self.layerParameter
		# find weight split scheme
		macUtilization, partsByWidth, partsByHeight, widthSplitInfo = self.findWeightSplitSchemeWithFewestPartsOfHeight(ratio)
		self.denseSplitInfo[FC_OUT] = widthSplitInfo
		BasicOperation.customPrintF("inNeuron: {}, outNeuron: {}".format(inNeuron, outNeuron))
		BasicOperation.customPrintF("ratio: {}".format(ratio))
		BasicOperation.customPrintF("partsByWidth: {}, partsByHeight: {}".format(partsByWidth, partsByHeight))
		BasicOperation.customPrintF("------<<"*10+"\n")
		if macUtilization == 0:
			self.meetFlag = False
			return
		else:
			self.meetFlag = True
		# assert(macUtilization != 0), "macUtilization should not be 0."
		# Extract width split infomation
		widthOfLargeWidth = widthSplitInfo[0]
		widthOfLargeWidthAlign = FcLayerMapper.weightWidthAlign(widthOfLargeWidth)
		partsOfLargeWidth = widthSplitInfo[1]
		widthOfSmallWidth = widthSplitInfo[2]
		widthOfSmallWidthAlign = FcLayerMapper.weightWidthAlign(widthOfSmallWidth)
		partsOfSmallWidth = widthSplitInfo[3]
		# Extract output activation from width information
		outActivationOfLargeWidth = widthOfLargeWidth * MAC_OUT_BYTES
		outActivationOfLargeWidthAlign = FcLayerMapper.outActivationAlign(widthOfLargeWidth)
		BasicOperation.customPrintF("outActivationOfLargeWidth: {}".format(outActivationOfLargeWidth))
		BasicOperation.customPrintF("outActivationOfLargeWidthAlign: {}".format(outActivationOfLargeWidthAlign))
		outActivationOfSmallWidth = widthOfSmallWidth * MAC_OUT_BYTES
		outActivationOfSmallWidthAlign = FcLayerMapper.outActivationAlign(outActivationOfSmallWidth)
		# Obtain height splitting information
		heightSplitInfo = BasicOperation.oneDimSplitting(inNeuron, partsByHeight)
		self.denseSplitInfo[FC_IN] = heightSplitInfo
		self.clockComputation()
		heightOfLargeHeight = heightSplitInfo[0]
		partsOfLargeHeight = heightSplitInfo[1]
		heightOfSmallHeight = heightSplitInfo[2]
		partsOfSmallHeight = heightSplitInfo[3]
		# Extract input activation from height information
		inActivationOfLargeHeight = heightOfLargeHeight
		inActivationOfLargeHeightAlign = FcLayerMapper.inActivationAlign(inActivationOfLargeHeight)
		inActivationOfSmallHeight = heightOfSmallHeight
		inActivationOfSmallHeightAlign = FcLayerMapper.inActivationAlign(inActivationOfSmallHeight)
		# Add FCTL
		denseTlName = "FCTL_" + self.layerNumberStr
		peAllocationInfo = {}
		peAllocationInfo[PEs] = partsOfLargeWidth*partsOfLargeHeight
		peAllocationInfo[INACTI] = inActivationOfLargeHeight
		peAllocationInfo[INACTIALIG] = inActivationOfLargeHeightAlign
		peAllocationInfo[WEIGHT] = widthOfLargeWidth*heightOfLargeHeight
		peAllocationInfo[WEIGHTALIG] = widthOfLargeWidthAlign*heightOfLargeHeight
		peAllocationInfo[OUTACTI] = outActivationOfLargeWidth
		peAllocationInfo[OUTACTIALIG] = outActivationOfLargeWidthAlign
		self.inserDenseLayerMap(denseTlName, peAllocationInfo, clockIndex=0)
		# Add FCTR
		if partsOfSmallWidth > 0:
			denseTrName = "FCTR_" + self.layerNumberStr
			peAllocationInfo = {}
			peAllocationInfo[PEs] = partsOfSmallWidth*partsOfLargeHeight
			peAllocationInfo[INACTI] = inActivationOfLargeHeight
			peAllocationInfo[INACTIALIG] = inActivationOfLargeHeightAlign
			peAllocationInfo[WEIGHT] = widthOfSmallWidth*heightOfLargeHeight
			peAllocationInfo[WEIGHTALIG] = widthOfSmallWidthAlign*heightOfLargeHeight
			peAllocationInfo[OUTACTI] = outActivationOfSmallWidth
			peAllocationInfo[OUTACTIALIG] = outActivationOfSmallWidthAlign
			self.inserDenseLayerMap(denseTrName, peAllocationInfo, clockIndex=1)		
		# Add FCBL
		if partsOfSmallHeight > 0:
			denseBlName = "FCBL_" + self.layerNumberStr
			peAllocationInfo = {}
			peAllocationInfo[PEs] = partsOfLargeWidth*partsOfSmallHeight
			peAllocationInfo[INACTI] = inActivationOfSmallHeight
			peAllocationInfo[INACTIALIG] = inActivationOfSmallHeightAlign
			peAllocationInfo[WEIGHT] = widthOfLargeWidth*heightOfSmallHeight
			peAllocationInfo[WEIGHTALIG] = widthOfLargeWidthAlign*heightOfSmallHeight
			peAllocationInfo[OUTACTI] = outActivationOfLargeWidth
			peAllocationInfo[OUTACTIALIG] = outActivationOfLargeWidthAlign
			self.inserDenseLayerMap(denseBlName, peAllocationInfo, clockIndex=2)
		# Add FCBR
		if partsOfSmallWidth > 0 and partsOfSmallHeight > 0:
			denseBrName = "FCBR_" + self.layerNumberStr
			peAllocationInfo = {}
			peAllocationInfo[PEs] = partsOfSmallWidth*partsOfSmallHeight
			peAllocationInfo[INACTI] = inActivationOfSmallHeight
			peAllocationInfo[INACTIALIG] = inActivationOfSmallHeightAlign
			peAllocationInfo[WEIGHT] = widthOfSmallWidth*heightOfSmallHeight
			peAllocationInfo[WEIGHTALIG] = widthOfSmallWidthAlign*heightOfSmallHeight
			peAllocationInfo[OUTACTI] = outActivationOfSmallWidth
			peAllocationInfo[OUTACTIALIG] = outActivationOfSmallWidthAlign
			self.inserDenseLayerMap(denseBrName, peAllocationInfo, clockIndex=3)

	def findWeightSplitSchemeWithFewestPartsOfWidth(self, ratio):
		'''
		Called by self.weightSplittingMap(). Find the optimum splitting scheme.
		When splitting weight, it has following rules:
		1. Muss keep the maximum MAC utilization rate
		2. Muss Memory limit must be met. Splitting weight will affect the size of input and output in each PE.
		3. Under the condition that the 1 and 2 are met, tend to select the splitting scheme, which split width 
			as few parts as possible, increasing data reuse.

		Args:
			ratio: the ratio between needed memory size for input+weight+output and limitation of SRAM.
					This ratio comes from the Step 1.
					Give Step 2 the information that the minimum parts it needs to split the weight.
		'''
		weightHeight, weightWidth = self.layerParameter
		maxSizeOfPart = math.floor(weightHeight*weightWidth/ratio)
		minPartsSplitting = math.ceil(weightHeight*weightWidth/maxSizeOfPart)
		# Obtain the optimized partsByWidth and partsByHeight
		maxPartsByWidth = math.ceil(weightWidth / MAC_ARRAY_COLUMN) + 1
		maxPartsByHeight = math.ceil(minPartsSplitting / self.scale)
		macUtilization = 0
		partsByWidth = maxPartsByWidth
		partsByHeight = maxPartsByHeight
		widthSplitInfo = None
		# Note the iteration order of partsByWidthTemp and partsByHeightTemp is from large --> small
		# It means we tend to select as few parts as possible 
		# for partsByWidthTemp in range(maxPartsByWidth, 0, -1):
		for partsByWidthTemp in range(maxPartsByWidth, 0, -1):
			minPartsByHeight = math.floor(minPartsSplitting / partsByWidthTemp)-1 if minPartsSplitting >= partsByWidthTemp else 0
			for partsByHeightTemp in range(maxPartsByHeight, minPartsByHeight, -1):
				# Get value of larger height when splitting along height
				weightHeightSplitInfoTemp = BasicOperation.oneDimSplitting(weightHeight, partsByHeightTemp)
				heightOfLargeHeight = weightHeightSplitInfoTemp[0]
				if heightOfLargeHeight % 4 != 0 or heightOfLargeHeight > FC_MAX_WEIGHT_HEIGHT:
					continue
				heightOfSmallHeight = weightHeightSplitInfoTemp[2]
				if heightOfSmallHeight % 4 != 0:
					continue
				# Get value of larger width when splitting along width
				weightWidthSplitInfoTemp = BasicOperation.oneDimSplitting(weightWidth, partsByWidthTemp)
				widthOfLargeWidth = weightWidthSplitInfoTemp[0]
				partsOfLargeWidth = weightWidthSplitInfoTemp[1]
				widthOfSmallWidth = weightWidthSplitInfoTemp[2]
				partsOfSmallWidth = weightWidthSplitInfoTemp[3]
				if widthOfLargeWidth % 16 != 0 or widthOfLargeWidth > FC_MAX_WEIGHT_WIDTH:
					continue
				if widthOfSmallWidth % 16 != 0:
					continue
				# Judge if the largest part meet the SRAM limitation
				inActivationSizeAlign = FcLayerMapper.inActivationAlign(heightOfLargeHeight)
				weightSizeAlign = FcLayerMapper.weightAlign(heightOfLargeHeight, widthOfLargeWidth)
				outActivationSizeAlign = FcLayerMapper.outActivationAlign(widthOfLargeWidth)
				largestPartSize = inActivationSizeAlign + weightSizeAlign + outActivationSizeAlign
				if largestPartSize > self.sramAvailable:
					continue
				if (largestPartSize/self.sramAvailable) < self.limitedSramUtilization:
					continue
				# Obtain MAC array utilication rate
				macUtilizationTemp = partsOfLargeWidth * FcLayerMapper.weightWidthAlign(widthOfLargeWidth) + \
									partsOfSmallWidth * FcLayerMapper.weightWidthAlign(widthOfSmallWidth)	
				if macUtilizationTemp != self.maxMacUtilizationAlongColumn:
					continue
				# Update macUtilization
				if partsByWidthTemp <= partsByWidth:
					if partsByWidth*partsByHeight >= partsByWidthTemp*partsByHeightTemp:
						macUtilization = macUtilizationTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = weightWidthSplitInfoTemp
		return macUtilization, partsByWidth, partsByHeight, widthSplitInfo

	def findWeightSplitSchemeWithFewestPartsOfHeight(self, ratio):
		'''
		Called by self.weightSplittingMap(). Find the optimum splitting scheme.
		When splitting weight, it has following rules:
		1. Muss keep the maximum MAC utilization rate
		2. Muss Memory limit must be met. Splitting weight will affect the size of input and output in each PE.
		3. Under the condition that the 1 and 2 are met, tend to select the splitting scheme, which split width 
			as few parts as possible, increasing data reuse.

		Args:
			ratio: the ratio between needed memory size for input+weight+output and limitation of SRAM.
					This ratio comes from the Step 1.
					Give Step 2 the information that the minimum parts it needs to split the weight.
		'''
		weightHeight, weightWidth = self.layerParameter
		maxSizeOfPart = math.floor(weightHeight*weightWidth/ratio)
		minPartsSplitting = math.ceil(weightHeight*weightWidth/maxSizeOfPart)
		# Obtain the optimized partsByWidth and partsByHeight
		maxPartsByWidth = math.ceil(weightWidth / MAC_ARRAY_COLUMN) + 1
		maxPartsByHeight = math.ceil(minPartsSplitting / self.scale)
		macUtilization = 0
		partsByWidth = maxPartsByWidth
		partsByHeight = maxPartsByHeight
		widthSplitInfo = None
		# Note the iteration order of partsByWidthTemp and partsByHeightTemp is from large --> small
		# It means we tend to select as few parts as possible 
		for partsByHeightTemp in range(maxPartsByHeight, 0, -1):
			minPartsByWidth = math.floor(minPartsSplitting / partsByHeightTemp)-1 if minPartsSplitting >= partsByHeightTemp else 0
			for partsByWidthTemp in range(maxPartsByWidth, minPartsByWidth, -1):
				if self.forSpiNNaker2:
				# Fit for 32 QPE in 4 double-blocks
					if partsByWidthTemp % 32 != 0:
						continue
				# Get value of larger height when splitting along height
				weightHeightSplitInfoTemp = BasicOperation.oneDimSplitting(weightHeight, partsByHeightTemp)
				heightOfLargeHeight = weightHeightSplitInfoTemp[0]
				if heightOfLargeHeight % 4 != 0 or heightOfLargeHeight > FC_MAX_WEIGHT_HEIGHT:
					continue
				heightOfSmallHeight = weightHeightSplitInfoTemp[2]
				if heightOfSmallHeight % 4 != 0:
					continue
				# Get value of larger width when splitting along width
				weightWidthSplitInfoTemp = BasicOperation.oneDimSplitting(weightWidth, partsByWidthTemp)
				widthOfLargeWidth = weightWidthSplitInfoTemp[0]
				partsOfLargeWidth = weightWidthSplitInfoTemp[1]
				widthOfSmallWidth = weightWidthSplitInfoTemp[2]
				partsOfSmallWidth = weightWidthSplitInfoTemp[3]
				if widthOfLargeWidth % 16 != 0 or widthOfLargeWidth > FC_MAX_WEIGHT_WIDTH:
					continue
				if widthOfSmallWidth % 16 != 0:
					continue
				# Judge if the largest part meet the SRAM limitation
				inActivationSizeAlign = FcLayerMapper.inActivationAlign(heightOfLargeHeight)
				weightSizeAlign = FcLayerMapper.weightAlign(heightOfLargeHeight, widthOfLargeWidth)
				outActivationSizeAlign = FcLayerMapper.outActivationAlign(widthOfLargeWidth)
				largestPartSize = inActivationSizeAlign + weightSizeAlign + outActivationSizeAlign
				if not self.forSpiNNaker2:
					largestPartSize += widthOfLargeWidth * MAC_OUT_BYTES
				if largestPartSize > self.sramAvailable:
					continue
				# if (largestPartSize/self.sramAvailable) < self.limitedSramUtilization:
				# 	continue
				# Obtain MAC array utilication rate
				macUtilizationTemp = partsOfLargeWidth * FcLayerMapper.weightWidthAlign(widthOfLargeWidth) + \
									partsOfSmallWidth * FcLayerMapper.weightWidthAlign(widthOfSmallWidth)	
				if macUtilizationTemp != self.maxMacUtilizationAlongColumn:
					continue
				# Update macUtilization
				if partsByHeightTemp <= partsByHeight:
					if partsByWidth*partsByHeight >= partsByWidthTemp*partsByHeightTemp:
						macUtilization = macUtilizationTemp
						partsByWidth = partsByWidthTemp
						partsByHeight = partsByHeightTemp
						widthSplitInfo = weightWidthSplitInfoTemp
		assert(widthSplitInfo is not None), "No splitting scheme for FC is found."
		return macUtilization, partsByWidth, partsByHeight, widthSplitInfo

	def findWeightSplitSchemeWithMostPartsOfWidth(self, ratio):
		'''
		Called by self.weightSplittingMap(). Find the optimum splitting scheme.
		When splitting weight, it has following rules:
		1. Muss keep the maximum MAC utilization rate.
		2. Muss Memory limit must be met. Splitting weight will affect the size of input and output in each PE.
		3. Under the condition that the 1 and 2 are met, splitting width as most parts as possible, decreasing the
			computation amount of ARM core.

		Args:
			ratio: float, the ratio between needed memory size for input+weight+output and limitation of SRAM.
					This ratio comes from the Step 1.
					Give information to Step 2 that the minimum parts it needs to split the weight
		'''
		weightHeight, weightWidth = self.layerParameter
		maxSizeOfPart = math.floor(weightHeight*weightWidth/ratio)
		minPartsSplitting = math.ceil(weightHeight*weightWidth/maxSizeOfPart)
		# Obtain the optimized partsByWidth and partsByHeight
		maxPartsByHeight = math.ceil(minPartsSplitting / self.scale)
		macUtilization = 0
		partsByWidth = math.ceil(weightWidth / MAC_ARRAY_COLUMN)
		partsByHeight = maxPartsByHeight
		widthSplitInfo = None
		# Note the iteration order of partsByWidthTemp and partsByHeightTemp is from large --> small
		# It means we tend to select as few parts as possible 
		# for partsByWidthTemp in range(maxPartsByWidth, 0, -1):
		minPartsByHeight = max(math.floor(minPartsSplitting / partsByWidth)-1, 1) - 1
		for partsByHeightTemp in range(maxPartsByHeight, minPartsByHeight, -1):
			# Get value of larger height when splitting along height
			# try:
			# 	weightHeightSplitInfoTemp = BasicOperation.oneDimSplitting(weightHeight, partsByHeightTemp)
			# except Exception as e:
			# 	print(e)
			# 	print("math.floor(minPartsSplitting / partsByWidth): ", math.floor(minPartsSplitting / partsByWidth))
			# 	print("minPartsSplitting: ", minPartsSplitting)
			# 	print("partsByWidth: ", partsByWidth)
			weightHeightSplitInfoTemp = BasicOperation.oneDimSplitting(weightHeight, partsByHeightTemp)
			heightOfLargeHeight = weightHeightSplitInfoTemp[0]
			if heightOfLargeHeight % 4 != 0 or heightOfLargeHeight > FC_MAX_WEIGHT_HEIGHT:
				continue
			heightOfSmallHeight = weightHeightSplitInfoTemp[2]
			if heightOfSmallHeight % 4 != 0:
				continue
			# Get value of larger width when splitting along width
			weightWidthSplitInfoTemp = BasicOperation.oneDimSplitting(weightWidth, partsByWidth)
			widthOfLargeWidth = weightWidthSplitInfoTemp[0]
			partsOfLargeWidth = weightWidthSplitInfoTemp[1]
			widthOfSmallWidth = weightWidthSplitInfoTemp[2]
			partsOfSmallWidth = weightWidthSplitInfoTemp[3]
			if widthOfLargeWidth % 16 != 0 or widthOfLargeWidth > FC_MAX_WEIGHT_WIDTH:
				continue
			if widthOfSmallWidth % 16 != 0:
				continue
			# Judge if the largest part meet the SRAM limitation
			inActivationSizeAlign = FcLayerMapper.inActivationAlign(heightOfLargeHeight)
			weightSizeAlign = FcLayerMapper.weightAlign(heightOfLargeHeight, widthOfLargeWidth)
			outActivationSizeAlign = FcLayerMapper.outActivationAlign(widthOfLargeWidth)
			largestPartSize = inActivationSizeAlign + weightSizeAlign + outActivationSizeAlign
			if largestPartSize > self.sramAvailable:
				continue
			# if (largestPartSize/self.sramAvailable) < self.limitedSramUtilization:
			# 	continue
			# Obtain MAC array utilication rate
			macUtilizationTemp = partsOfLargeWidth * FcLayerMapper.weightWidthAlign(widthOfLargeWidth) + \
								partsOfSmallWidth * FcLayerMapper.weightWidthAlign(widthOfSmallWidth)	
			if macUtilizationTemp != self.maxMacUtilizationAlongColumn:
				continue		
			# Update macUtilization
			if partsByWidth*partsByHeight >= partsByWidth*partsByHeightTemp:
				macUtilization = macUtilizationTemp
				partsByHeight = partsByHeightTemp
				widthSplitInfo = weightWidthSplitInfoTemp
		return macUtilization, partsByWidth, partsByHeight, widthSplitInfo

	# ==================================================================================================
	# 									Insert sum over layer
	# ==================================================================================================
	def getDenseExtraLayer(self):
		# Extract splitting information
		inNeurons = self.denseSplitInfo[FC_IN]
		if not isinstance(inNeurons, list):
			return
		if inNeurons[1] + inNeurons[3] == 1:
			return
		partsOfHeight = inNeurons[1] + inNeurons[3]
		outNeurons = self.denseSplitInfo[FC_OUT]
		# Generate input/output activation and weight
		inActivationDimOfExtraLayer = [outNeurons, partsOfHeight]
		weightDimOfExtraLayer = [0]
		outActivationDimOfExtraLayer = [outNeurons, 1]
		# Compute clocks
		computationClocks = outNeurons[0]
		# PEs
		requiredPEs = BasicOperation.getTotalPartsFromSplitInfo(outNeurons) * (partsOfHeight - 1)
		# Generate extra layer
		layerName = "FCE_" + self.layerNumberStr
		denseExtraLayerParameter = (LayerType.FCE, inActivationDimOfExtraLayer, weightDimOfExtraLayer,
			outActivationDimOfExtraLayer, computationClocks, requiredPEs)
		self.denseExtraLayerInfo = (layerName, denseExtraLayerParameter)

	# ==================================================================================================
	# 								Insert layer Mapping information
	# ==================================================================================================
	def inserDenseLayerMap(self, layerName, peAllocationInfo, clockIndex=0, invalid=False):
		'''
		Insert the splitting information into dictionary self.layerMapInfo

		Args:
			layerName: the allocation layer name.
			peAllocationInfo: information of all PEs with same size of input/weight/output.
			invalid: True for indicate this peAllocationInfo is invalid, which this splitting scheme doesn't meet the
						SRAM limitation.
		'''
		# Extract pe allocation infomation
		numOfPE = peAllocationInfo[PEs]
		inActivationSize = peAllocationInfo[INACTI]
		inActivationSizeAlign = peAllocationInfo[INACTIALIG]
		weightSize = peAllocationInfo[WEIGHT]
		weightSizeAlign = peAllocationInfo[WEIGHTALIG]
		outActivationSize = peAllocationInfo[OUTACTI]
		outActivationSizeAlign =peAllocationInfo[OUTACTIALIG]
		# invalid allocation, it means the layer will be remap again
		if invalid:
			layerName = layerName + "*"
		else:
			assert(inActivationSizeAlign + weightSizeAlign + outActivationSizeAlign <= self.sramAvailable), \
				"{}: SRAM overflow".format(layerName)
		# Get sram utilization
		requiredSramSize = inActivationSizeAlign + weightSizeAlign + outActivationSizeAlign
		sramUtilization = requiredSramSize/ self.sramAvailable
		fcSize = inActivationSize + weightSize + outActivationSize
		sramUtilizationNoAlign = fcSize / self.sramAvailable
		# Percentage (Alignment) calculation for input activation
		inActivationSizePercentage = inActivationSize / inActivationSizeAlign
		# Percentage (Alignment) calculation for weight
		weightSizePercentage = weightSize/weightSizeAlign
		# Percentage (Alignment) calculation for output activation
		outActivationSizePercentage = outActivationSize / outActivationSizeAlign
		# acceleration ratio
		speedRatio = self.theoreticalComputationClock[clockIndex] / self.actualComputationClock[clockIndex]
		# Insert layer mapping information
		self.layerMapInfo[layerName] = [numOfPE, \
			"{} [{}({:.2%})]".format(inActivationSizeAlign, \
										inActivationSize, \
										inActivationSizePercentage), \
			"{} [{}({:.2%})]".format(weightSizeAlign, \
										weightSize, \
										weightSizePercentage), \
			"{} [{}({:.2%})]".format(outActivationSizeAlign, \
										outActivationSize, \
										outActivationSizePercentage), \
			"{} [{}({:.2f})]".format(self.actualComputationClock[clockIndex], \
										self.theoreticalComputationClock[clockIndex], \
										speedRatio),
			"{:.2f}[{:.2f}]".format(sramUtilization, sramUtilizationNoAlign)]	

	# ==================================================================================================
	# 										Computation clocks
	# ==================================================================================================
	def clockComputation(self):
		heights = [0, 0]
		widths = [0, 0]
		if isinstance(self.denseSplitInfo[FC_IN], list):
			heights[0] = self.denseSplitInfo[FC_IN][0]
			heights[1] = self.denseSplitInfo[FC_IN][2]
		else:
			heights[0] = self.denseSplitInfo[FC_IN]
			heights[1] = 0
		if isinstance(self.denseSplitInfo[FC_OUT], list):
			widths[0] = self.denseSplitInfo[FC_OUT][0]
			widths[1] = self.denseSplitInfo[FC_OUT][2]
		else:
			widths[0] = self.denseSplitInfo[FC_OUT]
			widths[1] = 0
		index = 0
		for height in heights:
			for width in widths:
				self.theoreticalComputationClock[index] = FcLayerMapper.denseTheoreticalClocks(height, width)
				self.actualComputationClock[index] = FcLayerMapper.denseActualClocks(height, width)
				index = index + 1

	@staticmethod
	def denseTheoreticalClocks(height, width):
		return height * width

	@staticmethod
	def denseActualClocks(height, width):
		return height * math.ceil(width/MAC_ARRAY_COLUMN)	

	# ==================================================================================================
	# 										Common function
	#  	These functions could be upgraded when understanding the hardware deeper.
	# 	These functions are all about alignment problem
	# ==================================================================================================
	def getMacUtilization(self, weightWidth):
		return BasicOperation.ceilOf(weightWidth, MAC_ARRAY_COLUMN)

	@staticmethod
	def weightWidthAlign(weightWidth):
		return BasicOperation.ceilOf(weightWidth, MAC_ARRAY_COLUMN)

	@staticmethod
	def inActivationAlign(inNeuron):
		'''
		Input Activation is a vector. For MAC Array, the input alignment is MAC_ARRAY_ROW vectors. 
		'''
		return FcLayerMapper.align4(inNeuron) * MAC_ARRAY_ROW
		# return FcLayerMapper.align16(inNeuron * MAC_ARRAY_ROW)

	@staticmethod
	def weightAlign(inNeuron, outNeuron):
		'''
		Weight is a matrix with dimension [width, height].
			width = outNeuron --> need align to MAC_ARRAY_COLUMN
			height = inNeuron 
		'''
		# BasicOperation.ceilOf(outNeuron, MAC_ARRAY_COLUMN)
		return FcLayerMapper.align4(inNeuron) * FcLayerMapper.align16(outNeuron)

	@staticmethod
	def outActivationAlign(outNeuron):
		# outActivationSizeAlign = BasicOperation.ceilOf(outNeuron, MAC_ARRAY_COLUMN) * MAC_OUT_BYTES
		# outActivationSizeAlign = outActivationSizeAlign * MAC_ARRAY_ROW	
		return 	FcLayerMapper.align16(outNeuron) * MAC_OUT_BYTES * MAC_ARRAY_ROW

	@staticmethod
	def align16(size):
		return BasicOperation.ceilOf(size, 16)

	@staticmethod
	def align4(size):
		return BasicOperation.ceilOf(size, 4)