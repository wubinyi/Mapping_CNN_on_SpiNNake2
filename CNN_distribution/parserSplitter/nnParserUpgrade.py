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
import math
from actiLayerMapper import ActivationType

class NNParserUpgrade(object):
	'''
	INPUT: ([Width, Height, Channels])
	CONV:  ([F_Width, F_Height, Channels, Output_Channels], stride, same)
	POOL:  ([P_Width, P_Height], stride)
	FC: (output)
	'''
	def __init__(self, nnHyperparameter):
		'''
		nnHyperparameter = {layerType_0:layerParameter_0, layerType_1:layerParameter_1, layerType_2:layerParameter_2, ...}
		'''
		self.nnHyperparameter = nnHyperparameter
		self.layerCounter = 0

	def insertInputLayer(self, layerParameter, layerParameters):
		'''
		@Input
			layerParameter = ([Width, Height, Channels])
		@Ouput
			layerParameters 
				|---> append NAME:("input activation dimension", "output activation dimension")
		@Return
			the output activation size
		'''
		# Insert Input layer
		inputLayerName = "{}_{}".format(LayerType.INPUT.name, self.layerCounter)
		# Get input dimension of layer
		inActivationDim = layerParameter.copy()
		# Get output dimension of layer
		outputActivationDim = inActivationDim.copy()
		inputLayerParameter = (inActivationDim, outputActivationDim)
		layerParameters[inputLayerName] = (LayerType.INPUT, inputLayerParameter)
		return outputActivationDim

	def insertConvLayer(self, inActivationDim, layerParameter, layerParameters, poolStride=([1,1], 1), 
		shortCut=False, shortcutSource=None):
		'''
		@Input
			layerParameter = ([F_Width, F_Height, Channels, Output_Channels], stride, paddType, actiType)
		@Ouput
			layerParameters
				|---> append NAME:("input activation DIM", "output activation DIM", [paddDim, overlapSize]) for PADD if necessary 
				|		(paddType=PaddType.SAME)
				|---> append NAME:("input activation DIM", "filter DIM", stride, "output activation DIM") for CONV
				|---> append NAME:("input activation DIM", "output activation DIM", actiType) for CONV
		@Note
			Padding -> Convolution
		'''
		filterDim, stride, paddType, actiType = layerParameter
		inActivationChannel = inActivationDim[2]
		assert(inActivationChannel == filterDim[2]), \
			"input channels of input-activation [{}] and filter [{}] should be equal!".\
			format(inActivationChannel, filterDim[2])
		assert(filterDim[0] == filterDim[1]), "Only support convolution for width=height"
		# Insert Padding layer
		paddOutputActivationDim = inActivationDim.copy()
		if paddType == PaddType.SAME and filterDim[0] != 1:
			# (in + p - f) / s + 1 = out  ->  in + p - f = out - s  ->  in + p = out - s + f (in = out)
			if shortCut:
				paddingLayerName = "{}_{}".format(LayerType.PADD_SC.name, self.layerCounter)
			else:
				paddingLayerName = "{}_{}".format(LayerType.PADD.name, self.layerCounter)
			widthPaddingSize = filterDim[0] - stride
			paddOutputActivationDim[0] = paddOutputActivationDim[0] + widthPaddingSize
			heightPaddingSize = filterDim[1] - stride
			paddOutputActivationDim[1] = paddOutputActivationDim[1] + heightPaddingSize
			paddDim = [math.ceil(widthPaddingSize/2), widthPaddingSize//2, \
						math.ceil(heightPaddingSize/2), heightPaddingSize//2]
			overlapSize = filterDim[0] - stride
			if shortCut:
				paddingLayerParameter = (inActivationDim.copy(), paddOutputActivationDim.copy(), (paddDim, overlapSize), shortcutSource)
				layerParameters[paddingLayerName] = (LayerType.PADD_SC, paddingLayerParameter)
			else:
				paddingLayerParameter = (inActivationDim.copy(), paddOutputActivationDim.copy(), (paddDim, overlapSize))
				layerParameters[paddingLayerName] = (LayerType.PADD, paddingLayerParameter)
		# Insert Convolution layer
		if shortCut:
			convLayerName = "{}_{}".format(LayerType.CONV_SC.name, self.layerCounter)
		else:
			convLayerName = "{}_{}".format(LayerType.CONV.name, self.layerCounter)
		convOutActivationDim = []
		if paddType == PaddType.SAME:
			convOutActivationDim.append(math.ceil(inActivationDim[0] / stride))
			convOutActivationDim.append(math.ceil(inActivationDim[0] / stride))
			convOutActivationDim.append(filterDim[3])
		else:
			if (inActivationDim[0] - filterDim[0]) % stride != 0:
				BasicOperation.customPrintT("{} loss data along width when doing pooling".format(convLayerName))
			if (inActivationDim[1] - filterDim[1]) % stride != 0:
				BasicOperation.customPrintT("{} loss data along height when doing pooling".format(convLayerName))
			convOutActivationDim.append(math.ceil((inActivationDim[0] - poolingDim[0] + 1) / stride))
			convOutActivationDim.append(math.ceil((inActivationDim[1] - poolingDim[1] + 1) / stride))
			convOutActivationDim.append(filterDim[3])
		if shortCut:
			convLayerParameter = ((paddOutputActivationDim.copy(), filterDim, stride, 
				convOutActivationDim.copy()), poolStride, shortcutSource)
			layerParameters[convLayerName] = (LayerType.CONV_SC, convLayerParameter)
		else:
			convLayerParameter = ((paddOutputActivationDim.copy(), filterDim, stride, 
				convOutActivationDim.copy()), poolStride)
			layerParameters[convLayerName] = (LayerType.CONV, convLayerParameter)
		# Insert Activation layer
		if ActivationType.NONE == actiType:
			return convOutActivationDim
		if shortCut:
			actiLayerName = "{}_{}".format(LayerType.ACTI_SC.name, self.layerCounter)
			actiLayerParameter = (convOutActivationDim.copy(), convOutActivationDim.copy(), actiType, shortcutSource)
			layerParameters[actiLayerName] = (LayerType.ACTI_SC, actiLayerParameter)
		else:
			actiLayerName = "{}_{}".format(LayerType.ACTI.name, self.layerCounter)
			actiLayerParameter = (convOutActivationDim.copy(), convOutActivationDim.copy(), actiType)
			layerParameters[actiLayerName] = (LayerType.ACTI, actiLayerParameter)
		# Insert Quantization layer
		self.insertQuantizationLayer(convOutActivationDim.copy(), layerParameters, 
			shortCut=shortCut, shortcutSource=shortcutSource)
		return convOutActivationDim

	def insertFcLayer(self, inActivationDim, layerParameter, layerParameters):
		'''
		@Input
			layerParameter = (outNeuron)
		@Ouput
			layerParameters
				|---> append (inNeuron, outNeuron) for FC
		'''
		# Insert FC layer
		outNeuron, actiType = layerParameter
		if isinstance(inActivationDim, list):
			inNeuron = BasicOperation.listInnerProduct(inActivationDim)
		else:
			inNeuron = inActivationDim
		denseLayerName = "{}_{}".format(LayerType.FC.name, self.layerCounter)
		denseLayerParameter = (inNeuron, outNeuron)
		layerParameters[denseLayerName] = (LayerType.FC, denseLayerParameter)
		# Insert activation layer
		actiLayerName = "{}_{}".format(LayerType.ACTI.name, self.layerCounter)
		actiLayerParameter = (outNeuron, outNeuron, actiType)
		layerParameters[actiLayerName] = (LayerType.ACTI, actiLayerParameter)		
		return outNeuron

	def insertQuantizationLayer(self, inActivationDim, layerParameters, shortCut=False, shortcutSource=None):
		'''
		@Input
			CONV-inActivationDim = 
		@Ouput
			layerParameters
				|---> append (inNeuron, outNeuron) for FC
		'''
		if shortCut:
			quantizationLayerName = "{}_{}".format(LayerType.QUAN_SC.name, self.layerCounter)
			quantizationLayerParameter = (inActivationDim, inActivationDim, shortcutSource)
			layerParameters[quantizationLayerName] = (LayerType.QUAN_SC, quantizationLayerParameter)
		else:
			quantizationLayerName = "{}_{}".format(LayerType.QUAN.name, self.layerCounter)
			quantizationLayerParameter = (inActivationDim, inActivationDim)
			layerParameters[quantizationLayerName] = (LayerType.QUAN, quantizationLayerParameter)

	def insertPoolLayer(self, inActivationDim, layerParameter, layerParameters):
		'''
		@Input
			layerParameter = ([P_Width, P_Height], stride, poolType)
		@Ouput
			layerParameters
				|---> append NAME:("input activation DIM", "pooling filter DIM", stride, poolType, "output activation DIM") 
						for POOL
		'''
		poolingDim, stride, paddType, poolType = layerParameter
		assert(poolingDim[0] == poolingDim[1]), "Only support pooling for width=height"
		# Insert Padding layer
		if poolingDim[0] != stride and paddType == PaddType.SAME:
			paddOutputActivationDim = inActivationDim.copy()
			# (in + p - f) / s + 1 = out  ->  in + p - f = out - s  ->  in + p = out - s + f (in = out)
			paddingLayerName = "{}_{}".format(LayerType.PADD.name, self.layerCounter)
			widthPaddingSize = poolingDim[0] - stride
			paddOutputActivationDim[0] = paddOutputActivationDim[0] + widthPaddingSize
			heightPaddingSize = poolingDim[1] - stride
			paddOutputActivationDim[1] = paddOutputActivationDim[1] + heightPaddingSize
			paddDim = [math.ceil(widthPaddingSize/2), math.floor(widthPaddingSize/2), \
						math.ceil(heightPaddingSize/2), math.floor(heightPaddingSize/2)]
			overlapSize = poolingDim[0] - stride
			paddingLayerParameter = (inActivationDim.copy(), paddOutputActivationDim.copy(), (paddDim, overlapSize))
			layerParameters[paddingLayerName] = (LayerType.PADD, paddingLayerParameter)
		else:
			paddOutputActivationDim = inActivationDim.copy()
		# Insert Pooling layer
		poolLayerName = "{}_{}".format(LayerType.POOL.name, self.layerCounter)
		outputActivationDim = []
		if PaddType.SAME == paddType:
			outputActivationDim.append(math.ceil(inActivationDim[0] / stride))
			outputActivationDim.append(math.ceil(inActivationDim[1] / stride))
			outputActivationDim.append(inActivationDim[2])
		elif PaddType.VALID == paddType:
			outputActivationDim.append(math.ceil((inActivationDim[0] - poolingDim[0] + 1) / stride))
			outputActivationDim.append(math.ceil((inActivationDim[1] - poolingDim[1] + 1) / stride))
			outputActivationDim.append(inActivationDim[2])			
		poolLayerParameter = (paddOutputActivationDim.copy(), poolingDim, stride, poolType, outputActivationDim.copy())
		layerParameters[poolLayerName] = (LayerType.POOL, poolLayerParameter)
		return outputActivationDim

	def poolQuanReorder(self, layerParameters):
		# Reoder the layer name
		# Exchange "QUAN" and "POOL"
		reorderLayerKeys = []
		layerKeys = list(layerParameters.keys())
		for index in range(len(layerKeys)):
			layerName = layerKeys[index]
			if "QUAN" in layerName:
				if index+1 >= len(layerKeys):
					# If last layer is QUAN, add it
					reorderLayerKeys.append(layerName)
					continue
				nextLayerName = layerKeys[index+1]
				if "POOL" in nextLayerName:
					reorderLayerKeys.append(nextLayerName)
				reorderLayerKeys.append(layerName)
				continue
			if "POOL" in layerName:
				continue
			reorderLayerKeys.append(layerName)
		# According to layerName list reorder layerParameter
		reorderLayerParameter = {}
		preLayerName = "None"
		preLayerTypeParameter = None
		for layerName in reorderLayerKeys:
			currentLayerTypeParameter = layerParameters[layerName]
			if "POOL" in preLayerName and "QUAN" in layerName:
				preLayerType, preLayerParameter = preLayerTypeParameter
				_, _, _, _, outputActivationDim = preLayerParameter
				quanLayerTypeParameter = (LayerType.QUAN, (outputActivationDim.copy(), outputActivationDim.copy()))
			else:
				quanLayerTypeParameter = layerParameters[layerName]
			reorderLayerParameter[layerName] = quanLayerTypeParameter
			preLayerName = layerName
			preLayerTypeParameter = layerParameters[layerName]
		return reorderLayerParameter

	def nnParser(self):
		# Parser Layers
		layerParameters = {}
		outputActivationDim = None
		layerNames = list(self.nnHyperparameter.keys())
		for index in range(len(layerNames)):
			layerTypeParameter = self.nnHyperparameter[layerNames[index]]
			layerType, layerParameter = layerTypeParameter
			# Input layer
			if LayerType.INPUT == layerType:
				outputActivationDim = self.insertInputLayer(layerParameter, layerParameters)
			# Convolutional layer
			elif LayerType.CONV == layerType:
				if (index+1) < len(layerNames):
					nextLayerName = layerNames[index+1]
					nextLayerTypeParameter = self.nnHyperparameter[nextLayerName]
					nextlayerType, nextLayerParameter = nextLayerTypeParameter
				else:
					nextlayerType = None
				if LayerType.POOL == nextlayerType:
					poolingDim, stride, paddType, poolType = nextLayerParameter
					outputActivationDim = self.insertConvLayer(outputActivationDim, layerParameter, layerParameters, (poolingDim, stride))
				else:
					outputActivationDim = self.insertConvLayer(outputActivationDim, layerParameter, layerParameters)
			# Pooling layer
			elif LayerType.POOL == layerType:
				outputActivationDim = self.insertPoolLayer(outputActivationDim, layerParameter, layerParameters)
			# Fully-connected layer
			elif LayerType.FC == layerType:
				outputActivationDim = self.insertFcLayer(outputActivationDim, layerParameter, layerParameters)
				self.insertQuantizationLayer(outputActivationDim, layerParameters)
			elif LayerType.CONV_BLOCK == layerType:
				outputActivationDim = self.insertConvBlockLayers(outputActivationDim, layerParameter, layerParameters)
			else:
				assert(False), "Unknown layer type: {}".format(layerType)
			self.layerCounter = self.layerCounter + 1
		# Layer reorder
		# reorderLayerParameter = self.poolQuanReorder(layerParameters)
		# return reorderLayerParameter
		return layerParameters

	def insertConvBlockLayers(self, inActivationDim, layerParameter, layerParameters):
		convBlockParam, convStride, paddType, actiType = layerParameter
		shortcutSource = list(layerParameters)[-1]
		# Insert first CONV of conv block
		convParam = (convBlockParam[0], convStride, paddType, actiType)
		outputActivationDim = self.insertConvLayer(inActivationDim, convParam, layerParameters)
		# Insert mittle CONVs of conv block
		for convLayerIndex in range(1, len(convBlockParam)-1):
			self.layerCounter = self.layerCounter + 1
			convParam = (convBlockParam[convLayerIndex], 1, paddType, actiType)
			outputActivationDim = self.insertConvLayer(outputActivationDim, convParam, layerParameters)
		# Insert last CONV of conv block
			# Add conv layer without activation layer but with quantization
		self.layerCounter = self.layerCounter + 1
		convParam = (convBlockParam[-1], 1, paddType, ActivationType.NONE)
		outputActivationDim = self.insertConvLayer(outputActivationDim, convParam, layerParameters)
			# Add projection shortcut
		channelIncreaseFlag = outputActivationDim[2] != inActivationDim[2]
		sizeDecreaseFlag = outputActivationDim[0] != inActivationDim[0]
		noProjShortCut = None
		if channelIncreaseFlag or sizeDecreaseFlag:
			noProjShortCut = False
			if channelIncreaseFlag and sizeDecreaseFlag:
				convParam = ([1,1,inActivationDim[2], outputActivationDim[2]], convStride, 
					PaddType.SAME, ActivationType.NONE)
				shortCutOutput = self.insertConvLayer(inActivationDim, convParam, layerParameters, 
					shortCut=True, shortcutSource=shortcutSource)
				shortcutSource = list(layerParameters)[-1]
			elif channelIncreaseFlag:
				convParam = ([1,1,inActivationDim[2], outputActivationDim[2]], 1, PaddType.SAME, 
					ActivationType.NONE)
				shortCutOutput = self.insertConvLayer(inActivationDim, convParam, layerParameters,
					shortCut=True, shortcutSource=shortcutSource)
				shortcutSource = list(layerParameters)[-1]
			elif sizeDecreaseFlag:
				convParam = ([1,1,inActivationDim[2], outputActivationDim[2]], convStride, 
					PaddType.SAME, ActivationType.NONE)
				shortCutOutput = self.insertConvLayer(inActivationDim, convParam, layerParameters, 
					shortCut=True, shortcutSource=shortcutSource)
				shortcutSource = list(layerParameters)[-1]
			else:
				assert(False), "Why no projection shortcut"
		else:
			noProjShortCut = True
			shortCutOutput = inActivationDim.copy()
			# Add element-wise addition
		assert(shortCutOutput==outputActivationDim), \
			"After shortcut, the dimension should be the same {}-{}".format(outputActivationDim, inActivationDim)
		self.insertMatrixElementLayer(outputActivationDim, layerParameters, shortcutSource)
			# Add activation layer
		if ActivationType.NONE == actiType:
			return outputActivationDim
		actiLayerName = "{}_{}".format(LayerType.ACTI.name, self.layerCounter)
		actiLayerParameter = (outputActivationDim.copy(), outputActivationDim.copy(), actiType)
		layerParameters[actiLayerName] = (LayerType.ACTI, actiLayerParameter)
			# Add Quantization layer
		self.insertQuantizationLayer(outputActivationDim.copy(), layerParameters, 
			shortCut=False, shortcutSource=shortcutSource)
		return outputActivationDim

	def insertMatrixElementLayer(self, inActivationDim, layerParameters, shortcutSource):
		matrixElementLayerName = "{}_{}".format(LayerType.MAT_ELE.name, self.layerCounter)
		matrixElementLayerParameter = (inActivationDim, inActivationDim, shortcutSource)
		layerParameters[matrixElementLayerName] = (LayerType.MAT_ELE, matrixElementLayerParameter)