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

class NNParser(object):
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

	def insertConvLayer(self, inActivationDim, layerParameter, layerParameters, poolStride=1):
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
		# Determine if this convolution is valid
		if paddType == PaddType.SAME:
			maxOutActivationWidth = math.ceil((inActivationDim[0] - 1) / stride) + 1
			maxOutActivationHeight = math.ceil((inActivationDim[1] - 1) / stride) + 1
			assert(maxOutActivationWidth >= inActivationDim[0]), "Unvalid convolution operation along width"
			assert(maxOutActivationHeight >= inActivationDim[1]), "Unvalid convolution operation along height"
		# Insert Padding layer
		paddOutputActivationDim = inActivationDim.copy()
		if paddType == PaddType.SAME:
			paddingLayerName = "{}_{}".format(LayerType.PADD.name, self.layerCounter)
			widthPaddingSize = stride * inActivationDim[0] - stride + filterDim[0] - inActivationDim[0]
			paddOutputActivationDim[0] = paddOutputActivationDim[0] + widthPaddingSize
			heightPaddingSize = stride * inActivationDim[1] - stride + filterDim[1] - inActivationDim[1]
			paddOutputActivationDim[1] = paddOutputActivationDim[1] + heightPaddingSize
			paddDim = [math.ceil(widthPaddingSize/2), math.floor(widthPaddingSize/2), \
						math.ceil(heightPaddingSize/2), math.floor(heightPaddingSize/2)]
			overlapSize = filterDim[0] - stride
			paddingLayerParameter = (inActivationDim.copy(), paddOutputActivationDim.copy(), (paddDim, overlapSize))
			layerParameters[paddingLayerName] = (LayerType.PADD, paddingLayerParameter)
		# Insert Convolution layer
		convLayerName = "{}_{}".format(LayerType.CONV.name, self.layerCounter)
		convOutActivationDim = inActivationDim.copy()
		if paddType == PaddType.SAME:
			convOutActivationDim[2] = filterDim[3]
		else:
			if (inActivationDim[0] - filterDim[0]) % stride != 0:
				BasicOperation.customPrintT("{} loss data along width when doing pooling".format(convLayerName))
			if (inActivationDim[1] - filterDim[1]) % stride != 0:
				BasicOperation.customPrintT("{} loss data along height when doing pooling".format(convLayerName))
			convOutActivationDim[0] = math.floor((inActivationDim[0] - filterDim[0]) / stride) + 1
			convOutActivationDim[1] = math.floor((inActivationDim[1] - filterDim[1]) / stride) + 1
			convOutActivationDim[2] = filterDim[3]
		convLayerParameter = ((paddOutputActivationDim.copy(), filterDim, stride, convOutActivationDim.copy()), poolStride)
		layerParameters[convLayerName] = (LayerType.CONV, convLayerParameter)
		# Insert Activation layer
		actiLayerName = "{}_{}".format(LayerType.ACTI.name, self.layerCounter)
		actiLayerParameter = (convOutActivationDim.copy(), convOutActivationDim.copy(), actiType)
		layerParameters[actiLayerName] = (LayerType.ACTI, actiLayerParameter)
		return convOutActivationDim

	def insertPoolLayer(self, inActivationDim, layerParameter, layerParameters):
		'''
		@Input
			layerParameter = ([P_Width, P_Height], stride, poolType)
		@Ouput
			layerParameters
				|---> append NAME:("input activation DIM", "pooling filter DIM", stride, poolType, "output activation DIM") 
						for POOL
		'''
		# Insert Pooling layer
		poolingDim, stride, poolType = layerParameter
		poolLayerName = "{}_{}".format(LayerType.POOL.name, self.layerCounter)
		outputActivationDim = []
		if (inActivationDim[0] - poolingDim[0]) % stride != 0:
			BasicOperation.customPrintT("{} loss data along width when doing pooling".format(poolLayerName))
		if (inActivationDim[1] - poolingDim[1]) % stride != 0:
			BasicOperation.customPrintT("{} loss data along height when doing pooling".format(poolLayerName))
		outputActivationDim.append(math.floor((inActivationDim[0] - poolingDim[0]) / stride) + 1)
		outputActivationDim.append(math.floor((inActivationDim[1] - poolingDim[1]) / stride) + 1)
		outputActivationDim.append(inActivationDim[2])
		inActivationDim[0] = (outputActivationDim[0] - 1) * stride + poolingDim[0]
		inActivationDim[1] = (outputActivationDim[1] - 1) * stride + poolingDim[1]
		poolLayerParameter = (inActivationDim.copy(), poolingDim, stride, poolType, outputActivationDim.copy())
		layerParameters[poolLayerName] = (LayerType.POOL, poolLayerParameter)
		return outputActivationDim

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

	def insertQuantizationLayer(self, inActivationDim, layerParameters):
		'''
		@Input
			CONV-inActivationDim = 
		@Ouput
			layerParameters
				|---> append (inNeuron, outNeuron) for FC
		'''
		quantizationLayerName = "{}_{}".format(LayerType.QUAN.name, self.layerCounter)
		quantizationLayerParameter = (inActivationDim, inActivationDim)
		layerParameters[quantizationLayerName] = (LayerType.QUAN, quantizationLayerParameter)

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
			if LayerType.INPUT == layerType:
				outputActivationDim = self.insertInputLayer(layerParameter, layerParameters)
			if LayerType.CONV == layerType:
				# Currently, only support pool_stride = pool_width or pool_height
				nextLayerName = layerNames[index+1]
				nextLayerTypeParameter = self.nnHyperparameter[nextLayerName]
				nextlayerType, nextLayerParameter = nextLayerTypeParameter
				BasicOperation.customPrintF("------> nextlayerType: {}".format(nextlayerType))
				if LayerType.POOL == nextlayerType:
					poolingDim, stride, poolType = nextLayerParameter
					assert(poolingDim[0] == stride and poolingDim[1] == stride), "Limitation of Pooling"
					outputActivationDim = self.insertConvLayer(outputActivationDim, layerParameter, layerParameters, stride)
				else:
					outputActivationDim = self.insertConvLayer(outputActivationDim, layerParameter, layerParameters)
				self.insertQuantizationLayer(outputActivationDim, layerParameters)
			if LayerType.POOL == layerType:
				outputActivationDim = self.insertPoolLayer(outputActivationDim, layerParameter, layerParameters)
			if LayerType.FC == layerType:
				outputActivationDim = self.insertFcLayer(outputActivationDim, layerParameter, layerParameters)
				self.insertQuantizationLayer(outputActivationDim, layerParameters)
			self.layerCounter = self.layerCounter + 1
		return layerParameters
		# Layer reorder
		# reorderLayerParameter = self.poolQuanReorder(layerParameters)
		# return reorderLayerParameter