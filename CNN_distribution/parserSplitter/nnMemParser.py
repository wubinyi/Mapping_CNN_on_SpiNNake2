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

class NNMemParser:

	@staticmethod
	def inputLayerMem(layerParameter):
		'''
		@input
			layerParameter = ("input activation DIM", "output activation DIM")
				"input activation DIM" = [I_Width, I_Height, I_Channels]
				"output activation DIM" = [O_Width, O_Height, O_Channels]
		@Return
			memory need for Input layer [IN, WEIGHT, OUT]
		'''
		inActivationDim, outActivationDim = layerParameter
		inSize = BasicOperation.listInnerProduct(inActivationDim)
		weightSize = 0
		outSize = BasicOperation.listInnerProduct(outActivationDim)
		return [inSize, weightSize, outSize]

	def paddingLayerMem(layerParameter):
		'''
		@input
			layerParameter = ("input activation DIM", "output activation DIM")
				"input activation DIM" = [I_Width, I_Height, I_Channels]
				"output activation DIM" = [O_Width, O_Height, O_Channels]
		@Return
			memory need for padding layer [IN, WEIGHT, OUT]
		'''
		inActivationDim, outActivationDim, _ = layerParameter
		inSize = BasicOperation.listInnerProduct(inActivationDim)
		weightSize = 0
		outSize = BasicOperation.listInnerProduct(outActivationDim)
		return [inSize, weightSize, outSize]

	@staticmethod
	def convolutionLayerMem(layerParameter):
		'''
		@input
			layerParameter = ("input activation DIM", "filter DIM", stride, "output activation DIM")
				"input activation DIM" = [I_Width, I_Height, I_Channels]
				"filter DIM" = [F_Width, F_Height, F_Input_channels, F_Output_channels]
				"output activation DIM" = [O_Width, O_Height, O_Channels]
		@output
			memory need for convolutional layer [IN, WEIGHT, OUT]
		'''
		layerParameter, poolStride = layerParameter
		inActivationDim, filtersDim, stride, outActivationDim = layerParameter
		# assert (len(inActivationDim) == 3), "Dimension of activation should be 3"
		# assert (len(filtersDim) == 4), "Dimension of filters should be 4"
		# assert (inActivationDim[2] == filtersDim[2]), "Input channels should be equal"
		inSize = BasicOperation.listInnerProduct(inActivationDim)
		weightSize = BasicOperation.listInnerProduct(filtersDim)
		outSize = BasicOperation.listInnerProduct(outActivationDim) * MAC_OUT_BYTES
		return [inSize, weightSize, outSize]

	@staticmethod
	def poolingLayerMem(layerParameter):
		'''
		@input
			layerParameter = ("input activation DIM", "pooling filter DIM", stride, "output activation DIM")
				"input activation DIM" = [I_Width, I_Height, I_Channels]
				"pooling filter DIM" = [P_Width, P_Height]
				"output activation DIM" = [O_Width, O_Height, O_Channels]
		@output
			memory need for pooling layer [IN, WEIGHT, OUT]
		'''
		inActivationDim, poolingDim, stride, poolType, outActivationDim = layerParameter
		inSize = BasicOperation.listInnerProduct(inActivationDim)
		weightSize = 0
		outSize = BasicOperation.listInnerProduct(outActivationDim)
		return [inSize, weightSize, outSize]

	@staticmethod
	def denseLayerMem(layerParameter):
		'''
		@input
			layerParameter = (inNeuron, outNeuron)
		@output
			memory need for dense layer [IN, WEIGHT, OUT]
		'''
		inNeuron, outNeuron = layerParameter
		inSize = inNeuron
		weightSize = inNeuron * outNeuron
		outSize = outNeuron * MAC_OUT_BYTES
		return [inSize, weightSize, outSize]

	def quantizationLayerMem(layerParameter):
		inActivationDim, _ = layerParameter
		inSize = BasicOperation.listInnerProduct(inActivationDim) * 4
		weightSize = 0
		outSize = BasicOperation.listInnerProduct(inActivationDim)
		return [inSize, weightSize, outSize]

	def activationLayerMem(layerParameter):
		'''
		For activation layer, after using the input result, the output result will
		directly overwrite the input result
		'''
		inActivationDim, outActivationDim, actiType = layerParameter
		inSize = BasicOperation.listInnerProduct(inActivationDim)
		weightSize = 0
		outSize = inSize
		return [inSize, weightSize, outSize]

	@staticmethod
	def layerMemParser(layerTypeParameter):
		'''
		layerTypeParameter = (layerType, layerParameter)
		'''
		layerType, layerParameter = layerTypeParameter
		if LayerType.INPUT == layerType:
			return NNMemParser.inputLayerMem(layerParameter)
		elif LayerType.PADD == layerType:
			return NNMemParser.paddingLayerMem(layerParameter)
		elif LayerType.CONV == layerType:
			return NNMemParser.convolutionLayerMem(layerParameter)
		elif LayerType.POOL == layerType:
			return NNMemParser.poolingLayerMem(layerParameter)
		elif LayerType.FC == layerType:
			return NNMemParser.denseLayerMem(layerParameter)
		elif LayerType.QUAN == layerType:
			return NNMemParser.quantizationLayerMem(layerParameter)
		elif layerType.ACTI == layerType:
			return NNMemParser.activationLayerMem(layerParameter)
		elif layerType.PADD_SC == layerType:
			(inActiDim, outputActiDim, (paddDim, overlapSize), shortcutSource) = layerParameter
			paddLayerParameter = (inActiDim, outputActiDim, (paddDim, overlapSize))
			return NNMemParser.paddingLayerMem(paddLayerParameter)
		elif LayerType.CONV_SC == layerType:
			((inActiDim, filterDim, stride, outActiDim), poolStride, shortcutSource) = layerParameter
			convLayerParameter = ((inActiDim, filterDim, stride, outActiDim), poolStride)
			return NNMemParser.convolutionLayerMem(convLayerParameter)
		elif LayerType.QUAN_SC == layerType:
			(inActiDim, outActiDim, shortcutSource) = layerParameter
			quanLayerParameter = (inActiDim, outActiDim)
			return NNMemParser.quantizationLayerMem(quanLayerParameter)
		elif LayerType.ACTI_SC == layerType:
			(inActiDim, outActiDim, actiType, shortcutSource) = layerParameter
			actiLayerParameter = (inActiDim, outActiDim, actiType)
			return NNMemParser.activationLayerMem(actiLayerParameter)
		elif LayerType.MAT_ELE == layerType:
			return NNMemParser.matrixElemLayerMem(layerParameter)
		else:
			assert(False), "Unknown layer type: {}".format(layerType)

	@staticmethod
	def matrixElemLayerMem(layerParameter):
		(inActivationDim, inActivationDim, shortcutSource) = layerParameter
		inSize = BasicOperation.listInnerProduct(inActivationDim)
		weightSize = 0
		outSize = inSize
		return [inSize, weightSize, outSize]