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
from poolLayerMapper import *


class PaddLayerMapper():
	def __init__(self, layerParameter, postLayerSplitInfo):
		self.mapperReset(layerParameter, postLayerSplitInfo)

	def mapperReset(self, layerParameter, postLayerSplitInfo):
		self.layerParameter = layerParameter
		self.postLayerSplitInfo = postLayerSplitInfo
		self.paddSplitInfo = {}

	def getSplitInfo(self):
		return self.paddSplitInfo

	def map(self):
		inActivationDim, outActivationDim, paddDimOverlapSize = self.layerParameter
		self.paddSplitInfo[PADD_DIM_OVERLAP] = paddDimOverlapSize
		if self.postLayerSplitInfo != None:
			paddDim, _ = paddDimOverlapSize
			# CONV
			if CONV_IN_WIDTH in self.postLayerSplitInfo:
				postOutActiWidth = self.postLayerSplitInfo[CONV_IN_WIDTH]
				self.paddSplitInfo[PADD_OUT_WIDTH] = postOutActiWidth
				self.paddSplitInfo[PADD_IN_WIDTH] = self.paddOutToIn(postOutActiWidth, paddDim[:2])
				postOutActiHeight = self.postLayerSplitInfo[CONV_IN_HEIGHT]
				self.paddSplitInfo[PADD_OUT_HEIGHT] = postOutActiHeight
				self.paddSplitInfo[PADD_IN_HEIGHT] = self.paddOutToIn(postOutActiHeight, paddDim[2:])
				postOutActiChannel = self.postLayerSplitInfo[CONV_IN_CHANNEL]
				self.paddSplitInfo[PADD_OUT_CHANNEL] = postOutActiChannel
				self.paddSplitInfo[PADD_IN_CHANNEL] = postOutActiChannel
			# POOL
			elif POOL_IN_WIDTH in self.postLayerSplitInfo:
				postOutActiWidth = self.postLayerSplitInfo[POOL_IN_WIDTH]
				self.paddSplitInfo[PADD_OUT_WIDTH] = postOutActiWidth
				self.paddSplitInfo[PADD_IN_WIDTH] = self.paddOutToIn(postOutActiWidth, paddDim[:2])
				postOutActiHeight = self.postLayerSplitInfo[POOL_IN_HEIGHT]
				self.paddSplitInfo[PADD_OUT_HEIGHT] = postOutActiHeight
				self.paddSplitInfo[PADD_IN_HEIGHT] = self.paddOutToIn(postOutActiHeight, paddDim[2:])
				postOutActiChannel = self.postLayerSplitInfo[POOL_CHANNEL]
				self.paddSplitInfo[PADD_OUT_CHANNEL] = postOutActiChannel
				self.paddSplitInfo[PADD_IN_CHANNEL] = postOutActiChannel
			else:
				assert(False), "Unsupport postLayerSplitInfo"
		else:
			self.paddSplitInfo[PADD_IN_WIDTH] = inActivationDim[0]
			self.paddSplitInfo[PADD_IN_HEIGHT] = inActivationDim[1]
			self.paddSplitInfo[PADD_IN_CHANNEL] = inActivationDim[2]
			self.paddSplitInfo[PADD_OUT_WIDTH] = outActivationDim[0]
			self.paddSplitInfo[PADD_OUT_HEIGHT] = outActivationDim[1]
			self.paddSplitInfo[PADD_OUT_CHANNEL] = outActivationDim[2]

	def paddOutToIn(self, outWidthHeightDim, paddDim):
		inWidthHeightDim = []
		splitParts = BasicOperation.getSplitPartsFromSplitInfo(outWidthHeightDim)
		if 0 not in splitParts:
			# Head
			inWidthHeightDim.append(outWidthHeightDim[0]-paddDim[0])
			inWidthHeightDim.append(1)
			# Main parts
			if splitParts[0] > 1:
				inWidthHeightDim.append(outWidthHeightDim[0])
				inWidthHeightDim.append(outWidthHeightDim[1]-1)
			if splitParts[1] > 1:
				inWidthHeightDim.append(outWidthHeightDim[2])
				inWidthHeightDim.append(outWidthHeightDim[3]-1)	
			# Last
			inWidthHeightDim.append(outWidthHeightDim[2]-paddDim[1])
			inWidthHeightDim.append(1) 
		else:
			# Head
			if splitParts[0] == 1:
				inWidthHeightDim.append(outWidthHeightDim[0]-paddDim[0]-paddDim[1])
			else:
				inWidthHeightDim.append(outWidthHeightDim[0]-paddDim[0])
			inWidthHeightDim.append(1)
			# Main parts
			if splitParts[0] > 2:
				inWidthHeightDim.append(outWidthHeightDim[0])
				inWidthHeightDim.append(outWidthHeightDim[1]-2)
			# Last
			if splitParts[0] > 1:
				inWidthHeightDim.append(outWidthHeightDim[0]-paddDim[1])
				inWidthHeightDim.append(1)
		return inWidthHeightDim