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
from enum import Enum, auto

class ActivationType(Enum):
	NONE = auto()
	ReLU = auto()


class ActiLayerMapper():
	def __init__(self, preLayerName, preLayerSplitInfo):
		self.mapperReset(preLayerName, preLayerSplitInfo)

	def mapperReset(self, preLayerName, preLayerSplitInfo):
		assert("CONV" in preLayerName or "FC" in preLayerName or "QUAN" in preLayerName or "MAT_ELE" in preLayerName), \
			"pre-layer should be CONV or FC"
		self.preLayerName = preLayerName
		self.preLayerSplitInfo = preLayerSplitInfo
		self.actiSplitInfo = {}
		if "CONV" in self.preLayerName:
			self.theoreticalComputationClock = [0]*8
			self.actualComputationClock = [0]*8
		elif "FC" in preLayerName:
			self.theoreticalComputationClock = [0]*4
			self.actualComputationClock = [0]*4
		elif "QUAN" in preLayerName:
			self.theoreticalComputationClock = [0]*8
			self.actualComputationClock = [0]*8	
		elif "MAT_ELE" in preLayerName:
			self.theoreticalComputationClock = [0]*8
			self.actualComputationClock = [0]*8			

	def getSplitInfo(self):
		return self.actiSplitInfo

	def getClocks(self):
		return max(self.actualComputationClock)

	def map(self):
		if "CONV" in self.preLayerName:
			self.actiSplitInfo[ACTI_WIDTH] = self.preLayerSplitInfo[CONV_OUT_WIDTH]
			self.actiSplitInfo[ACTI_HEIGHT] = self.preLayerSplitInfo[CONV_OUT_HEIGHT]
			self.actiSplitInfo[ACTI_CHANNEL] = self.preLayerSplitInfo[CONV_OUT_CHANNEL]
		elif "FC" in self.preLayerName:
			self.actiSplitInfo[ACTI_WIDTH] = self.preLayerSplitInfo[FC_OUT]
			self.actiSplitInfo[ACTI_HEIGHT] = 1
		elif "QUAN" in self.preLayerName:
			self.actiSplitInfo[ACTI_WIDTH] = self.preLayerSplitInfo[QUAN_WIDTH]
			self.actiSplitInfo[ACTI_HEIGHT] = self.preLayerSplitInfo[QUAN_HEIGHT]
			if QUAN_CHANNEL in self.preLayerSplitInfo:
				self.actiSplitInfo[ACTI_CHANNEL] = self.preLayerSplitInfo[QUAN_CHANNEL]
		elif "MAT_ELE" in self.preLayerName:
			self.actiSplitInfo[ACTI_WIDTH] = self.preLayerSplitInfo[MAT_ELE_WIDTH]
			self.actiSplitInfo[ACTI_HEIGHT] = self.preLayerSplitInfo[MAT_ELE_HEIGHT]
			if MAT_ELE_CHANNEL in self.preLayerSplitInfo:
				self.actiSplitInfo[ACTI_CHANNEL] = self.preLayerSplitInfo[MAT_ELE_CHANNEL]
		self.clockComputation()

	# ==================================================================================================
	# 										Computation clocks
	# ==================================================================================================
	def clockComputation(self):
		widths = BasicOperation.getSplitDimFromSplitInfo(self.actiSplitInfo[ACTI_WIDTH])
		if isinstance(self.actiSplitInfo[ACTI_HEIGHT], list):
			heights = BasicOperation.getSplitDimFromSplitInfo(self.actiSplitInfo[ACTI_HEIGHT])
		else:
			heights = [1, 0]
		if ACTI_CHANNEL in self.actiSplitInfo:
			channels = BasicOperation.getSplitDimFromSplitInfo(self.actiSplitInfo[ACTI_CHANNEL])
		else:
			channels = [1]
		index = 0
		for height in heights:
			for width in widths:
				for channel in channels:
					self.theoreticalComputationClock[index] = ActiLayerMapper.actiTheoreticalClocks(width, height, channel)
					self.actualComputationClock[index] = ActiLayerMapper.actiActualClocks(width, height, channel)
					index = index + 1

	@staticmethod
	def actiTheoreticalClocks(inWidth, inHeight, inChannel):
		return inWidth * inHeight * inChannel

	@staticmethod
	def actiActualClocks(inWidth, inHeight, inChannel):
		return inWidth * inHeight * inChannel
		