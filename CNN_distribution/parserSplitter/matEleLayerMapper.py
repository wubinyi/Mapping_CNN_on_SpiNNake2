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

class MatEleLayerMapper():
	def __init__(self, preLayerName, preLayerSplitInfo):
		self.mapperReset(preLayerName, preLayerSplitInfo)

	def mapperReset(self, preLayerName, preLayerSplitInfo):
		assert("CONV" in preLayerName), "pre-layer should be CONV"
		self.preLayerName = preLayerName
		self.preLayerSplitInfo = preLayerSplitInfo
		self.matEleSplitInfo = {}
		if "CONV" in preLayerName:
			self.theoreticalComputationClock = [0]*8
			self.actualComputationClock = [0]*8

	def getSplitInfo(self):
		return self.matEleSplitInfo

	def getClocks(self):
		return max(self.actualComputationClock)

	def map(self):
		if "CONV" in self.preLayerName:
			# print("---------preLayerSplitInfo: {}".format(self.preLayerSplitInfo))
			self.matEleSplitInfo[MAT_ELE_WIDTH] = self.preLayerSplitInfo[CONV_OUT_WIDTH]
			self.matEleSplitInfo[MAT_ELE_HEIGHT] = self.preLayerSplitInfo[CONV_OUT_HEIGHT]
			if CONV_OUT_CHANNEL in self.preLayerSplitInfo:
				self.matEleSplitInfo[MAT_ELE_CHANNEL] = self.preLayerSplitInfo[CONV_OUT_CHANNEL]
		self.clockComputation()

	# ==================================================================================================
	# 										Computation clocks
	# ==================================================================================================
	def clockComputation(self):
		widths = BasicOperation.getSplitDimFromSplitInfo(self.matEleSplitInfo[MAT_ELE_WIDTH])
		if isinstance(self.matEleSplitInfo[MAT_ELE_HEIGHT], list):
			heights = BasicOperation.getSplitDimFromSplitInfo(self.matEleSplitInfo[MAT_ELE_HEIGHT])
		else:
			heights = [1, 0]
		if MAT_ELE_CHANNEL in self.matEleSplitInfo:
			channels = BasicOperation.getSplitDimFromSplitInfo(self.matEleSplitInfo[MAT_ELE_CHANNEL])
		else:
			channels = [1]
		index = 0
		for height in heights:
			for width in widths:
				for channel in channels:
					self.theoreticalComputationClock[index] = MatEleLayerMapper.matEleTheoreticalClocks(width, height, channel)
					self.actualComputationClock[index] = MatEleLayerMapper.matEleActualClocks(width, height, channel)
					index = index + 1

	@staticmethod
	def matEleTheoreticalClocks(inWidth, inHeight, inChannel):
		return inWidth * inHeight * inChannel

	@staticmethod
	def matEleActualClocks(inWidth, inHeight, inChannel):
		return inWidth * inHeight * inChannel