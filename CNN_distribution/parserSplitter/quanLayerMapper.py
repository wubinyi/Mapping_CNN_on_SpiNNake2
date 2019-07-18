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
# from parserSplitter.actiLayerMapper import *


class QuanType(Enum):
	TenserQuant = auto()


class QuanLayerMapper():
	def __init__(self, preLayerName, preLayerSplitInfo):
		self.mapperReset(preLayerName, preLayerSplitInfo)

	def mapperReset(self, preLayerName, preLayerSplitInfo):
		'''
		If pre-layer is POOL, "preLayerSplitInfo" contains width, height, channel.
		If pre-layer is CONV-ACTI, "preLayerSplitInfo" contains width, height, channel.
		If pre-layer is FC-ACTI, "preLayerSplitInfo" contains only width, height (height=1).
		'''
		assert("POOL" in preLayerName or "ACTI" in preLayerName or "CONV" in preLayerName), \
			"pre-layer should be POOL or ACTI"
		self.preLayerName = preLayerName
		self.preLayerSplitInfo = preLayerSplitInfo
		self.quanSplitInfo = {}
		self.theoreticalComputationClock = [0] * 8
		self.actualComputationClock = [0] * 8

	def getSplitInfo(self):
		return self.quanSplitInfo

	def getClocks(self):
		return max(self.actualComputationClock)

	def map(self):
		self.convertSplitInfo()
		self.clockComputation()

	def convertSplitInfo(self):
		if "POOL" in self.preLayerName:
			self.quanSplitInfo[QUAN_WIDTH] = self.preLayerSplitInfo[POOL_OUT_WIDTH]
			self.quanSplitInfo[QUAN_HEIGHT] = self.preLayerSplitInfo[POOL_OUT_HEIGHT]
			self.quanSplitInfo[QUAN_CHANNEL] = self.preLayerSplitInfo[POOL_CHANNEL]
		elif "ACTI" in self.preLayerName:
			self.quanSplitInfo[QUAN_WIDTH] = self.preLayerSplitInfo[ACTI_WIDTH]
			self.quanSplitInfo[QUAN_HEIGHT] = self.preLayerSplitInfo[ACTI_HEIGHT]
			if ACTI_CHANNEL in self.preLayerSplitInfo:
				self.quanSplitInfo[QUAN_CHANNEL] = self.preLayerSplitInfo[ACTI_CHANNEL]
		elif "CONV" in self.preLayerName:
			self.quanSplitInfo[QUAN_WIDTH] = self.preLayerSplitInfo[CONV_OUT_WIDTH]
			self.quanSplitInfo[QUAN_HEIGHT] = self.preLayerSplitInfo[CONV_OUT_HEIGHT]
			self.quanSplitInfo[QUAN_CHANNEL] = self.preLayerSplitInfo[CONV_OUT_CHANNEL]

	# ==================================================================================================
	# 										Computation clocks
	# ==================================================================================================
	def clockComputation(self):
		widths = BasicOperation.getSplitDimFromSplitInfo(self.quanSplitInfo[QUAN_WIDTH])
		if isinstance(self.quanSplitInfo[QUAN_HEIGHT], list):
			heights = BasicOperation.getSplitDimFromSplitInfo(self.quanSplitInfo[QUAN_HEIGHT])
		else:
			heights = [1, 0]
		if QUAN_CHANNEL in self.quanSplitInfo:
			channels = BasicOperation.getSplitDimFromSplitInfo(self.quanSplitInfo[QUAN_CHANNEL])
		else:
			channels = [1, 0]
		index = 0
		for height in heights:
			for width in widths:
				for channel in channels:
					self.theoreticalComputationClock[index] = QuanLayerMapper.actiTheoreticalClocks(width, height, channel)
					self.actualComputationClock[index] = QuanLayerMapper.actiActualClocks(width, height, channel)
					index = index + 1

	@staticmethod
	def actiTheoreticalClocks(inWidth, inHeight, inChannel):
		return inWidth * inHeight * inChannel

	@staticmethod
	def actiActualClocks(inWidth, inHeight, inChannel):
		return inWidth * inHeight * inChannel