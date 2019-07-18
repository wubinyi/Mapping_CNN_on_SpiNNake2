import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
s2sFolderPath = os.path.join(projectFolder, "spiNNaker2Simulator")
if s2sFolderPath not in sys.path:
	sys.path.insert(0, s2sFolderPath)
distriFolderPath = os.path.join(projectFolder, "distributor")
if distriFolderPath not in sys.path:
	sys.path.insert(0, distriFolderPath)
icproValidatePath = os.path.join(projectFolder, "icproValidate")
if icproValidatePath not in sys.path:
	sys.path.insert(0, icproValidatePath)
parserSplitterPath = os.path.join(projectFolder, "parserSplitter")
if parserSplitterPath not in sys.path:
	sys.path.insert(0, parserSplitterPath)

from enum import Enum, auto
import datetime
from nn2SpiNNaker2 import NNModelToSpiNNaker2
from nnModel import NNModel
from nnGeneral import LayerType, PaddType
from actiLayerMapper import ActivationType
from poolLayerMapper import PoolType
from spiNNakerSimulatorGeneral import *
import math

DIS_LAYER_NAME = "dlNam"
DIS_LAYER_TOTAL_CLKS = "dlTtClk"
DIS_LAYER_USED_PES = "dlPes"
DIS_LAYER_MLA_CLKS = "dlMClk"
DIS_LAYER_MLA_COMPUTE_CLKS = "dlMCmpClk"
DIS_LAYER_MLA_WR_OUTACTI_SRAM_CLKS = "dlMWOClk"
DIS_LAYER_SIMULATION_SECOND = "dlSiSec"


# class QpeDistriScheme(Enum):
# 	# No data reuse
# 	QPE_DISTRI_NO_DATA_REUSED = auto()
# 	# weight-reuse
# 	QPE_DISTRI_DATA_REUSED_WEIGHT = auto()
# 	# inActivation-reuse
# 	QPE_DISTRI_DATA_REUSED_INACTI = auto()
# 	# inActivation-reuse or filter-reuse
# 	QPE_DISTRI_DATA_REUSED_DYNAMIC = auto()
# 	# inActivation-reuse, fully PE utilization
# 	QPE_DISTRI_DATA_REUSED_INACTI_FULL = auto()
# 	# filter-reuse, fully PE utilization
# 	QPE_DISTRI_DATA_REUSED_WEIGHT_FULL = auto()
# 	# inActivation-reuse or filter-reuse, fully PE utilization
# 	QPE_DISTRI_DATA_REUSED_DYNAMIC_FULL = auto()


class PeState():
	IDLE = auto()
	READ_DRAM = auto()
	READ_DRAM_INACTI = auto()
	READ_DRAM_WEIGHT = auto()
	MLA_EXE_READY = auto()
	MLA_EXE_DRAM = auto()
	WRITE_OUTACTI = auto()


class DistributionGeneralClass():
	def __init__(self, printFlag):
		self.printFlag = printFlag
		# self.clockCounter = 0
		# self.simulationTimeSecond = 0
		# self.beginTime = 0
		# self.endTime = 0
		# self.compClockCounter = 0
		# self.wrOutActiIntoSramClockCounter = 0
		# self.finishedPeCounter = 0
		self.layerDistriResultReset()
		self.layerName = None
		self.distriResults = {}

	def getDistriResults(self):
		return self.distriResults

	def layerDistriResultReset(self):
		self.beginTime = 0
		self.endTime = 0
		self.clockCounter = 0
		self.simulationTimeSecond = 0
		self.mlaRunClks = 0
		self.compClockCounter = 0
		self.wrOutActiIntoSramClockCounter = 0
		self.finishedPeCounter = 0	

	def updateModelSplitInfoParameter(self, modelLayersSplitInfo, modelLayersParameter):
		self.modelLayersSplitInfo = modelLayersSplitInfo
		self.modelLayersParameter = modelLayersParameter

	def customAssert(self, condition, content):
		# funcName = sys._getframe().f_code.co_name
		# lineNumber = sys._getframe().f_lineno
		# fileName = sys._getframe().f_code.co_filename
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def customPrint(self, content):
		if self.printFlag:
			print(content)

	def printInfo(self, content):
		if self.printFlag:
			print("---> {:<7}: {}-{}".format(self.clockCounter, self.layerName, content))

	def logBeginTime(self):
		self.beginTime = self.timeConverToSecond(datetime.datetime.now())

	def logEndTime(self):
		self.endTime = self.timeConverToSecond(datetime.datetime.now())
		self.simulationTimeSecond = self.endTime - self.beginTime
		self.printInfo("Simulation time (second): {}".format(self.simulationTimeSecond))

	def timeConverToSecond(self, currentTime):
		hours = currentTime.hour
		minutes = currentTime.minute
		second = currentTime.second
		totalSecond = (hours * 60 + minutes) * 60 + second
		return totalSecond	

	def getPartLengthFromSplitInfo(self, splitInfo, index):
		if isinstance(splitInfo, list):
			# totalParts = BasicOperation.getTotalPartsFromSplitInfo(splitInfo)
			# assert(index <= totalParts), "index-{} muss not larger than {}".format(index, totalParts)
			if index < splitInfo[1]:
				return splitInfo[0]
			else:
				return splitInfo[2]
		else:
			# assert(index == 0), "index-{} should be 1".format(index)
			return splitInfo

	def getTotalPartsFromSplitInfo(self, splitInfo):
		if isinstance(splitInfo, list):
			# assert(len(splitInfo) != 0), "BasicOperation.listCeilOf: splitInfo contains 0 element"
			# assert(len(splitInfo)%2 == 0), "BasicOperation.listCeilOf: splitInfo contains {} elements".format(len(splitInfo))
			return splitInfo[1]+splitInfo[3]		
		else:
			return 1

	@staticmethod
	def vgg(sramBlocks=1, decreaseSize=False, forSpiNNaker2=False, logEn=False, printFlag=True):
		nnModel = NNModel()
		nnModel.addLayer(LayerType.INPUT, ([224, 224, 3]))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 3, 64], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 64, 64], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 64, 128], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 128], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 256], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 512], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
		nnModel.addLayer(LayerType.FC, (4096, ActivationType.ReLU))
		nnModel.addLayer(LayerType.FC, (4096, ActivationType.ReLU))
		nnModel.addLayer(LayerType.FC, (1024, ActivationType.ReLU))
		# print("nnHyperparameter: ", nnModel.getNNModel())
		nnModelToSpiNNaker2 = NNModelToSpiNNaker2(nnModel.getNNModel(), sramBlocks=sramBlocks, 
													logEn=logEn, printFlag=printFlag)
		modelLayersSplitInfo = nnModelToSpiNNaker2.run(decreaseSize=decreaseSize, forSpiNNaker2=forSpiNNaker2)
		modelLayersParameter = nnModelToSpiNNaker2.getLayerParameters()
		return modelLayersSplitInfo, modelLayersParameter

	@staticmethod
	def resNet50(sramBlocks=1, decreaseSize=False, forSpiNNaker2=False, logEn=False, printFlag=True):
		nnModel = NNModel()
		nnModel.addLayer(LayerType.INPUT, ([224, 224, 3]))
		# CONV_1
		nnModel.addLayer(LayerType.CONV, ([7, 7, 3, 64], 2, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.POOL, ([3, 3], 2, PaddType.SAME, PoolType.MAXPOOL))
		# # conv_2x
		convBlockParam = [[1, 1, 64, 64], [3, 3, 64, 64], [1, 1, 64, 256]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 64, 64], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 64, 64], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 64, 256], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 256, 64], [3, 3, 64, 64], [1, 1, 64, 256]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 64], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 64, 64], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 64, 256], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 256, 64], [3, 3, 64, 64], [1, 1, 64, 256]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 64], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 64, 64], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 64, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# # CONV_3x
		convBlockParam = [[1, 1, 256, 128], [3, 3, 128, 128], [1, 1, 128, 512]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 2, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 128, 512], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 512, 128], [3, 3, 128, 128], [1, 1, 128, 512]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 512, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 128, 512], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 512, 128], [3, 3, 128, 128], [1, 1, 128, 512]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 512, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 128, 512], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 512, 128], [3, 3, 128, 128], [1, 1, 128, 512]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 512, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 128], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 128, 512], 1, PaddType.SAME, ActivationType.ReLU))
		# # CONV_4x
		convBlockParam = [[1, 1, 512, 256], [3, 3, 256, 256], [1, 1, 256, 1024]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 2, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 512, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 1024], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 1024, 256], [3, 3, 256, 256], [1, 1, 256, 1024]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 1024, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 1024], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 1024, 256], [3, 3, 256, 256], [1, 1, 256, 1024]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 1024, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 1024], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 1024, 256], [3, 3, 256, 256], [1, 1, 256, 1024]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 1024, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 1024], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 1024, 256], [3, 3, 256, 256], [1, 1, 256, 1024]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 1024, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 1024], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 1024, 256], [3, 3, 256, 256], [1, 1, 256, 1024]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 1024, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 256, 1024], 1, PaddType.SAME, ActivationType.ReLU))
		# # CONV_5x
		convBlockParam = [[1, 1, 1024, 512], [3, 3, 512, 512], [1, 1, 512, 2048]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 2, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 1024, 512], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 512, 2048], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 2048, 512], [3, 3, 512, 512], [1, 1, 512, 2048]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 2048, 512], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 512, 2048], 1, PaddType.SAME, ActivationType.ReLU))
		convBlockParam = [[1, 1, 2048, 512], [3, 3, 512, 512], [1, 1, 512, 2048]]
		nnModel.addLayer(LayerType.CONV_BLOCK, (convBlockParam, 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 2048, 512], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
		# nnModel.addLayer(LayerType.CONV, ([1, 1, 512, 2048], 1, PaddType.SAME, ActivationType.ReLU))
		nnModel.addLayer(LayerType.POOL, ([7, 7], 7, PaddType.SAME, PoolType.AVGPOOL))
		nnModel.addLayer(LayerType.FC, (1024, ActivationType.ReLU))
		nnModelToSpiNNaker2 = NNModelToSpiNNaker2(nnModel.getNNModel(), sramBlocks=sramBlocks, 
													logEn=logEn, printFlag=printFlag, upgrade=True)
		modelLayersSplitInfo = nnModelToSpiNNaker2.run(decreaseSize=decreaseSize, forSpiNNaker2=forSpiNNaker2)
		modelLayersParameter = nnModelToSpiNNaker2.getLayerParameters()
		return modelLayersSplitInfo, modelLayersParameter

	def targetToPairedPeIdGenerator(self, targetPeId, pePairShift=1):
		pairedPeId = targetPeId[0:2].copy()
		pairedPeId.append((targetPeId[2]+pePairShift)%NUM_PES_IN_QPE)
		return pairedPeId

	def pairedToTargetPeIdGenerator(self, pairedPeId, pePairShift=1):
		targetPeId = pairedPeId[0:2].copy()
		targetPeId.append((pairedPeId[2]+NUM_PES_IN_QPE-pePairShift)%NUM_PES_IN_QPE)
		return targetPeId

	def getTriBlockIndexAndBasicQpeId(self, peID):
		'''
		This Basic-QPE is not the HOST-QPE, but just used to calculate the index of QPE
		'''
		yId = peID[0]
		xId = peID[1]
		if xId < NUM_QPES_X_AXIS_HALF and yId < NUM_QPES_Y_AXIS_HALF:
			return 0, TRIBLOCK_BASIC[0]
		elif xId >= NUM_QPES_X_AXIS_HALF and yId < NUM_QPES_Y_AXIS_HALF:
			return 1, TRIBLOCK_BASIC[1]
		elif xId < NUM_QPES_X_AXIS_HALF and yId >= NUM_QPES_Y_AXIS_HALF:
			return 2, TRIBLOCK_BASIC[2]
		elif xId >= NUM_QPES_X_AXIS_HALF and yId >= NUM_QPES_Y_AXIS_HALF:
			return 3, TRIBLOCK_BASIC[3]
		else:
			self.customAssert(False, "Unkown X and Y axis of qpeId")

	def getTriBlockBasicQpeId(self, triBlockIndex):
		'''
		This Basic-QPE is not the HOST-QPE, but just used to calculate the index of QPE
		'''
		if 0 == triBlockIndex:
			return TRIBLOCK_BASIC[0]
		elif 1 == triBlockIndex:
			return TRIBLOCK_BASIC[1]
		elif 2 == triBlockIndex:
			return TRIBLOCK_BASIC[2]
		elif 3 == triBlockIndex:
			return TRIBLOCK_BASIC[3]
		else:
			self.customAssert(False, "Unkown triBlock index")	

	def getHostQpeId(self, douTriBlockIndex):
		# -> Cooperate with GeneralClass.hostQpeId()
		# -> Cooperate with SpiNNaker2TriBlock.hostTaskDistributor()
		return TRIBLOCK_HOST[douTriBlockIndex]

	def whichDouBlock(self, qpeId):
		qpeId = qpeId[Y_AXIS_INDEX:Z_AXIS_INDEX]
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if qpeId in DOUBLOCK_QPE_ID_LIST[douBlockIndex]:
				return douBlockIndex

	def align4(self, size):
		return int(math.ceil(size/4) * 4)

	def align16(self, size):
		return int(math.ceil(size/16) * 16)

	def alignMlaRow(self, size):
		return int(math.ceil(size/MLA_MAC_ROW_MOD) * MLA_MAC_ROW_MOD)

	def alignMlaColumn(self, size):
		return int(math.ceil(size/MLA_MAC_COLUMN_MOD) * MLA_MAC_COLUMN_MOD)

	def convInActiDimToSize(self, inActiDim, operatorFusion):
		if operatorFusion:
			return inActiDim[0] * inActiDim[1] * inActiDim[2]
		else:
			return self.align16(inActiDim[0]) * inActiDim[1] * inActiDim[2]

	def convWeightDimToSize(self, weightDim, operatorFusion):
		return self.align16(weightDim[0] * weightDim[1] * weightDim[2] * self.alignMlaRow(weightDim[3]))

	def convOutActiDimToSize(self, outActiDim, operatorFusion):
		if operatorFusion:
			return outActiDim[0] * outActiDim[1] * outActiDim[2]
		else:
			return self.align4(outActiDim[0]) * outActiDim[1] * outActiDim[2] * 4

	def fcInActiDimToSize(self, inActiDim, operatorFusion):
		if operatorFusion:
			return inActiDim[0] * inActiDim[1]
		else:
			return self.align4(inActiDim[0]) * self.alignMlaRow(inActiDim[1])

	def fcWeightDimToSize(self, weightDim, operatorFusion):
		return self.alignMlaColumn(weightDim[0]) * self.align4(weightDim[1])

	def fcOutActiDimToSize(self, outActiDim, operatorFusion):
		return self.alignMlaColumn(outActiDim[0]) * self.alignMlaRow(outActiDim[1]) * 4


if __name__ == "__main__":
	DistributionGeneralClass.vgg(decreaseSize=True, forSpiNNaker2=True)