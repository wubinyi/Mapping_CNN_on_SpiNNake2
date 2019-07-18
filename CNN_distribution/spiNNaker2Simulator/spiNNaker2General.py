import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
from enum import Enum, auto
from parserSplitter.convLayerMapper import *
from parserSplitter.fcLayerMapper import *

# DRAM BW (16 Bytes * 100 MHz) < NOC (16 Bytes * 250 MHz) = PE_SRAM (16 Bytes * 250 MHz)
NUM_PE_IN_QPE = 4
BYTE_TO_BIT = 8

NOC_FREQ = 500
PE_FREQ = 250
DRAM_FREQ = 250

DRAM_CLOCKS_PER_OPER = 2

NOC_MAX_DATA_WIDTH_BYTES = 16
DRAM_MAX_DATA_WIDTH_BYTES = 16

# NOC_CLOCKS_TO_PE_CLOCKS = PE_FREQ/NOC_FREQ

PES_ON_QPE = 4
MAC_ARRAY_SIZE = 16 * 4
MAC_OPER_PER_CLOCK = 2
MEGA_TO_GIGA = 1000
MEGA = 1000000

PAIR_PE_RUN_CLOCK_SCALE = (128+32) / 128

class OperationType(Enum):
	CONV = auto()
	FC = auto()

class AcceleratorType(Enum):
	QPE = auto()
	SpiNNaker2 = auto()

# Block info
OPER_TYPE = "operationType"
INPUT_DIM = "inputDim"
WEGIHT_DIM = "weightDim"
OUTPUT_DIM = "outputDim"
SIDE_OPER_TYPE = "sideOperationType"

# 
WR_CLOCKS = "wrClocks"
WR_INACTI_CLOCKS = "wrInActiClocks"
WR_WEIGHT_CLOCKS = "wrFilterClocks"
WR_OUTACTI_CLOCKS = "wrOutActiClocks"
ML_CLOCKS = "mlClocks"
ML_REAL_CLOCKS = "mlRealClocks"
RD_CLOCKS = "rdClocks"
TOTAL_USED_PE = "totalUsedPes"
TOTAL_RUN_PE = "totalRunPes"


class ConvDistriScheme(Enum):
	# No data reuse
	QPE_DISTRI_NO_DATA_REUSED = auto()
	# weight-reuse
	QPE_DISTRI_DATA_REUSED_WEIGHT = auto()
	# inActivation-reuse
	QPE_DISTRI_DATA_REUSED_INACTI = auto()
	# inActivation-reuse or filter-reuse
	QPE_DISTRI_DATA_REUSED_DYNAMIC = auto()
	# inActivation-reuse, fully PE utilization
	QPE_DISTRI_DATA_REUSED_INACTI_FULL = auto()
	# filter-reuse, fully PE utilization
	QPE_DISTRI_DATA_REUSED_WEIGHT_FULL = auto()
	# inActivation-reuse or filter-reuse, fully PE utilization
	QPE_DISTRI_DATA_REUSED_DYNAMIC_FULL = auto()


class SpiNNakerBasic():
	# CONV
	@staticmethod
	def convInActiDimAlign(inActiDim):
		inActiDim = inActiDim.copy()
		inActiWidth = inActiDim[0]
		inActiDim[0] = ConvLayerMapper.convInActivationWidthAlign(inActiWidth, 0)
		return inActiDim

	@staticmethod
	def convInActiSizeAlign(inActiDim):
		# inActiDim = inActiDim.copy()
		# inActiDim = SpiNNakerBasic.convInActiDimAlign(inActiDim)
		return ConvLayerMapper.convInActivationSizeAlign(inActiDim[0], inActiDim[1], inActiDim[2], 0)

	@staticmethod
	def convFilterDimAlign(filterDim):
		filterDim = filterDim.copy()
		filterOutChannel = filterDim[3]
		filterDim[3] = ConvLayerMapper.convFilterOutChannelAlign(filterOutChannel)
		return filterDim

	@staticmethod
	def convFilterSizeAlign(filterDim):
		# filterDim = filterDim.copy()
		# filterDim = SpiNNakerBasic.convFilterDimAlign(filterDim)
		return ConvLayerMapper.convFilterSizeAlign(filterDim[0], filterDim[1], filterDim[2], filterDim[3])

	@staticmethod
	def convOutActiDimAlign(outActiDim):
		outActiDim = outActiDim.copy()
		outActiWidth = outActiDim[0]
		outActiDim[0] = ConvLayerMapper.convOutActivationWidthAlign(outActiWidth)
		return outActiDim

	@staticmethod
	def convOutActiSizeAlgin(outActiDim):
		outActiDim = outActiDim.copy()
		# print("-----> outActiDim: ", outActiDim)
		# outActiDim = SpiNNakerBasic.convOutActiDimAlign(outActiDim)
		# print("-----> outActiDim: ", outActiDim)
		return ConvLayerMapper.convOutActivationSizeAlign(outActiDim[0], outActiDim[1], outActiDim[2])

	# FC
	@staticmethod
	def fcInActiSizeAlign(inActiDim):
		inActiDim = inActiDim.copy()
		return inActiDim[0] * BasicOperation.ceilOf(inActiDim[1], MAC_ARRAY_ROW)

	@staticmethod
	def fcInActiDimAlign(inActiDim):
		inActiDim = inActiDim.copy()
		inActiDim[1] = BasicOperation.ceilOf(inActiDim[1], MAC_ARRAY_ROW)
		return inActiDim

	@staticmethod
	def fcWeightSizeAlign(weightDim):
		return FcLayerMapper.weightAlign(weightDim[1], weightDim[0])

	@staticmethod
	def fcWeightDimAlign(weightDim):
		weightDim = weightDim.copy()
		weightDim[0] = FcLayerMapper.weightWidthAlign(weightDim[0])
		return weightDim

	@staticmethod
	def fcOutActiSizeAlign(outActiDim):
		return FcLayerMapper.outActivationAlign(outActiDim[0]) * outActiDim[1]

	@staticmethod
	def fcOutActiDimAlign(outActiDim):
		outActiDim = outActiDim.copy()
		outActiDim[0] = FcLayerMapper.outActivationAlign(outActiDim[0])
		return outActiDim

	@staticmethod
	def dictToLists(dict):
		keys = list(dict)
		values = []
		for key in keys:
			values.append(dict[key])
		return keys, values

	@staticmethod
	def listAdd(accumulator, adder):
		accumulator = accumulator.copy()
		if isinstance(adder, list):
			assert(len(accumulator) == len(adder)), "SpiNNakerBasic:listAdd: length unequal!!"
			for index in range(len(accumulator)):
				accumulator[index] += adder[index]
		else:
			for index in range(len(accumulator)):
				accumulator[index] += adder		
		return accumulator

	@staticmethod
	def listSubstrate(accumulator, substrator):
		accumulator = accumulator.copy()
		if isinstance(substrator, list):
			assert(len(accumulator) == len(substrator)), "SpiNNakerBasic:listSubstrate: length unequal!!"
			for index in range(len(accumulator)):
				accumulator[index] -= substrator[index]
		else:
			for index in range(len(accumulator)):
				accumulator[index] -= substrator
		return accumulator	

	@staticmethod
	def getNonZeroMin(elements):
		elements = elements.copy()
		elements.sort()	
		for element in elements:
			if element > 0:
				return element

	@staticmethod
	def getPartLengthFromSplitInfo(splitInfo, index):
		if isinstance(splitInfo, list):
			totalParts = BasicOperation.getTotalPartsFromSplitInfo(splitInfo)
			assert(index <= totalParts), "index-{} muss not larger than {}".format(index, totalParts)
			if index < splitInfo[1]:
				return splitInfo[0]
			else:
				return splitInfo[2]
		else:
			assert(index == 0), "index-{} should be 1".format(index)
			return splitInfo