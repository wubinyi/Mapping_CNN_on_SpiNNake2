from enum import Enum, auto
from collections import deque
import copy
import sys
import math

# ==============================================================
SRAM_DATA_BEGIN_ADDR = 0x8100
SRAM_END_ADDR = 0x20000

# ===============================================================
# 					SpiNNaker2 Simulator Frequency/Bandwidth
# ===============================================================
BYTE_TO_BIT = 8

NOC_FREQ = 500
PE_FREQ = 250
DRAM_FREQ = 250

DRAM_CLOCKS_PER_OPER = 2

NOC_MAX_DATA_WIDTH_BYTES = 16
DRAM_MAX_DATA_WIDTH_BYTES = 16

MAC_ARRAY_SIZE = 16 * 4
MAC_OPER_PER_CLOCK = 2
MEGA_TO_GIGA = 1000
MEGA = 1000000

# ===============================================================
# 					SpiNNaker2 Simulator Parameter
# ===============================================================
# number of components in PE: MLA, DMA, ARMP, SRAM
NUM_COMPS_IN_PE = 4

# DRAM
NUM_DRAMS = 4
# SRAM size
TOTAL_SRAM_SIZE = 128 * 1024
MLA_MAC_COLUMN = 16
MLA_MAC_COLUMN_MOD = MLA_MAC_COLUMN // 1
MLA_MAC_COLUMN_DECREASE_TIME = MLA_MAC_COLUMN // MLA_MAC_COLUMN_MOD
MLA_MAC_ROW = 4
MLA_MAC_ROW_MOD = MLA_MAC_ROW // 1
NOC_SRAM_BW_BYTES = 16
NOC_SRAM_BW_WORDS = 4

NUM_PES_IN_QPE = 4

SPINNAKER_BLOCK_DIM = 3

NUM_QPES_X_AXIS = 6
NUM_QPES_Y_AXIS = 6

NUM_QPES_XY_HALF = 3
NUM_QPES_X_AXIS_HALF = NUM_QPES_XY_HALF
NUM_QPES_Y_AXIS_HALF = NUM_QPES_XY_HALF

# These are basic-QPE Id not host-QPE Id
NUM_QPES_X_AXIS_TRIBLOCK = 3
NUM_QPES_Y_AXIS_TRIBLOCK = 3
FIRST_TRIBLOCK_BASIC = [0,0]
SECOND_TRIBLOCK_BASIC = [0,3]
THIRD_TRIBLOCK_BASIC = [3,0]
FOURTH_TRIBLOCK_BASIC = [3,3]
TRIBLOCK_BASIC = [FIRST_TRIBLOCK_BASIC, SECOND_TRIBLOCK_BASIC, THIRD_TRIBLOCK_BASIC, FOURTH_TRIBLOCK_BASIC]

# Following are host-QPE Id not basic-QPE Id
FIRST_TRIBLOCK_HOST = [2,2]
SECOND_TRIBLOCK_HOST = [2,3]
THIRD_TRIBLOCK_HOST = [3,2]
FOURTH_TRIBLOCK_HOST = [3,3]
TRIBLOCK_HOST = [FIRST_TRIBLOCK_HOST, SECOND_TRIBLOCK_HOST, THIRD_TRIBLOCK_HOST, FOURTH_TRIBLOCK_HOST]
NUM_OF_TRIBLOCKS = 4


FIRST_DOUBLOCK_ST_QPE_ID = [3,2]
SECOND_DOUBLOCK_ST_QPE_ID = [2,2]
THIRD_DOUBLOCK_ST_QPE_ID = [3,3]
FOURTH_DOUBLOCK_ST_QPE_ID = [2,3]
DOUBLOCK_ST_QPE_IDS = [FIRST_DOUBLOCK_ST_QPE_ID, SECOND_DOUBLOCK_ST_QPE_ID, 
	THIRD_DOUBLOCK_ST_QPE_ID, FOURTH_DOUBLOCK_ST_QPE_ID]
NUM_OF_DOUBLOCKS = 4
NUM_OF_BLOCKS_IN_DOUBLOCKS = 2
NUM_OF_BLOCKS = NUM_OF_DOUBLOCKS * NUM_OF_BLOCKS_IN_DOUBLOCKS
NUM_OF_QPE_IN_BLOCK = 4
NUM_OF_QPE_IN_DOUBLOCK = NUM_OF_QPE_IN_BLOCK * NUM_OF_BLOCKS_IN_DOUBLOCKS
NUM_OF_PES_IN_ALL_DOUBLOCK = NUM_OF_QPE_IN_DOUBLOCK * NUM_OF_DOUBLOCKS * NUM_PES_IN_QPE

FIRST_DOUBLOCK_QPE_IDS = [[0,1],[1,1],[0,0],[1,0],[2,0],[2,1],[3,0],[3,1]]
SECOND_DOUBLOCK_QPE_IDS = [[1,5],[1,4],[0,5],[0,4],[0,3],[1,3],[0,2],[1,2]]
THIRD_DOUBLOCK_QPE_IDS = [[4,0],[4,1],[5,0],[5,1],[5,2],[4,2],[5,3],[4,3]]
FOURTH_DOUBLOCK_QPE_IDS = [[5,4],[4,4],[5,5],[4,5],[3,5],[3,4],[2,5],[2,4]]
DOUBLOCK_QPE_ID_LIST = [FIRST_DOUBLOCK_QPE_IDS, SECOND_DOUBLOCK_QPE_IDS, THIRD_DOUBLOCK_QPE_IDS, FOURTH_DOUBLOCK_QPE_IDS]
STORAGE_QPE_IDS = [[3,2],[2,2],[3,3],[2,3]]
# ===============================================================
# 						SpiNNaker2 Component ID
# ===============================================================
# Component ID
# HOST = [X] -------------------------- x = {0}
# 										HOST: 			x = {0}
# 										HostInterface: 	x = {1}
# DRAM_NoC = [X] ----------------------	x = {4,5,6,7}
# 										DRAM:			x = {4,5,6,7}
# 										DramInterface:	x = {8,9,10,11}
# 										DramDma:		x = {12,13,14,15}
# QPE = [X, Y] ------------------------	x,y = {0,1,2,3,4,5}
# PE_NoC_PESRAM_MLA = [X, Y, Z] ------- x,y = {0,1,2,3,4,5}
# 										PE: 	z = {0,1,2,3}
# 										PESRAM: z = {4,5,6,7}
# 										PEMLA:	z = {8,9,10,11}
# 										MLAOPA: z = {12,13,14,15}
# 										MLAOPB: z = {16,17,18,19}
# 										MLAOPC: z = {20,21,22,23}
# 										PEARMP: z = {24,25,26,27}
# 										NoC: 	z = {28}
# 										DMA: 	z = {32,33,34,35}
Y_AXIS_INDEX = 0
X_AXIS_INDEX = 1
Z_AXIS_INDEX = 2

HOST_ID = [0]
HOST_INTERFACE_ID = [1]

DRAM_HOST_ID_OFFSET = 4

DRAM_ID_START = 4

DRAMINTER_DRAM_ID_OFFSET = 4
DRAMDMA_DRAM_ID_OFFSET = 8

COMPONENT_ID_INTERVAL = 4

PE_ID_START = 0
PESRAM_ID_START = 4
PEMLA_ID_START = 8
PEMLAOPA_ID_START = 12
PEMLAOPB_ID_START = 16
PEMLAOPC_ID_START = 20
PEARMP_ID_START = 24
NOC_ID = 28

PESRAM_PE_ID_OFFSET = 4
PEMLA_PE_ID_OFFSET = 8
PEARMP_PE_ID_OFFSET = 24
PEDMA_PE_ID_OFFSET = 32

PESRAM_PEMLA_ID_OFFSET = -4

PEMLAOPA_PEMLA_ID_OFFSET = 4
PEMLAOPB_PEMLA_ID_OFFSET = 8
PEMLAOPC_PEMLA_ID_OFFSET = 12

PESRAM_PEARMP_ID_OFFSET = -20

# ===============================================================
# 		SpiNNaker2 Simulator General Class (Parent Class)
# ===============================================================
class ComponentType(Enum):
	COMP_DRAM = auto()
	COMP_QPE = auto()
	COMP_PE = auto()
	COMP_SRAM = auto()
	COMP_MLA = auto()
	COMP_MLAOPA = auto()
	COMP_MLAOPB = auto()
	COMP_MLAOPC = auto()
	COMP_ARMP = auto()
	COMP_NOC = auto()
	COMP_UNKNOWN = auto()

class Status(Enum):
	FREE = auto()
	BUSY = auto()

class GeneralClass():
	def __init__(self, componentId, queueSize):
		self.componentId = componentId
		self.reset(queueSize)

	def reset(self, queueSize):
		self.queueSize = queueSize
		self.status = Status.FREE
		self.taskQueue = deque([], maxlen=self.queueSize)

	def customAssert(self, condition, content):
		# funcName = sys._getframe().f_code.co_name
		# lineNumber = sys._getframe().f_lineno
		# fileName = sys._getframe().f_code.co_filename
		if False == condition:
			callerName = sys._getframe().f_back.f_code.co_name
			assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def addTask(self, task):
		if self.queueSize != None:
			self.customAssert(len(self.taskQueue) < self.queueSize, "Task queue is full: {}".format(self.taskQueue))
		self.taskQueue.append(task)

	def addTaskToTop(self, task):
		if self.queueSize != None:
			self.customAssert(len(self.taskQueue) < self.queueSize, "Task queue is full")
		self.taskQueue.insert(0, task)

	def getNextTask(self):
		if len(self.taskQueue) == 0:
			return None
		nextTask = self.taskQueue.popleft()
		return nextTask

	def getNextTaskWithoutPop(self):
		if len(self.taskQueue) == 0:
			return None
		nextTask = self.taskQueue[0]
		return nextTask

	def canAddTask(self):
		if self.queueSize != None:
			return not (len(self.taskQueue) >= self.queueSize)
		return True
		
	def isIdle(self):
		return len(self.taskQueue) == 0

	def compIdParse(self, compId):
		if isinstance(compId, list):
			idLen = len(compId)
			if 1 == idLen and compId[0] < 4:
				# print("DRAM{}".format(compId[0]))
				return ComponentType.COMP_DRAM
			elif 2 == idLen and compId[0] < 6 and compId[1] < 6:
				# print("QPE{}-{}".format(compId[0], compId[1]))
				return ComponentType.COMP_QPE
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] >= PE_ID_START and compId[2] < PESRAM_ID_START:
				# print("PE{}-{}-{}".format(compId[0], compId[1], compId[2]))
				return ComponentType.COMP_PE
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] >= PESRAM_ID_START and compId[2] < PEMLA_ID_START:
				# print("PE{}-{}-{}-SRAM".format(compId[0], compId[1], compId[2]%4))
				return ComponentType.COMP_SRAM
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] >= PEMLA_ID_START and compId[2] < PEMLAOPA_ID_START:
				# print("PE{}-{}-{}-MLA".format(compId[0], compId[1], compId[2]%8))
				return ComponentType.COMP_MLA
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] >= PEMLAOPA_ID_START and compId[2] < PEMLAOPB_ID_START:
				# print("PE{}-{}-{}-MLA".format(compId[0], compId[1], compId[2]%8))
				return ComponentType.COMP_MLAOPA
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] >= PEMLAOPB_ID_START and compId[2] < PEMLAOPC_ID_START:
				# print("PE{}-{}-{}-MLA".format(compId[0], compId[1], compId[2]%8))
				return ComponentType.COMP_MLAOPB
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] >= PEMLAOPC_ID_START and compId[2] < PEARMP_ID_START:
				# print("PE{}-{}-{}-MLA".format(compId[0], compId[1], compId[2]%8))
				return ComponentType.COMP_MLAOPC
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] >= PEARMP_ID_START and compId[2] < NOC_ID:
				# print("PE{}-{}-{}-ARMP".format(compId[0], compId[1], compId[2]%8))
				return ComponentType.COMP_ARMP
			elif 3 == idLen and compId[0] < 6 and compId[1] < 6 and compId[2] == NOC_ID:
				# print("QPE{}-{}-NoC".format(compId[0], compId[1]))
				return ComponentType.COMP_NOC
			else:
				print("Invalid compId: {}".format(compId))
				return ComponentType.COMP_UNKNOWN
		else:
			print("Invalid compId: {}".format(compId))
			return ComponentType.COMP_UNKNOWN

	def align16(self, length):
		return math.ceil(length/MLA_MAC_COLUMN) * MLA_MAC_COLUMN

	def align4(self, length):
		return math.ceil(length/4) * 4

	def getDramId(self):
		qpeId = self.componentId[0:2]
		x = qpeId[1] // SPINNAKER_BLOCK_DIM
		y = qpeId[0] // SPINNAKER_BLOCK_DIM
		return [y*2 + x + DRAM_HOST_ID_OFFSET]

	def nearestDramQpeID(self, dramId):
		x = dramId[0] % 2 * 5
		y = self.componentId[0]
		return [y, x]

	def getHostQpeIdForDram(self):
		triBlockIndex = self.componentId[0] % DRAM_ID_START
		return TRIBLOCK_HOST[triBlockIndex].copy()
		# if 0 == triBlockIndex:
		# 	return [2,2]
		# elif 1 == triBlockIndex:
		# 	return [2,3]
		# elif 2 == triBlockIndex:
		# 	return [3,2]
		# elif 3 == triBlockIndex:
		# 	return [3,3]
		# else:
		# 	self.customAssert(False, "Unkown X and Y axis of qpeId")

	def hostQpeId(self):
		if len(self.componentId) == 1:
			return self.getHostQpeIdForDram()
		# # 1 HOST PE -> Cooperate with SpiNNaker2TriBlock.hostTaskDistributor()
		# # 		  -> Cooperate with DistributionGeneralClass.getHostQpeId()
		# return [0,0]
		# 4 HOST PE -> Cooperate with SpiNNaker2TriBlock.hostTaskDistributor()
		# 		    -> Cooperate with DistributionGeneralClass.getHostQpeId()
		qpeYAxis = self.componentId[0]
		qpeXAxis = self.componentId[1]
		if qpeXAxis < NUM_QPES_X_AXIS_HALF and qpeYAxis < NUM_QPES_Y_AXIS_HALF:
			# return [2,2]
			return TRIBLOCK_HOST[0].copy()
		elif qpeXAxis >= NUM_QPES_X_AXIS_HALF and qpeYAxis < NUM_QPES_Y_AXIS_HALF:
			# return [2,3]
			return TRIBLOCK_HOST[1].copy()
		elif qpeXAxis < NUM_QPES_X_AXIS_HALF and qpeYAxis >= NUM_QPES_Y_AXIS_HALF:
			# return [3,2]
			return TRIBLOCK_HOST[2].copy()
		elif qpeXAxis >= NUM_QPES_X_AXIS_HALF and qpeYAxis >= NUM_QPES_Y_AXIS_HALF:
			# return [3,3]
			return TRIBLOCK_HOST[3].copy()
		else:
			self.customAssert(False, "Unkown X and Y axis of qpeId")

	def isHostId(self, compId):
		return compId == HOST_ID

	def getStorageQpeId(self):
		localQpeId = self.componentId[Y_AXIS_INDEX:Z_AXIS_INDEX]
		if localQpeId in FIRST_DOUBLOCK_QPE_IDS:
			return STORAGE_QPE_IDS[0].copy()
		elif localQpeId in SECOND_DOUBLOCK_QPE_IDS:
			return STORAGE_QPE_IDS[1].copy()
		elif localQpeId in THIRD_DOUBLOCK_QPE_IDS:
			return STORAGE_QPE_IDS[2].copy()
		elif localQpeId in FOURTH_DOUBLOCK_QPE_IDS:
			return STORAGE_QPE_IDS[3].copy()
		else:
			self.customAssert(False, "Unkown X and Y axis of qpeId")		

# ===============================================================
# 			SpiNNaker2 Simulator TASK Description
# ===============================================================
TASK_NAME = "tNam"
TASK_DESTINATION = "des"
TASK_NEXT_DESTINATION = "nDes"
TASK_SRAM_ADDR = "smAdd"
TASK_DATA_LEN = "dLen"
TASK_SOURCE = "sou"
TASK_SOURCE_SRAM_ADDR = "sAdd"
TASK_DATA = "dat"
TASK_MLA_PARAM = "mlaPa"
TASK_OPER_A_PEID = "oAPeid"
TASK_OPER_A_ADDR = "oAAdd"
TASK_OPER_B_ADDR = "oBAdd"
TASK_OPER_C_ADDR = "oCAdd"
TASK_MIGRATION_DESTINATION = "migDes"
TASK_MIGRATION_SIZE = "migSiz"
TASK_MIGRA_SRAM_ADDR = "tSmAdd"
TASK_MIGRATION_SOURCE = "migSou"
TASK_MLA_TOTAL_CLKS = "mTotClk"
TASK_MLA_COMP_CLKS = "mCmpClk"
TASK_MLA_OUTACTI_WR_CLKS = "mOWrClk"
TASK_OUTACTI_DRAM_ADDR = "oDmAdd"

TASK_DESTINATION_2 = 'des2'
TASK_SRAM_ADDR_2 = "smAdd2"
TASK_DATA_LEN_2 = "dLen2"
TASK_SOURCE_2 = "sou2"

# For MLA_EXE and MLA_EXE_SRAM, determine the operator fusion
TASK_ADDITION = "addi"
# For MLA_EXE and MLA_EXE_SRAM, determine if to auto send compute result into SRAM/DRAM
TASK_ADDITION2 = "addi2"
# For MLA_EXE_SRAM, determine where to auto send compute result
TASK_ADDITION3 = "addi3"


class DramSramDataMigraType(Enum):
	EMPTY = auto()
	INACTIVE = auto()
	WEIGHT = auto()
	WEIGHT2 = auto()
	OUTACTIVE = auto()

class MlaOperType(Enum):
	CONV = auto()
	MM = auto()

class IcproValidation(Enum):
	MLA_FINISH = auto()

class Task(Enum):
	# emulate clock, only used by clock thread
	TASK_CLOCK = auto()
	TASK_NO_CLOCK = auto()
	# 
	TASK_NONE = auto()
	TASK_DOWN = auto()
	# 
	TASK_DOWN_NOTI = auto()
	# NoC/ARMP/MLA -> SRAM
	SRAM_WR = auto() # {Task, Destination, Address, DataLength, Data}
	SRAM_RD = auto() # {Task, Destination, Address, DataLength, Source}
	# SRAM -> ARM/NoC/MLA
	SRAM_DATA = auto() # {Task, Data, Destination}
	# SRAM_DATA_FINISH = auto() # useless 26.01.2019
	# NoC/ARMP -> MLA
	MLA_EXE = auto() # {Task, MLA_PARAM(CONV/FC), OPER_A_PE_ID, OPER_A_ADDR, OPER_B_ADDR, OPER_C_ADDR}
					 # From NoC: {Task, Destination, MLA_PARAM(CONV/FC), OPER_A_PE_ID, OPER_A_ADDR, OPER_B_ADDR, OPER_C_ADDR}
	MLA_EXE_SRAM = auto()
	MLA_EXE_SRAM_FINISH = auto()
	# MLA -> request data from SRAM and NoC
	NOC_SRAM_DATA_REQ = auto()# {Task, Destination1 (self PE), Address1, DataLength1, source1, 
							  # 	   Destination2 (other PE), Address2, DataLength2, source2}
	DOUBLE_SRAM_DATA_REQ = auto()# {Task, Destination1 (self PE), Address1, DataLength1, source1, 
							  # 	   Destination2 (other PE), Address2, DataLength2, source2}
	SRAM_WR_32 = auto()
	SRAM_RD_32 = auto()
	NOC_SRAM_RD = auto()
	SRAM_DATA_32 = auto()
	NOC_SRAM_DATA = auto()
	# MLA -> ARMP: IRQ
	MLA_FINISH = auto() # {Task, Source, MLA_PARAM(CONV/FC)}
	# Data migration: DMA functionality
	# HOST -> ARM -> DMA
	DATA_MIGRATION = auto() # {Task, Destination, Address, migrationSize, targetPEID, targetAddr}
	# DMA -> ARM -> HOST
	DATA_MIGRATION_FINISH = auto()
	# ARM -> target ARM
	DATA_MIGRATION_DEST_FINISH = auto()
	# target ARM -> HOST
	DATA_MIGRATION_ALL_FINISH = auto()
	SRAM_RD_TO_SRAM = auto() # DMA-> {Task, Destination, Address, DataLength, Source, Source_Addr}

	DATA_MIGRATION_32 = auto()
	DATA_MIGRATION_32_FINISH = auto()
	# Special task for triBlock-based-SpiNNaker2
	PROCESS_END = auto()
	# DRAM
		# DRAM->PE: {Task, Destination(DRAM), address, dataLength=16, Source, Source_Addr} 
		# DRAM->PE: {Task, Destination(DRAM), Source} 
		# Response: {SRAM_WR, Destination}
	DRAM_RD_16_BYTES = auto()
	# DATA from dram -> SRAM
	# {Task, Destination(DRAM), sourceAddr(DRAM), migrationSize, migraDest(PE), targetAddr(SRAM), addition(activation/weight)}
	# {Task, Destination(DRAM), migrationSize, migraDest(PE), addition(activation/weight)}
	# Cause DMA generate {DRAM_RD_16_BYTES and DRAM_DMA_REQUEST_FINISH}
	DRAM_SRAM_DATA_MIGRATION = auto()
	# Send DRAM_DMA_REQUEST_FINISH to PE, which response DRAM_SRAM_DATA_MIGRA_FINISH
	DRAM_DMA_REQUEST_FINISH = auto()
	# Transfer to ARM, then re-transfer to HOST 
	DRAM_SRAM_DATA_MIGRA_FINISH = auto()
	# MLA -> request data from SRAM: SRAM_RD
	# MLA -> request data from NoC: SRAM_RD
	# # MLA -> ARMP: IRQ
	# # NoC -> ARMP (Not support 19.01.2019)
	# MLA_FINISH = auto()
	# # ARMP -> NoC: MLA_FINISH


# MLA_PARAM
# CONV: (MlaTaskType, (IN_WIDTH, IN_HEIGHT, IN_CHANNEL, FILTER_WIDTH, FILTER_HEIGHT, OUT_CHANNEL, STRIDE))
# FC: (MlaTaskType, (IN_NEURON, OUT_NEURON))


if __name__ == "__main__":
	general = GeneralClass([2, 1], None)
	print("dram QPE ID: {}-[2,0]".format(general.nearestDramQpeID([4])))

	general = GeneralClass([0, 3], None)
	print("dram QPE ID: {}-[0,5]".format(general.nearestDramQpeID([5])))

	general = GeneralClass([3, 2], None)
	print("dram QPE ID: {}-[3,0]".format(general.nearestDramQpeID([6])))

	general = GeneralClass([4, 4], None)
	print("dram QPE ID: {}-[4,5]".format(general.nearestDramQpeID([7])))