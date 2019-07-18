from dramInterface import DramInterface
from dramSimulatorNoDataTran import DRAMNoData
from dramSimulator import DRAM
from spiNNakerSimulatorGeneral import *
from dramDmaSimulatorNoDataTran import DramDmaNoDataTran


class DramController():
	def __init__(self, componentId, noDataTran):
		self.noDataTran = noDataTran
		self.componentId = componentId
		# DRAM
		if self.noDataTran:
			self.dram = DRAMNoData(self.componentId)
		else:
			self.dram = DRAM(self.componentId)
		# DRAM Interface
		dramInterId = self.componentId.copy()
		dramInterId[0] += DRAMINTER_DRAM_ID_OFFSET
		self.dramInterface = DramInterface(componentId=dramInterId)
		# DRAM DMA
		dramDmaId = self.componentId.copy()
		dramDmaId[0] += DRAMDMA_DRAM_ID_OFFSET
		if self.noDataTran:
			self.dramDma = DramDmaNoDataTran(dramDmaId)
		else:
			self.customAssert("Not support DRAM DMA with data transfer")

	def customAssert(self, condition, content):
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def isToLocalDram(self, destination):
		if destination == self.componentId:
			return True
		else:
			return False

	def run(self):
		# Run DRAM interface
		interfaceOutTask = self.dramInterface.transfer()
		# Run DRAM
		dramOutTask = self.dram.writeRead()
		if (not isinstance(dramOutTask, list)):
			dramOutTask = [dramOutTask]
		for singleTask in dramOutTask:
			dramOutTaskName = singleTask[TASK_NAME]
			if Task.TASK_NONE != dramOutTaskName and Task.TASK_DOWN != dramOutTaskName:
				self.addTask(singleTask)
		# Run DRAM-DMA
		dramDmaTask = self.dramDma.dataTransfer()
		dramDmaTaskName = dramDmaTask[TASK_NAME]
		if Task.TASK_NONE != dramDmaTaskName and Task.TASK_DOWN != dramDmaTaskName:
			self.dram.addTask(dramDmaTask)		
		# DRAM Interface interfaceOutTask process
		taskName = interfaceOutTask[TASK_NAME]
		if Task.TASK_NONE == taskName:
			return interfaceOutTask
		# -> DRAM
		elif Task.SRAM_DATA_32 == taskName:
			self.dram.addTask(interfaceOutTask)
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.DRAM_RD_16_BYTES == taskName:
			self.dram.addTask(interfaceOutTask)
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.DRAM_DMA_REQUEST_FINISH == taskName:
			self.dram.addTask(interfaceOutTask)
			return {TASK_NAME:Task.TASK_DOWN}
		# -> QPE/HOST
		elif Task.SRAM_WR == taskName:
			self.router(interfaceOutTask)
			return interfaceOutTask			
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == taskName:
			self.router(interfaceOutTask)
			return interfaceOutTask
		# -> DRAM/HOST
		elif Task.DATA_MIGRATION_32_FINISH == taskName:
			if self.isToLocalDram(interfaceOutTask[TASK_DESTINATION]):
				self.dram.addTask(interfaceOutTask)
				return {TASK_NAME:Task.TASK_DOWN}
			else:
				self.router(interfaceOutTask)
				return interfaceOutTask
		# -> DMA: 
		elif Task.DRAM_SRAM_DATA_MIGRATION == taskName:
			self.dramDma.addTask(interfaceOutTask)
			return {TASK_NAME:Task.TASK_DOWN}
		else:
			self.customAssert(False, "Unsupport interfaceOutTask: {}".format(interfaceOutTask))

	def addTask(self, task):
		# print("task -> DRAM: {}".format(task))
		self.dramInterface.addTask(task)

	def router(self, task):
		'''
		Only support to QPE
		'''
		# print("dram router: {}".format(task))
		destination = task[TASK_DESTINATION]
		# -> HOST
		if destination == HOST_ID:
			task[TASK_NEXT_DESTINATION] = self.generateNextDestination(self.hostQpeId())
			# print("dram router to host: {}".format(task))
		# -> QPE
		else:
			destBlockIndex = self.whichBlock(destination)
			# DRAM: 4
			if self.componentId[0] == DRAM_ID_START:
				if (destBlockIndex // 2) == 0:
					task[TASK_NEXT_DESTINATION] = [destination[0], 0]
				else:
					task[TASK_NEXT_DESTINATION] = [2, 0]
			# DRAM: 5
			elif self.componentId[0] == (DRAM_ID_START+1):
				if (destBlockIndex // 2) == 0:
					task[TASK_NEXT_DESTINATION] = [destination[0], 5]
				else:
					task[TASK_NEXT_DESTINATION] = [2, 5]
			# DRAM: 6
			elif self.componentId[0] == (DRAM_ID_START+2):
				if (destBlockIndex // 2) == 0:
					task[TASK_NEXT_DESTINATION] = [3, 0]
				else:
					task[TASK_NEXT_DESTINATION] = [destination[0], 0]
			# DRAM: 7
			elif self.componentId[0] == (DRAM_ID_START+3):
				if (destBlockIndex // 2) == 0:
					task[TASK_NEXT_DESTINATION] = [3, 5]
				else:
					task[TASK_NEXT_DESTINATION] = [destination[0], 5]
			else:
				self.customAssert(False, "Unkown DRAM Controller")	

	def hostQpeId(self):
		triBlockIndex = self.componentId[0] % DRAM_ID_START
		if 0 == triBlockIndex:
			return [0,0]
		elif 1 == triBlockIndex:
			return [0,3]
		elif 2 == triBlockIndex:
			return [3,0]
		elif 3 == triBlockIndex:
			return [3,3]
		else:
			self.customAssert(False, "Unkown ID for dram Controller")

	def generateNextDestination(self, destination):
		'''
		Follow X axis firstly, then Y axis.
		'''
		if self.componentId[0] % 2 == 0:
			return [destination[0], 0]
		else:
			return [destination[0], 5]

	def whichBlock(self, qpeId):
		qpeYAxis = qpeId[0]
		qpeXAxis = qpeId[1]
		if qpeXAxis < NUM_QPES_X_AXIS_HALF and qpeYAxis < NUM_QPES_Y_AXIS_HALF:
			return 0
		elif qpeXAxis >= NUM_QPES_X_AXIS_HALF and qpeYAxis < NUM_QPES_Y_AXIS_HALF:
			return 1
		elif qpeXAxis < NUM_QPES_X_AXIS_HALF and qpeYAxis >= NUM_QPES_Y_AXIS_HALF:
			return 2
		elif qpeXAxis >= NUM_QPES_X_AXIS_HALF and qpeYAxis >= NUM_QPES_Y_AXIS_HALF:
			return 3
		else:
			self.customAssert(False, "Unkown X and Y axis of qpeId")


if __name__ == "__main__":
	componentId = [4]
	dramId = componentId

	dramController = DramController(componentId, noDataTran=False)
	print("dramController.dram.componentId: {}".format(dramController.dram.componentId))
	print("dramController.dramInterface.componentId: {}".format(dramController.dramInterface.componentId))

	dramController.addTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId, TASK_SRAM_ADDR:0x0, 
		TASK_DATA_LEN:16, TASK_DATA:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]})
	dramController.addTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId, TASK_SRAM_ADDR:0x80, 
		TASK_DATA_LEN:16, TASK_DATA:[10,11,12,13,14,15,16,17,18,19,110,111,112,113,114,115]})
	dramController.addTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,4,4], 
		TASK_SOURCE_SRAM_ADDR:0x10, TASK_SRAM_ADDR:0x0, TASK_DATA_LEN:16})
	dramController.addTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,6,5], 
		TASK_SOURCE_SRAM_ADDR:0x90, TASK_SRAM_ADDR:0x80, TASK_DATA_LEN:16})

	for clock in range(25):
		print("--> {}: dramController response: {}".format(clock+1, dramController.run()))