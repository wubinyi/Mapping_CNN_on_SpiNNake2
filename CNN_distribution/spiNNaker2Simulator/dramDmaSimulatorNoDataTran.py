from spiNNakerSimulatorGeneral import *

class DramDmaNoDataTran(GeneralClass):
	def __init__(self, componentId, queueSize=None):
		GeneralClass.__init__(self, componentId, queueSize)

	def localDramIdGenerator(self):
		return [self.componentId[0] - DRAMDMA_DRAM_ID_OFFSET]

	def dataTransfer(self):
		nextTask = self.getNextTask()
		if None == nextTask:
			return {TASK_NAME:Task.TASK_NONE}
		nextTaskName = nextTask[TASK_NAME]
		if Task.DRAM_SRAM_DATA_MIGRATION == nextTaskName:
			self.dramToSramDataMigration(migraSize=nextTask[TASK_MIGRATION_SIZE], 
				migraDest=nextTask[TASK_MIGRATION_DESTINATION], dataMigraType=nextTask[TASK_ADDITION])
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.DRAM_RD_16_BYTES == nextTaskName:
			return nextTask
		elif Task.DRAM_DMA_REQUEST_FINISH == nextTaskName:
			return nextTask
		else:
			self.customAssert(False, "Unsupport task: {}".format(nextTask))

	def dramToSramDataMigration(self, migraSize, migraDest, dataMigraType):
		localDramId = self.localDramIdGenerator()
		addrOffset = 0
		task = {TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:localDramId, TASK_SOURCE:migraDest}
		while addrOffset < migraSize:
			self.addTask(task)
			addrOffset += NOC_SRAM_BW_BYTES
		self.addTask({TASK_NAME:Task.DRAM_DMA_REQUEST_FINISH, TASK_DESTINATION:localDramId, 
			TASK_MIGRATION_SOURCE:localDramId, TASK_MIGRATION_SIZE:migraSize, TASK_ADDITION:dataMigraType, 
			TASK_MIGRATION_DESTINATION:migraDest})