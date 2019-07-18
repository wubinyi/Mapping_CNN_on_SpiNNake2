from spiNNakerSimulatorGeneral import *


class DMA(GeneralClass):
	def __init__(self, componentId, queueSize=None, noDataTran=True):
		GeneralClass.__init__(self, componentId, queueSize)
		self.noDataTran = noDataTran

	def localPeIdGenerator(self):
		localPeId = copy.deepcopy(self.componentId)
		localPeId[2] = localPeId[2] % PEDMA_PE_ID_OFFSET
		return localPeId

	def dataTransfer(self):
		nextTask = self.getNextTask()
		if None == nextTask:
			return {TASK_NAME:Task.TASK_NONE}
		elif Task.DATA_MIGRATION == nextTask[TASK_NAME]:
			self.dataMigration(readAddr=nextTask[TASK_SRAM_ADDR], migrationSize=nextTask[TASK_MIGRATION_SIZE], 
								targetPeId=nextTask[TASK_MIGRATION_DESTINATION], 
								targetSramAddr=nextTask[TASK_MIGRA_SRAM_ADDR])
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.DATA_MIGRATION_32 == nextTask[TASK_NAME]:
			self.migrateOutActiToHost(readAddr=nextTask[TASK_SRAM_ADDR], migrationSize=nextTask[TASK_MIGRATION_SIZE], 
								targetPeId=nextTask[TASK_MIGRATION_DESTINATION], 
								targetSramAddr=nextTask[TASK_MIGRA_SRAM_ADDR])
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.SRAM_RD_TO_SRAM == nextTask[TASK_NAME]:
			return nextTask
		elif Task.DATA_MIGRATION_FINISH == nextTask[TASK_NAME]:
			return nextTask
		elif Task.SRAM_RD_32 == nextTask[TASK_NAME]:
			return nextTask
		elif Task.DATA_MIGRATION_32_FINISH == nextTask[TASK_NAME]:
			return nextTask
		else:
			self.customAssert(False, "Unsupport task: {}".format(nextTask))

	# ===========================================================================
	# 							Typical DMA Functionality
	# ===========================================================================
	def dataMigration(self, readAddr, migrationSize, targetPeId, targetSramAddr):
		addrOffset = 0
		if self.noDataTran:
			task = {TASK_NAME:Task.SRAM_RD_TO_SRAM, TASK_DESTINATION:self.localPeIdGenerator(), TASK_SOURCE:targetPeId}
			while addrOffset < migrationSize:
				self.addTask(copy.deepcopy(task))
				addrOffset += NOC_SRAM_BW_BYTES
		else:
			task = {TASK_NAME:Task.SRAM_RD_TO_SRAM, TASK_DESTINATION:self.localPeIdGenerator(), TASK_SOURCE:targetPeId,
					TASK_DATA_LEN:NOC_SRAM_BW_BYTES}
			while addrOffset < migrationSize:
				task[TASK_SRAM_ADDR] = readAddr + addrOffset
				task[TASK_SOURCE_SRAM_ADDR] = targetSramAddr + addrOffset
				self.addTask(copy.deepcopy(task))
				addrOffset += NOC_SRAM_BW_BYTES
		# This task works like the DMA IRQ.
		self.addTask({TASK_NAME:Task.DATA_MIGRATION_FINISH, TASK_DESTINATION:HOST_ID, 
						TASK_MIGRATION_SOURCE:self.localPeIdGenerator(),
						TASK_MIGRATION_DESTINATION:targetPeId, TASK_SRAM_ADDR:readAddr,
						TASK_MIGRA_SRAM_ADDR:targetSramAddr})

	# ===========================================================================
	# 							Transfer outActi to HOST
	# ===========================================================================
	def migrateOutActiToHost(self, readAddr, migrationSize, targetPeId, targetSramAddr):
		# {Task, Destination, Address, DataLength, Source}
		addrOffset = 0
		if self.noDataTran:
			task = {TASK_NAME:Task.SRAM_RD_32, TASK_DESTINATION:self.localPeIdGenerator(), TASK_SOURCE:self.getDramId()}
			while addrOffset < migrationSize:
				self.addTask(copy.deepcopy(task))
				addrOffset += NOC_SRAM_BW_WORDS
		else:
			task = {TASK_NAME:Task.SRAM_RD_32, TASK_DESTINATION:self.localPeIdGenerator(), TASK_SOURCE:self.getDramId(),
					TASK_DATA_LEN:NOC_SRAM_BW_WORDS}
			while addrOffset < migrationSize:
				task[TASK_SRAM_ADDR] = readAddr + addrOffset
				task[TASK_SOURCE_SRAM_ADDR] = targetSramAddr + addrOffset
				self.addTask(copy.deepcopy(task))
				addrOffset += NOC_SRAM_BW_WORDS
		# This task works like the DMA IRQ.
		self.addTask({TASK_NAME:Task.DATA_MIGRATION_32_FINISH, TASK_DESTINATION:HOST_ID, 
						TASK_MIGRATION_SOURCE:self.localPeIdGenerator(),
						TASK_MIGRATION_DESTINATION:targetPeId, TASK_SRAM_ADDR:readAddr,
						TASK_MIGRA_SRAM_ADDR:targetSramAddr})