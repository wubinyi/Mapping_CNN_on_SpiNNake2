from spiNNakerSimulatorGeneral import *


class DMANoData(GeneralClass):
	def __init__(self, componentId, queueSize=None):
		GeneralClass.__init__(self, componentId, queueSize)

	def localPeIdGenerator(self):
		localPeId = copy.deepcopy(self.componentId)
		localPeId[2] = localPeId[2] % PEDMA_PE_ID_OFFSET
		return localPeId

	def dataTransfer(self):
		nextTask = self.getNextTask()
		if None == nextTask:
			return {TASK_NAME:Task.TASK_NONE}
		elif Task.DATA_MIGRATION == nextTask[TASK_NAME]:
			if TASK_ADDITION in nextTask:
				addition = nextTask[TASK_ADDITION]
			else:
				addition = -1
			self.dataMigration(migrationSize=nextTask[TASK_MIGRATION_SIZE], 
								targetPeId=nextTask[TASK_MIGRATION_DESTINATION],
								addition=addition)
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.DATA_MIGRATION_32 == nextTask[TASK_NAME]:
			self.migrateOutActiToDram(readAddr=nextTask[TASK_SRAM_ADDR], migrationSize=nextTask[TASK_MIGRATION_SIZE], 
								targetPeId=nextTask[TASK_MIGRATION_DESTINATION], 
								targetDramAddr=nextTask[TASK_MIGRA_SRAM_ADDR])
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.SRAM_RD_TO_SRAM == nextTask[TASK_NAME]:
			return nextTask
		elif Task.DATA_MIGRATION_FINISH == nextTask[TASK_NAME]:
			return nextTask
		elif Task.SRAM_RD_32 == nextTask[TASK_NAME]:
			return nextTask
		elif Task.DATA_MIGRATION_32_FINISH == nextTask[TASK_NAME]:
			return nextTask
		# elif Task.DRAM_SRAM_DATA_MIGRATION == nextTask[TASK_NAME]:
		# 	self.dramToSramMigration(migraSource=nextTask[TASK_MIGRATION_SOURCE], migraSize=nextTask[TASK_MIGRATION_SIZE], 
		# 		dataMigraType=nextTask[TASK_ADDITION])
		# 	return {TASK_NAME:Task.TASK_DOWN}
		# elif Task.DRAM_RD_16_BYTES == nextTask[TASK_NAME]:
		# 	return nextTask
		# elif Task.DRAM_DMA_REQUEST_FINISH == nextTask[TASK_NAME]:
		# 	return nextTask
		else:
			self.customAssert(False, "Unsupport task: {}".format(nextTask))

	# ===========================================================================
	# 							Typical DMA Functionality
	# ===========================================================================
	def dataMigration(self, migrationSize, targetPeId, addition):
		addrOffset = 0
		task = {TASK_NAME:Task.SRAM_RD_TO_SRAM, TASK_DESTINATION:self.localPeIdGenerator(), TASK_SOURCE:targetPeId}
		while addrOffset < migrationSize:
			self.addTask(task)
			addrOffset += NOC_SRAM_BW_BYTES
		# This task works like the DMA IRQ.
		self.addTask({TASK_NAME:Task.DATA_MIGRATION_FINISH, TASK_DESTINATION:HOST_ID, 
						TASK_MIGRATION_SOURCE:self.localPeIdGenerator(),
						TASK_MIGRATION_DESTINATION:targetPeId, TASK_MIGRATION_SIZE: migrationSize,
						TASK_ADDITION:addition})

	# ===========================================================================
	# 							Transfer outActi to DRAM
	# ===========================================================================
	def migrateOutActiToDram(self, readAddr, migrationSize, targetPeId, targetDramAddr):
		# {Task, Destination, Address, DataLength, Source}
		addrOffset = 0
		task = {TASK_NAME:Task.SRAM_RD_32, TASK_DESTINATION:self.localPeIdGenerator(), TASK_SOURCE:targetPeId}
		while addrOffset < migrationSize:
			self.addTask(task)
			# # Output 32-bit data
			# addrOffset += NOC_SRAM_BW_WORDS
			# Output 8-bit data, however need quantization stage
			addrOffset += NOC_SRAM_BW_BYTES
		# This task works like the DMA IRQ. -> ARM -> DRAM -> HOST
		self.addTask({TASK_NAME:Task.DATA_MIGRATION_32_FINISH, TASK_DESTINATION:targetPeId, 
						TASK_MIGRATION_SOURCE:self.localPeIdGenerator(),
						TASK_MIGRATION_DESTINATION:targetPeId, TASK_SRAM_ADDR:readAddr,
						TASK_MIGRA_SRAM_ADDR:targetDramAddr})

	# # ===========================================================================
	# # 							DRAM -> SRAM
	# # ===========================================================================
	# def dramToSramMigration(self, migraSource, migraSize, dataMigraType):
	# 	addrOffset = 0
	# 	task = {TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:migraSource, TASK_SOURCE:self.localPeIdGenerator()}
	# 	while addrOffset < migraSize:
	# 		self.addTask(task)
	# 		addrOffset += NOC_SRAM_BW_BYTES
	# 	# How to known sram get all dram data????
	# 	self.addTask({TASK_NAME:Task.DRAM_DMA_REQUEST_FINISH, TASK_DESTINATION:migraSource, 
	# 		TASK_MIGRATION_SOURCE:migraSource, TASK_MIGRATION_SIZE:migraSize, TASK_ADDITION:dataMigraType, 
	# 		TASK_MIGRATION_DESTINATION:self.localPeIdGenerator()})