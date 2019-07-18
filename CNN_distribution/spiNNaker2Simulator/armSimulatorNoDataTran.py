from spiNNakerSimulatorGeneral import *
import copy


class ARMPNoData(GeneralClass):
	'''
	ARM Proccessor
	'''
	def __init__(self, componentId, queueSize=None):
		GeneralClass.__init__(self, componentId, queueSize)
		self.irqQueue = deque([])
		# MLA: automatically send outActi to outside when finish MLA_TASK
		self.mlaAutoSendOutActiReset()

	def mlaAutoSendOutActiReset(self):
		self.mlaAutoSendOutActiFlag = False
		# self.mlaInActiStartAddr = None
		# self.mlaWeightStartAddr = None
		self.mlaOutActiStartAddr = None
		# self.mlaOperAPeId = None
		self.mlaTaskParam = None
		self.outActiAlignSize = None
		self.outActiDramAddr = None
		self.additionParam = None
		# 
		self.changeDefaultOutActiTargetSram = None
		self.outActiTargetSram = None

	def isIdle(self):
		return len(self.taskQueue) == 0 and len(self.irqQueue) == 0

	def process(self):
		if len(self.irqQueue) > 0:
			return self.irqProcess()
		elif len(self.taskQueue) > 0:
			task = self.getNextTask()
			taskName = task[TASK_NAME]
			# if Task.SRAM_RD_32 == taskName:
			# 	return task
			if Task.DATA_MIGRATION == taskName:
				return task
			elif Task.DATA_MIGRATION_DEST_FINISH == taskName:
				return task
			elif Task.DATA_MIGRATION_32 == taskName:
				return task
			# elif Task.DRAM_SRAM_DATA_MIGRATION == taskName:
			# 	return task
			# elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == taskName:
			# 	return task
			else:
				self.customAssert(False, "Unsupport task: {}".format(task))
		else:
			return {TASK_NAME:Task.TASK_NONE}

	def localPeIdGenerator(self):
		localPeId = copy.deepcopy(self.componentId)
		localPeId[2] = localPeId[2] % PEARMP_PE_ID_OFFSET
		return localPeId

	def irqProcess(self):
		task = self.irqQueue.popleft()
		if Task.MLA_FINISH == task[TASK_NAME]:
			# self.mlaAutoSendOutActiTasksGenerator()
			if self.mlaAutoSendOutActiFlag:
				dmaTask = {TASK_NAME:Task.DATA_MIGRATION_32, TASK_SRAM_ADDR:self.mlaOutActiStartAddr, 
					TASK_MIGRATION_SIZE:self.outActiAlignSize, TASK_MIGRATION_DESTINATION:self.getDramId(), 
					TASK_MIGRA_SRAM_ADDR:self.outActiDramAddr}
				self.addTask(dmaTask)
			self.mlaAutoSendOutActiReset()
			return task
		elif Task.MLA_EXE_SRAM_FINISH == task[TASK_NAME]:
			if self.mlaAutoSendOutActiFlag:
				if self.changeDefaultOutActiTargetSram:
					migraDest = self.outActiTargetSram
				else:
					migraDest = self.getStorageQpeId()
					migraDest.append(self.componentId[Z_AXIS_INDEX]%NUM_PES_IN_QPE)
				dmaTask = {TASK_NAME:Task.DATA_MIGRATION_32, TASK_SRAM_ADDR:self.mlaOutActiStartAddr, 
					TASK_MIGRATION_SIZE:self.outActiAlignSize, TASK_MIGRATION_DESTINATION:migraDest, 
					TASK_MIGRA_SRAM_ADDR:self.outActiDramAddr}
				self.addTask(dmaTask)
			self.mlaAutoSendOutActiReset()
			return task			
		elif Task.DATA_MIGRATION_FINISH == task[TASK_NAME]:
			self.addTask({TASK_NAME:Task.DATA_MIGRATION_DEST_FINISH, 
							TASK_DESTINATION:task[TASK_MIGRATION_DESTINATION],
							TASK_MIGRATION_SOURCE:task[TASK_MIGRATION_SOURCE],
							TASK_MIGRATION_SIZE:task[TASK_MIGRATION_SIZE],
							TASK_ADDITION:task[TASK_ADDITION]})
			return task
		elif Task.DATA_MIGRATION_DEST_FINISH == task[TASK_NAME]:
			return {TASK_NAME:Task.DATA_MIGRATION_ALL_FINISH, TASK_DESTINATION:HOST_ID,
					TASK_MIGRATION_SOURCE:task[TASK_MIGRATION_SOURCE],
					TASK_MIGRATION_DESTINATION:self.localPeIdGenerator(),
					TASK_MIGRATION_SIZE:task[TASK_MIGRATION_SIZE],
					TASK_ADDITION:task[TASK_ADDITION]}
		elif Task.DATA_MIGRATION_32_FINISH == task[TASK_NAME]:
			if self.localPeIdGenerator() == task[TASK_DESTINATION]:
				task[TASK_DESTINATION] = HOST_ID
			return task
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == task[TASK_NAME]:
			task[TASK_DESTINATION] = HOST_ID
			return task
		else:
			self.customAssert(False, "Unsupport task: {}".format(task))

	def irqInsert(self, task):
		# task = copy.deepcopy(task)
		self.irqQueue.append(task)

	# =================================================================
	# Simulate the Program of ARM： automatically send out outActi
	# =================================================================
	def mlaAutoSendOutActiSetting(self, task):
		# task = copy.deepcopy(task)
		# self.mlaInActiStartAddr = task[TASK_OPER_B_ADDR]
		# self.mlaWeightStartAddr = task[TASK_OPER_A_ADDR]
		self.mlaOutActiStartAddr = task[TASK_OPER_C_ADDR]
		self.outActiDramAddr = task[TASK_OUTACTI_DRAM_ADDR]
		# self.mlaOperAPeId = task[TASK_OPER_A_PEID]
		self.mlaTaskParam = task[TASK_MLA_PARAM]
		mlataskOperation, mlaParam = self.mlaTaskParam
		# self.customAssert(MlaOperType.CONV == mlataskOperation, "Unsupport MLA operation: {}".format(mlataskOperation))
		if MlaOperType.CONV == mlataskOperation:
			inWidth, inHeight, inChannel, filterWidth, filterHeight, outChannel, stride = mlaParam
			outWidth = (inWidth - filterWidth) // stride + 1
			outHeight = (inHeight - filterHeight) // stride + 1
			self.additionParam = task[TASK_ADDITION]
			# Operator fusion
			fuseFlag, poolSize = self.additionParam
			if fuseFlag:
				# Each Packet payload is 16 bytes
				self.outActiAlignSize = self.align16((outWidth * outHeight * outChannel) // poolSize)
			else:
				# Each row align to 16 bytes
				self.outActiAlignSize = self.align4(outWidth)  * 4 * outHeight * outChannel
			# print("FROM ARM: {}", self.outActiAlignSize)
		else:
			matrixAWidth, matrixAHeight, matrixBWidth = mlaParam
			self.additionParam = task[TASK_ADDITION]
			fuseFlag, poolSize = self.additionParam
			if fuseFlag:
				self.outActiAlignSize = self.align16(matrixAHeight * matrixBWidth)
			else:
				self.outActiAlignSize = self.align4(matrixBWidth) * 4 * matrixAHeight
			# print("FROM ARM: {}", self.outActiAlignSize)
		# extra information for indicating whether output data to DRAM after computing
		if TASK_ADDITION2 in task:
			self.mlaAutoSendOutActiFlag = False
		else:
			self.mlaAutoSendOutActiFlag = True
		if TASK_ADDITION3 in task:
			self.changeDefaultOutActiTargetSram = True
			self.outActiTargetSram = task[TASK_ADDITION3]
		else:
			self.changeDefaultOutActiTargetSram = False