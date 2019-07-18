from spiNNakerSimulatorGeneral import *
import copy


class ARMP(GeneralClass):
	'''
	ARM Proccessor
	'''
	def __init__(self, componentId, queueSize=None, noDataTran=True):
		GeneralClass.__init__(self, componentId, queueSize)
		self.noDataTran = noDataTran
		self.irqQueue = deque([])
		# MLA: automatically send outActi to outside when finish MLA_TASK
		self.mlaAutoSendOutActiReset()

	def mlaAutoSendOutActiReset(self):
		self.mlaAutoSendOutActiFlag = False
		self.mlaInActiStartAddr = None
		self.mlaWeightStartAddr = None
		self.mlaOutActiStartAddr = None
		self.mlaOperAPeId = None
		self.mlaTaskParam = None
		self.outActiAlignSize = None
		self.outActiDramAddr = None

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
			dmaTask = {TASK_NAME:Task.DATA_MIGRATION_32, TASK_SRAM_ADDR:self.mlaOutActiStartAddr, 
				TASK_MIGRATION_SIZE:self.outActiAlignSize, TASK_MIGRATION_DESTINATION:HOST_ID, 
				TASK_MIGRA_SRAM_ADDR:self.outActiDramAddr}
			self.addTask(dmaTask)
			self.mlaAutoSendOutActiReset()
			return task
		elif Task.DATA_MIGRATION_FINISH == task[TASK_NAME]:
			self.addTask({TASK_NAME:Task.DATA_MIGRATION_DEST_FINISH, 
							TASK_DESTINATION:copy.deepcopy(task[TASK_MIGRATION_DESTINATION]),
							TASK_MIGRATION_SOURCE:copy.deepcopy(task[TASK_MIGRATION_SOURCE]),
							TASK_SRAM_ADDR:task[TASK_SRAM_ADDR], TASK_MIGRA_SRAM_ADDR:task[TASK_MIGRA_SRAM_ADDR]})
			return task
		elif Task.DATA_MIGRATION_DEST_FINISH == task[TASK_NAME]:
			return {TASK_NAME:Task.DATA_MIGRATION_ALL_FINISH, TASK_DESTINATION:HOST_ID,
					TASK_MIGRATION_SOURCE:copy.deepcopy(task[TASK_MIGRATION_SOURCE]),
					TASK_MIGRATION_DESTINATION:self.localPeIdGenerator(),
					TASK_SRAM_ADDR:task[TASK_SRAM_ADDR], TASK_MIGRA_SRAM_ADDR:task[TASK_MIGRA_SRAM_ADDR]}
		elif Task.DATA_MIGRATION_32_FINISH == task[TASK_NAME]:
			return task
		else:
			self.customAssert(False, "Unsupport task: {}".format(task))

	def irqInsert(self, task):
		task = copy.deepcopy(task)
		self.irqQueue.append(task)

	# =================================================================
	# Simulate the Program of ARM： automatically send out outActi
	# =================================================================
	def mlaAutoSendOutActiSetting(self, task):
		task = copy.deepcopy(task)
		self.mlaInActiStartAddr = task[TASK_OPER_B_ADDR]
		self.mlaWeightStartAddr = task[TASK_OPER_A_ADDR]
		self.mlaOutActiStartAddr = task[TASK_OPER_C_ADDR]
		self.mlaOperAPeId = task[TASK_OPER_A_PEID]
		self.mlaTaskParam = task[TASK_MLA_PARAM]
		mlataskOperation, mlaParam = self.mlaTaskParam
		self.customAssert(MlaOperType.CONV == mlataskOperation, "Unsupport MLA operation: {}".format(mlataskOperation))
		inWidth, inHeight, inChannel, filterWidth, filterHeight, outChannel, stride = mlaParam
		outWidth = (inWidth - filterWidth) // stride + 1
		outHeight = (inHeight - filterHeight) // stride + 1
		self.outActiAlignSize = self.align4(outWidth) * outHeight * outChannel
		self.outActiDramAddr = task[TASK_OUTACTI_DRAM_ADDR]

	# def mlaAutoSendOutActiTasksGenerator(self):
	# 	# {Task, Destination, Address, DataLength, Source}
	# 	if self.noDataTran:
	# 		task = {TASK_NAME:Task.SRAM_RD_32, TASK_DESTINATION:self.localPeIdGenerator(), TASK_SOURCE:self.getDramId()}
	# 		addrOffset = 0
	# 		while addrOffset < self.outActiAlignSize:
	# 			self.addTask(copy.deepcopy(task))
	# 			addrOffset += NOC_SRAM_BW_WORDS
	# 	else:
	# 		task = {TASK_NAME:Task.SRAM_RD_32, TASK_DESTINATION:self.localPeIdGenerator(), \
	# 				TASK_SRAM_ADDR:0, TASK_DATA_LEN:NOC_SRAM_BW_WORDS, TASK_SOURCE:self.getDramId()}
	# 		addrOffset = 0
	# 		while addrOffset < self.outActiAlignSize:
	# 			task[TASK_SRAM_ADDR] = self.mlaOutActiStartAddr + addrOffset
	# 			self.addTask(copy.deepcopy(task))
	# 			addrOffset += NOC_SRAM_BW_WORDS
	# 		# self.addTask({TASK_NAME:Task.SRAM_DATA_FINISH}) # useless 26.01.2019
	# 	self.mlaAutoSendOutActiReset()