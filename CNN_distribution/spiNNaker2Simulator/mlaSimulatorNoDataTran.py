from spiNNakerSimulatorGeneral import *
import math
import copy


class MlaState(Enum):
	IDLE = auto()
	COMP_STAGE = auto()
	RESULT_OUT_STAGE = auto()
	FINISH_STAGE = auto()


class MLANoData(GeneralClass):
	'''
	Machine Learning Accelerator
	'''
	def __init__(self, componentId, queueSize=1, icproValidateNoti=False):
		GeneralClass.__init__(self, componentId, queueSize)
		self.operABufferId = self.operAIdGenerator()
		self.operBBufferId = self.operBIdGenerator()
		self.operCBufferId = self.operCIdGenerator()
		self.mlaConvReset()
		self.mlaMmReset()
		self.compClocksCounter = None
		self.writeOutActiClocksCounter = None
		self.mlaRunClks = None
		self.icproValidateNoti = icproValidateNoti

	def mlaGeneralReset(self):
		# Component ID of operA, operB, operC
		self.operABuffer = None
		self.operBBuffer = None
		self.operA = None
		self.operB = None
		self.alreadySendOperBReq = False
		self.alreadySendOperAReq = False
		# State of MLA
		self.mlaState = MlaState.IDLE
		self.mlaTaskParam = None
		self.operAPeId = None
		self.currentOperation = None
		self.mlaFinishFlag = None
		# Used for MLA_MAC_COLUMN_MOD == 8, fetch 128 bits / 2 clks 
		# at the beginging of fetching each row
		self.operandBfetchedClocks = 0

	def mlaConvReset(self):
		self.mlaGeneralReset()
		# CONV Parameter
		self.convFilterWidth = None
		self.convFilterHeight = None
		self.convFilterInChannel = None
		self.convStride = None
		# Loop Info
		self.convOutWidthRoundList = None
		self.convOutHeightRoundList = None
		self.convOutChannelRoundList = None
		self.convOutWidthRound = None
		self.convOutHeightRound = None
		self.convOutChannelRound = None
		self.convInChannelRound = None
		self.convInHeightRound = None
		self.convInWidthRound = None
		self.convOutWidthRoundCounter = None
		self.convOutHeightRoundCounter = None
		self.convOutChannelRoundCounter = None
		self.convInChannelRoundCounter = None
		self.convInHeightRoundCounter = None
		self.convInWidthRoundCounter = None
		# For continuely fetch operABuffer
		self.convFetchOperATimesReg = None
		self.convFetchOperATimesCounter = None
		self.convFetchOperAIntervalReg = None
		self.convFetchOperAIntervalCounter = None
		# For output result to SRAM
		self.convFilterOutChannelEachRound = None
		self.convOutActiWidthEachRound = None
		self.convOutActiWidthCounter = None
		self.convOutActiChannelCounter = None

	def mlaMmReset(self):
		self.mlaGeneralReset()
		# Control how to fetch operA
		self.mmfetchOperAFromSelf = None
		# Control the loop of MM
		self.mmOutWidthRoundList = None 
		self.mmOutHeightRoundList = None
		self.mmOneRoundReg = None
		self.mmOutWidthRoundReg = None
		self.mmOutHeightRoundReg = None
		self.mmOneRoundCounter = None
		self.mmOutWidthRoundCounter = None
		self.mmOutHeightRoundCounter = None
		# Used for fetch operand A
		self.mmFetchOperATimesReg = None
		self.mmFetchOperATimesCounter = None
		self.mmFetchOperAIntervalReg = None
		self.mmFetchOperAIntervalCounter = None
		# Output compute result: setting in mmCompute()

	def selfPeIdGenerator(self):
		selfPeId = copy.deepcopy(self.componentId)
		selfPeId[2] = selfPeId[2] % PEMLA_PE_ID_OFFSET
		return selfPeId

	def selfsramIdGenerator(self):
		selfSramId = copy.deepcopy(self.componentId)
		selfSramId[2] = selfSramId[2] + PESRAM_PEMLA_ID_OFFSET
		return selfSramId

	def pairSramIdGenerator(self):
		pairSramId = copy.deepcopy(self.operAPeId)
		pairSramId[2] = pairSramId[2] + PESRAM_PE_ID_OFFSET
		return pairSramId

	def operAIdGenerator(self):
		operAId = copy.deepcopy(self.componentId)
		operAId[2] = operAId[2] + PEMLAOPA_PEMLA_ID_OFFSET
		return operAId	

	def operBIdGenerator(self):
		operBId = copy.deepcopy(self.componentId)
		operBId[2] = operBId[2] + PEMLAOPB_PEMLA_ID_OFFSET
		return operBId	

	def operCIdGenerator(self):
		operCId = copy.deepcopy(self.componentId)
		operCId[2] = operCId[2] + PEMLAOPC_PEMLA_ID_OFFSET
		return operCId	

	def writeOperABuffer(self):
		# self.customAssert(len(data) != 0, "Data is empty")
		self.alreadySendOperAReq = False
		# if self.operABuffer == None:
		# 	self.operABuffer = data
		# else:
		# 	self.operABuffer.extend(data)
		self.operABuffer += 16

	def writeOperBBuffer(self):
		# self.customAssert(len(data) != 0, "Data is empty")
		self.alreadySendOperBReq = False
		# if None == self.operB:
		# 	self.operBBuffer = data
		# else:
		# 	self.operBBuffer = data[0:4]
		if 0 == self.operB:
			self.operBBuffer += MLA_MAC_COLUMN
		else:
			self.operBBuffer += 4

	def addTask(self, task):
		# task = copy.deepcopy(task)
		if self.queueSize != None:
			self.customAssert(len(self.taskQueue) < self.queueSize, "Task queue is full: {}".format(self.taskQueue))
		self.taskQueue.append(task)
		# self.mlaSetting()

	def addTaskToTop(self, task):
		# task = copy.deepcopy(task)
		if self.queueSize != None:
			self.customAssert(len(self.taskQueue) < self.queueSize, "Task queue is full: {}-{}".format(self.taskQueue, task))
		self.taskQueue.insert(0, task)
		# self.mlaSetting()

	def mlaSetting(self):
		self.mlaRunClks = 0
		task = self.getNextTaskWithoutPop()
		mlaTaskOperation, taskParam = task[TASK_MLA_PARAM]
		if MlaOperType.CONV == mlaTaskOperation:
			self.mlaConvSetting(task)
		else:
			self.mlaMmSetting(task)


	def mlaConvSetting(self, task):
		'''
		This function is automatically called when adding task to MLA
		'''
		# self.mlaRunClks = 0
		# task = self.getNextTaskWithoutPop()
		# self.customAssert(task != None, "There should be a task")
		self.operA = 0
		self.operABuffer = 0
		self.operB = 0
		self.operBBuffer = 0
		mlataskOperation, taskParam = task[TASK_MLA_PARAM]
		# self.customAssert(MlaOperType.CONV == mlataskOperation, "Unsupport MLA operation: {}".format(mlataskOperation))
		self.mlaTaskParam = (mlataskOperation, taskParam)
		self.operAPeId = task[TASK_OPER_A_PEID]
		self.mlaState = MlaState.COMP_STAGE
		self.currentOperation = mlataskOperation
		self.compClocksCounter = 0
		self.mlaFinishFlag = False
		inWidth, inHeight, inChannel, filterWidth, filterHeight, outChannel, stride = taskParam
		self.convFilterWidth = filterWidth
		self.convFilterHeight = filterHeight
		self.convFilterInChannel = inChannel
		# MLA has limitation of stride = 1
		# self.convStride = stride
		self.convStride = 1
		inWidthAlign = self.align16(inWidth)
		outWidth = (inWidth - filterWidth) // self.convStride + 1
		outHeight = (inHeight - filterHeight) // self.convStride + 1
		self.convOutWidthRoundList, self.convOutHeightRoundList, self.convOutChannelRoundList = \
			self.convGetRound(outWidth, outHeight, outChannel)
		self.convOutWidthRound = len(self.convOutWidthRoundList)
		self.convOutHeightRound = len(self.convOutHeightRoundList)
		self.convOutChannelRound = len(self.convOutChannelRoundList)
		self.convInWidthRound = self.convOutWidthToInWidth(self.convOutWidthRoundList[0])
		self.convInHeightRound = filterHeight
		self.convInChannelRound = inChannel
		self.convOutWidthRoundCounter = 0
		self.convOutHeightRoundCounter = 0
		self.convOutChannelRoundCounter = 0
		self.convInWidthRoundCounter = 0
		self.convInHeightRoundCounter = 0
		self.convInChannelRoundCounter = 0
		self.convFetchOperATimesReg = math.ceil(filterWidth * filterHeight * inChannel * MLA_MAC_ROW_MOD / NOC_SRAM_BW_BYTES)
		self.convFetchOperATimesCounter = 0
		self.convFetchOperAIntervalReg = NOC_SRAM_BW_BYTES // MLA_MAC_ROW_MOD
		self.convFetchOperAIntervalCounter = self.convFetchOperAIntervalReg
		# For output-activation
		outWidthAlign = self.align4(outWidth)
		self.writeOutActiClocksCounter = 0

	def mlaMmSetting(self, task):
		self.operA = 0
		self.operABuffer = 0
		self.operB = 0
		self.operBBuffer = 0
		# General setting
		mlataskOperation, taskParam = task[TASK_MLA_PARAM]
		self.mlaTaskParam = (mlataskOperation, taskParam)
		self.operAPeId = task[TASK_OPER_A_PEID]
		self.mmfetchOperAFromSelf = self.doesFetchOperAFromSelf()
		self.mlaState = MlaState.COMP_STAGE
		self.currentOperation = mlataskOperation
		self.mlaFinishFlag = False
		self.compClocksCounter = 0
		self.writeOutActiClocksCounter = 0
		# Control the loop of MM
		matrixAWidth, matrixAHeight, matrixBWidth = taskParam
		self.mmOutWidthRoundList, self.mmOutHeightRoundList = self.mmGetRound(outWidth=matrixBWidth, outHeight=matrixAHeight)
		self.mmOneRoundReg = matrixAWidth
		self.mmOutWidthRoundReg = len(self.mmOutWidthRoundList)
		self.mmOutHeightRoundReg = len(self.mmOutHeightRoundList)
		self.mmOneRoundCounter = 0
		self.mmOutWidthRoundCounter = 0
		self.mmOutHeightRoundCounter = 0
		# Used for fetch operand A
		self.mmFetchOperATimesReg = matrixAWidth // MLA_MAC_ROW_MOD
		self.mmFetchOperATimesCounter = 0
		self.mmFetchOperAIntervalReg = NOC_SRAM_BW_BYTES // MLA_MAC_ROW_MOD
		self.mmFetchOperAIntervalCounter = self.mmFetchOperAIntervalReg
		# if self.componentId[2] % 4 == 1:
		# 	print("self.mmOutWidthRoundList: {}".format(self.mmOutWidthRoundList))
		# 	print("self.mmOutHeightRoundList: {}".format(self.mmOutHeightRoundList))
		# print("self.mmOneRoundReg: {}".format(self.mmOneRoundReg))
		# print("self.mmOutWidthRoundReg: {}".format(self.mmOutWidthRoundReg))
		# print("self.mmOutHeightRoundReg: {}".format(self.mmOutHeightRoundReg))
		# print("self.mmFetchOperAIntervalCounter: {}".format(self.mmFetchOperAIntervalCounter))
		# print("------->>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		# Output compute result: setting in mmCompute()

	def doesFetchOperAFromSelf(self):
		selfPeId = self.selfPeIdGenerator()
		operAPeId = copy.deepcopy(self.operAPeId)
		operAPeId[2] = operAPeId[2] % COMPONENT_ID_INTERVAL
		if selfPeId == operAPeId:
			return True
		else:
			return False

	def accelerate(self):
		if not self.isRunning():
			return {TASK_NAME: Task.TASK_NONE}
		elif MlaOperType.CONV == self.currentOperation:
			self.mlaRunClks += 1
			return self.convRunning()
		elif MlaOperType.MM == self.currentOperation:
			self.mlaRunClks += 1
			return self.mmRunning()
		else:
			self.customAssert(False, "Unsupport MLA Operation: {}".format(self.currentOperation))

	def isRunning(self):
		if MlaState.IDLE == self.mlaState:
			if not self.isIdle():
				self.mlaSetting()
			return False
		else:
			return True
		# return MlaState.IDLE != self.mlaState

	# ===========================================================
	# 							CONV
	# ===========================================================
	def convRunning(self):
		'''
		Every clock request oper_A
		Every 4 clocks request oper_B
		oper_A: always fetch 128-bit
		oper_B: At the first clock of each round, fetch 128-bit, afterward fetch 32-bit.
		'''
		if MlaState.COMP_STAGE == self.mlaState:
			return self.convCompute()
		elif MlaState.RESULT_OUT_STAGE == self.mlaState:
			return self.convOutputResult()
		elif MlaState.FINISH_STAGE == self.mlaState:
			mlaTaskParam = copy.deepcopy(self.mlaTaskParam)
			self.mlaConvReset()
			# Let task queue empty
			task = self.getNextTask()
			taskName = task[TASK_NAME]
			if Task.MLA_EXE == taskName:
				finishTaskName = Task.MLA_FINISH
			elif Task.MLA_EXE_SRAM == taskName:
				finishTaskName = Task.MLA_EXE_SRAM_FINISH
			else:
				self.customAssert(False, "Unsupport MLA task: {}".format(task))
			# return {TASK_NAME:Task.MLA_FINISH, TASK_SOURCE: self.selfPeIdGenerator(), 
			# 		TASK_MLA_PARAM: mlaTaskParam, TASK_DESTINATION: HOST_ID}
			return {TASK_NAME:finishTaskName, TASK_SOURCE: self.selfPeIdGenerator(), 
					TASK_MLA_PARAM: mlaTaskParam, TASK_DESTINATION: HOST_ID, TASK_MLA_TOTAL_CLKS:self.mlaRunClks,
					TASK_MLA_COMP_CLKS:self.compClocksCounter, TASK_MLA_OUTACTI_WR_CLKS:self.writeOutActiClocksCounter}
		else:
			self.customAssert(False, "Unsupport state: {}".format(self.mlaState))

	# ===================== CONV Computation ====================
	def convCompute(self):
		fetchAFlag = self.convOperABufferPop()
		fetchBFlag = self.convOperBBufferPop()
		# Fetch 128-bit value / 2 clocks at the begining of fetching row of input
		self.operandBfetchedClocks += 1
		if MLA_MAC_COLUMN_MOD != MLA_MAC_COLUMN and self.operandBfetchedClocks < MLA_MAC_COLUMN_DECREASE_TIME:
			fetchBFlag = False
		if self.isOperABvalid():
			# print("*******CONV******")
			# Update operA and operB
			self.compClocksCounter += 1
			self.operA = 0
			self.operB -= 1
			# Determine if Row/Round/CONV is finish
			if self.convInWidthRoundCounter == self.convInWidthRound:
				# Row for one Round finish
				self.operandBfetchedClocks = 0
				self.operBBuffer = 0
				self.operB = 0
				fetchBFlag = True
				self.convInWidthRoundCounter = 0
				self.convInHeightRoundCounter += 1

				if self.convInHeightRoundCounter == self.convInHeightRound:
					self.convInHeightRoundCounter = 0
					self.convInChannelRoundCounter += 1

					if self.convInChannelRoundCounter == self.convInChannelRound:
						self.operABuffer = 0
						self.operA = 0
						self.operBBuffer = 0
						self.operB = 0
						fetchAFlag = False
						fetchBFlag = False
						self.convFetchOperATimesCounter = 0
						self.mlaState = MlaState.RESULT_OUT_STAGE
						# One Round finish				
						self.convInChannelRoundCounter = 0
						# Update the outWidth, outChannal and base addr of this round for output to SRAM
						self.convOutActiWidthCounter = 0
						self.convOutActiChannelCounter = 0
						self.convFilterOutChannelEachRound = self.convOutChannelRoundList[self.convOutChannelRoundCounter]
						self.convOutActiWidthEachRound = self.convOutWidthRoundList[self.convOutWidthRoundCounter]			
						self.convOutWidthRoundCounter += 1
						if self.convOutWidthRoundCounter == self.convOutWidthRound:
							self.convOutWidthRoundCounter = 0
							self.convOutHeightRoundCounter += 1
							if self.convOutHeightRoundCounter == self.convOutHeightRound:
								self.convOutHeightRoundCounter = 0
								self.convOutChannelRoundCounter += 1
								if self.convOutChannelRoundCounter == self.convOutChannelRound:
									# CONV finish
									self.convOutChannelRoundCounter = 0
									self.mlaFinishFlag = True
						self.convInWidthRound = self.convOutWidthToInWidth(self.convOutWidthRoundList[self.convOutWidthRoundCounter])
		return self.convRequestData(fetchAFlag, fetchBFlag)

	def convGetRound(self, width, height, channelBatch):
		outWidthRound = []
		while width > 0:
			if width > MLA_MAC_COLUMN_MOD:
				outWidthRound.append(MLA_MAC_COLUMN_MOD)
			else:
				outWidthRound.append(width)
			width -= MLA_MAC_COLUMN_MOD
		outHeightRound = [1]*height
		outChannelRound = []
		while channelBatch > 0:
			if channelBatch >= MLA_MAC_ROW_MOD:
				outChannelRound.append(MLA_MAC_ROW_MOD)
			else:
				outChannelRound.append(channelBatch)
			channelBatch -= MLA_MAC_ROW_MOD			
		return outWidthRound, outHeightRound, outChannelRound

	def convOutWidthToInWidth(self, outWidth):
		return (outWidth - 1) * self.convStride + self.convFilterWidth

	def convInWidthToOutWidth(self, inWidth):
		return (inWidth - self.convFilterWidth) // self.convStride + 1

	def convRequestData(self, fetchAFlag, fetchBFlag):
		fetchBFlag = fetchBFlag & (not self.alreadySendOperBReq)
		if fetchAFlag and fetchBFlag and self.operB == 0:
			task = {TASK_NAME: Task.NOC_SRAM_DATA_REQ, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
													TASK_SOURCE_2: self.operBBufferId,
													TASK_DESTINATION: self.pairSramIdGenerator(),
													TASK_SOURCE: self.operABufferId}
			self.alreadySendOperBReq = True
		else:
			if fetchAFlag and fetchBFlag and self.operB != 0:
				task = {TASK_NAME: Task.NOC_SRAM_DATA_REQ, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
														TASK_SOURCE_2: self.operBBufferId,
														TASK_DESTINATION: self.pairSramIdGenerator(),
														TASK_SOURCE: self.operABufferId}
				self.alreadySendOperBReq = True
			elif fetchAFlag:
				task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION: self.pairSramIdGenerator(), 
														TASK_SOURCE: self.operABufferId}
			elif fetchBFlag and self.operB == 0:
				task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
														TASK_SOURCE_2: self.operBBufferId}
				self.alreadySendOperBReq = True
			elif fetchBFlag and self.operB != 0:
				task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
														TASK_SOURCE_2: self.operBBufferId}
				self.alreadySendOperBReq = True
			else:
				task = {TASK_NAME:Task.TASK_NONE}
		return task

	def convOperBBufferPop(self):
		if self.operB == MLA_MAC_COLUMN_MOD:
			return self.operBBuffer == 0
		if self.operBBuffer > 0:
			if self.operB == 0:
				self.operB = MLA_MAC_COLUMN_MOD
				self.operBBuffer -= MLA_MAC_COLUMN_MOD
				if self.convOutWidthRoundList[self.convOutWidthRoundCounter] >= MLA_MAC_COLUMN_MOD:
					self.convInWidthRoundCounter += MLA_MAC_COLUMN_MOD
				else:
					self.convInWidthRoundCounter = self.convInWidthToOutWidth(self.convInWidthRound)
			else:
				self.operB = MLA_MAC_COLUMN_MOD
				self.operBBuffer -= 1 
				self.convInWidthRoundCounter += 1
		# Indicate if need to fetch new data
		return self.operBBuffer == 0

	def convOperABufferPop(self):
		if self.operABuffer > 0 and self.operA == 0:
			# self.customAssert(len(self.operABuffer)>=4, "operABuffer not enough: {}".format(self.operABuffer))
			self.operABuffer -= MLA_MAC_ROW_MOD
			self.operA = MLA_MAC_ROW_MOD
			# if len(self.operABuffer) == 0:
			# 	self.operABuffer = None
		# Every 4 clocks fetch next OperA
		if self.convFetchOperATimesCounter < self.convFetchOperATimesReg:
			self.convFetchOperAIntervalCounter += 1
			if self.convFetchOperAIntervalCounter >= self.convFetchOperAIntervalReg:
				self.convFetchOperATimesCounter += 1
				self.convFetchOperAIntervalCounter = 0
				return True	# Fetch another 128-bit
			else:
				return False # Not fetch another 128-bit
		else:
			self.convFetchOperAIntervalCounter = self.convFetchOperAIntervalReg
			return False # Not fetch another 128-bit

	def isOperABvalid(self):
		# if self.componentId[2] == 8:
		# 	if self.operA == None or len(self.operA) < 4:
		# 		print("--- MLA --- {:<7}: {}".format(self.mlaRunClks, "oper A not enough"))
		# 	if self.operB == None or len(self.operB) < 16:
		# 		print("--- MLA --- {:<7}: {}".format(self.mlaRunClks, "oper B not enough"))		
		return self.operA == MLA_MAC_ROW_MOD and self.operB == MLA_MAC_COLUMN_MOD

	# ===================== CONV Output-Acti ====================
	def convOutputResult(self):
		# Fetch Oper A
		if not self.mlaFinishFlag:
			fetchAFlag = self.convOperABufferPop()
			fetchATask = self.convRequestData(fetchAFlag, False)
		else:
			fetchATask = {TASK_NAME:Task.TASK_NONE}
		# Write output
		self.writeOutActiClocksCounter += 1
		# 
		self.convOutActiWidthCounter += 4
		if self.convOutActiWidthCounter >=  self.convOutActiWidthEachRound:
			self.convOutActiWidthCounter = 0
			self.convOutActiChannelCounter += 1
			if self.convOutActiChannelCounter >= self.convFilterOutChannelEachRound:
				self.convOutActiChannelCounter = 0
				if self.mlaFinishFlag:
					self.mlaState = MlaState.FINISH_STAGE
				else:
					self.mlaState = MlaState.COMP_STAGE
		outputActiTask = {TASK_NAME:Task.SRAM_WR_32, TASK_DESTINATION:self.selfsramIdGenerator()}
		if self.icproValidateNoti and MlaState.FINISH_STAGE == self.mlaState:
			outputActiTask[TASK_ADDITION] = IcproValidation.MLA_FINISH
		return [fetchATask, outputActiTask]

	# ===========================================================
	# 							 MM
	# ===========================================================
	def mmRunning(self):
		if MlaState.COMP_STAGE == self.mlaState:
			return self.mmCompute()
		elif MlaState.RESULT_OUT_STAGE == self.mlaState:
			return self.mmOutputResult()
		elif MlaState.FINISH_STAGE == self.mlaState:
			mlaTaskParam = copy.deepcopy(self.mlaTaskParam)
			self.mlaMmReset()
			# Let task queue empty
			task = self.getNextTask()
			taskName = task[TASK_NAME]
			if Task.MLA_EXE == taskName:
				finishTaskName = Task.MLA_FINISH
			elif Task.MLA_EXE_SRAM == taskName:
				finishTaskName = Task.MLA_EXE_SRAM_FINISH
			else:
				self.customAssert(False, "Unsupport MLA task: {}".format(task))
			return {TASK_NAME:finishTaskName, TASK_SOURCE: self.selfPeIdGenerator(), 
					TASK_MLA_PARAM: mlaTaskParam, TASK_DESTINATION: HOST_ID, TASK_MLA_TOTAL_CLKS:self.mlaRunClks,
					TASK_MLA_COMP_CLKS:self.compClocksCounter, TASK_MLA_OUTACTI_WR_CLKS:self.writeOutActiClocksCounter}
		else:
			self.customAssert(False, "Unsupport state: {}".format(self.mlaState))	
	# ===================== MM ====================
	def mmGetRound(self, outWidth, outHeight):
		outWidthRound = []
		while outWidth > 0:
			if outWidth > MLA_MAC_COLUMN_MOD:
				outWidthRound.append(MLA_MAC_COLUMN_MOD)
			else:
				outWidthRound.append(outWidth)
			outWidth -= MLA_MAC_COLUMN_MOD
		outHeightRound = []
		while outHeight > 0:
			if outHeight > MLA_MAC_ROW_MOD:
				outHeightRound.append(MLA_MAC_ROW_MOD)
			else:
				outHeightRound.append(outHeight)
			outHeight -= MLA_MAC_ROW_MOD
		return outWidthRound, outHeightRound

	def mmCompute(self):
		fetchAFlag = self.mmOperABufferPop()
		fetchBFlag = self.mmOperBBufferPop()
		if self.isOperABvalid():
			# Update operA and operB
			self.compClocksCounter += 1
			# input("---compute one time: {}".format(self.compClocksCounter))
			self.operA = 0
			self.operB = 0
			self.mmOneRoundCounter += 1
			if self.mmOneRoundCounter == self.mmOneRoundReg:
				# input("---finish one round")
				self.mmOneRoundCounter = 0
				# For output computation result
				self.mlaState = MlaState.RESULT_OUT_STAGE
				self.mmOutActiWidthCounter = 0
				self.mmOutActiHeightCounter = 0
				self.mmOutActiWidthReg = self.mmOutWidthRoundList[self.mmOutWidthRoundCounter]
				self.mmOutActiHeightReg = self.mmOutHeightRoundList[self.mmOutHeightRoundCounter]
				# For fetching operA and operB
				fetchAFlag = False
				fetchBFlag = False
				self.operABuffer = 0
				self.operBBuffer = 0
				self.operA = 0
				self.operB = 0
				self.mmFetchOperATimesCounter = 0
				self.mmFetchOperAIntervalCounter = self.mmFetchOperAIntervalReg
				# 
				self.mmOutWidthRoundCounter += 1	
				if self.mmOutWidthRoundCounter == self.mmOutWidthRoundReg:
					self.mmOutWidthRoundCounter = 0
					self.mmOutHeightRoundCounter += 1
					if self.mmOutHeightRoundCounter == self.mmOutHeightRoundReg:
						self.mlaFinishFlag = True
		return self.mmRequestData(fetchAFlag, fetchBFlag)

	def mmOperABufferPop(self):
		if self.operABuffer > 0 and self.operA == 0:
			# self.customAssert(len(self.operABuffer)>=4, "operABuffer not enough: {}".format(self.operABuffer))
			self.operABuffer -= MLA_MAC_ROW_MOD
			self.operA = MLA_MAC_ROW_MOD
		if self.mmfetchOperAFromSelf:
			return self.operABuffer <= 8
		else:
			if self.mmFetchOperATimesCounter < self.mmFetchOperATimesReg:
				if (self.operABuffer <= 8) and (self.alreadySendOperAReq == False):
					self.mmFetchOperATimesCounter += 1
					return True
				else:
					return False
			else:
				return False

	def mmOperBBufferPop(self):
		if self.operB == MLA_MAC_COLUMN_MOD:
			return self.operBBuffer == 0
		if self.operBBuffer > 0 and self.operB == 0:
			self.operB = MLA_MAC_COLUMN_MOD
			self.operBBuffer = 0
		# Indicate if need to fetch new data
		return self.operBBuffer == 0

	def mmRequestData(self, fetchAFlag, fetchBFlag):
		# if self.mmfetchOperAFromSelf:
		fetchAFlag = fetchAFlag & (not self.alreadySendOperAReq)
		fetchBFlag = fetchBFlag & (not self.alreadySendOperBReq)
		if fetchAFlag and fetchBFlag:
			if self.mmfetchOperAFromSelf:
				taskName = Task.DOUBLE_SRAM_DATA_REQ
			else:
				taskName = Task.NOC_SRAM_DATA_REQ
			task = {TASK_NAME: taskName, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
				TASK_SOURCE_2: self.operBBufferId, TASK_DESTINATION: self.pairSramIdGenerator(),
				TASK_SOURCE: self.operABufferId}
			self.alreadySendOperAReq = True
			self.alreadySendOperBReq = True			
		else:
			if fetchAFlag:
				if self.mmfetchOperAFromSelf:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.pairSramIdGenerator(), 
															TASK_SOURCE_2: self.operABufferId}
				else:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION: self.pairSramIdGenerator(), 
															TASK_SOURCE: self.operABufferId}
				self.alreadySendOperAReq = True
			elif fetchBFlag:
				task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
														TASK_SOURCE_2: self.operBBufferId}
				self.alreadySendOperBReq = True
			else:
				task = {TASK_NAME:Task.TASK_NONE}
		return task	

	def mmOutputResult(self):
		# Fetch Oper A

		# Write output
		self.writeOutActiClocksCounter += 1
		# 
		self.mmOutActiWidthCounter += 4
		if self.mmOutActiWidthCounter >=  self.mmOutActiWidthReg:
			self.mmOutActiWidthCounter = 0
			self.mmOutActiHeightCounter += 1
			if self.mmOutActiHeightCounter >= self.mmOutActiHeightReg:
				self.mmOutActiHeightCounter = 0
				if self.mlaFinishFlag:
					self.mlaState = MlaState.FINISH_STAGE
				else:
					self.mlaState = MlaState.COMP_STAGE
		outputActiTask = {TASK_NAME:Task.SRAM_WR_32, TASK_DESTINATION:self.selfsramIdGenerator()}
		if self.icproValidateNoti and MlaState.FINISH_STAGE == self.mlaState:
			outputActiTask[TASK_ADDITION] = IcproValidation.MLA_FINISH
		if self.mmfetchOperAFromSelf:
			return outputActiTask
		else:
			# if not self.mlaFinishFlag:
			# 	fetchAFlag = self.mmOperABufferPop()
			# 	fetchATask = self.mmRequestData(fetchAFlag, False)
			# else:
			# 	fetchATask = {TASK_NAME: Task.TASK_NONE}
			# return [fetchATask, outputActiTask]
			return outputActiTask
		