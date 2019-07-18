from spiNNakerSimulatorGeneral import *
import math
import copy
from collections import deque
import tensorflow as tf
import numpy as np
from icproValidate.dataGenerator import convDataSplitter

class MlaState(Enum):
	IDLE = auto()
	COMP_STAGE = auto()
	RESULT_OUT_STAGE = auto()
	FINISH_STAGE = auto()


class MLA(GeneralClass):
	'''
	Machine Learning Accelerator
	'''
	def __init__(self, componentId, queueSize=1, noDataTran=True):
		GeneralClass.__init__(self, componentId, queueSize)
		self.mlaReset()
		self.noDataTran = noDataTran
		self.compClocksCounter = None
		self.writeOutActiClocksCounter = None

	def mlaReset(self):
		# Component ID of operA, operB, operC
		self.operABufferId = self.operAIdGenerator()
		self.operBBufferId = self.operBIdGenerator()
		self.operCBufferId = self.operCIdGenerator()
		self.operABuffer = None
		self.operBBuffer = None
		self.operA = None
		self.operB = None
		# self.alreadySendOperAReq = False
		self.alreadySendOperBReq = False
		# State of MLA
		self.mlaState = MlaState.IDLE
		self.mlaTaskParam = None
		self.operAPeId = None
		self.currentOperation = None
		self.inputActi = None
		self.weight = None
		self.outputActi = None
		# self.compClocksCounter = None
		self.mlaFinishFlag = None
		# self.writeOutActiClocksCounter = None
		# CONV Parameter
		self.convFilterWidth = None
		self.convFilterHeight = None
		self.convFilterInChannel = None
		self.convStride = None
			# For input-activation
		self.convChannelAddrOffset = None
		self.convHeightAddrOffset = None
			# For weight
		self.convOutChannelAddrOffset = None
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
			# Start Address of input-activation, weight, output-activation
		self.convInActiStartAddr = None
		self.convWeightStartAddr = None
		self.convOutActiStartAddr = None
			# For input-activation
		self.convInActiChannelAddrOffset = None
		self.convInActiHeightAddrOffset = None
		self.convInActiWidthAddrOffset = None
			# For weight
		self.convWeightAddr = None
		self.convWeightOutChannelAddrOffset = None
			# For continuely fetch operABuffer
		self.convFetchOperATimesReg = None
		self.convFetchOperATimesCounter = None
		self.convFetchOperAIntervalReg = None
		self.convFetchOperAIntervalCounter = None
		# For output result to SRAM
		self.convFilterOutChannelEachRound = None
		self.convOutActiWidthEachRound = None
		self.convOutActiChannelAddrOffset = None
		self.convOutActiHeightAddrOffset = None
		self.convOutActiBaseAddr = None
		self.convOutActiWidthCounter = None
		self.convOutActiChannelCounter = None
		self.convOutActiArrayBaseIndex = None

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

	def writeOperABuffer(self, data):
		self.customAssert(len(data) != 0, "Data is empty")
		# self.alreadySendOperAReq = False
		if self.operABuffer == None:
			self.operABuffer = deque(data)
		else:
			self.operABuffer.extend(data)

	def writeOperBBuffer(self, data):
		self.customAssert(len(data) != 0, "Data is empty")
		self.alreadySendOperBReq = False
		self.operBBuffer = deque(data)

	def addTask(self, task):
		task = copy.deepcopy(task)
		if self.queueSize != None:
			self.customAssert(len(self.taskQueue) < self.queueSize, "Task queue is full: {}".format(self.taskQueue))
		self.taskQueue.append(task)
		self.mlaSetting()

	def addTaskToTop(self, task):
		task = copy.deepcopy(task)
		if self.queueSize != None:
			self.customAssert(len(self.taskQueue) < self.queueSize, "Task queue is full")
		self.taskQueue.insert(0, task)
		self.mlaSetting()

	def mlaSetting(self):
		'''
		This function is automatically called when adding task to MLA
		'''
		task = self.getNextTaskWithoutPop()
		self.customAssert(task != None, "There should be a task")
		mlataskOperation, taskParam = task[TASK_MLA_PARAM]
		self.customAssert(MlaOperType.CONV == mlataskOperation, "Unsupport MLA operation: {}".format(mlataskOperation))
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
		self.convStride = stride
		inWidthAlign = self.align16(inWidth)
		self.convChannelAddrOffset = inWidthAlign * inHeight
		self.convHeightAddrOffset = inWidthAlign
		self.convOutChannelAddrOffset = filterWidth * filterHeight * inChannel * MLA_MAC_ROW
		outWidth = (inWidth - filterWidth) // stride + 1
		outHeight = (inHeight - filterHeight) // stride + 1
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
		self.convInActiStartAddr = task[TASK_OPER_B_ADDR]
		self.convWeightStartAddr = task[TASK_OPER_A_ADDR]
		self.convOutActiStartAddr = task[TASK_OPER_C_ADDR]
		self.convWeightAddr = self.convWeightStartAddr
		self.convInActiChannelAddrOffset = self.convGetInActiAddr()
		self.convInActiHeightAddrOffset = 0
		self.convInActiWidthAddrOffset = 0
		self.convWeightOutChannelAddrOffset = 0
		self.convFetchOperATimesReg = math.ceil(filterWidth * filterHeight * inChannel * 4 / NOC_SRAM_BW_BYTES)
		self.convFetchOperATimesCounter = 0
		self.convFetchOperAIntervalReg = 4
		self.convFetchOperAIntervalCounter = self.convFetchOperAIntervalReg
		# For output-activation
		outWidthAlign = self.align4(outWidth)
		self.convOutActiChannelAddrOffset = outWidthAlign * outHeight
		self.convOutActiHeightAddrOffset = outWidthAlign
		self.writeOutActiClocksCounter = 0

	def accelerate(self):
		if not self.isRunning():
			return {TASK_NAME: Task.TASK_NONE}
		elif MlaOperType.CONV == self.currentOperation:
			return self.convRunning()
		else:
			self.customAssert(False, "Unsupport MLA Operation: {}".format(self.currentOperation))

	def isRunning(self):
		return MlaState.IDLE != self.mlaState	
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
			self.mlaReset()
			# Let task queue empty
			self.getNextTask()
			return {TASK_NAME:Task.MLA_FINISH, TASK_SOURCE: self.selfPeIdGenerator(), 
					TASK_MLA_PARAM: mlaTaskParam, TASK_DESTINATION: HOST_ID, TASK_MLA_COMP_CLKS:self.compClocksCounter,
					TASK_MLA_OUTACTI_WR_CLKS:self.writeOutActiClocksCounter}
		else:
			self.customAssert(False, "Unsupport state: {}".format(self.mlaState))

	# ===================== CONV Computation ====================
	def convCompute(self):
		fetchAFlag = self.convOperABufferPop()
		fetchBFlag = self.convOperBBufferPop()
		if self.convOperABvalid():
			# print("*******CONV******")
			# Update operA and operB
			self.compClocksCounter += 1
			self.operA = None
			self.operB.popleft()
			# Determine if Row/Round/CONV is finish
			if self.convInWidthRoundCounter == self.convInWidthRound:
				# Row for one Round finish
				self.convInActiWidthAddrOffset = 0
				self.operBBuffer = None
				self.operB = None
				fetchBFlag = True
				self.convInWidthRoundCounter = 0
				self.convInHeightRoundCounter += 1
				self.convInActiHeightAddrOffset += self.convHeightAddrOffset

				if self.convInHeightRoundCounter == self.convInHeightRound:
					self.convInActiHeightAddrOffset = 0
					self.convInHeightRoundCounter = 0
					self.convInChannelRoundCounter += 1
					self.convInActiChannelAddrOffset += self.convChannelAddrOffset

					if self.convInChannelRoundCounter == self.convInChannelRound:
						self.operABuffer = None
						self.operA = None
						self.operBBuffer = None
						self.operB = None
						fetchAFlag = False
						fetchBFlag = False
						self.convFetchOperATimesCounter = 0
						self.mlaState = MlaState.RESULT_OUT_STAGE
						self.convComputeOutActi()
						# print("---------------> one round finish")
						# One Round finish
						self.convWeightAddr = self.convWeightStartAddr					
						self.convInChannelRoundCounter = 0
						# Update the outWidth, outChannal and base addr of this round for output to SRAM
						self.convOutActiWidthCounter = 0
						self.convOutActiChannelCounter = 0
						self.convGetOutActiArrayIndex()
						self.updateOutChannelEachRound()
						self.updateOutActiWidthEachRound()
						self.convOutActiBaseAddr = self.convGetOutActiAddr()					
						self.convOutWidthRoundCounter += 1
						if self.convOutWidthRoundCounter == self.convOutWidthRound:
							self.convOutWidthRoundCounter = 0
							self.convOutHeightRoundCounter += 1
							if self.convOutHeightRoundCounter == self.convOutHeightRound:
								self.convOutHeightRoundCounter = 0
								self.convOutChannelRoundCounter += 1
								self.convWeightOutChannelAddrOffset += self.convOutChannelAddrOffset 
								if self.convOutChannelRoundCounter == self.convOutChannelRound:
									# CONV finish
									self.convOutChannelRoundCounter = 0
									self.mlaFinishFlag = True
								# I think this is wrong. 22.01.2019-17:17
								# else:
								# 	# Get current out-channel (avoiding over-indexing)
								# 	self.updateOutChannelEachRound()
						self.convInWidthRound = self.convOutWidthToInWidth(self.convOutWidthRoundList[self.convOutWidthRoundCounter])
						# print("self.convInWidthRound: {}".format(self.convInWidthRound))
						# Update after updating the out-counter
						self.convInActiChannelAddrOffset = self.convGetInActiAddr()
						# print("---------->convInActiChannelAddrOffset: {}".format(self.convInActiChannelAddrOffset))
		return self.convRequestData(fetchAFlag, fetchBFlag)

	def updateOutChannelEachRound(self):
		self.convFilterOutChannelEachRound = self.convOutChannelRoundList[self.convOutChannelRoundCounter]

	def updateOutActiWidthEachRound(self):
		self.convOutActiWidthEachRound = self.convOutWidthRoundList[self.convOutWidthRoundCounter]

	def convGetRound(self, width, height, channelBatch):
		outWidthRound = []
		while width > 0:
			if width > MLA_MAC_COLUMN:
				outWidthRound.append(MLA_MAC_COLUMN)
			else:
				outWidthRound.append(width)
			width -= MLA_MAC_COLUMN
		outHeightRound = [1]*height
		outChannelRound = []
		while channelBatch > 0:
			if channelBatch >= MLA_MAC_ROW:
				outChannelRound.append(MLA_MAC_ROW)
			else:
				outChannelRound.append(channelBatch)
			channelBatch -= MLA_MAC_ROW			
		return outWidthRound, outHeightRound, outChannelRound

	def convOutWidthToInWidth(self, outWidth):
		return (outWidth - 1) * self.convStride + self.convFilterWidth

	def convInWidthToOutWidth(self, inWidth):
		return (inWidth - self.convFilterWidth) // self.convStride + 1

	def convGetInActiAddr(self):
		addr = 0
		for index in range(self.convOutWidthRoundCounter):
			addr += self.convOutWidthRoundList[index]
		addr += self.convOutHeightRoundCounter * self.convHeightAddrOffset
		addr += self.convInActiStartAddr
		return addr 

	def convRequestData(self, fetchAFlag, fetchBFlag):
		# fetchAFlag = fetchAFlag & (not self.alreadySendOperAReq)
		fetchBFlag = fetchBFlag & (not self.alreadySendOperBReq)
		convInActiAddr = self.convInActiChannelAddrOffset + self.convInActiHeightAddrOffset + self.convInActiWidthAddrOffset
		convFilterAddr = self.convWeightAddr + self.convWeightOutChannelAddrOffset
		if fetchAFlag and fetchBFlag and self.operB == None:
			if self.noDataTran:
				task = {TASK_NAME: Task.NOC_SRAM_DATA_REQ, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
														TASK_SOURCE_2: copy.deepcopy(self.operBBufferId),
														TASK_DESTINATION: self.pairSramIdGenerator(),
														TASK_SOURCE: copy.deepcopy(self.operABufferId)}
			else:					
				task = {TASK_NAME: Task.NOC_SRAM_DATA_REQ, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
														TASK_SRAM_ADDR_2: convInActiAddr, 
														TASK_DATA_LEN_2: 16,
														TASK_SOURCE_2: copy.deepcopy(self.operBBufferId),
														TASK_DESTINATION: self.pairSramIdGenerator(),
														TASK_SRAM_ADDR: convFilterAddr,
														TASK_DATA_LEN: 16,
														TASK_SOURCE: copy.deepcopy(self.operABufferId)}
			self.convWeightAddr += 16
			self.convInActiWidthAddrOffset += 16
			# self.alreadySendOperAReq = True
			self.alreadySendOperBReq = True
		else:
			if fetchAFlag and fetchBFlag and self.operB != None:
				if self.noDataTran:
					task = {TASK_NAME: Task.NOC_SRAM_DATA_REQ, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
															TASK_SOURCE_2: copy.deepcopy(self.operBBufferId),
															TASK_DESTINATION: self.pairSramIdGenerator(),
															TASK_SOURCE: copy.deepcopy(self.operABufferId)}
				else:
					task = {TASK_NAME: Task.NOC_SRAM_DATA_REQ, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
															TASK_SRAM_ADDR_2: convInActiAddr, 
															TASK_DATA_LEN_2: 4,
															TASK_SOURCE_2: copy.deepcopy(self.operBBufferId),
															TASK_DESTINATION: self.pairSramIdGenerator(),
															TASK_SRAM_ADDR: convFilterAddr,
															TASK_DATA_LEN: 16,
															TASK_SOURCE: copy.deepcopy(self.operABufferId)}
				self.convWeightAddr += 16
				self.convInActiWidthAddrOffset += 4
				# self.alreadySendOperAReq = True
				self.alreadySendOperBReq = True
			elif fetchAFlag:
				if self.noDataTran:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION: self.pairSramIdGenerator(), 
															TASK_SOURCE: copy.deepcopy(self.operABufferId)}
				else:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION: self.pairSramIdGenerator(), 
															TASK_SRAM_ADDR: convFilterAddr, 
															TASK_DATA_LEN: 16,
															TASK_SOURCE: copy.deepcopy(self.operABufferId)}
				self.convWeightAddr += 16
				# self.alreadySendOperAReq = True
			elif fetchBFlag and self.operB == None:
				if self.noDataTran:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
															TASK_SOURCE_2: copy.deepcopy(self.operBBufferId)}
				else:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
															TASK_SRAM_ADDR_2: convInActiAddr, 
															TASK_DATA_LEN_2: 16,
															TASK_SOURCE_2: copy.deepcopy(self.operBBufferId)}
				self.convInActiWidthAddrOffset += 16
				self.alreadySendOperBReq = True
			elif fetchBFlag and self.operB != None:
				if self.noDataTran:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
															TASK_SOURCE_2: copy.deepcopy(self.operBBufferId)}
				else:
					task = {TASK_NAME: Task.SRAM_RD, TASK_DESTINATION_2: self.selfsramIdGenerator(), 
															TASK_SRAM_ADDR_2: convInActiAddr, 
															TASK_DATA_LEN_2: 4,
															TASK_SOURCE_2: copy.deepcopy(self.operBBufferId)}
				self.convInActiWidthAddrOffset += 4
				self.alreadySendOperBReq = True
			else:
				task = {TASK_NAME:Task.TASK_NONE}
		return task

	def convOperBBufferPop(self):
		if self.operB != None and len(self.operB) == 16:
			return self.operBBuffer == None
		if self.operBBuffer != None:
			if self.operB == None:
				self.operB = deque([])
				length = 16
			else:
				length = 1
			for index in range(length):
				data = self.operBBuffer.popleft()
				self.operB.append(data)
				if self.inputActi is None:
					# self.inputActi = []
					self.inputActi = np.array([])
				if self.convOutWidthRoundList[self.convOutWidthRoundCounter] >= 16:
					# self.inputActi.append(data)
					self.inputActi = np.append(self.inputActi, data)
					self.convInWidthRoundCounter += 1
			# For outActi width < 16: inputActi
			if self.convOutWidthRoundList[self.convOutWidthRoundCounter] < 16:
				operBCopyList = copy.deepcopy(list(self.operB))
				if length == 16:
					# self.inputActi.extend(operBCopyList[0:self.convInWidthToOutWidth(self.convInWidthRound)])
					self.inputActi = np.append(self.inputActi, operBCopyList[0:self.convInWidthToOutWidth(self.convInWidthRound)])
				else:
					indexTemp = self.convInWidthToOutWidth(self.convInWidthRound)-1
					# self.inputActi.extend([operBCopyList[indexTemp]])
					self.inputActi = np.append(self.inputActi, operBCopyList[indexTemp])
				# if self.convInWidthRound == 4:
				# 	print("self.inputActi: {}".format(self.inputActi))
			# For outActi width < 16: inputActi length
			if self.convOutWidthRoundList[self.convOutWidthRoundCounter] < 16:
				if length == 16:
					self.convInWidthRoundCounter = self.convInWidthToOutWidth(self.convInWidthRound)
				else:
					self.convInWidthRoundCounter += length
				# self.inputActi = self.inputActi[0: self.convInWidthRound]
			self.customAssert(len(self.operB) <= 16, "operB overflow: {}".format(self.operB))
			if len(self.operBBuffer) == 0:
				self.operBBuffer = None
		# Indicate if need to fetch new data
		return self.operBBuffer == None

	def convOperABufferPop(self):
		if self.operABuffer != None and self.operA == None:
			self.operA = deque([])
			self.customAssert(len(self.operABuffer)>=4, "operABuffer not enough: {}".format(self.operABuffer))
			for index in range(4):
				data = self.operABuffer.popleft()
				self.operA.append(data)
				if self.weight == None:
					self.weight = []
				self.weight.append(data)
			if len(self.operABuffer) == 0:
				self.operABuffer = None
		
		if self.convFetchOperATimesCounter < self.convFetchOperATimesReg:
			self.convFetchOperAIntervalCounter += 1
			if self.convFetchOperAIntervalCounter >= self.convFetchOperAIntervalReg:
				self.convFetchOperATimesCounter += 1
				self.convFetchOperAIntervalCounter = 1
				return True	# Fetch another 128-bit
			else:
				return False # Not fetch another 128-bit
		else:
			self.convFetchOperAIntervalCounter = self.convFetchOperAIntervalReg
			return False # Not fetch another 128-bit

	def convOperABvalid(self):
		return self.operA != None and self.operB != None and len(self.operA) == 4 and len(self.operB) == 16

	# ===================== CONV Output-Acti ====================
	def convOutputResult(self):
		self.writeOutActiClocksCounter += 1
		if not self.noDataTran:
			# Get SRAM Addr
			outAddr = self.convOutActiBaseAddr + self.convOutActiChannelCounter * self.convOutActiChannelAddrOffset + \
						self.convOutActiWidthCounter
			# Get 4 uint32 data
			# [batch, outchannel, height, width]
			beginIndex = self.convOutActiArrayBaseIndex + self.convOutActiWidthCounter
			if self.convOutActiWidthCounter + 4 > self.convOutActiWidthEachRound:
				data = self.outputActi[:, self.convOutActiChannelCounter, :, beginIndex:beginIndex+self.convOutActiWidthEachRound]
				data = np.ravel(data).tolist()
				data.extend([0]*(self.convOutActiWidthCounter + 4 - self.convOutActiWidthEachRound))
			else:
				data = self.outputActi[:, self.convOutActiChannelCounter, :, beginIndex:beginIndex+4]
				data = np.ravel(data).tolist()
		# 
		self.convOutActiWidthCounter += 4
		if self.convOutActiWidthCounter >=  self.convOutActiWidthEachRound:
			self.convOutActiWidthCounter = 0
			self.convOutActiChannelCounter += 1
			if self.convOutActiChannelCounter >= self.convFilterOutChannelEachRound:
				self.convOutActiChannelCounter = 0
				self.outputActi = None
				if self.mlaFinishFlag:
					self.mlaState = MlaState.FINISH_STAGE
				else:
					self.mlaState = MlaState.COMP_STAGE
		# Send Task
		if not self.noDataTran:
			return {TASK_NAME:Task.SRAM_WR_32, TASK_DESTINATION:self.selfsramIdGenerator(), 
					TASK_SRAM_ADDR:outAddr, TASK_DATA:data, TASK_DATA_LEN:4}
		else:
			return {TASK_NAME:Task.SRAM_WR_32, TASK_DESTINATION:self.selfsramIdGenerator()}

	def convGetOutActiArrayIndex(self):
		# arrayIndex = 0
		# for index in range(self.convOutWidthRoundCounter):
		# 	arrayIndex += MLA_MAC_COLUMN
		# self.convOutActiArrayBaseIndex = arrayIndex
		self.convOutActiArrayBaseIndex = 0

	def convGetOutActiAddr(self):
		if self.noDataTran:
			return 0
		addr = 0
		for index in range(self.convOutChannelRoundCounter):
			addr +=  MLA_MAC_ROW * self.convOutActiChannelAddrOffset
		for index in range(self.convOutHeightRoundCounter):
			addr += self.convOutActiHeightAddrOffset
		for index in range(self.convOutWidthRoundCounter):
			addr += MLA_MAC_COLUMN
		addr += self.convOutActiStartAddr
		return addr		

	def convComputeOutActi(self):
		if self.noDataTran:
			return
		# Input-activation
			# [batch, inchannel, height, width]
		inputActi = np.reshape(np.array(self.inputActi), (1, self.convFilterInChannel, self.convFilterHeight, -1))
			# [batch, inchannel, height, width] -> [batch, height, width, inchannel]
		inputActi = np.moveaxis(inputActi, 1, 3)
		self.inputActi = None
		# Weight
			# [outchannel/4, inchannel, height, width, 4]
		weight = np.reshape(np.array(self.weight), (-1, self.convFilterInChannel, self.convFilterHeight, self.convFilterWidth, 4))
			# [outchannel/4, inchannel, height, width, 4] -> [inchannel, height, width, outchannel/4, 4]
		weight = np.moveaxis(weight, 0, 3)
			# [inchannel, height, width, outchannel/4, 4] -> [inchannel, height, width, outchannel]
		weight = np.reshape(weight, (self.convFilterInChannel, self.convFilterHeight, self.convFilterWidth, -1))
			# [inchannel, height, width, outchannel] -> [height, width, inchannel, outchannel]
		weight = np.moveaxis(weight, 0, 2)
		self.weight = None
		inputActiFloat = np.array(inputActi, dtype=np.float32)
		weightFloat = np.array(weight, dtype=np.float32)
		# TensorFlow
		inputActiDim = inputActi.shape
		inputActiWidth = inputActiDim[2]
		weightDim = weight.shape
		weightOutChannel = weightDim[3]
		X = tf.placeholder(tf.float32, [1,self.convFilterHeight,inputActiWidth,self.convFilterInChannel])
		W = tf.placeholder(tf.float32, [self.convFilterHeight,self.convFilterWidth,self.convFilterInChannel,weightOutChannel])
		Y = tf.nn.conv2d(X, W, strides = [self.convStride,self.convStride,self.convStride,self.convStride], padding='VALID')
		sess = tf.Session()
		outActiFloat = sess.run(Y, feed_dict={X: inputActiFloat, W: weightFloat})
		outActi = np.array(outActiFloat, dtype=np.uint32)
		# [batch, height, width, outchannel] -> [batch, outchannel, height, width]
		outActi = np.moveaxis(outActi, 3, 1)
		self.outputActi = outActi
		# print("-"*20)
		# print("self.outputActi: \n{}".format(self.outputActi))
		# print("-"*20)

def mlaValidate(inWidth=33, inHeight=4, inChannel=3, filterWidth=3, filterHeight=3, outChannel=5, stride=1):
	outWidth = (inWidth - filterWidth) // stride + 1
	outHeight = (inHeight - filterHeight) // stride + 1
	layerSplitInfo = (0, [[inWidth, 1, 0, 0], [inHeight, 1, 0, 0], inChannel], 
						[[filterWidth, filterHeight, inChannel, [outChannel, 1, 0, 0]], stride], 
						[[outWidth, 1, 0, 0], [outHeight, 1, 0, 0], [outChannel, 1, 0, 0]], 32256, 1)
	layerTypeParameter = (0, (([inWidth,inHeight,inChannel],
								[filterWidth,filterHeight,inChannel,outChannel],stride,
								[outWidth,outHeight,outChannel]), 1))
	inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
	inActiAlign = inActiAlignBlocks[0]
	weightAlign = weightAlignBlocks[0]
	outActiAlign = outActiAlignBlocks[0]
	# output-activation
	print("-"*20)
	print("outActiAlign: \n{}".format(outActiAlign))
	print("outActiAlign: \n")
	outWidthAlign = math.ceil(outWidth/4) * 4
	for channelIndex in range(outChannel):
		for heightIndex in range(outHeight):
			baseIndex = (heightIndex + channelIndex * outHeight) * outWidthAlign
			print(outActiAlign[baseIndex:baseIndex+outWidthAlign].tolist())
		print("\n")
	print("-"*20)	
	# # weight
	# print("-"*20)
	# print("weightAlign: \n{}".format(weightAlign))
	# print("outActiAlign: \n")
	# for channelIndex in range(math.ceil(outChannel/4)):
	# 	for index in range(filterWidth*filterHeight*inChannel):
	# 		baseIndex = (filterWidth*filterHeight*inChannel*channelIndex + index) * 4
	# 		print(weightAlign[baseIndex:baseIndex+4].tolist())
	# 	print("\n")
	# print("-"*20)
	# # input-activation
	# print("-"*20)
	# print("inActiAlign: \n{}".format(inActiAlign))
	# print("inActiAlign: \n")
	# inWidthAlign = math.ceil(inWidth/16) * 16
	# for channelIndex in range(inChannel):
	# 	for heightIndex in range(inHeight):
	# 		baseIndex = (heightIndex + channelIndex * inHeight) * inWidthAlign
	# 		print(inActiAlign[baseIndex:baseIndex+inWidthAlign].tolist())
	# 	print("\n")
	# print("-"*20)
	# 
	mla = MLA(componentId=[0,0,8])
	mlaParam = (MlaOperType.CONV, (inWidth,inHeight,inChannel,filterWidth,filterHeight,outChannel,stride))
	task = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_PEID:[0,0,1], 
			TASK_OPER_A_ADDR: weightBaseAddr, TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr}
	mla.addTaskToTop(task)
	mla.mlaSetting()
	clocksCounter = 0
	weightNocSramData = deque([])
	weightNocSramData2ndStage = deque([])
	weightNocSramData3rdStage = deque([])
	weightNocSramData4thStage = deque([])
	weightNocSramData5thStage = deque([])
	while True:
		clocksCounter += 1
		if len(weightNocSramData5thStage) == 1:
			data = weightNocSramData5thStage.popleft()
			mla.writeOperABuffer(data)
		if len(weightNocSramData4thStage) == 1:
			data = weightNocSramData4thStage.popleft()
			# mla.writeOperABuffer(data)	
			weightNocSramData5thStage.append(data)		
		if len(weightNocSramData3rdStage) == 1:
			data = weightNocSramData3rdStage.popleft()
			# mla.writeOperABuffer(data)	
			weightNocSramData4thStage.append(data)	
		if len(weightNocSramData2ndStage) == 1:
			data = weightNocSramData2ndStage.popleft()
			# mla.writeOperABuffer(data)
			weightNocSramData3rdStage.append(data)
			# print("Write into operABuffer: {}".format(data))
		if len(weightNocSramData) == 1:
			data = weightNocSramData.popleft()
			weightNocSramData2ndStage.append(data)
			# print("Write into NocSramData2ncStage: {}".format(data))
		# print("operA: {}".format(mla.operA))
		# print("operABuffer: {}".format(mla.operABuffer))
		# print("operB: {}".format(mla.operB))
		# print("operBBuffer: {}".format(mla.operBBuffer))
		rsp = mla.accelerate()
		# print("-"*20)
		# print("operA: {}".format(mla.operA))
		# print("operABuffer: {}".format(mla.operABuffer))
		# print("operB: {}".format(mla.operB))
		# print("operBBuffer: {}".format(mla.operBBuffer))
		# print("-"*20)
		if Task.TASK_NONE == rsp[TASK_NAME]:
			pass
		elif Task.MLA_FINISH == rsp[TASK_NAME]:
			break
			# if mla.mlaFinishFlag:
			# 	break
		elif Task.SRAM_WR_32 == rsp[TASK_NAME]:
			data = rsp[TASK_DATA]
			print("data: {}".format(data))
		else:
			assert(TASK_DESTINATION in rsp or TASK_DESTINATION_2 in rsp), "Unknown task:{}".format(rsp)
			if TASK_DESTINATION in rsp:
				dataLen = rsp[TASK_DATA_LEN]
				weightAddr = rsp[TASK_SRAM_ADDR]
				# print("weightAddr: {}, len: {}".format(hex(rsp[TASK_SRAM_ADDR]), dataLen))
				# mla.writeOperABuffer(weightAlign[weightAddr-weightBaseAddr:weightAddr-weightBaseAddr+dataLen])
				data = weightAlign[weightAddr-weightBaseAddr:weightAddr-weightBaseAddr+dataLen]
				# print("beginAddr: {}".format(weightAddr-weightBaseAddr))
				# print("endAddr: {}".format(weightAddr-weightBaseAddr+dataLen))
				# print("longer Data: {}".format(weightAlign[weightAddr-weightBaseAddr:weightAddr-weightBaseAddr+dataLen+dataLen]))
				weightNocSramData.append(data)
				# print("Write into weightNocSramData: {}".format(data))
			if TASK_DESTINATION_2 in rsp:
				dataLen = rsp[TASK_DATA_LEN_2]
				inActiAddr = rsp[TASK_SRAM_ADDR_2]
				# print("inActiAddr: {}, len: {}".format(hex(rsp[TASK_SRAM_ADDR_2]), dataLen))
				inActiData = inActiAlign[inActiAddr-inActiBaseAddr:inActiAddr-inActiBaseAddr+dataLen]
				# print("inActiData: {}".format(inActiData))
				mla.writeOperBBuffer(inActiData)
		# print("Press Enter to continue\n")
		# input("Press Enter to continue\n")
	print("\nclocksCounter: {}".format(clocksCounter))
	print("compClocksCounter: {}".format(mla.compClocksCounter))
	print("writeOutActiClocksCounter: {}".format(mla.writeOutActiClocksCounter))

if __name__ == "__main__":
	# mlaValidate(inWidth=4, inHeight=4, inChannel=3, outChannel=5)
	# mlaValidate(inWidth=17, inHeight=4, inChannel=3, outChannel=5)
	# mlaValidate(inWidth=17, inHeight=4, filterWidth=4, filterHeight=4, inChannel=3, outChannel=5)
	# mlaValidate(inWidth=20, inHeight=4, inChannel=3, outChannel=5)
	mlaValidate(inWidth=20, inHeight=4, filterWidth=3, filterHeight=3, inChannel=3, outChannel=5)