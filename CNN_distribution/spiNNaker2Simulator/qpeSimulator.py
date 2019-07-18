import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
from spiNNakerSimulatorGeneral import *
from peSimulator import ProcessElement
from peSimulatorNoDataTran import ProcessElementNoData
from nocSimulator import NoC
import copy
from icproValidate.dataGenerator import convDataSplitter
import numpy as np
import datetime
import time


class QuadPE(GeneralClass):
	def __init__(self, componentId, queueSize=None, nocDoubleFreq=False, noDataTran=True, printFlag=True,
		sramHalfFreq=False, icproValidateNoti=False):
		self.customAssert(isinstance(componentId, list) and len(componentId)==2, 
			"It is not a QPE ID: {}".format(componentId))
		GeneralClass.__init__(self, componentId, queueSize)
		self.printFlag = printFlag
		self.noDataTran = noDataTran
		self.nocDoubleFreq = nocDoubleFreq
		if self.nocDoubleFreq:
			self.nocPeProcessTimes = 2
		else:
			self.nocPeProcessTimes = 1
		self.sramHalfFreq = sramHalfFreq
		self.icproValidateNoti = icproValidateNoti
		self.noc = NoC(self.nocIdGenerator())
		self.pesInitiator()
		self.peRunTime = 0
		self.nocRunTime = 0
		self.clockCounter = 0
		# Used for NoC PE channel, works as buffer to store the tasks from PE
		self.peToNocChannelQueue = [[],[],[],[]]

	def printInfo(self, content):
		if self.printFlag:
			print("---> {:<7}: {}-{}".format(self.clockCounter, "Qpe", content))

	def nocIdGenerator(self):
		nocCompId = copy.deepcopy(self.componentId)
		nocCompId.append(NOC_ID)
		return nocCompId

	def peIdGenerator(self, peIndex):
		peCompId = copy.deepcopy(self.componentId)
		peCompId.append(PE_ID_START+peIndex)
		return peCompId

	def pesInitiator(self):
		self.peArray = []
		for peIndex in range(NUM_PES_IN_QPE):
			if self.noDataTran:
				pe = ProcessElementNoData(self.peIdGenerator(peIndex), noc=self.noc, 
					sramHalfFreq=self.sramHalfFreq, icproValidateNoti=self.icproValidateNoti, 
					nocDoubleFreq=self.nocDoubleFreq)
			else:
				pe = ProcessElement(self.peIdGenerator(peIndex), noc=self.noc, noDataTran=self.noDataTran)
			self.peArray.append(pe)

	def isForLocalQpe(self, compId):
		if len(compId) < 2:
			return False
		return compId[0:2] == self.componentId

	def whichPe(self, compId):
		return compId[2] % 4

	def isIdle(self):
		idleFlag = self.noc.isIdle()
		for peIndex in range(NUM_PES_IN_QPE):
			idleFlag = idleFlag & self.peArray[peIndex].isIdle()
		return idleFlag

	def run(self):
		# self.printInfo(self.noc.taskQueue)
		self.clockCounter += 1
		self.peOperationProcess()
		self.nocOperationProcess(1)
		if self.nocDoubleFreq:
			self.nocOperationProcess(2)
		if self.nocDoubleFreq:
			return self.doubleCommandTo()
		else:
			return self.commandTo()

	def peOperationProcess(self):
		for peIndex in range(NUM_PES_IN_QPE):
			tasksToNoc = self.peArray[peIndex].run()
			self.peToNocChannelQueue[peIndex].extend(tasksToNoc)
		# input("---> {:<7}: pe running -> {}".format(self.clockCounter, self.numOfRemainingPeToNocTasks))

	def nocOperationProcess(self, index):
		rsp = self.noc.transfer()
		if Task.TASK_NONE == rsp[TASK_NAME]:
			pass
		elif TASK_DESTINATION in rsp:
			# TODO: to local PE
			if self.isForLocalQpe(rsp[TASK_DESTINATION]):
				peIndex = self.whichPe(rsp[TASK_DESTINATION])
				self.peArray[peIndex].commandFromNoC(rsp)
				# self.nocToPeChannelQueue[peIndex].append(rsp)
			# TODO: to other QPE
			else:
				self.addTask(rsp)
		else:
			self.customAssert(False, "Unsupport task: {}".format(rsp))
		self.processPeTasksToNoc()

	def processPeTasksToNoc(self):
		for peIndex in range(NUM_PES_IN_QPE):
			if len(self.peToNocChannelQueue[peIndex]) > 0:
				self.commandFrom(self.peToNocChannelQueue[peIndex].pop(0))

	# ===========================================================
	# 				(NoC) Interface to outside
	# ===========================================================
	def commandFrom(self, task):
		self.noc.addTask(task)

	def commandTo(self):
		# TODO: to other QPE
		if len(self.taskQueue) > 0:
			# return self.taskQueue.popleft()
			return self.getNextTask()
		else:
			return {TASK_NAME: Task.TASK_NONE}

	def doubleCommandTo(self):
		if len(self.taskQueue) >= 2:
			return [self.getNextTask(), self.getNextTask()]
		elif len(self.taskQueue) == 1:
			return [self.getNextTask()]
		else:
			return [{TASK_NAME: Task.TASK_NONE}]



# ===========================================================
# 						Validation
# ===========================================================
def sramWrValidate():
	qpe = QuadPE(componentId=[0,0])
	taskQueue = deque([])
	# Write data to SRAM:0x8100
	taskQueue.append({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION: [0,0,0], TASK_SRAM_ADDR: 0x8100, TASK_DATA_LEN: 8, 
						TASK_DATA: [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]})
	taskQueue.append({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION: [0,0,1], TASK_SRAM_ADDR: 0x8100, TASK_DATA_LEN: 16, 
						TASK_DATA: [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x0]})
	taskQueue.append({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION: [0,0,2], TASK_SRAM_ADDR: 0x8100, TASK_DATA_LEN: 16, 
						TASK_DATA: [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf]})
	taskQueue.append({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION: [0,0,3], TASK_SRAM_ADDR: 0x8100, TASK_DATA_LEN: 8, 
						TASK_DATA: [0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01]})
	# 
	clockCounter = 0
	while not (len(taskQueue) == 0 and qpe.isIdle()):
		if len(taskQueue) > 0:
			qpe.commandFrom(taskQueue.popleft())
		qpe.run()
		clockCounter += 1
	# 
	print("clockCounter: {}".format(clockCounter))
	for peIndex in range(NUM_PES_IN_QPE):
		data = qpe.peArray[peIndex].sram.readSram(0x8100, 16)
		print("PE{}: {}".format(peIndex, data))	

# mlaValidate(inWidth=30, inHeight=10, inChannel=256, outChannel=4)
def mlaValidate(inWidth=20, inHeight=4, inChannel=3, filterWidth=3, filterHeight=3, outChannel=5, stride=1, peIndex=0):
	qpe = QuadPE(componentId=[0,0])
	taskQueue = deque([])
	# MLA_EXE
	inActiBaseAddr = 0x8100
	weightBaseAddr = 0x8500
	outActiBaseAddr = 0x8900
	mlaParam = (MlaOperType.CONV, (inWidth,inHeight,inChannel,filterWidth,filterHeight,outChannel,stride))
	task = {TASK_NAME:Task.MLA_EXE, TASK_DESTINATION:[0,0,peIndex], TASK_MLA_PARAM:mlaParam, TASK_OPER_A_PEID:[0,0,0], 
			TASK_OPER_A_ADDR: weightBaseAddr, TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr}
	taskQueue.append(task)
	# # 
	# clockCounter = 0
	# while not (len(taskQueue) == 0 and qpe.isIdle()):
	# 	if len(taskQueue) > 0:
	# 		qpe.commandFrom(taskQueue.popleft())
	# 	qpe.run()
	# 	clockCounter += 1
	# 
	clockCounter = 0
	while True:
		if len(taskQueue) > 0:
			qpe.commandFrom(taskQueue.popleft())
		clockCounter += 1
		rsp = qpe.run()
		if Task.MLA_FINISH == rsp[TASK_NAME]:
			print("MLA task source: {}".format(rsp[TASK_SOURCE]))
			print("MLA Param: {}".format(rsp[TASK_MLA_PARAM]))
			break
	# 
	print("clockCounter: {}".format(clockCounter))
	print("compClocksCounter: {}".format(qpe.peArray[peIndex].mla.compClocksCounter))
	print("writeOutActiClocksCounter: {}".format(qpe.peArray[peIndex].mla.writeOutActiClocksCounter))

def mlaValidateWithData(inWidth=33, inHeight=4, inChannel=3, filterWidth=3, filterHeight=3, outChannel=5, stride=1,
						printInActi=False, printWeight=False, printOutActi=False, peIndex=0, pairPeIndex=1, noDataTran = True):
	'''
	For mlaValidateWithData() When inActi is too large, outActiArray and outActiAlign has a little bit different.
	For "inWidth=30, inHeight=10, inChannel=128, outChannel=4", it works good.
	However, for "inWidth=30, inHeight=10, inChannel=256, outChannel=4", it failed.
	Then I decrease the inwdith for each loop to half, like "inWidth=8, inHeight=10, inChannel=256, outChannel=4",
		It works again.
	The largest input-acti size "inWidth=14, inHeight=10, inChannel=256, outChannel=4"
	I think it because the size is too large, which is limitation of list. However, I have change the self.inActi 
		in MLA to be numpy.ndarray, but it didn't help.
	'''
	# Generate validate data, inActi, weight and outActi
	print(datetime.datetime.now())
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
	# print output-activation, weight, input-activation
	if printOutActi:
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
	if printWeight:
		print("-"*20)
		print("weightAlign: \n{}".format(weightAlign))
		print("outActiAlign: \n")
		for channelIndex in range(math.ceil(outChannel/4)):
			for index in range(filterWidth*filterHeight*inChannel):
				baseIndex = (filterWidth*filterHeight*inChannel*channelIndex + index) * 4
				print(weightAlign[baseIndex:baseIndex+4].tolist())
			print("\n")
		print("-"*20)
	if printInActi:
		print("-"*20)
		print("inActiAlign: \n{}".format(inActiAlign))
		print("inActiAlign: \n")
		inWidthAlign = math.ceil(inWidth/16) * 16
		for channelIndex in range(inChannel):
			for heightIndex in range(inHeight):
				baseIndex = (heightIndex + channelIndex * inHeight) * inWidthAlign
				print(inActiAlign[baseIndex:baseIndex+inWidthAlign].tolist())
			print("\n")
		print("-"*20)
	# Generate MLA task
	taskQueue = deque([])
	qpeId = [0, 0]
	otherQpeId = [0, 1]
	targetPeId =  copy.deepcopy(qpeId)
	targetPeId.append(peIndex)
	pairPeId = copy.deepcopy(qpeId)
	pairPeId.append(pairPeIndex)
	mlaParam = (MlaOperType.CONV, (inWidth,inHeight,inChannel,filterWidth,filterHeight,outChannel,stride))
	task = {TASK_NAME:Task.MLA_EXE, TASK_DESTINATION:targetPeId, TASK_MLA_PARAM:mlaParam, 
			TASK_OPER_A_PEID:pairPeId, TASK_OPER_A_ADDR: weightBaseAddr, TASK_OPER_B_ADDR:inActiBaseAddr, 
			TASK_OPER_C_ADDR:outActiBaseAddr}
	taskQueue.append(task)
	# Instance QPE
	qpe = QuadPE(componentId=qpeId, noDataTran=noDataTran)
	clockCounter = 0
	nocWriteClocks = 0
	nocReadClocks = 0
	# Load inActi
	baseIndex = 0
	taskName = Task.SRAM_WR
	dataLen = NOC_SRAM_BW_BYTES
	destination = targetPeId
	address = inActiBaseAddr
	task = {TASK_NAME:taskName, TASK_DESTINATION:destination, TASK_DATA_LEN: dataLen, TASK_SRAM_ADDR: address}
	while baseIndex < len(inActiAlign):
		data = inActiAlign[baseIndex: baseIndex+dataLen]
		task[TASK_DATA] = data
		task[TASK_SRAM_ADDR] = address
		qpe.commandFrom(copy.deepcopy(task))
		qpe.run()
		baseIndex += dataLen
		address += dataLen
		clockCounter += 1
		nocWriteClocks += 1
	# Load weight
	baseIndex = 0
	taskName = Task.SRAM_WR
	dataLen = NOC_SRAM_BW_BYTES
	destination = pairPeId
	address = weightBaseAddr
	task = {TASK_NAME:taskName, TASK_DESTINATION:destination, TASK_DATA_LEN: dataLen, TASK_SRAM_ADDR: address}
	while baseIndex < len(weightAlign):
		if baseIndex+dataLen <= len(weightAlign):
			data = weightAlign[baseIndex: baseIndex+dataLen]
		else:
			data = weightAlign[baseIndex: baseIndex+dataLen]
			data = np.append(data, np.zeros((baseIndex+dataLen-len(weightAlign), 1)))
		task[TASK_DATA] = data
		task[TASK_SRAM_ADDR] = address
		qpe.commandFrom(copy.deepcopy(task))
		qpe.run()
		baseIndex += dataLen
		address += dataLen
		clockCounter += 1
		nocWriteClocks += 1
	# Compute and receive outActi
	while True:
		if len(taskQueue) > 0:
			qpe.commandFrom(taskQueue.popleft())
		clockCounter += 1
		rsp = qpe.run()
		if Task.MLA_FINISH == rsp[TASK_NAME]:
			print("MLA_FINISH: {}".format(rsp))
			print("MLA task source: {}".format(rsp[TASK_SOURCE]))
			print("MLA Param: {}".format(rsp[TASK_MLA_PARAM]))
			break
	# Automatically get outActi
	outActiAlignSize = len(outActiAlign)
	outActi = []
	addrOffset = 0
	while True:
		clockCounter += 1
		nocReadClocks += 1
		rsp = qpe.run()
		if Task.TASK_NONE == rsp[TASK_NAME]:
			pass
		elif Task.SRAM_DATA_32 == rsp[TASK_NAME]:	
			assert([4]==rsp[TASK_DESTINATION]), "invalid DRAM ID"
			assert([4]==rsp[TASK_NEXT_DESTINATION]), "invalid next destination"
			if not noDataTran:
				data = rsp[TASK_DATA]
				outActi.extend(data)				
			addrOffset += 4
		if addrOffset == outActiAlignSize:
			break
	# # Mannuly get outActi
	# outActiAlignSize = len(outActiAlign)
	# # outActi = qpe.peArray[peIndex].sram.readSram(outActiBaseAddr, outActiAlignSize)
	# outActi = []
	# addrOffset = 0
	# readOutActiTaskQueue = deque([])
	# while addrOffset < outActiAlignSize:
	# 	task = {TASK_NAME:Task.SRAM_RD_32, TASK_DESTINATION:targetPeId, TASK_SRAM_ADDR:outActiBaseAddr+addrOffset, 
	# 			TASK_DATA_LEN:4, TASK_SOURCE:otherQpeId}
	# 	readOutActiTaskQueue.append(task)
	# 	addrOffset += 4
	# receDataCounter = 0
	# while True:
	# 	if len(readOutActiTaskQueue) > 0:
	# 		qpe.commandFrom(readOutActiTaskQueue.popleft())
	# 	clockCounter += 1
	# 	nocReadClocks += 1
	# 	rsp = qpe.run()
	# 	if Task.TASK_NONE == rsp[TASK_NAME]:
	# 		pass
	# 	elif Task.SRAM_DATA_32 == rsp[TASK_NAME]:	
	# 		data = rsp[TASK_DATA]
	# 		receDataCounter += 1
	# 		outActi.extend(data)
	# 	if receDataCounter == outActiAlignSize // 4:
	# 		break
	# Determine if PASS
	if noDataTran:
		print("--"*20 + " Not available " + "--"*20)
	else:
		outActiArray = np.array(outActi, dtype=np.uint32)
		if np.array_equal(outActiArray, outActiAlign):
			print("--"*20 + " PASS " + "--"*20)
		else:
			print("--"*20 + " FAILED " + "--"*20)
	# 
	print("clockCounter: {}".format(clockCounter))
	print("compClocksCounter: {}".format(qpe.peArray[peIndex].mla.compClocksCounter))
	print("writeOutActiClocksCounter: {}".format(qpe.peArray[peIndex].mla.writeOutActiClocksCounter))
	print("nocWriteClocks: {}".format(nocWriteClocks))
	print("nocReadClocks: {}".format(nocReadClocks))
	# print("outActiAlign: \n{}".format(outActiAlign))
	# np.savetxt("outActiAlign.txt", outActiAlign, delimiter=' ')
	# print("outActiArray: \n{}".format(outActiArray))
	# np.savetxt("outActiArray.txt", outActiArray, delimiter=' ')
	print(datetime.datetime.now())

if __name__ == "__main__":
	# mlaValidate(inWidth=30, inHeight=10, inChannel=256, outChannel=4)
	# mlaValidate()
	# sramWrValidate()
	# For mlaValidateWithData() When inActi is too large, outActiArray and outActiAlign has a little bit different
	# mlaValidateWithData(inWidth=14, inHeight=10, inChannel=256, outChannel=4, noDataTran=False)
	# mlaValidateWithData(inWidth=14, inHeight=10, inChannel=256, outChannel=4, noDataTran=True)
	qpe = QuadPE(componentId=[0,0])
	tasksToNoc = [[1], [], [2,3,4], [5,6]]
	print("----")
	qpe.processTasksToNoc(tasksToNoc)