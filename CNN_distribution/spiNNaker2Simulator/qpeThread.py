import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
# print("System path: {}".format(sys.path))
from spiNNakerSimulatorGeneral import *
from qpeSimulator import QuadPE
import queue
import threading
import time
from icproValidate.dataGenerator import convDataSplitter
import numpy as np
import datetime


class QpeThread(threading.Thread):
	def __init__(self, clockQueue, inQueue, outQueue, threadIdQpeId, nocDoubleFreq=False, noDataTran=True):
		'''
		threadIdQpeId = [y,x]
		'''
		threading.Thread.__init__(self)
		self.nocDoubleFreq = nocDoubleFreq
		self.noDataTran = noDataTran
		self.qpe = QuadPE(threadIdQpeId, nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)
		self.clockQueue = clockQueue
		self.inQueue = inQueue
		self.outQueue = outQueue
		self.threadId = threadIdQpeId
		self.clockCounter = 0

	def customAssert(self, condition, content):
		funcName = sys._getframe().f_code.co_name
		lineNumber = sys._getframe().f_lineno
		fileName = sys._getframe().f_code.co_filename
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def run(self):
		while True:
			clockTask = self.clockQueue.get()
			if Task.TASK_CLOCK == clockTask:
				self.qpeSingleStep()
			else:
				self.customAssert(Task.TASK_NO_CLOCK==clockTask, "Clock task-{} should be TASK_NO_CLOCK".format(clockTask))
				break

	def qpeSingleStep(self):
		# Get out all tasks out of inQueue
		try:
			while True:
				task = self.inQueue.get(block=False)
				self.qpe.commandFrom(task)
		except queue.Empty as emptyExce:
			pass
		# Run QuadPE one time and put task into outQueue if necessary
		rsp = self.qpe.run()
		self.outQueue.put(rsp)
		# if self.nocDoubleFreq:
		# 	for singleRsp in rsp:
		# 		self.outQueue.put(singleRsp)
		# else:
		# 	self.outQueue.put(rsp)


# ===========================================================
# 						Validation
# ===========================================================
def qpeThreadValidate():
	inQueue = queue.Queue()
	outQueue = queue.Queue()
	clockQueue = queue.Queue()
	qpeThread = QpeThread(clockQueue, inQueue, outQueue, [0,0])
	qpeThread.start()
	for clockIndex in range(30):
		if clockIndex == 10:
			print("Put sram_wr task")
			inQueue.put({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION: [0,0,0], TASK_SRAM_ADDR: 0x8100, TASK_DATA_LEN: 8, 
						TASK_DATA: [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]})
		if clockIndex == 15:
			print("Put sram_rd task")
			inQueue.put({TASK_NAME:Task.SRAM_RD, TASK_DESTINATION: [0,0,0], TASK_SRAM_ADDR: 0x8100, TASK_DATA_LEN: 8, 
						TASK_SOURCE: [0, 1, 13]})
		clockQueue.put(Task.TASK_CLOCK)
		# Get outTask Every Task
		task = outQueue.get()
		if task[TASK_NAME] != Task.TASK_NONE:
			print("TASK from qpe[0,0]: {}".format(task))		
	clockQueue.put(Task.TASK_NO_CLOCK)
	qpeThread.join()

# mlaValidate(inWidth=30, inHeight=10, inChannel=256, outChannel=4)
def qpeMlaValidate(inWidth=20, inHeight=4, inChannel=3, filterWidth=3, filterHeight=3, outChannel=5, stride=1, peIndex=0):
	inQueue = queue.Queue()
	outQueue = queue.Queue()
	clockQueue = queue.Queue()
	qpeThread = QpeThread(clockQueue, inQueue, outQueue, [0,0])
	qpeThread.start()
	# MLA_EXE task
	inActiBaseAddr = 0x8100
	weightBaseAddr = 0x8500
	outActiBaseAddr = 0x8900
	mlaParam = (MlaOperType.CONV, (inWidth,inHeight,inChannel,filterWidth,filterHeight,outChannel,stride))
	task = {TASK_NAME:Task.MLA_EXE, TASK_DESTINATION:[0,0,peIndex], TASK_MLA_PARAM:mlaParam, TASK_OPER_A_PEID:[0,0,0], 
			TASK_OPER_A_ADDR: weightBaseAddr, TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr}
	outWidth = (inWidth-filterWidth) // stride + 1
	outWidthAlign = math.ceil(outWidth/4) * 4
	outHeight = (inHeight-filterHeight) // stride + 1
	outActiTimesReg = outWidthAlign * outHeight * outChannel // 4
	# Running
	clockCounter = 0
	outActiCounter = 0
	while True:
		if clockCounter == 0:
			inQueue.put(task)
		clockQueue.put(Task.TASK_CLOCK)
		outTask = outQueue.get()
		clockCounter += 1
		if outTask[TASK_NAME] == Task.MLA_FINISH:
			print("MLA_FINISH: {}".format(outTask))	
		elif outTask[TASK_NAME] == Task.SRAM_DATA_32:
			print("SRAM_DATA_32: {}".format(outTask))	
			outActiCounter += 1
			if outActiCounter == outActiTimesReg:
				break
	# End qpe thread
	clockQueue.put(Task.TASK_NO_CLOCK)
	qpeThread.join()
	# Print result
	print("clockCounter: {}".format(clockCounter))
	print("compClocksCounter: {}".format(qpeThread.qpe.peArray[peIndex].mla.compClocksCounter))
	print("writeOutActiClocksCounter: {}".format(qpeThread.qpe.peArray[peIndex].mla.writeOutActiClocksCounter))	

def qpeMlaValidateWithData(inWidth=33, inHeight=4, inChannel=3, filterWidth=3, filterHeight=3, outChannel=5, stride=1,
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
	# Generate taskqueue
	taskQueue = deque([])
	qpeId = [0, 0]
	targetPeId =  copy.deepcopy(qpeId)
	targetPeId.append(peIndex)
	pairPeId = copy.deepcopy(qpeId)
	pairPeId.append(pairPeIndex)
	# Generate inActi load task
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
		taskQueue.append(copy.deepcopy(task))
		baseIndex += dataLen
		address += dataLen
	# Generate weight load task
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
		taskQueue.append(copy.deepcopy(task))
		baseIndex += dataLen
		address += dataLen
	# Generate MLA task
	mlaParam = (MlaOperType.CONV, (inWidth,inHeight,inChannel,filterWidth,filterHeight,outChannel,stride))
	mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_DESTINATION:targetPeId, TASK_MLA_PARAM:mlaParam, 
			TASK_OPER_A_PEID:pairPeId, TASK_OPER_A_ADDR: weightBaseAddr, TASK_OPER_B_ADDR:inActiBaseAddr, 
			TASK_OPER_C_ADDR:outActiBaseAddr, TASK_OUTACTI_DRAM_ADDR: 0x5550}
	taskQueue.append(copy.deepcopy(mlaTask))
	# Load inActi/weight, Compute and receive outActi
	outActiTimesReg = len(outActiAlign) // 4
	inQueue = queue.Queue()
	outQueue = queue.Queue()
	clockQueue = queue.Queue()
	qpeThread = QpeThread(clockQueue, inQueue, outQueue, [0,0], noDataTran=noDataTran)
	qpeThread.start()
	clockCounter = 0
	outActiCounter = 0
	outActi = []
	while True:
		if len(taskQueue) > 0:
			nextTask = taskQueue.popleft()
			# print("---> {}".format(nextTask))
			inQueue.put(nextTask)
		clockQueue.put(Task.TASK_CLOCK)
		outTask = outQueue.get()
		clockCounter += 1
		if outTask[TASK_NAME] == Task.MLA_FINISH:
			print("MLA_FINISH: {}".format(outTask))	
		elif outTask[TASK_NAME] == Task.SRAM_DATA_32:
			# print("SRAM_DATA_32: {}".format(outTask))
			assert([4]==outTask[TASK_DESTINATION]), "invalid DRAM ID"
			assert([4]==outTask[TASK_NEXT_DESTINATION]), "invalid next destination"
			if noDataTran == False:
				assert(outTask[TASK_SRAM_ADDR] >= 0x5550), "invalid outActi Addr"
			if not noDataTran:
				data = outTask[TASK_DATA]
				outActi.extend(data)
			outActiCounter += 1
			if outActiCounter == outActiTimesReg:
				# break
				print("Receive all outActi")
		elif outTask[TASK_NAME] == Task.DATA_MIGRATION_32_FINISH:
			break
	clockQueue.put(Task.TASK_NO_CLOCK)
	qpeThread.join()
	# Determine if PASS
	if noDataTran:
		print("--"*20 + " Not available " + "--"*20)
	else:
		outActiArray = np.array(outActi, dtype=np.uint32)
		if np.array_equal(outActiArray, outActiAlign):
			print("--"*20 + " PASS " + "--"*20)
		else:
			print("--"*20 + " FAILED " + "--"*20)
	# Print result
	print("clockCounter: {}".format(clockCounter))
	print("compClocksCounter: {}".format(qpeThread.qpe.peArray[peIndex].mla.compClocksCounter))
	print("writeOutActiClocksCounter: {}".format(qpeThread.qpe.peArray[peIndex].mla.writeOutActiClocksCounter))
	print(datetime.datetime.now())

if __name__ == "__main__":
	# qpeThreadValidate()
	# qpeMlaValidate()
	qpeMlaValidateWithData(inWidth=14, inHeight=10, inChannel=256, outChannel=4, noDataTran=False)
	print("\n\n")
	qpeMlaValidateWithData(inWidth=14, inHeight=10, inChannel=256, outChannel=4, noDataTran=True)