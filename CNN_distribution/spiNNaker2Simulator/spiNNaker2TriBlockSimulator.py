from spiNNakerSimulatorGeneral import *
from triBlockQpeSimulatorProcess import TriBlockQpeProcess
from dramControllerSimulator import DramController
from spiNNakerHostInterface import SpiNNakerHostInterface
from collections import deque
import multiprocessing
import queue


class SpiNNaker2TriBlock():
	def __init__(self, nocDoubleFreq=False, noDataTran=True):
		self.nocDoubleFreq = nocDoubleFreq
		self.noDataTran = noDataTran
		self.ssInQueue = deque([])
		self.ssOutQueue = deque([])
		self.triBlockQpeArrayGenerator()
		self.ssHostInterface = SpiNNakerHostInterface(HOST_INTERFACE_ID, self.ssInQueue, self.ssOutQueue)

	def triBlockQpeArrayGenerator(self):
		self.triBlockQpeClockQueueArray = []
		self.triBlockQpeInQueueArray = []
		self.triBlockQpeOutQueueArray = []
		self.triBlockQpeProcessArray =[]
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			clockQueue = multiprocessing.Queue()
			inQueue = multiprocessing.Queue()
			outQueue = multiprocessing.Queue()
			triBlockQpeProcess = TriBlockQpeProcess(blockIndex, clockQueue, inQueue, outQueue,
				nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)
			triBlockQpeProcess.start()
			self.triBlockQpeClockQueueArray.append(clockQueue)
			self.triBlockQpeInQueueArray.append(inQueue)
			self.triBlockQpeOutQueueArray.append(outQueue)
			self.triBlockQpeProcessArray.append(triBlockQpeProcess)

	# def dramControllerGenerator(self):
	# 	self.dramControllerArray = []
	# 	for sramIndex in range(NUM_DRAMS):
	# 		dramController = DramController([sramIndex+DRAM_ID_START], noDataTran=self.noDataTran)
	# 		self.dramControllerArray.append(dramController)

	# def dramWriteData(self, dramIndex, addr, dataLen, data):
	# 	self.dramControllerArray[dramIndex].dram.writeData(addr=addr, dataLen=dataLen, data=data)

	def customAssert(self, condition, content):
		# funcName = sys._getframe().f_code.co_name
		# lineNumber = sys._getframe().f_lineno
		# fileName = sys._getframe().f_code.co_filename
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def getNextOutTask(self):
		if len(self.ssOutQueue) == 0:
			return {TASK_NAME: Task.TASK_NONE}
		nextTask = self.ssOutQueue.popleft()
		return nextTask

	def addInTask(self, task):
		# self.ssInQueue.append(task)
		self.ssHostInterface.addTask(task)

	def addOutTask(self, task):
		# self.ssOutQueue.append(task)
		self.ssHostInterface.addTask(task)

	def getNextInTask(self):
		if len(self.ssInQueue) == 0:
			return None
		nextTask = self.ssInQueue.popleft()
		return nextTask

	def isFotHost(self, destination):
		return destination == HOST_ID

	# =========================================================================
	# 								Run function
	# =========================================================================
	def spiNNaker2Stop(self):
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			self.triBlockQpeClockQueueArray[blockIndex].put(Task.TASK_NO_CLOCK)
			self.triBlockQpeProcessArray[blockIndex].join()
		
	def spiNNaker2Run(self):
		# Run SpiNNaker2 Host Interface
		self.ssHostInterface.transfer()
		# Processing task from outside of spiNNaker2
		self.hostTaskDistributor()
		# Wakeup all Tri-Block-QPE Process
		self.activeClockGenerate()
		# Get all output Task from all Tri-Block-QPE
		tasks = self.getTasksFromTriBlockQpe()	
		# Run DRAM to get task
		# dramOutTasks = self.getTasksFromDramController()
		# Distribute all tasks into next destination
		# tasks.extend(dramOutTasks)
		self.processOutTasks(tasks)

	def hostTaskDistributor(self):
		task = self.getNextInTask()
		if task == None:
			return
		# # 1 HOST PE -> Cooperate with GeneralClass.hostQpeId()
		# #           -> Cooperate with DistributionGeneralClass.getHostQpeId()
		# task[TASK_NEXT_DESTINATION] = [0,0]
		# self.triBlockQpeInQueueArray[0].put(task)
		# 4 HOST PE -> Cooperate with GeneralClass.hostQpeId()
		# 			-> Cooperate with DistributionGeneralClass.getHostQpeId()
		destId = task[TASK_DESTINATION]
		if len(destId) == 1:
			hostQpeId = self.getHostQpeIdForDram(destId)
			qpeYAxis = hostQpeId[0]
			qpeXAxis = hostQpeId[1]			
		else:
			qpeYAxis = destId[0]
			qpeXAxis = destId[1]
		if qpeXAxis < NUM_QPES_X_AXIS_HALF and qpeYAxis < NUM_QPES_Y_AXIS_HALF:
			task[TASK_NEXT_DESTINATION] = TRIBLOCK_HOST[0]
			self.triBlockQpeInQueueArray[0].put(task)
		elif qpeXAxis >= NUM_QPES_X_AXIS_HALF and qpeYAxis < NUM_QPES_Y_AXIS_HALF:
			task[TASK_NEXT_DESTINATION] = TRIBLOCK_HOST[1]
			self.triBlockQpeInQueueArray[1].put(task)
		elif qpeXAxis < NUM_QPES_X_AXIS_HALF and qpeYAxis >= NUM_QPES_Y_AXIS_HALF:
			task[TASK_NEXT_DESTINATION] = TRIBLOCK_HOST[2]
			self.triBlockQpeInQueueArray[2].put(task)
		elif qpeXAxis >= NUM_QPES_X_AXIS_HALF and qpeYAxis >= NUM_QPES_Y_AXIS_HALF:
			task[TASK_NEXT_DESTINATION] = TRIBLOCK_HOST[3]
			self.triBlockQpeInQueueArray[3].put(task)
		else:
			self.customAssert(False, "Unkown X and Y axis of qpeId")

	def getHostQpeIdForDram(self, destId):
		triBlockIndex = destId[0] % DRAM_ID_START
		return TRIBLOCK_HOST[triBlockIndex]
		# if 0 == triBlockIndex:
		# 	return [0,0]
		# elif 1 == triBlockIndex:
		# 	return [0,3]
		# elif 2 == triBlockIndex:
		# 	return [3,0]
		# elif 3 == triBlockIndex:
		# 	return [3,3]
		# else:
		# 	self.customAssert(False, "Unkown Destination ID")

	def activeClockGenerate(self):
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			self.triBlockQpeClockQueueArray[blockIndex].put(Task.TASK_CLOCK)

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

	# def getTasksFromTriBlockQpe(self):
	# 	tasks = []
	# 	for blockIndex in range(NUM_OF_TRIBLOCKS):
	# 		try:
	# 			while True:
	# 				task = self.triBlockQpeOutQueueArray[blockIndex].get()
	# 				if Task.PROCESS_END == task[TASK_NAME]:
	# 					break
	# 				tasks.append(task)
	# 		except queue.Empty as emptyExce:
	# 			pass
	# 	return tasks

	def getTasksFromTriBlockQpe(self):
		tasks = []
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			while True:
				task = self.triBlockQpeOutQueueArray[blockIndex].get()
				if Task.PROCESS_END == task[TASK_NAME]:
					break
				tasks.append(task)
		return tasks

	# def getTasksFromDramController(self):
	# 	tasks = []
	# 	for dramController in self.dramControllerArray:
	# 		interfaceOutTask = dramController.run()
	# 		if Task.TASK_NONE != interfaceOutTask[TASK_NAME] and Task.TASK_DOWN != interfaceOutTask[TASK_NAME]:
	# 			tasks.append(interfaceOutTask)
	# 	return tasks

	def processOutTasks(self, tasks):
		for task in tasks:
			taskNextDest = task[TASK_NEXT_DESTINATION]
			# -> DRAM/HOST
			if len(taskNextDest) == 1:
				# print("task -> dram/host in spiNNaker2: {}".format(task))
				if self.isFotHost(taskNextDest):
					self.addOutTask(task)
				else:
					# print("task -> dram in spiNNaker2: {}".format(task))
					# dramControllerIndex = taskNextDest[0] - DRAM_ID_START
					# self.dramControllerArray[dramControllerIndex].addTask(task)
					self.customAssert(False, "Only HOST ID can be appear here")
			# -> QPE
			else:
				targetBlockIndex = self.whichBlock(taskNextDest)
				self.triBlockQpeInQueueArray[targetBlockIndex].put(task)

if __name__ == "__main__":
	ss = SpiNNaker2TriBlock(nocDoubleFreq=True, noDataTran=False)
	for blockIndex in range(NUM_OF_TRIBLOCKS):
		for qpe in ss.triBlockQpeProcessArray[blockIndex].qpeArray:
			print("qpeId: {}".format(qpe.componentId))
	for dramController in ss.dramControllerArray:
		print("dramInterface Id: {}".format(dramController.dramInterface.componentId))
		print("dram Id: {}".format(dramController.dram.componentId))
	# ss.spiNNaker2Stop()
	print("sram componentId: {}".format(ss.triBlockQpeProcessArray[0].qpeArray[4].peArray[2].sram.componentId))
	ss.triBlockQpeProcessArray[0].qpeArray[4].peArray[2].sram.sram[0:16] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	print("sram: {}".format(ss.triBlockQpeProcessArray[0].qpeArray[4].peArray[2].sram.sram[0:16]))

	# dramId = [4]
	# ss.addInTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId, TASK_SRAM_ADDR:0x0, 
	# 	TASK_DATA_LEN:16, TASK_DATA:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]})
	# ss.addInTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId, TASK_SRAM_ADDR:0x80, 
	# 	TASK_DATA_LEN:16, TASK_DATA:[10,11,12,13,14,15,16,17,18,19,110,111,112,113,114,115]})
	# ss.addInTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,4,4], 
	# 	TASK_SOURCE_SRAM_ADDR:0x10, TASK_SRAM_ADDR:0x0, TASK_DATA_LEN:16})
	# ss.addInTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,3,5], 
	# 	TASK_SOURCE_SRAM_ADDR:0x90, TASK_SRAM_ADDR:0x80, TASK_DATA_LEN:16})

	# for _ in range(30):
	# 	ss.spiNNaker2Run()
	# 	rsp = ss.getNextOutTask()
	# 	print("rsp: {}".format(rsp))

	# print("DRAM[0:16]: {}".format(ss.dramControllerArray[0].dram.dram[0:16]))
	# print("DRAM[0x80:0x90]: {}".format(ss.dramControllerArray[0].dram.dram[0x80:0x90]))


	# ss.addInTask({TASK_NAME:Task.SRAM_RD, TASK_DESTINATION:[3,4,4], TASK_SOURCE:[0], TASK_SRAM_ADDR:0x10, TASK_DATA_LEN:16})
	# ss.addInTask({TASK_NAME:Task.SRAM_RD, TASK_DESTINATION:[3,3,5], TASK_SOURCE:[0], TASK_SRAM_ADDR:0x90, TASK_DATA_LEN:16})
	# for _ in range(40):
	# 	ss.spiNNaker2Run()
	# 	rsp = ss.getNextOutTask()
	# 	print("rsp: {}".format(rsp))

	ss.spiNNaker2Stop()