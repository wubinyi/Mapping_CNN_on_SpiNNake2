from spiNNakerSimulatorGeneral import *
from enum import Enum, auto

class SramState(Enum):
	ADDR_HOLD = auto()
	DATA_INOUT = auto()

class SRAMNoData(GeneralClass):
	'''
	SRAM
	'''
	def __init__(self, componentId, queueSize=None, halfFreq=False, icproValidateNoti=False):
		GeneralClass.__init__(self, componentId, queueSize)
		self.sramState = SramState.ADDR_HOLD
		self.halfFreq = halfFreq
		self.icproValidateNoti = icproValidateNoti

	def addMlaSramRdTask(self, task):
		'''
		Use to accelerate the speed of MLA
		'''
		if self.queueSize != None:
			self.customAssert(len(self.taskQueue) < self.queueSize, "Task queue is full")
		if len(self.taskQueue) > 0:
			lastTaskName = self.taskQueue[-1][TASK_NAME]
			if Task.NOC_SRAM_RD == lastTaskName:
				self.taskQueue.insert(len(self.taskQueue)-1, task)
			else:
				self.taskQueue.append(task)
		else:
			self.taskQueue.append(task)

	def writeRead(self):
		if self.halfFreq:
			return self.writeReadHalfFreq()
		else:
			return self.writeReadFullFreq()

	def writeReadFullFreq(self):
		nextTask = self.getNextTask()
		if None == nextTask:
			return {TASK_NAME:Task.TASK_NONE}
		elif Task.SRAM_WR == nextTask[TASK_NAME]:
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.SRAM_RD == nextTask[TASK_NAME]:
			return {TASK_NAME:Task.SRAM_DATA, TASK_DESTINATION: nextTask[TASK_SOURCE]}
		elif Task.NOC_SRAM_RD == nextTask[TASK_NAME]:
			return {TASK_NAME:Task.NOC_SRAM_DATA, TASK_DESTINATION: nextTask[TASK_SOURCE]}
		elif Task.SRAM_WR_32 == nextTask[TASK_NAME]:
			if self.icproValidateNoti and (TASK_ADDITION in nextTask) and (IcproValidation.MLA_FINISH == nextTask[TASK_ADDITION]):
				return {TASK_NAME:Task.TASK_DOWN_NOTI, TASK_DESTINATION:HOST_ID}
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.SRAM_RD_32 == nextTask[TASK_NAME]:
			return {TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION: nextTask[TASK_SOURCE]}	
		elif Task.SRAM_RD_TO_SRAM == nextTask[TASK_NAME]:	
			return {TASK_NAME:Task.SRAM_WR, TASK_DESTINATION:nextTask[TASK_SOURCE]}	 
		else:
			self.customAssert(False, "Unsupport Task: {}".format(nextTask))

	def writeReadHalfFreq(self):
		if SramState.ADDR_HOLD == self.sramState:
			if not self.isIdle():
				self.sramState = SramState.DATA_INOUT
			return {TASK_NAME:Task.TASK_NONE}
		elif SramState.DATA_INOUT == self.sramState:
			self.sramState = SramState.ADDR_HOLD
			return self.writeReadFullFreq()
		else:
			self.customAssert(False, "Unsupport DRAM state: {}".format(self.sramState))

if __name__ == "__main__":
	clock = 0
	sramId = [0,0,4]
	sram = SRAMNoData(componentId=sramId, halfFreq=True)
	for _ in range(16):
		clock += 1
		sram.addTask({TASK_NAME:Task.SRAM_WR_32, TASK_DESTINATION:sramId})
		rsp = sram.writeRead()
		if rsp[TASK_NAME] != Task.TASK_NONE:
			print("---> {:<7}: {}".format(clock, rsp))
	for _ in range(20):
		clock += 1
		rsp = sram.writeRead()
		if rsp[TASK_NAME] != Task.TASK_NONE:
			print("---> {:<7}: {}".format(clock, rsp))