from spiNNakerSimulatorGeneral import *
from enum import Enum, auto


class DramState(Enum):
	ADDR_HOLD = auto()
	DATA_INOUT = auto()


MIGA_BYTES = 1024
class DRAM(GeneralClass):
	def __init__(self, componentId, queueSize=None, dramSize=MIGA_BYTES*20):
		GeneralClass.__init__(self, componentId, queueSize)
		self.dramState = DramState.ADDR_HOLD
		self.dramSize = dramSize
		self.dram = [0] * self.dramSize

	def writeRead(self):
		if DramState.ADDR_HOLD == self.dramState:
			if not self.isIdle():
				self.dramState = DramState.DATA_INOUT
			return {TASK_NAME:Task.TASK_NONE}
		elif DramState.DATA_INOUT == self.dramState:
			self.dramState = DramState.ADDR_HOLD
			task = self.getNextTask()
			taskName = task[TASK_NAME]
			# SRAM -> DRAM
			if Task.SRAM_DATA_32 == taskName:
				self.writeData(task[TASK_SRAM_ADDR], task[TASK_DATA_LEN], task[TASK_DATA])
				return {TASK_NAME:Task.TASK_DOWN}
			# DRAM -> SRAM
			elif Task.DRAM_RD_16_BYTES == taskName:
				data = self.readData(task[TASK_SRAM_ADDR], task[TASK_DATA_LEN])
				return {TASK_NAME:Task.SRAM_WR, TASK_DESTINATION:task[TASK_SOURCE], 
						TASK_SRAM_ADDR:task[TASK_SOURCE_SRAM_ADDR], TASK_DATA_LEN:task[TASK_DATA_LEN],
						TASK_DATA:data}		 
			else:
				self.customAssert(False, "Unsupport Task: {}".format(task))
		else:
			self.customAssert(False, "Unsupport DRAM state: {}".format(self.dramState))

	def writeData(self, addr, dataLen, data):
		for index in range(dataLen):
			self.dram[addr+index] = data[index]

	def readData(self, addr, dataLen):
		data = []
		for index in range(dataLen):
			data.append(self.dram[addr+index])
		return data



if __name__ == "__main__":
	dramId = [0]
	dram = DRAM(dramId)
	print("DRAM ID: {}".format(dram.componentId))
	print("DRAM State: {}".format(dram.dramState))
	dram.addTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId, TASK_SRAM_ADDR:0x0, 
		TASK_DATA_LEN:16, TASK_DATA:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]})
	dram.addTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId, TASK_SRAM_ADDR:0x80, 
		TASK_DATA_LEN:16, TASK_DATA:[10,11,12,13,14,15,16,17,18,19,110,111,112,113,114,115]})
	dram.addTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,4,4], 
		TASK_SOURCE_SRAM_ADDR:0x10, TASK_SRAM_ADDR:0x0, TASK_DATA_LEN:16})
	dram.addTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,6,5], 
		TASK_SOURCE_SRAM_ADDR:0x90, TASK_SRAM_ADDR:0x80, TASK_DATA_LEN:16})
	for _ in range(10):
		print("DRAM Response: {}".format(dram.writeRead()))