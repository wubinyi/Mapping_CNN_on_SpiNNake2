from spiNNakerSimulatorGeneral import *
from enum import Enum, auto


class DramState(Enum):
	ADDR_HOLD = auto()
	DATA_INOUT = auto()


class DRAMNoData(GeneralClass):
	def __init__(self, componentId, queueSize=None):
		GeneralClass.__init__(self, componentId, queueSize)
		self.dramState = DramState.ADDR_HOLD

	def writeRead(self):
		if DramState.ADDR_HOLD == self.dramState:
			if not self.isIdle():
				# Process non-dram task
				dmaMigrationFinishTask = []
				while (not self.isIdle()):
					tempTask = self.getNextTaskWithoutPop()
					if Task.DRAM_DMA_REQUEST_FINISH == tempTask[TASK_NAME]:
						tempTask[TASK_NAME] = Task.DRAM_SRAM_DATA_MIGRA_FINISH
						tempTask[TASK_DESTINATION] = tempTask[TASK_MIGRATION_DESTINATION]
						self.getNextTask()
						# if not self.isIdle():
						# 	self.dramState = DramState.DATA_INOUT
						dmaMigrationFinishTask.append(tempTask)
					elif Task.DATA_MIGRATION_32_FINISH == tempTask[TASK_NAME]:
						tempTask[TASK_DESTINATION] = HOST_ID
						self.getNextTask()
						# if not self.isIdle():
						# 	self.dramState = DramState.DATA_INOUT	
						dmaMigrationFinishTask.append(tempTask)
					else:
						self.dramState = DramState.DATA_INOUT
						if len(dmaMigrationFinishTask) > 0:
							return dmaMigrationFinishTask
						else:
							return {TASK_NAME:Task.TASK_NONE}
				return dmaMigrationFinishTask
			return {TASK_NAME:Task.TASK_NONE}
		elif DramState.DATA_INOUT == self.dramState:
			self.dramState = DramState.ADDR_HOLD
			task = self.getNextTask()
			taskName = task[TASK_NAME]
			# SRAM -> DRAM
			if Task.SRAM_DATA_32 == taskName:
				return {TASK_NAME:Task.TASK_DOWN}
			# SRAM -> DRAM -> SRAM
			elif Task.DRAM_RD_16_BYTES == taskName:
				# print("-----task------: {}".format(task))
				return {TASK_NAME:Task.SRAM_WR, TASK_DESTINATION:task[TASK_SOURCE]}
			else:
				self.customAssert(False, "Unsupport Task: {}".format(task))
		else:
			self.customAssert(False, "Unsupport DRAM state: {}".format(self.dramState))


if __name__ == "__main__":
	dramId = [0]
	dramNoData = DRAMNoData(dramId)
	print("DRAM ID: {}".format(dramNoData.componentId))
	print("DRAM State: {}".format(dramNoData.dramState))
	dramNoData.addTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId})
	dramNoData.addTask({TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION:dramId})
	dramNoData.addTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,4,4]})
	dramNoData.addTask({TASK_NAME:Task.DRAM_RD_16_BYTES, TASK_DESTINATION:dramId, TASK_SOURCE:[3,4,4]})
	for _ in range(10):
		print("DRAM Response: {}".format(dramNoData.writeRead()))