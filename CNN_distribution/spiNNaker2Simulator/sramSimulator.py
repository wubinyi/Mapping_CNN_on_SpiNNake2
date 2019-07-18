from spiNNakerSimulatorGeneral import *
from parserSplitter.nnGeneral import PE_SRAM_LIMIT


class SRAM(GeneralClass):
	'''
	SRAM
	'''
	def __init__(self, componentId, queueSize=None, sramSize=PE_SRAM_LIMIT, baseAddr=TOTAL_SRAM_SIZE-PE_SRAM_LIMIT,
				noDataTran=True):
		GeneralClass.__init__(self, componentId, queueSize)
		self.noDataTran = noDataTran
		self.sramSize = PE_SRAM_LIMIT
		self.baseAddr = baseAddr
		self.endAddr = sramSize+baseAddr-1
		self.sram = [0] * self.sramSize

	def writeRead(self):
		if self.noDataTran:
			return self.writeReadWithoutDataTran()
		nextTask = self.getNextTask()
		if None == nextTask:
			# self.customAssert(False, "SRAM task sould not be None")
			return {TASK_NAME:Task.TASK_NONE}
		elif Task.SRAM_WR == nextTask[TASK_NAME]:
			# if nextTask[TASK_DESTINATION] == [1,1,1] and nextTask[TASK_SRAM_ADDR] >= 0x8900 and nextTask[TASK_NEXT_DESTINATION] == [1,1]:
			# 	print("noc migration task in target sram{}: {}".format(self.componentId, nextTask))
			# 	# if nextTask[TASK_SRAM_ADDR] == 0x91f0:
			# 	# 	print("sram{} data: {}".format(self.componentId, self.sram[]))
			self.writeSram(nextTask[TASK_SRAM_ADDR], nextTask[TASK_DATA_LEN], nextTask[TASK_DATA])
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.SRAM_RD == nextTask[TASK_NAME]:
			data = self.readSram(nextTask[TASK_SRAM_ADDR], nextTask[TASK_DATA_LEN])
			return {TASK_NAME:Task.SRAM_DATA, TASK_DATA: data, TASK_DESTINATION: nextTask[TASK_SOURCE]}
		elif Task.NOC_SRAM_RD == nextTask[TASK_NAME]:
			data = self.readSram(nextTask[TASK_SRAM_ADDR], nextTask[TASK_DATA_LEN])
			return {TASK_NAME:Task.NOC_SRAM_DATA, TASK_DATA: data, TASK_DESTINATION: nextTask[TASK_SOURCE]}
		elif Task.SRAM_WR_32 == nextTask[TASK_NAME]:
			self.writeSram(nextTask[TASK_SRAM_ADDR], nextTask[TASK_DATA_LEN], nextTask[TASK_DATA])
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.SRAM_RD_32 == nextTask[TASK_NAME]:
			data = self.readSram(nextTask[TASK_SRAM_ADDR], nextTask[TASK_DATA_LEN])
			return {TASK_NAME:Task.SRAM_DATA_32, TASK_DATA: data, TASK_DESTINATION: nextTask[TASK_SOURCE], 
				TASK_SRAM_ADDR:nextTask[TASK_SOURCE_SRAM_ADDR]}
		elif Task.SRAM_RD_TO_SRAM == nextTask[TASK_NAME]:
			data = self.readSram(nextTask[TASK_SRAM_ADDR], nextTask[TASK_DATA_LEN])		
			return {TASK_NAME:Task.SRAM_WR, TASK_DATA:data, TASK_DATA_LEN:nextTask[TASK_DATA_LEN], 
					TASK_DESTINATION:nextTask[TASK_SOURCE], TASK_SRAM_ADDR:nextTask[TASK_SOURCE_SRAM_ADDR]} 
		else:
			self.customAssert(False, "Unsupport Task: {}".format(nextTask))

	def writeReadWithoutDataTran(self):
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
			return {TASK_NAME:Task.TASK_DOWN}
		elif Task.SRAM_RD_32 == nextTask[TASK_NAME]:
			return {TASK_NAME:Task.SRAM_DATA_32, TASK_DESTINATION: nextTask[TASK_SOURCE]}	
		elif Task.SRAM_RD_TO_SRAM == nextTask[TASK_NAME]:	
			return {TASK_NAME:Task.SRAM_WR, TASK_DESTINATION:nextTask[TASK_SOURCE]} 		 
		else:
			self.customAssert(False, "Unsupport Task: {}".format(nextTask))

	def writeSram(self, addr, dataLen, data):
		self.customAssert(dataLen==len(data), "Data length unmatch {}-{}".format(dataLen, data))
		addr = addr - self.baseAddr
		for index in range(dataLen):
			self.sram[addr+index] = data[index]

	def readSram(self, addr, dataLen):
		data = []
		addr = addr - self.baseAddr
		for index in range(dataLen):
			data.append(self.sram[addr+index])
		return data