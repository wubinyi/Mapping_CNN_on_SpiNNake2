from spiNNakerSimulatorGeneral import *
import threading, queue
from collections import deque
from sramSimulator import SRAM
from armSimulator import ARMP
from mlaSimulator import MLA
from dmaSimulator import DMA
import copy


'''
QUAD_PE connect to outside through NoC.
QUAD_PE has 4 PE and 1 NoC, all PEs connect to the NoC, which is the interface connecting to outside.  
Each PE has three components: ARM Processor (ARMP), ML Accelerator (MLA) and SRAM.
	ARM Processor (ARMP): 
		Status:				Free, Busy
		Active behavior:	Write/Read SRAM, Command MLA, Write NoC
		Passive behavior:	Command by NoC, Command by MLA (IRQ)
		other behavior:		Read data from other 3 PE through NoC,DMA
	ML Accelerator (MLA): 
		Status:				Free, Busy
		Active behavior:	Write/Read SRAM, Command ARMP (IRQ), 
		Passive behavior:	Command by NoC/ARMP
		other behavior:		Read data from other 3 PE through NoC,DMA
	SRAM:
		Status:				Free, Busy(In/Out)
		Active behavior:	-
		Passive behavior:	Write/Read by NoC/ARMP/MLA
NoC Functionality:
		Status:				Free, Busy(In/Out)
		Active behavior:	Write/Read SRAM, Command MLA/ARMP
		Passive behavior:	Write by ARMP, Request data by ARMP/MLA through NoC,DMA

Note that DMA is mentioned above, however, it is simulated. Because it appearance always with other components. 	
'''	
ARMP_RUN_FLAG_INDEX = 0
MLA_RUN_FLAG_INDEX = 1
SRAM_RUN_FLAG_INDEX = 2
class ProcessElement(GeneralClass):
	def __init__(self, componentId, noc=None, queueSize=None, noDataTran=True):
		GeneralClass.__init__(self, componentId, queueSize)
		self.noDataTran = noDataTran
		self.armp = ARMP(self.armIdGenerator(), noDataTran=self.noDataTran)
		self.mla = MLA(self.mlaIdGenerator(), noDataTran=self.noDataTran)
		self.sram = SRAM(self.sramIdGenerator(), noDataTran=self.noDataTran)
		self.noc = noc
		self.dma = DMA(self.dmaIdGenerator(), noDataTran=self.noDataTran)

	def armIdGenerator(self):
		armId = copy.deepcopy(self.componentId)
		armId[2] = armId[2] + PEARMP_PE_ID_OFFSET
		return armId

	def mlaIdGenerator(self):
		mlaId = copy.deepcopy(self.componentId)
		mlaId[2] = mlaId[2] + PEMLA_PE_ID_OFFSET
		return mlaId

	def sramIdGenerator(self):
		sramId = copy.deepcopy(self.componentId)
		sramId[2] = sramId[2] + PESRAM_PE_ID_OFFSET
		return sramId

	def dmaIdGenerator(self):
		dmaId = copy.deepcopy(self.componentId)
		dmaId[2] = dmaId[2] + PEDMA_PE_ID_OFFSET
		return dmaId

	def islocalId(self, compId):
		if len(compId) != 3:
			return False
		if self.componentId[0:2] != compId[0:2]:
			return False
		return self.componentId[2] == (compId[2]%4)

	def isIdle(self):
		return self.sram.isIdle() and self.mla.isIdle() and self.armp.isIdle()

	def run(self):
		'''
		NoC -> ARM -> MLA -> SRAM
		'''
		# Run SRAM firstly
		self.sramOperationProcess()
		self.dmaOperationProcess()
		self.mlaOperationProcesss()
		self.armOperationProcess()

	# ===========================================================
	# 							SRAM Operation
	# ===========================================================
	def sramOperationProcess(self):
		'''
		SRAM WR/RD Operation
		'''
		response = self.sram.writeRead()
		responseName = response[TASK_NAME]
		if Task.TASK_DOWN == responseName or Task.TASK_NONE == responseName:
			pass
		elif Task.SRAM_DATA == responseName:
			self.sramDataProcess(response)
		elif Task.NOC_SRAM_DATA == responseName:
			self.commandToNoC(response)
		elif Task.SRAM_DATA_32 == responseName:
			self.sramDataProcess(response)
		elif Task.SRAM_WR == responseName:
			if self.islocalId(response[TASK_DESTINATION]):
				# print("local DMA task: {}".format(response))
				self.sram.addTask(response)
			else:
				# print("DMA task: {}".format(response))
				self.commandToNoC(response)
		else:
			self.customAssert(False, "Unsupport task: {}".format(response))

	def sramDataProcess(self, response):
		'''
		SRAM Data -> MLA/DRAM(NoC)/SRAM(DMA-NoC)
		'''
		if self.islocalId(response[TASK_DESTINATION]):
			destType = self.compIdParse(response[TASK_DESTINATION])
			# SRAM data for MLA
			if ComponentType.COMP_MLAOPA == destType:
				if not self.noDataTran:
					self.mla.writeOperABuffer(response[TASK_DATA])
				else:
					self.mla.writeOperABuffer(copy.deepcopy([0]*16))
			elif ComponentType.COMP_MLAOPB == destType:
				if not self.noDataTran:
					self.mla.writeOperBBuffer(response[TASK_DATA])
				else:
					self.mla.writeOperBBuffer(copy.deepcopy([0]*16))
			else:
				self.customAssert(False, "Unsupport Component Type: {} in TASK: {}".format(destType, response))
		else:
			self.commandToNoC(response)

	# ===========================================================
	# 							MLA Operation
	# ===========================================================
	def mlaOperationProcesss(self):
		response = self.mla.accelerate()
		responseName = response[TASK_NAME]
		if Task.TASK_NONE == responseName:
			pass
		elif Task.NOC_SRAM_DATA_REQ == responseName:
			# operA Buffer: Through NoC
			destination1 = response[TASK_DESTINATION]
			source1 = response[TASK_SOURCE]
			if self.noDataTran:
				task1 = {TASK_NAME:Task.NOC_SRAM_RD, TASK_DESTINATION:destination1, 
						TASK_SOURCE:source1}
			else:
				dataLen1 = response[TASK_DATA_LEN]
				sramAddr1 = response[TASK_SRAM_ADDR]
				task1 = {TASK_NAME:Task.NOC_SRAM_RD, TASK_DESTINATION:destination1, 
							TASK_SRAM_ADDR:sramAddr1, TASK_DATA_LEN:dataLen1, 
							TASK_SOURCE:source1}
			self.commandToNoC(task1)
			# print("ToNoC : {}".format(task1))
			# operB Buffer: local SRAM
			destination2 = response[TASK_DESTINATION_2]
			source2 = response[TASK_SOURCE_2]
			if self.noDataTran:
				task2 = {TASK_NAME:Task.SRAM_RD, TASK_DESTINATION:destination2, 
							TASK_SOURCE:source2}
			else:
				dataLen2 = response[TASK_DATA_LEN_2]
				sramAddr2 = response[TASK_SRAM_ADDR_2]
				task2 = {TASK_NAME:Task.SRAM_RD, TASK_DESTINATION:destination2, 
							TASK_SRAM_ADDR:sramAddr2, TASK_DATA_LEN:dataLen2, 
							TASK_SOURCE:source2}
			self.sram.addTask(task2)
			# print("ToSRAM: {}".format(task2))
		elif Task.SRAM_RD == responseName:
			if TASK_DESTINATION in response:
				destination = response[TASK_DESTINATION]
				source = response[TASK_SOURCE]
				taskName = Task.NOC_SRAM_RD
				if not self.noDataTran:
					dataLen = response[TASK_DATA_LEN]
					sramAddr = response[TASK_SRAM_ADDR]
			else:
				destination = response[TASK_DESTINATION_2]
				source = response[TASK_SOURCE_2]
				taskName = Task.SRAM_RD
				if not self.noDataTran:
					dataLen = response[TASK_DATA_LEN_2]
					sramAddr = response[TASK_SRAM_ADDR_2]
			if not self.noDataTran:
				task = {TASK_NAME:taskName, TASK_DESTINATION:destination, 
							TASK_SRAM_ADDR:sramAddr, TASK_DATA_LEN:dataLen, 
							TASK_SOURCE:source}
			else:
				task = {TASK_NAME:taskName, TASK_DESTINATION:destination, 
							TASK_SOURCE:source}				
			if taskName == Task.SRAM_RD:
				self.sram.addTask(task)
				# print("ToSRAM: {}".format(task))
			else:
				self.commandToNoC(task)
				# print("ToNoC : {}".format(task))
				pass
		elif Task.SRAM_WR_32 == responseName:
			self.sram.addTask(response)
		elif Task.MLA_FINISH == responseName:
				# TODO -> ARM
				# Actually an IRQ should be transfer to ARM and then ARM generate a notification packet to NoC
				# self.commandToNoC(response)
				# print("-------------------------> MLA Finish")
				self.armp.irqInsert(response)
				pass
	# ===========================================================
	# 						ARMP Operation
	# ===========================================================
	def armOperationProcess(self):
		response = self.armp.process()
		responseName = response[TASK_NAME]
		if Task.TASK_NONE == responseName or Task.TASK_DOWN == responseName:
			pass
		elif Task.MLA_FINISH == responseName:
			self.commandToNoC(response)
		# elif Task.SRAM_RD_32 == responseName:
		# 	self.sram.addTask(response)
		elif Task.DATA_MIGRATION_FINISH == responseName:
			self.commandToNoC(response)
		elif Task.DATA_MIGRATION == responseName:
			self.dma.addTask(response)
		elif Task.DATA_MIGRATION_DEST_FINISH == responseName:
			self.commandToNoC(response)
		elif Task.DATA_MIGRATION_ALL_FINISH == responseName:
			self.commandToNoC(response)
		elif Task.DATA_MIGRATION_32 == responseName:
			self.dma.addTask(response)
		elif Task.DATA_MIGRATION_32_FINISH == responseName:
			self.commandToNoC(response)
		else:
			self.customAssert(False, "Unsupport task: {}".format(response))

	# ===========================================================
	# 						DMA Operation
	# ===========================================================
	def dmaOperationProcess(self):
		response = self.dma.dataTransfer()
		responseName = response[TASK_NAME]
		if Task.TASK_NONE == responseName or Task.TASK_DOWN == responseName:
			pass
		elif Task.SRAM_RD_TO_SRAM == responseName:
			self.sram.addTask(response) 
		elif Task.DATA_MIGRATION_FINISH == responseName:
			self.armp.irqInsert(response)
		elif Task.SRAM_RD_32 == responseName:
			self.sram.addTask(response)
		elif Task.DATA_MIGRATION_32_FINISH == responseName:
			self.armp.irqInsert(response)
		else:
			self.customAssert(False, "Unsupport task: {}".format(response))

	# ===========================================================
	# 				(NoC) Interface to outside
	# ===========================================================
	def commandToNoC(self, task):
		self.customAssert(self.noc != None, "NoC is none")
		self.noc.addTask(task)

	def commandFromNoC(self, task):
		taskName = task[TASK_NAME]
		if Task.SRAM_WR == taskName or Task.SRAM_RD == taskName:
			self.sram.addTask(task)
		elif Task.MLA_EXE == taskName:
			self.customAssert(self.mla.isIdle(), "MLA is busy now")
			self.mla.addTaskToTop(task)
			self.armp.mlaAutoSendOutActiSetting(task)
		elif Task.SRAM_DATA == taskName:
			self.sramDataProcess(task)
		elif Task.NOC_SRAM_RD == taskName:
			self.sram.addTask(task)
		elif Task.NOC_SRAM_DATA == taskName:
			self.sramDataProcess(task)
		elif Task.SRAM_RD_32 == taskName:
			self.sram.addTask(task)
		elif Task.DATA_MIGRATION == taskName:
			self.armp.addTask(task)
		elif Task.DATA_MIGRATION_DEST_FINISH == taskName:
			# task[TASK_NAME] = Task.DATA_MIGRATION_ALL_FINISH
			self.armp.irqInsert(task)
		else:
			self.customAssert(False, "Unsupport task: {}".format(task))

# Abanden after adding NoC
# if __name__ == "__main__":
# 	pe = ProcessElement(componentId=[0,0,0])
# 	taskQueue = deque([])
# 	# Write 16 bytes to SRAM:0x8080
# 	taskQueue.append({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION: [0,0,0], TASK_SRAM_ADDR: 0x8100, TASK_DATA_LEN: 16, 
# 						TASK_DATA: [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 
# 											0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0]})
# 	# # Read 5 bytes to local MLA::operABuffer
# 	# taskQueue.append({TASK_NAME:Task.SRAM_RD, TASK_DESTINATION: [0,0,0], TASK_SRAM_ADDR: 0x8080, TASK_DATA_LEN: 5,
# 	# 					TASK_SOURCE: [0,0,12]})
# 	# # Read 2 bytes to local MLA::operBBuffer
# 	# taskQueue.append({TASK_NAME:Task.SRAM_RD, TASK_DESTINATION: [0,0,0], TASK_SRAM_ADDR: 0x8080, TASK_DATA_LEN: 2,
# 	# 					TASK_SOURCE: [0,0,16]})	
# 	# MLA_EXE
# 	inWidth = 20
# 	inHeight = 4
# 	inChannel = 3
# 	filterWidth = 3
# 	filterHeight = 3
# 	outChannel = 5
# 	stride = 1
# 	inActiBaseAddr = 0x8100
# 	weightBaseAddr = 0x8500
# 	outActiBaseAddr = 0x8900
# 	mlaParam = (MlaOperType.CONV, (inWidth,inHeight,inChannel,filterWidth,filterHeight,outChannel,stride))
# 	task = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_PEID:[0,0,0], 
# 			TASK_OPER_A_ADDR: weightBaseAddr, TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr}
# 	taskQueue.append(task)
# 	# PE
# 	clockCounter = 0
# 	while not (len(taskQueue) == 0 and pe.isIdle()):
# 		if pe.canAddTask() and len(taskQueue) > 0:
# 			pe.commandFromNoC(taskQueue.popleft())
# 		pe.run()
# 		clockCounter += 1
# 	# clock counter
# 	print("clockCounter: {}".format(clockCounter))
# 	# Check
# 	data = pe.sram.readSram(0x8100, 16)
# 	print("sram: {}".format(data))
# 	print("compute clocks: {}".format(pe.mla.compClocksCounter))
# 	print("write outActi clocks: {}".format(pe.mla.writeOutActiClocksCounter))
