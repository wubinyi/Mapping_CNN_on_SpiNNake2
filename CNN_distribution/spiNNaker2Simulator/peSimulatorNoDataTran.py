from spiNNakerSimulatorGeneral import *
import threading, queue
from collections import deque
from sramSimulatorNoDataTran import SRAMNoData
from armSimulatorNoDataTran import ARMPNoData
from mlaSimulatorNoDataTran import MLANoData
from dmaSimulatorNoDataTran import DMANoData
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

class ProcessElementNoData(GeneralClass):
	def __init__(self, componentId, noc=None, queueSize=None, sramHalfFreq=False, icproValidateNoti=False, 
		nocDoubleFreq=False):
		GeneralClass.__init__(self, componentId, queueSize)
		self.sramHalfFreq = sramHalfFreq
		self.icproValidateNoti = icproValidateNoti
		self.armp = ARMPNoData(self.armIdGenerator())
		self.mla = MLANoData(self.mlaIdGenerator(), icproValidateNoti=self.icproValidateNoti)
		self.sram = SRAMNoData(self.sramIdGenerator(), halfFreq=sramHalfFreq, icproValidateNoti=self.icproValidateNoti)
		# self.noc = noc
		# self.nocDoubleFreq = nocDoubleFreq
		self.tasksToNoc = []
		self.dma = DMANoData(self.dmaIdGenerator())

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
		self.tasksToNoc = []
# original
		# if not self.mla.isIdle():
		# 	self.mlaOperationProcesss()
		# if not self.dma.isIdle():
		# 	self.dmaOperationProcess()
		# if not self.armp.isIdle():
		# 	self.armOperationProcess()
		# if not self.sram.isIdle():
		# 	self.sramOperationProcess()
# correct
		if not self.armp.isIdle():
			self.armOperationProcess()
		if not self.dma.isIdle():
			self.dmaOperationProcess()
		if not self.sram.isIdle():
			self.sramOperationProcess()
		if not self.mla.isIdle():
			self.mlaOperationProcesss()
		return self.nocPeChannelProcess()

	def nocPeChannelProcess(self):
		# Process task from NoC to PE
		task = self.getNextTask()
		if task != None:
			self.processTaskFromNoc(task)
		# Process task from PE to NoC
		return self.tasksToNoc

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
				self.sram.addTask(response)
			else:
				self.commandToNoC(response)
		elif Task.TASK_DOWN_NOTI == responseName:
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
				self.mla.writeOperABuffer()
			elif ComponentType.COMP_MLAOPB == destType:
				self.mla.writeOperBBuffer()
			else:
				self.customAssert(False, "Unsupport Component Type: {} in TASK: {}".format(destType, response))
		else:
			self.commandToNoC(response)

	# ===========================================================
	# 							MLA Operation
	# ===========================================================
	def mlaOperationProcesss(self):
		response = self.mla.accelerate()
		if isinstance(response, list):
			for rsp in response:
				self.mlaSingleRspsProcess(rsp)
		else:
			self.mlaSingleRspsProcess(response)

	def mlaSingleRspsProcess(self, response):
		responseName = response[TASK_NAME]
		if Task.TASK_NONE == responseName:
			pass
		elif Task.NOC_SRAM_DATA_REQ == responseName:
			# operA Buffer: Through NoC
			destination1 = response[TASK_DESTINATION]
			source1 = response[TASK_SOURCE]
			task1 = {TASK_NAME:Task.NOC_SRAM_RD, TASK_DESTINATION:destination1, TASK_SOURCE:source1}
			self.commandToNoC(task1)
			# operB Buffer: local SRAM
			destination2 = response[TASK_DESTINATION_2]
			source2 = response[TASK_SOURCE_2]
			task2 = {TASK_NAME:Task.SRAM_RD, TASK_DESTINATION:destination2, TASK_SOURCE:source2}
			# If not half the sram frequance, fetching operB task should be add to the head of task queue.
			# 		Because maybe at the same time, fetching operA task comes in from NoC.
			# 		Adding to the head means, the priority of fetching operB is higher.
			# If half the sram frequence, not to add to tha task head. 
			# 		As maybe there are tasks waiting in the sram task queue, put fetching operB before them 
			# 		is not make sense.
			if self.sramHalfFreq:
				self.sram.addTask(task2)
			else:
				# "addMlaSramRdTask()" will increase the computation speed
				# "addTask()" slower than "addMlaSramRdTask()"
				# self.sram.addMlaSramRdTask(task2)
				self.sram.addTask(task2)
		elif Task.DOUBLE_SRAM_DATA_REQ == responseName:
			# Fetching operB's priority is higher than fetching operA
			# operB Buffer: local SRAM
			destination2 = response[TASK_DESTINATION_2]
			source2 = response[TASK_SOURCE_2]
			task2 = {TASK_NAME:Task.SRAM_RD, TASK_DESTINATION:destination2, TASK_SOURCE:source2}
			self.sram.addTask(task2)
			# operA Buffer: local SRAM
			destination1 = response[TASK_DESTINATION]
			source1 = response[TASK_SOURCE]
			task1 = {TASK_NAME:Task.NOC_SRAM_RD, TASK_DESTINATION:destination1, TASK_SOURCE:source1}
			self.sram.addTask(task1)			
		elif Task.SRAM_RD == responseName:
			if TASK_DESTINATION in response:
				destination = response[TASK_DESTINATION]
				source = response[TASK_SOURCE]
				taskName = Task.NOC_SRAM_RD
			else:
				destination = response[TASK_DESTINATION_2]
				source = response[TASK_SOURCE_2]
				taskName = Task.SRAM_RD
			task = {TASK_NAME:taskName, TASK_DESTINATION:destination, TASK_SOURCE:source}				
			if taskName == Task.SRAM_RD:
				if self.sramHalfFreq:
					self.sram.addTask(task)
				else:
					# "addMlaSramRdTask()" will increase the computation speed
					# "addTask()" slower than "addMlaSramRdTask()"
					# self.sram.addMlaSramRdTask(task)
					self.sram.addTask(task)
			else:
				self.commandToNoC(task)
		elif Task.SRAM_WR_32 == responseName:
			self.sram.addTask(response)
		elif Task.MLA_FINISH == responseName:
			self.armp.irqInsert(response)
		elif Task.MLA_EXE_SRAM_FINISH == responseName:
			self.armp.irqInsert(response)
		else:
			self.customAssert(False, "Unsupport task: {}".format(response))
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
			# Not necessay to send out to HOST any more
			# pass
		elif Task.MLA_EXE_SRAM_FINISH == responseName:
			self.commandToNoC(response)
			# Not necessay to send out to HOST any more
			# pass		
		elif Task.DATA_MIGRATION_FINISH == responseName:
			# self.commandToNoC(response)
			# Not necessay to send out to HOST any more
			pass
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
		# elif Task.DRAM_SRAM_DATA_MIGRATION == responseName:
		# 	self.dma.addTask(response)
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == responseName:
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
		elif Task.DRAM_RD_16_BYTES == responseName:
			self.commandToNoC(response)
		# elif Task.DRAM_DMA_REQUEST_FINISH == responseName:
		# 	self.commandToNoC(response)
		else:
			self.customAssert(False, "Unsupport task: {}".format(response))

	# ===========================================================
	# 				(NoC) Interface to outside
	# ===========================================================
	def commandToNoC(self, task):
		# self.customAssert(self.noc != None, "NoC is none")
		# self.noc.addTask(task)
		self.tasksToNoc.append(task)

	def commandFromNoC(self, task):
		self.addTask(task)

	def processTaskFromNoc(self, task):
		taskName = task[TASK_NAME]
		if Task.SRAM_WR == taskName or Task.SRAM_RD == taskName:
			self.sram.addTask(task)
		elif Task.MLA_EXE == taskName:
			# self.customAssert(self.mla.isIdle(), "MLA is busy now")
			self.mla.addTaskToTop(task)
			self.armp.mlaAutoSendOutActiSetting(task)
		elif Task.MLA_EXE_SRAM == taskName:
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
		# elif Task.DRAM_SRAM_DATA_MIGRATION == taskName:
		# 	self.armp.addTask(task)
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == taskName:
			self.armp.irqInsert(task)
		elif Task.SRAM_DATA_32 == taskName:
			task[TASK_NAME] = Task.SRAM_WR_32
			self.sram.addTask(task)
		elif Task.DATA_MIGRATION_32_FINISH == taskName:
			# If add into IRQ, need to identify whether it is from other QPE/local dma
			self.armp.irqInsert(task)
		else:
			self.customAssert(False, "Unsupport task: {}".format(task))