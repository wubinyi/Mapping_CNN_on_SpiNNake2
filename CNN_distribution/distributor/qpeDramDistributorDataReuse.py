import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
s2sFolderPath = os.path.join(projectFolder, "spiNNaker2Simulator")
if s2sFolderPath not in sys.path:
	sys.path.insert(0, s2sFolderPath)
distriFolderPath = os.path.join(projectFolder, "distributor")
if distriFolderPath not in sys.path:
	sys.path.insert(0, distriFolderPath)
icproValidatePath = os.path.join(projectFolder, "icproValidate")
if icproValidatePath not in sys.path:
	sys.path.insert(0, icproValidatePath)
from distributionGeneral import *
from qpeDramSimulator import QpeDram
from spiNNakerSimulatorGeneral import *
from dataGenerator import convSpittedDataDim, fcSpittedDataDim
import copy

class QpeDramDistributorDataReuse(DistributionGeneralClass):
	'''
	Step 1: Simulator --> Create an instance of simulator:SpiNNaker2Distributor
	Step 2: Splitter --> Generate splitting scheme of NN model
			Currently, there is only VGG
	Step 3: Distributor --> Distribute CONV-layer into simulator
			SS 1: 	inActi: DRAM -> SRAM
					weight: DRAM -> SRAM
			SS 2: 	Run QpeDram with data reuse
	'''
	def __init__(self, nocDoubleFreq=True, printFlag=True, sramHalfFreq=False, icproValidateNoti=False, 
		operatorFusion=True, hostInterfaceLatency=10):
		'''
		Step 1: Simulator
		'''
		DistributionGeneralClass.__init__(self, printFlag=printFlag)
		self.nocDoubleFreq = nocDoubleFreq
		self.sramHalfFreq = sramHalfFreq
		self.icproValidateNoti = icproValidateNoti
		self.operatorFusion = operatorFusion
		self.hostInterfaceLatency = hostInterfaceLatency
		self.qpeId=[2,2]
		self.dramId=[4]
		self.qpeDram = QpeDram(qpeId=self.qpeId, dramId=self.dramId, nocDoubleFreq=self.nocDoubleFreq, 
			sramHalfFreq=sramHalfFreq, icproValidateNoti=self.icproValidateNoti, 
			hostInterfaceLatency=self.hostInterfaceLatency)
		self.freeFlag = True
		self.peMlaInfoArray = [None] * NUM_PES_IN_QPE
		self.peStateArray = [PeState.IDLE] * NUM_PES_IN_QPE
		# CONV (InActi Reuse): index for weight migration tasks
		self.convWeightMigraIndex = None
		# CONV: inActi/weight DRAM->SRAM migration tasks
		self.convInActiMigraTasks = None
		self.convWeightMigraTasks = None
		# CONV: MLA_EXE tasks
		self.convMlaTask = None
		# CONV: pairshift for data reuse
		self.pairShift = None
		# CONV: DRAM addr for data migration SRAM->DRAM
		self.dramOutAddr = None
		# CONV: number of InActi, which are loaded into QPE,
		# 		used for different processing
		self.loadInActiParts = None
		# CONV: used for self.loadInActiParts < 4
		# 		Data migration from SRAM->SRAM
		self.sramMigraTasks = None
		# CONV: used for self.loadInActiParts < 4
		# 		Indicating if can load InActi from sram
		self.sramMigraFlag = None
		# Speical for self.loadInActiParts = 3
		self.sramMigraSourcePeIndexFor3InActi = None
		# 
		self.isConv = None
		self.inActiReuseTimes = None

	def qpeDramRunOneClock(self):
		self.clockCounter += 1
		self.qpeDram.run()
		rsp = self.qpeDram.getNextOutTask()
		return rsp

	def qpeDramAddTask(self, task):
		# self.printInfo(str(task))
		self.qpeDram.addInTask(task)

	def vggModelSplitter(self):
		'''
		Step 2: Splitter (VGG)
		'''
		modelLayersSplitInfo, modelLayersParameter = QpeDramDistributorDataReuse.vgg(decreaseSize=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def resNetModelSplitter(self):
		'''
		Step 2: Splitter (ResNet)
		'''
		modelLayersSplitInfo, modelLayersParameter = QpeDramDistributorDataReuse.resNet50(decreaseSize=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def distribute(self):
		'''
		Step 3: Distributor
		'''
		self.convLayersDistribution()
		
	def convLayersDistribution(self):
		'''
		CONV_1:  2282308 clocks -> 2301100 -> 2319736
		CONV_3:  5641344
		CONV_4: 10686632
		CONV_6:	 5776704
		CONV_10: 5587232
		'''
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "CONV_15" not in layerName:
				continue
			self.layerName = layerName
			self.logBeginTime()
			self.singleConvDistribution()
			self.logEndTime()
			break

	def singleConvDistribution(self):
		if "CONV_" in self.layerName:
			self.isConv = True
		else:
			self.isConv = False
		# S1: Generate computation data (inputActi and weight) and validation data (outputActi)
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		layerTypeParameter = self.modelLayersParameter[self.layerName]
		if self.isConv:
			# inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, \
			# 	outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
			inActiDimBlocks, weightDimBlocks, outActiDimBlocks = convSpittedDataDim(layerSplitInfo, layerTypeParameter)
		else:
			inActiDimBlocks, weightDimBlocks, outActiDimBlocks = fcSpittedDataDim(layerSplitInfo, widthHeightOrder=False)
		# S2: Distribute Data into DRAM
		# As parts of weight is always be times of 4, therefore, weights is equally divided into 4 DRAMs
		# InputActi are totally copied into 4 DRAMs
		# TODO: For No-Data-Tran, it is not necessary
		# S3: Prepare for Running SpiNNaker2
		if self.isConv:
			self.convInActiMigraTasks, self.convWeightMigraTasks, self.convMlaTask = self.convBlockTasksGenerator(layerSplitInfo, 
				layerTypeParameter, inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		else:
			self.convInActiMigraTasks, self.convWeightMigraTasks, self.convMlaTask = self.fcBlockTasksGenerator(layerSplitInfo, 
				inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		# print("number of inActi migration tasks: {}".format(len(self.convInActiMigraTasks)))
		# print("number of wegiht migration tasks: {}".format(len(self.convWeightMigraTasks)))
		# for inActiMigraTask in self.convInActiMigraTasks:
		# 	print("inActi Task: {}".format(inActiMigraTask))
		# for weightMigraTask in self.convWeightMigraTasks:
		# 	print("weight Task: {}".format(weightMigraTask))
		# print("pemla Task: {}".format(self.convMlaTask))
		# input("---")		
		# S4: Running SpiNNaker2
		if self.isConv:
			self.inActiReuseTimes = len(self.convWeightMigraTasks)
		else:
			self.inActiReuseTimes = (len(self.convWeightMigraTasks) // len(self.convInActiMigraTasks)) * NUM_PES_IN_QPE
		# print("self.inActiReuseTimes: {}".format(self.inActiReuseTimes))
		# input("---")
		self.convWeightMigraIndex = 0
		self.dramOutAddr = 0
		while True:
			# SS1: Check if all finish
			if len(self.convInActiMigraTasks) == 0 and self.convWeightMigraIndex == 0:
			# if self.convWeightMigraIndex == 4: 	# For testing different with Data-Reuse
				self.printInfo("ALL FINISH")
				break				
			# SS2: Insert inActi/weight migration task
			self.addInActiWeightMigraTasks()
			# SS3: Generate PE MLA task
			mlaTasks = []
			if self.isConv:
				numOfMlaTasks = NUM_PES_IN_QPE*self.loadInActiParts
			else:
				numOfMlaTasks = NUM_PES_IN_QPE
			for _ in range(numOfMlaTasks):
				mlaTasks.append(copy.deepcopy(self.convMlaTask))
			# SS4: Run QPE
			self.pairShift = 0
			while True:
				# SSS1: Run QpeDram one clock
				try:
					rsp = self.qpeDramRunOneClock()
				except AssertionError as ae:
					self.printInfo(ae)
					self.customAssert(False, "")
				if self.loadInActiParts == NUM_PES_IN_QPE-3:
					self.processRspFromQpeDramForOneInActi(rsp)
				elif self.loadInActiParts == NUM_PES_IN_QPE-2:
					self.processRspFromQpeDramForTwoInActi(rsp)
				elif self.loadInActiParts == NUM_PES_IN_QPE-1:
					self.processRspFromQpeDramForThreeInActi(rsp)
				elif self.loadInActiParts == NUM_PES_IN_QPE:
					self.processRspFromQpeDram(rsp)
				else:
					self.customAssert(False, "Unsupport parts of InActi: {}".format(self.loadInActiParts))
				# SSS2: Insert MLA_EXE task - (A Data-Reuse round)
				self.addMlaExeTasks(mlaTasks, 1)
				# SSS3: Break Data-Reuse-Loop
				if self.freeFlag == True:

					break

	# =========================================================
	# Adding Migration Task into QPE-DRAM
	# =========================================================
	def addInActiWeightMigraTasks(self):
		if self.freeFlag:
			if 0 == self.convWeightMigraIndex:
				self.addInActiMigrationTasks()
				self.addWeightMigrationTasks(peState=PeState.READ_DRAM)
			else:
				self.addWeightMigrationTasks()
				# Spicial for for self.loadInActiParts = 3
				if self.loadInActiParts == NUM_PES_IN_QPE - 1:
					self.peStateArray[NUM_PES_IN_QPE-1] = PeState.READ_DRAM
					self.sramMigraSourcePeIndexFor3InActi = 0
					sramMigraTask = copy.deepcopy(self.sramMigraTasks[self.sramMigraSourcePeIndexFor3InActi])
					sramMigraTask[TASK_DESTINATION] = [self.qpeId[0], self.qpeId[1], self.sramMigraSourcePeIndexFor3InActi]
					sramMigraTask[TASK_MIGRATION_DESTINATION] = [self.qpeId[0], self.qpeId[1], NUM_PES_IN_QPE-1]
					self.qpeDramAddTask(sramMigraTask)
					self.sramMigraSourcePeIndexFor3InActi += 1			
			self.freeFlag = False

	def addInActiMigrationTasks(self):
		numOfRemaindInActiTasks = len(self.convInActiMigraTasks)
		self.loadInActiParts = NUM_PES_IN_QPE if numOfRemaindInActiTasks > NUM_PES_IN_QPE else numOfRemaindInActiTasks
		if self.loadInActiParts < NUM_PES_IN_QPE:
			self.sramMigraFlag = True
		if self.loadInActiParts == NUM_PES_IN_QPE-1:
			self.sramMigraSourcePeIndexFor3InActi = 0
		self.sramMigraTasks = []
		for peIndex in range(self.loadInActiParts):
			targetPeId = [self.qpeId[0], self.qpeId[1], peIndex]
			inActiDim, inActiMigraTask = self.convInActiMigraTasks.pop(0)
			inActiMigraTask[TASK_MIGRATION_DESTINATION] = targetPeId
			self.qpeDramAddTask(inActiMigraTask)
			if self.loadInActiParts < NUM_PES_IN_QPE:
				# TASK_DESTINATION, TASK_MIGRATION_DESTINATION
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0,
					TASK_MIGRATION_SIZE:inActiMigraTask[TASK_MIGRATION_SIZE]}
				self.sramMigraTasks.append(sramMigraTask)
			self.peMlaInfoArray[peIndex] = [inActiDim, None]
		for peIndex in range(self.loadInActiParts, NUM_PES_IN_QPE):
			self.peMlaInfoArray[peIndex] = [inActiDim, None]

	def addWeightMigrationTasks(self, peState=PeState.READ_DRAM_WEIGHT):
		for peIndex in range(NUM_PES_IN_QPE):
			targetPeId = [self.qpeId[0], self.qpeId[1], peIndex]
			weightDim, weightMigraTask = self.convWeightMigraTasks[self.convWeightMigraIndex]
			weightMigraTask[TASK_MIGRATION_DESTINATION] = targetPeId
			self.qpeDramAddTask(weightMigraTask)
			self.convWeightMigraIndex += 1
			self.peMlaInfoArray[peIndex][1] = weightDim
			self.peStateArray[peIndex] = peState
		if self.convWeightMigraIndex >= self.inActiReuseTimes:
			self.convWeightMigraIndex = 0			

	# =========================================================
	# Adding MLA_EXE Task into QPE-DRAM
	# =========================================================
	def addMlaExeTasks(self, mlaTasks, outAddrSize):
		if self.peStateArray.count(PeState.MLA_EXE_READY) == NUM_PES_IN_QPE:
			if len(mlaTasks) > 0:
				self.printInfo("Write MLA_EXE task...")
				for peIndex in range(NUM_PES_IN_QPE):
					targetPeId = [self.qpeId[0], self.qpeId[1], peIndex]
					pairedPeId = self.targetToPairedPeIdGenerator(targetPeId=targetPeId, pePairShift=self.pairShift)
					mlaTask = mlaTasks.pop(0)
					mlaTask[TASK_DESTINATION] = targetPeId
					mlaTask[TASK_OPER_A_PEID] = pairedPeId
					mlaTask[TASK_OUTACTI_DRAM_ADDR] = self.dramOutAddr
					if self.isConv:
						mlaTask[TASK_MLA_PARAM] = (MlaOperType.CONV, self.getMlaParam(targetPeId, pairedPeId))
					else:
						mlaTask[TASK_MLA_PARAM] = (MlaOperType.MM, self.getMlaParam(targetPeId, pairedPeId))
					self.qpeDramAddTask(mlaTask)
					self.peStateArray[peIndex] = PeState.MLA_EXE_DRAM
					self.dramOutAddr += outAddrSize
				self.pairShift += 1
			else:
				self.printInfo("Round Finish...")
				self.peStateArray = [PeState.IDLE] * NUM_PES_IN_QPE
				self.freeFlag = True

	def getMlaParam(self, targetPeId, pairedPeId):
		inActiDim = tuple(self.peMlaInfoArray[targetPeId[2]][0])
		weightDim = tuple(self.peMlaInfoArray[pairedPeId[2]][1])
		return inActiDim+weightDim

	# =========================================================
	# QPE-DRAM reponse processing
	# =========================================================
	def processRspFromQpeDramForOneInActi(self, rsp):
		rspName = rsp[TASK_NAME]
		if Task.TASK_NONE == rspName:
			pass
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
			self.printInfo(str(rsp))
			# DRAM -> PE_SRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraSize = rsp[TASK_MIGRATION_SIZE]
			migraDataType = rsp[TASK_ADDITION]
			peIndex = migraDest[2]
			peState = self.peStateArray[peIndex]
			if DramSramDataMigraType.INACTIVE == migraDataType:
				# Add sram->sram migration task
				if self.sramMigraFlag:
					sramMigraTask = copy.deepcopy(self.sramMigraTasks[0])
					sramMigraTask[TASK_DESTINATION] = migraDest
					sramMigraTask[TASK_MIGRATION_DESTINATION] = [self.qpeId[0], self.qpeId[1], 1]
					self.qpeDramAddTask(sramMigraTask)
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
				elif PeState.READ_DRAM_INACTI == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)")
			else:
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_INACTI
				elif PeState.READ_DRAM_WEIGHT == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (weight)")
		elif Task.DATA_MIGRATION_32_FINISH == rspName:
			self.printInfo(str(rsp))
			# PE_SRAM -> DRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraSourAddr = rsp[TASK_SRAM_ADDR]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
			peIndex = migraSour[2]
			self.peStateArray[peIndex] = PeState.MLA_EXE_READY
		elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			peIndex = migraDest[2]
			peState = self.peStateArray[peIndex]
			if migraSour == [self.qpeId[0], self.qpeId[1], 0] and migraDest == [self.qpeId[0], self.qpeId[1], 1] \
				and self.sramMigraFlag:
				sramMigraTask = copy.deepcopy(self.sramMigraTasks[0])
				sramMigraTask[TASK_DESTINATION] = [self.qpeId[0], self.qpeId[1], 0]
				sramMigraTask[TASK_MIGRATION_DESTINATION] = [self.qpeId[0], self.qpeId[1], 2]
				self.qpeDramAddTask(sramMigraTask)	
				sramMigraTask = copy.deepcopy(self.sramMigraTasks[0])
				sramMigraTask[TASK_DESTINATION] = [self.qpeId[0], self.qpeId[1], 1]
				sramMigraTask[TASK_MIGRATION_DESTINATION] = [self.qpeId[0], self.qpeId[1], 3]
				self.qpeDramAddTask(sramMigraTask)
				self.sramMigraFlag = False
			if PeState.READ_DRAM == peState:
				self.peStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
			elif PeState.READ_DRAM_INACTI == peState:
				self.peStateArray[peIndex] = PeState.MLA_EXE_READY
			else:
				self.customAssert(False, "Not supoort PE state when DATA_MIGRATION_ALL_FINISH (inActi):{}".format(peState))		
		else:
			self.printInfo(str(rsp))
			pass		

	def processRspFromQpeDramForTwoInActi(self, rsp):
		rspName = rsp[TASK_NAME]
		if Task.TASK_NONE == rspName:
			pass
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
			self.printInfo(str(rsp))
			# DRAM -> PE_SRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraSize = rsp[TASK_MIGRATION_SIZE]
			migraDataType = rsp[TASK_ADDITION]
			peIndex = migraDest[2]
			peState = self.peStateArray[peIndex]
			if DramSramDataMigraType.INACTIVE == migraDataType:
				# Add sram->sram migration task
				if self.sramMigraFlag:
					sramMigraTask = self.sramMigraTasks.pop(0)
					sramMigraTask[TASK_DESTINATION] = migraDest
					sramMigraTask[TASK_MIGRATION_DESTINATION] = [self.qpeId[0], self.qpeId[1], migraDest[2]+2]
					self.qpeDramAddTask(sramMigraTask)
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
				elif PeState.READ_DRAM_INACTI == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)")
			else:
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_INACTI
				elif PeState.READ_DRAM_WEIGHT == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (weight)")
		elif Task.DATA_MIGRATION_32_FINISH == rspName:
			self.printInfo(str(rsp))
			# PE_SRAM -> DRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraSourAddr = rsp[TASK_SRAM_ADDR]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
			peIndex = migraSour[2]
			self.peStateArray[peIndex] = PeState.MLA_EXE_READY
		elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
			self.printInfo(str(rsp))
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			peIndex = migraDest[2]
			peState = self.peStateArray[peIndex]
			if PeState.READ_DRAM == peState:
				self.peStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
			elif PeState.READ_DRAM_INACTI == peState:
				self.peStateArray[peIndex] = PeState.MLA_EXE_READY
			else:
				self.customAssert(False, "Not supoort PE state when DATA_MIGRATION_ALL_FINISH (inActi):{}".format(peState))		
		else:
			self.printInfo(str(rsp))
			pass	

	def processRspFromQpeDramForThreeInActi(self, rsp):
		rspName = rsp[TASK_NAME]
		if Task.TASK_NONE == rspName:
			pass
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
			self.printInfo(str(rsp))
			# DRAM -> PE_SRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraSize = rsp[TASK_MIGRATION_SIZE]
			migraDataType = rsp[TASK_ADDITION]
			peIndex = migraDest[2]
			peState = self.peStateArray[peIndex]
			if DramSramDataMigraType.INACTIVE == migraDataType:
				# Add sram->sram migration task
				if self.sramMigraFlag and migraDest[2] == 0:
					sramMigraTask = copy.deepcopy(self.sramMigraTasks[self.sramMigraSourcePeIndexFor3InActi])
					self.sramMigraSourcePeIndexFor3InActi += 1
					sramMigraTask[TASK_DESTINATION] = migraDest
					sramMigraTask[TASK_MIGRATION_DESTINATION] = [self.qpeId[0], self.qpeId[1], NUM_PES_IN_QPE-1]
					self.qpeDramAddTask(sramMigraTask)
					self.sramMigraFlag = False
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
				elif PeState.READ_DRAM_INACTI == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)")
			else:
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_INACTI
				elif PeState.READ_DRAM_WEIGHT == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (weight)")
		elif Task.DATA_MIGRATION_32_FINISH == rspName:
			self.printInfo(str(rsp))
			# PE_SRAM -> DRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraSourAddr = rsp[TASK_SRAM_ADDR]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
			peIndex = migraSour[2]
			self.peStateArray[peIndex] = PeState.MLA_EXE_READY
			# Add sram->sram migration task
			if (self.peStateArray.count(PeState.MLA_EXE_READY) == NUM_PES_IN_QPE) and \
				(self.sramMigraSourcePeIndexFor3InActi < NUM_PES_IN_QPE-1):
				self.peStateArray[NUM_PES_IN_QPE-1] = PeState.READ_DRAM_INACTI
				sramMigraTask = copy.deepcopy(self.sramMigraTasks[self.sramMigraSourcePeIndexFor3InActi])
				sramMigraTask[TASK_DESTINATION] = [self.qpeId[0], self.qpeId[1], self.sramMigraSourcePeIndexFor3InActi]
				sramMigraTask[TASK_MIGRATION_DESTINATION] = [self.qpeId[0], self.qpeId[1], NUM_PES_IN_QPE-1]
				# print("---> sramMigraTask: {}".format(sramMigraTask))
				# input("")
				self.sramMigraSourcePeIndexFor3InActi += 1
				self.qpeDramAddTask(sramMigraTask)
		elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
			self.printInfo(str(rsp))
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			peIndex = migraDest[2]
			peState = self.peStateArray[peIndex]
			if PeState.READ_DRAM == peState:
				self.peStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
			elif PeState.READ_DRAM_INACTI == peState:
				self.peStateArray[peIndex] = PeState.MLA_EXE_READY
			else:
				self.customAssert(False, "Not supoort PE state when DATA_MIGRATION_ALL_FINISH (inActi):{}".format(peState))		
		else:
			self.printInfo(str(rsp))
			pass	

	def processRspFromQpeDram(self, rsp):
		rspName = rsp[TASK_NAME]
		if Task.TASK_NONE == rspName:
			pass
		elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
			self.printInfo(str(rsp))
			# DRAM -> PE_SRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraSize = rsp[TASK_MIGRATION_SIZE]
			migraDataType = rsp[TASK_ADDITION]
			peIndex = migraDest[2]
			peState = self.peStateArray[peIndex]
			if DramSramDataMigraType.INACTIVE == migraDataType:
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
				elif PeState.READ_DRAM_INACTI == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)")
			else:
				if PeState.READ_DRAM == peState:
					self.peStateArray[peIndex] = PeState.READ_DRAM_INACTI
				elif PeState.READ_DRAM_WEIGHT == peState:
					self.peStateArray[peIndex] = PeState.MLA_EXE_READY
				else:
					self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (weight)")
		elif Task.DATA_MIGRATION_32_FINISH == rspName:
			self.printInfo(str(rsp))
			# PE_SRAM -> DRAM
			migraSour = rsp[TASK_MIGRATION_SOURCE]
			migraSourAddr = rsp[TASK_SRAM_ADDR]
			migraDest = rsp[TASK_MIGRATION_DESTINATION]
			migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
			peIndex = migraSour[2]
			self.peStateArray[peIndex] = PeState.MLA_EXE_READY
		else:
			self.printInfo(str(rsp))
			pass

	# =========================================================
	# Migration and MLA_EXE Task Generation
	# =========================================================
	def convBlockTasksGenerator(self, layerSplitInfo, layerTypeParameter, inActiDimBlocks, 
		weightDimBlocks, outActiDimBlocks):
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		layerType, layerParameter = layerTypeParameter
		if 2 == len(layerParameter):
			(_, _, convStride, _), poolStride = layerParameter
		else:
			(_, _, convStride, _), poolStride, _ = layerParameter
		if isinstance(poolStride, tuple):
			poolDim, poolStride = poolStride
			if poolDim[0] != poolStride:
				poolStride = 1
		# 
		largestInActiBlockAlignSize = self.convInActiDimToSize(inActiDimBlocks[0], operatorFusion=False)
		largestWeightBlockAlignSize = self.convWeightDimToSize(weightDimBlocks[0], operatorFusion=False)
		largestOutActiBlockAlignSize = self.convOutActiDimToSize(outActiDimBlocks[0], operatorFusion=False)
		largestOutActiBlockAlignSize *= (convStride * convStride) 
		inActiBaseAddr = SRAM_DATA_BEGIN_ADDR
		weightBaseAddr = inActiBaseAddr + largestInActiBlockAlignSize
		outActiBaseAddr = weightBaseAddr + largestWeightBlockAlignSize
		self.customAssert(outActiBaseAddr+largestOutActiBlockAlignSize < SRAM_END_ADDR, "sram overflow")
		# Conv inActi migration task
		convInActiMigraTasks = []
		for inActiDim in inActiDimBlocks:
			inActiBlockSize = self.convInActiDimToSize(inActiDim, self.operatorFusion)
			inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
				TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:inActiBlockSize, 
				TASK_ADDITION:DramSramDataMigraType.INACTIVE}
			convInActiMigraTasks.append(((inActiDim[0], inActiDim[1], inActiDim[2]), inActiDramToSramMigraTask))
		# Conv weight migration task
		convWeightMigraTasks = []
		for weightDim in weightDimBlocks:
			weightBlockSize = self.convWeightDimToSize(weightDim, self.operatorFusion)
			# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
			weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
				TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:weightBlockSize, 
				TASK_ADDITION:DramSramDataMigraType.WEIGHT}
			convWeightMigraTasks.append(((weightDim[0], weightDim[1], weightDim[3], weightStrideSplitInfo[1]), weightDramToSramMigraTask))	
		# Conv execution task
		# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe), 
		# 						 TASK_MLA_PARAM, TASK_OUTACTI_DRAM_ADDR
		additionParam = (self.operatorFusion, poolStride*poolStride) # (no operator fusion, poolSize)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_OPER_A_ADDR: weightBaseAddr, 
			TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
			TASK_ADDITION:additionParam}
		return convInActiMigraTasks, convWeightMigraTasks, mlaTask

	def fcBlockTasksGenerator(self, layerSplitInfo, inActiDimBlocks, weightDimBlocks, outActiDimBlocks):
		layerType, inActiSplitInfo, weightSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		# 
		largestInActiBlockAlignSize = self.fcInActiDimToSize(inActiDimBlocks[0], operatorFusion=False)
		largestWeightBlockAlignSize = self.fcWeightDimToSize(weightDimBlocks[0], operatorFusion=False)
		largestOutActiBlockAlignSize = self.fcOutActiDimToSize(outActiDimBlocks[0], operatorFusion=False)
		inActiBaseAddr = SRAM_DATA_BEGIN_ADDR
		weightBaseAddr = inActiBaseAddr + largestInActiBlockAlignSize
		outActiBaseAddr = weightBaseAddr + largestWeightBlockAlignSize
		self.customAssert(outActiBaseAddr+largestOutActiBlockAlignSize < SRAM_END_ADDR, "sram overflow")
		# Conv inActi migration task
		convInActiMigraTasks = []
		for inActiDim in inActiDimBlocks:
			inActiBlockSize = self.fcInActiDimToSize(inActiDim, self.operatorFusion)
			inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
				TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:inActiBlockSize, 
				TASK_ADDITION:DramSramDataMigraType.INACTIVE}
			convInActiMigraTasks.append((inActiDim, inActiDramToSramMigraTask))
		# Conv weight migration task
		convWeightMigraTasks = []
		for weightDim in weightDimBlocks:
			weightBlockSize = self.fcWeightDimToSize(weightDim, self.operatorFusion)
			# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
			weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
				TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:weightBlockSize, 
				TASK_ADDITION:DramSramDataMigraType.WEIGHT}
			convWeightMigraTasks.append(([weightDim[0]], weightDramToSramMigraTask))	
		# Conv execution task
		# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe), 
		# 						 TASK_MLA_PARAM, TASK_OUTACTI_DRAM_ADDR
		additionParam = (self.operatorFusion, 1) # (no operator fusion, poolSize)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_OPER_A_ADDR: weightBaseAddr, 
			TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
			TASK_ADDITION:additionParam}
		return convInActiMigraTasks, convWeightMigraTasks, mlaTask

	def getInActiDim(self, layerSplitInfo, inActiBlockIndex):
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		inWidthTotalParts = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		inWidthIndex = inActiBlockIndex % inWidthTotalParts
		inHeightIndex = inActiBlockIndex // inWidthTotalParts
		inWidth = self.getPartLengthFromSplitInfo(inActiSplitInfo[0], inWidthIndex)
		inHeight = self.getPartLengthFromSplitInfo(inActiSplitInfo[1], inHeightIndex)
		inChannel = inActiSplitInfo[2]
		return (inWidth, inHeight, inChannel)

	def getWeightDim(self, layerSplitInfo, weightBlockIndex):
		outChannelIndex = weightBlockIndex
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		filterWidth = weightStrideSplitInfo[0][0]
		filterHeight = weightStrideSplitInfo[0][1]
		outChannel = self.getPartLengthFromSplitInfo(weightStrideSplitInfo[0][3], outChannelIndex)
		stride = weightStrideSplitInfo[1]
		return (filterWidth,filterHeight,outChannel,stride)

	# =========================================================
	# Simple Validation
	# =========================================================
	def dramSramDataMigraValidation(self):
		# S+: Validation
		self.qpeDramAddTask({TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
			TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:0x1000, TASK_ADDITION:DramSramDataMigraType.INACTIVE, 
			TASK_MIGRATION_DESTINATION:[self.qpeId[0],self.qpeId[1],0]})
		for _ in range(8000):
			rsp = self.qpeDramRunOneClock()
			if Task.TASK_NONE != rsp[TASK_NAME]:
				self.printInfo(str(rsp))
				if Task.DRAM_SRAM_DATA_MIGRA_FINISH == rsp[TASK_NAME]:
					break

	def mlaConvExeValidate(self):
		# inWidth, inHeight, inChannel, filterWidth, filterHeight, outChannel, stride
		mlaParam = (MlaOperType.CONV, (28,14,128,5,5,4,1))
		# mlaParam = (MlaOperType.CONV, (30,9,256,3,3,4,1))
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR:48864, 
					TASK_OPER_B_ADDR:33024, TASK_OPER_C_ADDR:48976, TASK_OUTACTI_DRAM_ADDR:0, 
					TASK_DESTINATION:[self.qpeId[0],self.qpeId[1],0], 
					TASK_OPER_A_PEID:[self.qpeId[0],self.qpeId[1],0],
					TASK_ADDITION: (False, 1), TASK_ADDITION2: False}
		self.qpeDramAddTask(mlaTask)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR:48864, 
					TASK_OPER_B_ADDR:33024, TASK_OPER_C_ADDR:48976, TASK_OUTACTI_DRAM_ADDR:0, 
					TASK_DESTINATION:[self.qpeId[0],self.qpeId[1],1], 
					TASK_OPER_A_PEID:[self.qpeId[0],self.qpeId[1],1],
					TASK_ADDITION: (False, 1), TASK_ADDITION2: False}
		self.qpeDramAddTask(mlaTask)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR:48864, 
					TASK_OPER_B_ADDR:33024, TASK_OPER_C_ADDR:48976, TASK_OUTACTI_DRAM_ADDR:0, 
					TASK_DESTINATION:[self.qpeId[0],self.qpeId[1],2], 
					TASK_OPER_A_PEID:[self.qpeId[0],self.qpeId[1],2],
					TASK_ADDITION: (False, 1), TASK_ADDITION2: False}
		self.qpeDramAddTask(mlaTask)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR:48864, 
					TASK_OPER_B_ADDR:33024, TASK_OPER_C_ADDR:48976, TASK_OUTACTI_DRAM_ADDR:0, 
					TASK_DESTINATION:[self.qpeId[0],self.qpeId[1],3], 
					TASK_OPER_A_PEID:[self.qpeId[0],self.qpeId[1],3],
					TASK_ADDITION: (False, 1), TASK_ADDITION2: False}
		self.qpeDramAddTask(mlaTask)
		finishCounter = 0
		for _ in range(1000000):
			rsp = self.qpeDramRunOneClock()
			if Task.TASK_NONE != rsp[TASK_NAME]:
				self.printInfo(str(rsp))
				if Task.TASK_DOWN_NOTI == rsp[TASK_NAME]:
					finishCounter += 1
					if finishCounter == 4:
						break

	def mlaMmExeValidate(self, matrixAWidth, matrixAHeight, matrixBWidth, numOfUsedPes=NUM_PES_IN_QPE, peShift=0):
		# matrixAWidth, matrixAHeight, matrixBWidth --> matrixAHeight, matrixBWidth
		matrixAWidth = self.align4(matrixAWidth)
		matrixAHeight = self.alignMlaRow(matrixAHeight)
		matrixBWidth = self.align16(matrixBWidth)
		mlaParam = (MlaOperType.MM, (matrixAWidth,matrixAHeight,matrixBWidth))
		for peIndex in range(numOfUsedPes):
			mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR:48864, 
						TASK_OPER_B_ADDR:33024, TASK_OPER_C_ADDR:48976, TASK_OUTACTI_DRAM_ADDR:0, 
						TASK_DESTINATION:[self.qpeId[0],self.qpeId[1],peIndex], 
						TASK_OPER_A_PEID:[self.qpeId[0],self.qpeId[1],(peIndex+peShift)%NUM_PES_IN_QPE],
						TASK_ADDITION: (False, 1), TASK_ADDITION2: False}
			self.qpeDramAddTask(mlaTask)
		finishCounter = NUM_PES_IN_QPE - numOfUsedPes
		for _ in range(1000000):
			rsp = self.qpeDramRunOneClock()
			if Task.TASK_NONE != rsp[TASK_NAME]:
				self.printInfo(str(rsp))
				if Task.TASK_DOWN_NOTI == rsp[TASK_NAME]:
					finishCounter += 1
					if finishCounter == 4:
						for _ in range(100):
							self.qpeDramRunOneClock()
						break

def qpeDramConvValidatedWithIcpro():
	# ssd = QpeDramDistributorDataReuse(nocDoubleFreq=True, sramHalfFreq=False, hostInterfaceLatency=1, icproValidateNoti=True)
	# ssd.mlaConvExeValidate()
	# print("")
	ssd = QpeDramDistributorDataReuse(nocDoubleFreq=True, sramHalfFreq=True, hostInterfaceLatency=1, icproValidateNoti=True)
	ssd.mlaConvExeValidate()

def qpeDramMmValidatedWithIcpro(matrixAWidth, matrixAHeight, matrixBWidth, numOfUsedPes=NUM_PES_IN_QPE, peShift=0):
	# ssd = QpeDramDistributorDataReuse(nocDoubleFreq=True, sramHalfFreq=False, hostInterfaceLatency=1, icproValidateNoti=True)
	# ssd.mlaMmExeValidate(matrixAWidth, matrixAHeight, matrixBWidth, numOfUsedPes=numOfUsedPes, peShift=peShift)
	# print("")
	ssd = QpeDramDistributorDataReuse(nocDoubleFreq=True, sramHalfFreq=True, hostInterfaceLatency=1, icproValidateNoti=True)
	ssd.mlaMmExeValidate(matrixAWidth, matrixAHeight, matrixBWidth, numOfUsedPes=numOfUsedPes, peShift=peShift)

def qpeDramSimulation():
	ssd = QpeDramDistributorDataReuse(nocDoubleFreq=True, sramHalfFreq=False, operatorFusion=True)
	ssd.vggModelSplitter()
	# ssd.resNetModelSplitter()
	ssd.distribute()	

if __name__ == "__main__":
	isSimulation = True
	isConv = False
	if isSimulation:
		qpeDramSimulation()
	else:
		if isConv:
			qpeDramConvValidatedWithIcpro()
		else:
			qpeDramMmValidatedWithIcpro(matrixAWidth=64, matrixAHeight=1, matrixBWidth=1024, 
				numOfUsedPes=4, peShift=0)