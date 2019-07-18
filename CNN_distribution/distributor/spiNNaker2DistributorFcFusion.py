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
from spiNNaker2TriBlockSimulator import SpiNNaker2TriBlock
from spiNNakerSimulatorGeneral import *
from dataGenerator import fcSpittedDataDim

class SpiNNaker2DistributorFcFusion(DistributionGeneralClass):
	'''
	Step 1: Simulator --> Create an instance of simulator:SpiNNaker2Distributor
	Step 2: Splitter --> Generate splitting scheme of NN model
			Currently, there is only VGG
	Step 3: Distributor --> Distribute CONV-layer into simulator
	'''
	def __init__(self, nocDoubleFreq=True, printFlag=True, operatorFuse=True):
		'''
		Step 1: Simulator
		'''
		DistributionGeneralClass.__init__(self, printFlag=printFlag)
		self.nocDoubleFreq = nocDoubleFreq
		self.operatorFuse = operatorFuse
		self.spiNNaker2 = SpiNNaker2TriBlock(nocDoubleFreq=self.nocDoubleFreq, noDataTran=True)
		self.spiNNakerPeTasksContainer()
		self.peStateArrayGenerator()

	def spiNNakerPeTasksContainer(self):
		yContainer = []
		for yIndex in range(NUM_QPES_Y_AXIS):
			xContainer = []
			for xIndex in range(NUM_QPES_X_AXIS):
				qpeContainer = []
				for peIndex in range(NUM_PES_IN_QPE):
					qpeContainer.append([])
				xContainer.append(qpeContainer)
			yContainer.append(xContainer)
		self.spiNNakerContainer = yContainer

	def peStateArrayGenerator(self):
		self.spiNNakerPeStateArray = []
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			blockPeStateArray = []
			for yAxisIndex in range(NUM_QPES_Y_AXIS_TRIBLOCK):
				for xAxisIndex in range(NUM_QPES_X_AXIS_TRIBLOCK):
					for peIndex in range(NUM_PES_IN_QPE):
						blockPeStateArray.append(PeState.IDLE)
			self.spiNNakerPeStateArray.append(blockPeStateArray)

	def spiNNakerRunOneClock(self):
		self.clockCounter += 1
		self.spiNNaker2.spiNNaker2Run()
		rsp = self.spiNNaker2.getNextOutTask()
		return rsp

	def spiNNakerStop(self):
		self.spiNNaker2.spiNNaker2Stop()

	def spiNNakerAddTask(self, task):
		# self.printInfo(">>>>"+str(task))
		self.spiNNaker2.addInTask(task)

	def vggModelSplitter(self):
		'''
		Step 2: Splitter (VGG)
		'''
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorFcFusion.vgg(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def resNetModelSplitter(self):
		'''
		Step 2: Splitter (ResNet)
		'''
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorFcFusion.resNet50(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def distribute(self):
		'''
		Step 3: Distributor
		'''
		self.fcLayersDistribution()
		self.spiNNakerStop()
		
	def fcLayersDistribution(self):
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "FC_20" not in layerName:
				continue
			self.layerName = layerName
			self.logBeginTime()
			if self.operatorFuse:
				self.singleFcDistributionFusion()
			else:
				self.singleFcDistributionNoFusion()
			self.logEndTime()
			break


	def singleFcDistributionNoFusion(self):
		# S1: Generate computation data (inputActi and weight) and validation data (outputActi)
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		layerTypeParameter = self.modelLayersParameter[self.layerName]
		inActiDimBlocks, weightDimBlocks, outActiDimBlocks = fcSpittedDataDim(layerSplitInfo, widthHeightOrder=True)
		# S2: Distribute Data into DRAM
		# As parts of weight is always be times of 4, therefore, weights is equally divided into 4 DRAMs
		# InputActi are totally copied into 4 DRAMs
		# TODO: For No-Data-Tran, it is not necessary
		# S3: Prepare for Running SpiNNaker2
		fcPesTasks = self.fcBlockTasksGeneratorNoFusion(inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		# print("number of used pe: {}".format(len(fcPesTasks)))
		# for fcPeTasks in fcPesTasks:
		# 	print("-"*30)
		# 	print("number of tasks in each pe: {}".format(len(fcPeTasks)))
		# 	for fcPeTask in fcPeTasks:
		# 		if len(fcPeTask) == 3:
		# 			print(fcPeTask[0])
		# 			print(fcPeTask[1])
		# 			print(fcPeTask[2])
		# 		else:
		# 			print(fcPeTask)
		# 	print("\n"*2)
		# input("")
		# S5: Running SpiNNaker2
		for yIndex in range(NUM_QPES_Y_AXIS):
			for xIndex in range(NUM_QPES_X_AXIS):
				for peIndex in range(NUM_PES_IN_QPE):
					peId = [yIndex, xIndex, peIndex]
					triBlockIndex, peIndexInTriBlock = self.getPeIndex(peId)
					if self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] == PeState.IDLE:
						fcPeTask = fcPesTasks[triBlockIndex].pop(0)
						fcPeTask[0][TASK_MIGRATION_DESTINATION] = peId.copy()
						fcPeTask[1][TASK_MIGRATION_DESTINATION] = peId.copy()
						fcPeTask[2][TASK_DESTINATION] = peId.copy()
						fcPeTask[2][TASK_OPER_A_PEID] = peId.copy()
						self.spiNNakerAddTask(fcPeTask.pop(0))
						self.spiNNakerAddTask(fcPeTask.pop(0))
						self.spiNNakerContainer[yIndex][xIndex][peIndex].append(fcPeTask.pop(0))
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM
		while True:
			# Run SpiNNaker2
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(str(rsp))
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraSize = rsp[TASK_MIGRATION_SIZE]
				migraDataType = rsp[TASK_ADDITION]
				triBlockIndex, peIndexInTriBlock = self.getPeIndex(migraDest)
				peState = self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock]
				if DramSramDataMigraType.WEIGHT == migraDataType:
					if PeState.READ_DRAM == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM_INACTI
					elif PeState.READ_DRAM_WEIGHT == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.MLA_EXE_DRAM
						fcPeTask = self.spiNNakerContainer[migraDest[0]][migraDest[1]][migraDest[2]].pop(0)
						self.spiNNakerAddTask(fcPeTask)
					else:
						self.customAssert(False, "Not supoort PE state [{}] when DRAM_SRAM_DATA_MIGRA_FINISH (weight)".format(peState))
				else:
					if PeState.READ_DRAM == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM_WEIGHT
					elif PeState.READ_DRAM_INACTI == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.MLA_EXE_DRAM
						fcPeTask = self.spiNNakerContainer[migraDest[0]][migraDest[1]][migraDest[2]].pop(0)
						self.spiNNakerAddTask(fcPeTask)
					else:
						self.customAssert(False, "Not supoort PE state [{}] when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)".format(str(peState)))
			elif Task.DATA_MIGRATION_32_FINISH == rspName:
				self.printInfo(str(rsp))
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraSourAddr = rsp[TASK_SRAM_ADDR]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
				triBlockIndex, peIndexInTriBlock = self.getPeIndex(migraSour)
				peState = self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock]
				self.customAssert(PeState.MLA_EXE_DRAM == peState, "DATA_MIGRATION_32_FINISH muss be for PeState.MLA_EXE_DRAM")
				self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.IDLE
				# 
				if len(fcPesTasks[triBlockIndex]) > 0:
					peId = [migraSour[0], migraSour[1], migraSour[2]]
					fcPeTask = fcPesTasks[triBlockIndex].pop(0)
					fcPeTask[0][TASK_MIGRATION_DESTINATION] = peId.copy()
					fcPeTask[1][TASK_MIGRATION_DESTINATION] = peId.copy()
					fcPeTask[2][TASK_DESTINATION] = peId.copy()
					fcPeTask[2][TASK_OPER_A_PEID] = peId.copy()
					self.spiNNakerAddTask(fcPeTask.pop(0))
					self.spiNNakerAddTask(fcPeTask.pop(0))
					self.spiNNakerContainer[migraSour[0]][migraSour[1]][migraSour[2]].append(fcPeTask.pop(0))
					self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM				
				else:
					# Determine if finish
					if self.allTasksFinish():
						self.printInfo("ALL FINISH")
						break
			else:
				self.printInfo("Unused---" + str(rsp))
				pass

	def singleFcDistributionFusion(self):
		# S1: Generate computation data (inputActi and weight) and validation data (outputActi)
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		layerTypeParameter = self.modelLayersParameter[self.layerName]
		inActiDimBlocks, weightDimBlocks, outActiDimBlocks = fcSpittedDataDim(layerSplitInfo, widthHeightOrder=True)
		# S2: Distribute Data into DRAM
		# As parts of weight is always be times of 4, therefore, weights is equally divided into 4 DRAMs
		# InputActi are totally copied into 4 DRAMs
		# TODO: For No-Data-Tran, it is not necessary
		# S3: Prepare for Running SpiNNaker2
		fcPesTasks = self.fcBlockTasksGeneratorFusion(inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		# print("number of used pe: {}".format(len(fcPesTasks)))
		# for fcPeTasks in fcPesTasks:
		# 	print("-"*30)
		# 	print("number of tasks in each pe: {}".format(len(fcPeTasks)))
		# 	for fcPeTask in fcPeTasks:
		# 		if len(fcPeTask) == 3:
		# 			print(fcPeTask[0])
		# 			print(fcPeTask[1])
		# 			print(fcPeTask[2])
		# 		else:
		# 			print(fcPeTask)
		# 	print("\n"*2)
		# input("")
		# S4: Allocate all tasks into spiNNakerContainer
		self.allocateTasksIntoContainer(fcPesTasks)
		# totalUsedPe = 0
		# for yIndex in range(NUM_QPES_Y_AXIS):
		# 	for xIndex in range(NUM_QPES_X_AXIS):
		# 		for peIndex in range(NUM_PES_IN_QPE):
		# 			if len(self.spiNNakerContainer[yIndex][xIndex][peIndex]) > 0:
		# 				print("Task length of pe[{}:{}:{}]: {}".format(yIndex, xIndex, peIndex, len(self.spiNNakerContainer[yIndex][xIndex][peIndex])))
		# 				totalUsedPe += 1
		# 				for fcPeTask in self.spiNNakerContainer[yIndex][xIndex][peIndex]:
		# 					if len(fcPeTask) == 3:
		# 						print(fcPeTask[0])
		# 						print(fcPeTask[1])
		# 						print(fcPeTask[2])
		# 					else:
		# 						print(fcPeTask)
		# 				print("\n"*2)
		# print("totalUsedPe: {}".format(totalUsedPe))
		# S5: Running SpiNNaker2
		for yIndex in range(NUM_QPES_Y_AXIS):
			for xIndex in range(NUM_QPES_X_AXIS):
				for peIndex in range(NUM_PES_IN_QPE):
					if len(self.spiNNakerContainer[yIndex][xIndex][peIndex]) > 0:
						self.spiNNakerAddTask(self.spiNNakerContainer[yIndex][xIndex][peIndex][0].pop(0))
						self.spiNNakerAddTask(self.spiNNakerContainer[yIndex][xIndex][peIndex][0].pop(0))
						triBlockIndex, peIndexInTriBlock = self.getPeIndex([yIndex, xIndex, peIndex])
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM
		while True:
			# Run SpiNNaker2
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(str(rsp))
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraSize = rsp[TASK_MIGRATION_SIZE]
				migraDataType = rsp[TASK_ADDITION]
				self.customAssert(DramSramDataMigraType.WEIGHT == migraDataType, "DRAM_SRAM_DATA_MIGRA_FINISH Muss be weight")
				triBlockIndex, peIndex = self.getPeIndex(migraDest)
				peState = self.spiNNakerPeStateArray[triBlockIndex][peIndex]
				if PeState.READ_DRAM == peState:
					self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.READ_DRAM_INACTI
				elif PeState.READ_DRAM_WEIGHT == peState:
					self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.MLA_EXE_DRAM
					fcPeTask = self.spiNNakerContainer[migraDest[0]][migraDest[1]][migraDest[2]].pop(0)
					self.spiNNakerAddTask(fcPeTask.pop(0))
				else:
					self.customAssert(False, "Not supoort PE state [{}] when DRAM_SRAM_DATA_MIGRA_FINISH (weight)".format(peState))

			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(str(rsp))
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDataType = rsp[TASK_ADDITION]
				# self.customAssert(DramSramDataMigraType.INACTIVE == migraDataType, "DATA_MIGRATION_FINISH Muss be inActi")
				if DramSramDataMigraType.OUTACTIVE == migraDataType:
					triBlockIndex, peIndexInTriBlock = self.getPeIndex(migraSour)
					peState = self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock]
					if PeState.WRITE_OUTACTI == peState:
						yIndex = migraSour[0]
						xIndex = migraSour[1]
						peIndex = migraSour[2]
						triBlockIndex, peIndexInTriBlock = self.getPeIndex([yIndex, xIndex, peIndex])
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.IDLE
						if len(self.spiNNakerContainer[yIndex][xIndex][peIndex]) > 0:
							self.spiNNakerAddTask(self.spiNNakerContainer[yIndex][xIndex][peIndex][0].pop(0))
							self.spiNNakerAddTask(self.spiNNakerContainer[yIndex][xIndex][peIndex][0].pop(0))
							self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM
						else:
							# Determine if finish
							if self.allTasksFinish():
								self.printInfo("ALL FINISH")
								break
					else:
						self.customAssert(False, "Not supoort PE state [{}] when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)".format(str(peState)))
				else:
					triBlockIndex, peIndexInTriBlock = self.getPeIndex(migraDest)
					peState = self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock]
					if PeState.READ_DRAM == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM_WEIGHT
					elif PeState.READ_DRAM_INACTI == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.MLA_EXE_DRAM
						fcPeTask = self.spiNNakerContainer[migraDest[0]][migraDest[1]][migraDest[2]].pop(0)
						self.spiNNakerAddTask(fcPeTask.pop(0))
					else:
						self.customAssert(False, "Not supoort PE state [{}] when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)".format(str(peState)))

			elif Task.DATA_MIGRATION_32_FINISH == rspName:
				self.printInfo(str(rsp))
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraSourAddr = rsp[TASK_SRAM_ADDR]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
				triBlockIndex, peIndex = self.getPeIndex(migraSour)
				peState = self.spiNNakerPeStateArray[triBlockIndex][peIndex]
				self.customAssert(PeState.MLA_EXE_DRAM == peState, "DATA_MIGRATION_32_FINISH muss be for PeState.MLA_EXE_DRAM")
				self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.IDLE
				# 
				yIndex = migraSour[0]
				xIndex = migraSour[1]
				peIndex = migraSour[2]
				triBlockIndex, peIndexInTriBlock = self.getPeIndex([yIndex, xIndex, peIndex])
				if len(self.spiNNakerContainer[yIndex][xIndex][peIndex]) > 0:
					if len(self.spiNNakerContainer[yIndex][xIndex][peIndex][0]) == 3:
						self.spiNNakerAddTask(self.spiNNakerContainer[yIndex][xIndex][peIndex][0].pop(0))
						self.spiNNakerAddTask(self.spiNNakerContainer[yIndex][xIndex][peIndex][0].pop(0))
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.READ_DRAM
					elif len(self.spiNNakerContainer[yIndex][xIndex][peIndex][0]) == 1:
						migraOutActi = self.spiNNakerContainer[yIndex][xIndex][peIndex].pop(0)
						self.spiNNakerAddTask(migraOutActi.pop(0))
						self.spiNNakerPeStateArray[triBlockIndex][peIndexInTriBlock] = PeState.WRITE_OUTACTI
					else:
						self.customAssert(False, "Unknown")
				else:
					# Determine if finish
					if self.allTasksFinish():
						self.printInfo("ALL FINISH")
						break
			else:
				self.printInfo("Unused---" + str(rsp))
				pass


	def allTasksFinish(self):
		for triBlockIndex in range(NUM_OF_TRIBLOCKS):
			for peState in self.spiNNakerPeStateArray[triBlockIndex]:
				if PeState.IDLE != peState:
					return False
		for yIndex in range(NUM_QPES_Y_AXIS):
			for xIndex in range(NUM_QPES_X_AXIS):
				for peIndex in range(NUM_PES_IN_QPE):
					if len(self.spiNNakerContainer[yIndex][xIndex][peIndex]) > 0:
						return False
		return True

	def allocateTasksIntoContainer(self, fcPesTasks):
		usedPesInQpe = len(fcPesTasks) // (NUM_OF_BLOCKS * NUM_OF_QPE_IN_BLOCK)
		# print("usedPesInQpe: {}".format(usedPesInQpe))
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			storageQpeId = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			storagePeId = storageQpeId + [0]
			for yIndex in range(NUM_QPES_Y_AXIS):
				for xIndex in range(NUM_QPES_X_AXIS):
					douBlockIndexTemp = self.getDouBlockIndex([yIndex, xIndex])
					if douBlockIndexTemp is None or douBlockIndexTemp != douBlockIndex:
						continue
					for peIndex in range(usedPesInQpe):
						# print("{}-[{}:{}:{}]".format(douBlockIndex, yIndex, xIndex, peIndex))
						fcPeTasks = fcPesTasks.pop(0)
						for fcPeTask in fcPeTasks:
							if len(fcPeTask) == 3:
								# inActi migration task
								fcPeTask[0][TASK_DESTINATION] = storagePeId
								fcPeTask[0][TASK_MIGRATION_DESTINATION] = [yIndex, xIndex, peIndex]
								# weight migration task
								fcPeTask[1][TASK_DESTINATION] = [douBlockIndex+DRAM_ID_START]
								fcPeTask[1][TASK_MIGRATION_DESTINATION] = [yIndex, xIndex, peIndex]
								# mla task
								fcPeTask[2][TASK_DESTINATION] = [yIndex, xIndex, peIndex]
								fcPeTask[2][TASK_OPER_A_PEID] = [yIndex, xIndex, peIndex]
								fcPeTask[2][TASK_ADDITION3] = [yIndex, xIndex, peIndex+2]
								# print("fcPeTask: {}".format(fcPeTask))
								self.spiNNakerContainer[yIndex][xIndex][peIndex].append(fcPeTask)
							elif len(fcPeTask) == 0:
								# Add task: migrate final result to storage PE
								mlaTaskTemp = self.spiNNakerContainer[yIndex][xIndex][peIndex][-1][-1]
								mlaParamTemp = mlaTaskTemp[TASK_MLA_PARAM]
								mlataskOperation, mlaParam = mlaParamTemp
								matrixAWidth, matrixAHeight, matrixBWidth = mlaParam
								outActiTranSize = matrixAHeight * matrixBWidth
								outActiMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_MIGRATION_SIZE:outActiTranSize, 
									TASK_ADDITION:DramSramDataMigraType.OUTACTIVE, TASK_DESTINATION: [yIndex, xIndex, peIndex], 
									TASK_MIGRATION_DESTINATION:storagePeId.copy()}
								self.spiNNakerContainer[yIndex][xIndex][peIndex].append([outActiMigraTask])
							else:
								self.customAssert(False, "Unknown")
						# self.spiNNakerContainer[yIndex][xIndex][peIndex] = fcPesTasks.pop(0)

	def getDouBlockIndex(self, qpeId):
		qpeId = qpeId[Y_AXIS_INDEX:Z_AXIS_INDEX]
		for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
			if qpeId in DOUBLOCK_QPE_ID_LIST[douBlockIndex]:
				return douBlockIndex
		return None

	def getPeIndex(self, peId):
		triBlockIndex, basicQpeId = self.getTriBlockIndexAndBasicQpeId(peId)
		peIndex = (peId[0] - basicQpeId[0]) * NUM_QPES_X_AXIS_TRIBLOCK + (peId[1] - basicQpeId[1])
		peIndex = peIndex * NUM_PES_IN_QPE + peId[2] % NUM_PES_IN_QPE
		return triBlockIndex, peIndex

	def getPeId(self, triBlockIndex, peIndex):
		qpeIndex = peIndex // NUM_PES_IN_QPE
		qpeIdY = qpeIndex // NUM_QPES_X_AXIS_TRIBLOCK
		qpeIdX = qpeIndex - (qpeIdY * NUM_QPES_X_AXIS_TRIBLOCK)
		qpeIdZ = peIndex - (qpeIdY * NUM_QPES_X_AXIS_TRIBLOCK + qpeIdX) * NUM_PES_IN_QPE
		qpeBasicId = self.getTriBlockBasicQpeId(triBlockIndex)
		qpeIdY += qpeBasicId[0]
		qpeIdX += qpeBasicId[1]
		return [qpeIdY, qpeIdX, qpeIdZ]

	def fcBlockTasksGeneratorFusion(self, inActiDimBlocks, weightDimBlocks, outActiDimBlocks):
		numOfInActiBlocks = len(inActiDimBlocks)
		numOfWeightBlocks = len(weightDimBlocks)
		numOfOutActiBlocks = len(outActiDimBlocks)
		self.customAssert(numOfInActiBlocks*numOfOutActiBlocks == numOfWeightBlocks, "fc unmatch 1")
		#
		inActiAlignSize = self.fcInActiDimToSize(inActiDimBlocks[0], operatorFusion=False)
		weightAlignSize = self.fcWeightDimToSize(weightDimBlocks[0], operatorFusion=False)
		outActiAlignSize = self.fcOutActiDimToSize(outActiDimBlocks[0], operatorFusion=False)	
		inActiBaseAddr = SRAM_DATA_BEGIN_ADDR
		weightBaseAddr = inActiBaseAddr + inActiAlignSize
		outActiBaseAddr = weightBaseAddr + weightAlignSize
		self.customAssert(outActiBaseAddr+outActiAlignSize <= SRAM_END_ADDR, "sram overflow")
		# 
		fcLayerTasks = []
		numOfUsedPes = numOfOutActiBlocks
		self.customAssert(numOfUsedPes <= NUM_OF_BLOCKS * NUM_OF_QPE_IN_BLOCK * 2, "Unsupport")
		for _ in range(numOfUsedPes):
			fcLayerTasks.append([])
		outActiDramAddr = 0
		partsOfWeightAlongColumn = numOfWeightBlocks//numOfInActiBlocks
		columnsOfEachPes = math.ceil(partsOfWeightAlongColumn / numOfUsedPes)
		for weightColumnBlockIndex in range(partsOfWeightAlongColumn):
			outActiDim = outActiDimBlocks[weightColumnBlockIndex]
			peIndex = weightColumnBlockIndex // columnsOfEachPes
			for weightRowBlockIndex in range(numOfInActiBlocks):
				inActiDim = inActiDimBlocks[weightRowBlockIndex]
				weightDim = weightDimBlocks[weightColumnBlockIndex*numOfInActiBlocks + weightRowBlockIndex]
				self.customAssert(inActiDim[0] == weightDim[1] and outActiDim[0] == weightDim[0], "fc unmatch 2")
				inActiTranSize = self.fcInActiDimToSize(inActiDim, self.operatorFuse)
				weightTranSize = self.fcWeightDimToSize(weightDim, self.operatorFuse)
				# DRAM_RD_MIGRATION (inActi): TASK_DESTINATION, TASK_MIGRATION_DESTINATION (targetPe)
				inActiDramToSramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, 
					TASK_MIGRATION_SIZE:inActiTranSize, 
					TASK_ADDITION:DramSramDataMigraType.INACTIVE}
				# DRAM_RD_MIGRATION (weight): TASK_DESTINATION, TASK_MIGRATION_DESTINATION (pairedPe)
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_MIGRATION_SIZE:weightTranSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT}
				# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe)
				mlaTask = self.fcMlaTaskGenerator(weightDim, inActiBaseAddr, weightBaseAddr, 
					outActiBaseAddr, outActiDramAddr, notSendResult=False)
				fcLayerTasks[peIndex].append([inActiDramToSramMigraTask, weightDramToSramMigraTask, mlaTask])
				# 
				outActiDramAddr += 100
			# Add outActi migration marker
			fcLayerTasks[peIndex].append([])
		self.peTasksLen = len(fcLayerTasks[0])
		return fcLayerTasks


	def fcBlockTasksGeneratorNoFusion(self, inActiDimBlocks, weightDimBlocks, outActiDimBlocks):
		numOfInActiBlocks = len(inActiDimBlocks)
		numOfWeightBlocks = len(weightDimBlocks)
		numOfOutActiBlocks = len(outActiDimBlocks)
		self.customAssert(numOfInActiBlocks*numOfOutActiBlocks == numOfWeightBlocks, "fc unmatch 1")
		#
		inActiAlignSize = self.fcInActiDimToSize(inActiDimBlocks[0], operatorFusion=False)
		weightAlignSize = self.fcWeightDimToSize(weightDimBlocks[0], operatorFusion=False)
		outActiAlignSize = self.fcOutActiDimToSize(outActiDimBlocks[0], operatorFusion=False)	
		inActiBaseAddr = SRAM_DATA_BEGIN_ADDR
		weightBaseAddr = inActiBaseAddr + inActiAlignSize
		outActiBaseAddr = weightBaseAddr + weightAlignSize
		self.customAssert(outActiBaseAddr+outActiAlignSize <= SRAM_END_ADDR, "sram overflow")
		# 
		fcLayerTasks = [[],[],[],[]]
		outActiDramAddr = 0
		partsOfWeightAlongColumn = numOfWeightBlocks//numOfInActiBlocks
		columnsOfEachTriBlock = partsOfWeightAlongColumn // NUM_OF_TRIBLOCKS
		for weightColumnBlockIndex in range(partsOfWeightAlongColumn):
			outActiDim = outActiDimBlocks[weightColumnBlockIndex]
			triBlockIndex = weightColumnBlockIndex // columnsOfEachTriBlock
			for weightRowBlockIndex in range(numOfInActiBlocks):
				inActiDim = inActiDimBlocks[weightRowBlockIndex]
				weightDim = weightDimBlocks[weightColumnBlockIndex*numOfInActiBlocks + weightRowBlockIndex]
				self.customAssert(inActiDim[0] == weightDim[1] and outActiDim[0] == weightDim[0], "fc unmatch 2")
				inActiTranSize = self.fcInActiDimToSize(inActiDim, False)
				weightTranSize = self.fcWeightDimToSize(weightDim, False)
				# DRAM_RD_MIGRATION (inActi): TASK_MIGRATION_DESTINATION (targetPe)
				inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_MIGRATION_SIZE:inActiTranSize, TASK_DESTINATION: [triBlockIndex+DRAM_ID_START],
					TASK_ADDITION:DramSramDataMigraType.INACTIVE}
				# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_MIGRATION_SIZE:weightTranSize, TASK_DESTINATION: [triBlockIndex+DRAM_ID_START],
					TASK_ADDITION:DramSramDataMigraType.WEIGHT}
				# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe)
				mlaTask = self.fcMlaTaskGenerator(weightDim, inActiBaseAddr, weightBaseAddr, 
					outActiBaseAddr, outActiDramAddr, notSendResult=False, mlaExeSram=False)
				fcLayerTasks[triBlockIndex].append([inActiDramToSramMigraTask, weightDramToSramMigraTask, mlaTask])
				# 
				outActiDramAddr += 100
		self.peTasksLen = len(fcLayerTasks[0])
		return fcLayerTasks

	def fcMlaTaskGenerator(self, weightDim, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, 
		outActiDramAddr, notSendResult, mlaExeSram=True):
		# matrixAWidth, matrixAHeight, matrixBWidth
		mlaParam = (MlaOperType.MM, (weightDim[1], 1, weightDim[0]))
		# For spinnaker, when operator fusion, the outActi should still be 32-bit.
		additionParam = (False, 1) # (no operator fusion, poolSize=1)
		if mlaExeSram:
			mlaTask = {TASK_NAME:Task.MLA_EXE_SRAM, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR: weightBaseAddr, 
						TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
						TASK_OUTACTI_DRAM_ADDR: outActiDramAddr, TASK_ADDITION:additionParam}
		else:
			mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR: weightBaseAddr, 
						TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
						TASK_OUTACTI_DRAM_ADDR: outActiDramAddr, TASK_ADDITION:additionParam}			
		if notSendResult:
			mlaTask[TASK_ADDITION2] = True
		return mlaTask


if __name__ == "__main__":
	ssd = SpiNNaker2DistributorFcFusion(operatorFuse=True)
	ssd.vggModelSplitter()
	# ssd.resNetModelSplitter()
	ssd.distribute()
	# ssd.dramSramDataMigraValidation()
	# ssd.mlaExeValidate()