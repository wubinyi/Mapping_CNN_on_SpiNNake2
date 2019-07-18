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
from dataGenerator import convSpittedDataDim

class SpiNNaker2DistributorNoDataReuse(DistributionGeneralClass):
	'''
	Step 1: Simulator --> Create an instance of simulator:SpiNNaker2Distributor
	Step 2: Splitter --> Generate splitting scheme of NN model
			Currently, there is only VGG
	Step 3: Distributor --> Distribute CONV-layer into simulator
	'''
	def __init__(self, nocDoubleFreq=True, printFlag=True, operatorFuse=False):
		'''
		Step 1: Simulator
		'''
		DistributionGeneralClass.__init__(self, printFlag=printFlag)
		self.nocDoubleFreq = nocDoubleFreq
		self.operatorFuse = operatorFuse
		self.spiNNaker2 = SpiNNaker2TriBlock(nocDoubleFreq=self.nocDoubleFreq, noDataTran=True)
		self.peStateArrayGenerator()
		# print(self.spiNNakerPeStateArray)
		self.peTasksLen = None

	def peStateArrayGenerator(self):
		self.spiNNakerPeStateArray = []
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			blockPeStateArray = []
			for yAxisIndex in range(NUM_QPES_Y_AXIS_TRIBLOCK):
				for xAxisIndex in range(NUM_QPES_X_AXIS_TRIBLOCK):
					for peIndex in range(NUM_PES_IN_QPE):
						blockPeStateArray.append(PeState.IDLE)
			self.spiNNakerPeStateArray.append(blockPeStateArray)
		self.freePeCounter = NUM_QPES_X_AXIS * NUM_QPES_Y_AXIS * NUM_PES_IN_QPE

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

	def dramSramDataMigraValidation(self):
		# S+: Validation
		self.spiNNakerAddTask({TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
			TASK_DESTINATION:[4], TASK_MIGRATION_SIZE:0x1000, TASK_ADDITION:DramSramDataMigraType.INACTIVE, 
			TASK_MIGRATION_DESTINATION:[0,0,0]})
		for _ in range(8000):
			rsp = self.spiNNakerRunOneClock()
			if Task.TASK_NONE != rsp[TASK_NAME]:
				self.printInfo(str(rsp))
				if Task.DRAM_SRAM_DATA_MIGRA_FINISH == rsp[TASK_NAME]:
					self.spiNNakerStop()
					break

	def mlaExeValidate(self):
		mlaParam = (MlaOperType.CONV, (224,22,3,3,3,4,1))
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR:48864, 
					TASK_OPER_B_ADDR:33024, TASK_OPER_C_ADDR:48976, TASK_OUTACTI_DRAM_ADDR:0, 
					TASK_DESTINATION:[0,0,0], TASK_OPER_A_PEID:[0,0,1]}
		self.spiNNakerAddTask(mlaTask)
		for _ in range(40000):
			rsp = self.spiNNakerRunOneClock()
			if Task.TASK_NONE != rsp[TASK_NAME]:
				self.printInfo(str(rsp))
				if Task.DATA_MIGRATION_32_FINISH == rsp[TASK_NAME]:
					self.spiNNakerStop()
					break
		self.spiNNakerStop()

	def vggModelSplitter(self):
		'''
		Step 2: Splitter (VGG)
		'''
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorNoDataReuse.vgg(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def resNetModelSplitter(self):
		'''
		Step 2: Splitter (ResNet)
		'''
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorNoDataReuse.resNet50(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def distribute(self):
		'''
		Step 3: Distributor
		'''
		self.convLayersDistribution()
		self.spiNNakerStop()
		
	def convLayersDistribution(self):
		'''
		CONV_1: 534008 clocks
		CONV_1: 523097 clocks
		CONV_1: 523083 clocks
		CONV_1: 520297 clocks
		'''
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "CONV_43" not in layerName:
				continue
			self.layerName = layerName
			self.logBeginTime()
			self.singleConvDistribution()
			self.logEndTime()
			break

	def singleConvDistribution(self):
		# S1: Generate computation data (inputActi and weight) and validation data (outputActi)
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		layerTypeParameter = self.modelLayersParameter[self.layerName]
		# inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, \
		# 	outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
		inActiDimBlocks, weightDimBlocks, outActiDimBlocks = convSpittedDataDim(layerSplitInfo, layerTypeParameter)
		# S2: Distribute Data into DRAM
		# As parts of weight is always be times of 4, therefore, weights is equally divided into 4 DRAMs
		# InputActi are totally copied into 4 DRAMs
		# TODO: For No-Data-Tran, it is not necessary
		# S3: Prepare for Running SpiNNaker2
		# convLayerTasks = self.convBlockTasksGenerator(layerSplitInfo, inActiAlignBlocks, inActiBaseAddr, 
		# 	weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr)
		convLayerTasks = self.convBlockTasksGenerator(layerSplitInfo, layerTypeParameter, inActiDimBlocks, 
			weightDimBlocks, outActiDimBlocks)
		# for triblockIndex in range(NUM_OF_TRIBLOCKS):
		# 	for taskPair in convLayerTasks[triblockIndex]:
		# 		print("DRAM_SRAM_MIGRATION inActi Task: {}".format(taskPair[0]))
		# print("-"*30+"\n")
		# for triblockIndex in range(NUM_OF_TRIBLOCKS):
		# 	for taskPair in convLayerTasks[triblockIndex]:
		# 		print("DRAM_SRAM_MIGRATION weight Task: {}".format(taskPair[1]))
		# print("-"*30+"\n")
		# for triblockIndex in range(NUM_OF_TRIBLOCKS):
		# 	for taskPair in convLayerTasks[triblockIndex]:
		# 		print("MLA_EXE Task: {}".format(taskPair[2]))		
		# S4: Running SpiNNaker2
		while True:
			# Insert task into free PE
			if self.freePeCounter > 0:
				for triBlockIndex in range(NUM_OF_TRIBLOCKS):
					triBlockTasks = convLayerTasks[triBlockIndex]
					triBlockPeState = self.spiNNakerPeStateArray[triBlockIndex]
					findFlag, needPeTaskIndex, freePeIndex = self.findFreePeForTask(triBlockTasks, triBlockPeState)
					if findFlag:
						self.freePeCounter -= 1
						targetPeId = self.getPeId(triBlockIndex, freePeIndex)
						pairedPeId = self.targetToPairedPeIdGenerator(targetPeId)
						convLayerTasks[triBlockIndex][needPeTaskIndex][0][TASK_MIGRATION_DESTINATION] = targetPeId
						convLayerTasks[triBlockIndex][needPeTaskIndex][1][TASK_MIGRATION_DESTINATION] = pairedPeId
						convLayerTasks[triBlockIndex][needPeTaskIndex][2][TASK_DESTINATION] = targetPeId
						convLayerTasks[triBlockIndex][needPeTaskIndex][2][TASK_OPER_A_PEID] = pairedPeId
						self.spiNNakerAddTask(convLayerTasks[triBlockIndex][needPeTaskIndex].pop(0))
						self.spiNNakerAddTask(convLayerTasks[triBlockIndex][needPeTaskIndex].pop(0))
						self.spiNNakerPeStateArray[triBlockIndex][freePeIndex] = PeState.READ_DRAM
						break
			# Run SpiNNaker2
			rsp = self.spiNNakerRunOneClock()
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
				if DramSramDataMigraType.INACTIVE == migraDataType:
					triBlockIndex, peIndex = self.getPeIndex(migraDest)
					peState = self.spiNNakerPeStateArray[triBlockIndex][peIndex]
					if PeState.READ_DRAM == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.READ_DRAM_WEIGHT
					elif PeState.READ_DRAM_INACTI == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.MLA_EXE_DRAM
						for tasks in convLayerTasks[triBlockIndex]:
							if len(tasks) == 1 and tasks[0][TASK_DESTINATION] == migraDest:
								self.spiNNakerAddTask(tasks.pop(0))
					else:
						self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)")
				else:
					targetPeId = self.pairedToTargetPeIdGenerator(migraDest)
					triBlockIndex, peIndex = self.getPeIndex(targetPeId)
					peState = self.spiNNakerPeStateArray[triBlockIndex][peIndex]
					if PeState.READ_DRAM == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.READ_DRAM_INACTI
					elif PeState.READ_DRAM_WEIGHT == peState:
						self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.MLA_EXE_DRAM
						for tasks in convLayerTasks[triBlockIndex]:
							if len(tasks) == 1 and tasks[0][TASK_DESTINATION] == targetPeId:
								self.spiNNakerAddTask(tasks.pop(0))
					else:
						self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (weight)")
			elif Task.DATA_MIGRATION_32_FINISH == rspName:
				self.printInfo(str(rsp))
				# PE_SRAM -> DRAM
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraSourAddr = rsp[TASK_SRAM_ADDR]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
				triBlockIndex, peIndex = self.getPeIndex(migraSour)
				self.spiNNakerPeStateArray[triBlockIndex][peIndex] = PeState.IDLE
				self.freePeCounter += 1
				allFinishFlag = True
				for triBlockIndex in range(NUM_OF_TRIBLOCKS):
					for tasks in convLayerTasks[triBlockIndex]:
						if len(tasks) > 0:
							allFinishFlag = False
							break
					for peState in self.spiNNakerPeStateArray[triBlockIndex]:
						if PeState.IDLE != peState:
							allFinishFlag = False
							break
					if allFinishFlag == False:
						break
				if allFinishFlag:
					self.printInfo("ALL FINISH")
					break
			else:
				# self.printInfo(str(rsp))
				pass

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

	def findFreePeForTask(self, triBlockTasks, triBlockPeState):
		findFlag = False
		# Find Tasks which need Pe
		needPeTaskIndex = None
		for tasksIndex in range(len(triBlockTasks)):
			if self.peTasksLen == len(triBlockTasks[tasksIndex]):
				needPeTaskIndex = tasksIndex
				break
		# Find Free Pe
		freePeIndex = None
		if needPeTaskIndex != None:
			if PeState.IDLE in triBlockPeState:
				peIndex = triBlockPeState.index(PeState.IDLE)
				freePeIndex = peIndex
				findFlag = True
		return findFlag, needPeTaskIndex, freePeIndex

	def convBlockTasksGenerator(self, layerSplitInfo, layerTypeParameter, inActiDimBlocks, 
		weightDimBlocks, outActiDimBlocks):
		layerType, layerParameter = layerTypeParameter
		if len(layerParameter) == 2:
			(inActiDim, weightDim, convStride, outActiDim), poolStride = layerParameter
		else:
			(inActiDim, weightDim, convStride, outActiDim), poolStride, _ = layerParameter
		if isinstance(poolStride, tuple):
			poolDim, poolStride = poolStride
			if poolDim[0] != poolStride:
				poolStride = 1
		numOfInActiBlocks = len(inActiDimBlocks)
		numOfWeightBlocks = len(weightDimBlocks)
		largestInActiBlockAlignSize = self.convInActiDimToSize(inActiDimBlocks[0], operatorFusion=False)
		largestWeightBlockAlignSize = self.convWeightDimToSize(weightDimBlocks[0], operatorFusion=False)
		largestOutActiBlockAlignSize = self.convOutActiDimToSize(outActiDimBlocks[0], operatorFusion=False)
		# MLA in SpiNNaker2 only support convStride=1, need to scale it
		largestOutActiBlockAlignSize *= (convStride * convStride)
		inActiBaseAddr = SRAM_DATA_BEGIN_ADDR
		weightBaseAddr = inActiBaseAddr + largestInActiBlockAlignSize
		outActiBaseAddr = weightBaseAddr + largestWeightBlockAlignSize
		self.customAssert(outActiBaseAddr+largestOutActiBlockAlignSize < SRAM_END_ADDR, "sram overflow")
		# 
		numOfInActiBlocks = len(inActiDimBlocks)
		numOfWeightBlocksInTriblock = len(weightDimBlocks) // NUM_OF_TRIBLOCKS
		convLayerTasks = []
		# DRAM_RD_MIGRATION + MLA_DRAM_WR_MIGRATION
		for triblockIndex in range(NUM_OF_TRIBLOCKS):
			triBlockTasks = []
			outActiDramAddr = 0
			for inActiBlockIndex in range(numOfInActiBlocks):
				inActiDim = inActiDimBlocks[inActiBlockIndex]
				inActiTranSize = self.convInActiDimToSize(inActiDim, self.operatorFuse)
				for weightBlockIndexInTriblock in range(numOfWeightBlocksInTriblock):
					weightBlockIndex = triblockIndex * numOfWeightBlocksInTriblock + weightBlockIndexInTriblock
					weightDim = weightDimBlocks[weightBlockIndex]
					weightBlockAlignSize = self.convWeightDimToSize(weightDim, self.operatorFuse)					
					# DRAM_RD_MIGRATION (inActi): TASK_MIGRATION_DESTINATION (targetPe)
					inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
						TASK_DESTINATION:[triblockIndex+DRAM_ID_START], TASK_MIGRATION_SIZE:inActiTranSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE}
					# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
					weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
						TASK_DESTINATION:[triblockIndex+DRAM_ID_START], TASK_MIGRATION_SIZE:weightBlockAlignSize, 
						TASK_ADDITION:DramSramDataMigraType.WEIGHT}
					# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe)
					mlaTask = self.convMlaTaskGenerator(layerSplitInfo, poolStride, inActiBlockIndex, 
						weightBlockIndex, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, outActiDramAddr)
					triBlockTasks.append([inActiDramToSramMigraTask, weightDramToSramMigraTask, mlaTask])
					outActiDramAddr += 100
			convLayerTasks.append(triBlockTasks)
		self.peTasksLen = len(convLayerTasks[0][0])
		return convLayerTasks

	def convMlaTaskGenerator(self, layerSplitInfo, poolStride, inActiBlockIndex, weightBlockIndex, inActiBaseAddr, 
		weightBaseAddr, outActiBaseAddr, outActiDramAddr):
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		inWidthTotalParts = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		inWidthIndex = inActiBlockIndex % inWidthTotalParts
		inHeightIndex = inActiBlockIndex // inWidthTotalParts
		outChannelIndex = weightBlockIndex
		inWidth = self.getPartLengthFromSplitInfo(inActiSplitInfo[0], inWidthIndex)
		inHeight = self.getPartLengthFromSplitInfo(inActiSplitInfo[1], inHeightIndex)
		inChannel = inActiSplitInfo[2]
		filterWidth = weightStrideSplitInfo[0][0]
		filterHeight = weightStrideSplitInfo[0][1]
		outChannel = self.getPartLengthFromSplitInfo(weightStrideSplitInfo[0][3], outChannelIndex)
		stride = weightStrideSplitInfo[1]
		mlaParam = (MlaOperType.CONV, (inWidth,inHeight,inChannel,filterWidth,filterHeight,outChannel,stride))
		additionParam = (self.operatorFuse, poolStride*poolStride) # (no operator fusion, poolSize=1)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR: weightBaseAddr, 
					TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
					TASK_OUTACTI_DRAM_ADDR: outActiDramAddr, TASK_ADDITION:additionParam}
		return mlaTask


if __name__ == "__main__":
	ssd = SpiNNaker2DistributorNoDataReuse(operatorFuse=True)
	# ssd.vggModelSplitter()
	ssd.resNetModelSplitter()
	ssd.distribute()
	# ssd.dramSramDataMigraValidation()
	# ssd.mlaExeValidate()