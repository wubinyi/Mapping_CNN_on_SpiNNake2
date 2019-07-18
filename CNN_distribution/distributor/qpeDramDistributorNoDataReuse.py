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

class QpeDramDistributorNoDataReuse(DistributionGeneralClass):
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
		self.qpeId=[0,0]
		self.dramId=[4]
		self.qpeDram = QpeDram(qpeId=self.qpeId, dramId=self.dramId, nocDoubleFreq=self.nocDoubleFreq)
		self.qpePeStateArray = [PeState.IDLE] * NUM_PES_IN_QPE
		self.freePeCounter = NUM_PES_IN_QPE
		self.peTasksLen = None

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
		modelLayersSplitInfo, modelLayersParameter = QpeDramDistributorNoDataReuse.vgg(decreaseSize=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def resNetModelSplitter(self):
		'''
		Step 2: Splitter (ResNet)
		'''
		modelLayersSplitInfo, modelLayersParameter = QpeDramDistributorNoDataReuse.resNet50(decreaseSize=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def distribute(self):
		'''
		Step 3: Distributor
		'''
		self.convLayersDistribution()
		
	def convLayersDistribution(self):
		'''
		CONV_1: 2332472 clocks -> 2354264 -> 2361097
		CONV_3: 5858444
		CONV_4: 16210715
		CONV_6: 6260116
		CONV_10:6375261
		'''
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "CONV_1" not in layerName:
				continue
			self.layerName = layerName
			self.logBeginTime()
			self.singleConvDistribution()
			self.logEndTime()
			break

	def singleConvDistribution(self):
		if "CONV_" in self.layerName:
			isConv = True
		else:
			isConv = False
		# finishTaskCounter = 0		# For testing different with Data-Reuse
		# S1: Generate computation data (inputActi and weight) and validation data (outputActi)
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		layerTypeParameter = self.modelLayersParameter[self.layerName]
		if isConv:
			# inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, \
			# 	outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
			inActiDimBlocks, weightDimBlocks, outActiDimBlocks = convSpittedDataDim(layerSplitInfo, layerTypeParameter)
		else:
			inActiDimBlocks, weightDimBlocks, outActiDimBlocks = fcSpittedDataDim(layerSplitInfo, widthHeightOrder=True)
		# S2: Distribute Data into DRAM
		# As parts of weight is always be times of 4, therefore, weights is equally divided into 4 DRAMs
		# InputActi are totally copied into 4 DRAMs
		# TODO: For No-Data-Tran, it is not necessary
		# S3: Prepare for Running SpiNNaker2
		if isConv:
			# convLayerTasks = self.convBlockTasksGenerator(layerSplitInfo, inActiAlignBlocks, inActiBaseAddr, 
			# 	weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, outActiBaseAddr)
			convLayerTasks = self.convBlockTasksGenerator(layerSplitInfo, layerTypeParameter, inActiDimBlocks, 
				weightDimBlocks, outActiDimBlocks)
		else:
			convLayerTasks = self.fcBlockTasksGenerator(inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		# print("length of convLayerTasks: {}".format(len(convLayerTasks)))
		# for convLayerTask in convLayerTasks:
		# 	print(convLayerTask[0])
		# 	print(convLayerTask[1])
		# 	print(convLayerTask[2])
		# 	print("")
		# S4: Running SpiNNaker2
		while True:
			# Insert task into free PE
			if self.freePeCounter > 0:
				tasksIndex, freePeId = self.findFreePeAndTask(convLayerTasks)
				if tasksIndex != None:
					self.freePeCounter -= 1
					targetPeId = freePeId
					pairedPeId = self.targetToPairedPeIdGenerator(targetPeId)					
					convLayerTasks[tasksIndex][0][TASK_MIGRATION_DESTINATION] = targetPeId
					convLayerTasks[tasksIndex][1][TASK_MIGRATION_DESTINATION] = pairedPeId
					convLayerTasks[tasksIndex][2][TASK_DESTINATION] = targetPeId
					convLayerTasks[tasksIndex][2][TASK_OPER_A_PEID] = pairedPeId
					self.qpeDramAddTask(convLayerTasks[tasksIndex].pop(0))
					self.qpeDramAddTask(convLayerTasks[tasksIndex].pop(0))
					peIndex = freePeId[2]
					self.qpePeStateArray[peIndex] = PeState.READ_DRAM
			# Run SpiNNaker2
			rsp = self.qpeDramRunOneClock()
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
					targetPe = migraDest
					peIndex = migraDest[2]
					peState = self.qpePeStateArray[peIndex]
					if PeState.READ_DRAM == peState:
						self.qpePeStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
					elif PeState.READ_DRAM_INACTI == peState:
						self.qpePeStateArray[peIndex] = PeState.MLA_EXE_DRAM
						for tasks in convLayerTasks:
							if len(tasks) == 1 and tasks[0][TASK_DESTINATION] == migraDest:
								self.qpeDramAddTask(tasks.pop(0))
								break
					else:
						self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)")
				else:
					targetPeId = self.pairedToTargetPeIdGenerator(migraDest)
					peIndex = targetPeId[2]
					peState = self.qpePeStateArray[peIndex]
					if PeState.READ_DRAM == peState:
						self.qpePeStateArray[peIndex] = PeState.READ_DRAM_INACTI
					elif PeState.READ_DRAM_WEIGHT == peState:
						self.qpePeStateArray[peIndex] = PeState.MLA_EXE_DRAM
						for tasks in convLayerTasks:
							if len(tasks) == 1 and tasks[0][TASK_DESTINATION] == targetPeId:
								self.qpeDramAddTask(tasks.pop(0))
								break
					else:
						self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (weight)")
			elif Task.DATA_MIGRATION_32_FINISH == rspName:
				self.printInfo(str(rsp))
				# PE_SRAM -> DRAM
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraSourAddr = rsp[TASK_SRAM_ADDR]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
				self.freePeCounter += 1
				peIndex = migraSour[2]
				self.qpePeStateArray[peIndex] = PeState.IDLE
				# Check if all tasks are finish
				if self.allTasksFinish(convLayerTasks):
					self.printInfo("ALL FINISH")
					break
				# finishTaskCounter += 1 		# For testing different with Data-Reuse
				# if finishTaskCounter == 16: 	# For testing different with Data-Reuse
				# 	break						# For testing different with Data-Reuse	
			else:
				self.printInfo(str(rsp))
				pass

	def allTasksFinish(self, layerTasks):
		for tasks in layerTasks:
			if len(tasks) > 0:
				return False
		for peState in self.qpePeStateArray:
			if PeState.IDLE != peState:
				return False
		return True

	def findFreePeAndTask(self, layerTasks):
		# Find Task
		tasksIndex = None
		for index in range(len(layerTasks)):
			if len(layerTasks[index]) == self.peTasksLen:
				tasksIndex = index
				break
		# Find Free PE
		freePeId = [self.qpeId[0], self.qpeId[1], self.qpePeStateArray.index(PeState.IDLE)]
		return tasksIndex, freePeId

	def convBlockTasksGenerator(self, layerSplitInfo, layerTypeParameter, inActiDimBlocks, 
		weightDimBlocks, outActiDimBlocks):
		layerType, layerParameter = layerTypeParameter
		if 2 == len(layerParameter):
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
		largestOutActiBlockAlignSize *= (convStride * convStride)
		inActiBaseAddr = SRAM_DATA_BEGIN_ADDR
		weightBaseAddr = inActiBaseAddr + largestInActiBlockAlignSize
		outActiBaseAddr = weightBaseAddr + largestWeightBlockAlignSize
		self.customAssert(outActiBaseAddr+largestOutActiBlockAlignSize < SRAM_END_ADDR, "sram overflow")
		convLayerTasks = []
		# DRAM_RD_MIGRATION + MLA_DRAM_WR_MIGRATION
		outActiDramAddr = 0
		for inActiBlockIndex in range(numOfInActiBlocks):
			inActiDim = inActiDimBlocks[inActiBlockIndex]
			inActiTranSize = self.convInActiDimToSize(inActiDim, self.operatorFuse)
			for weightBlockIndex in range(numOfWeightBlocks):
				weightDim = weightDimBlocks[weightBlockIndex]
				weightBlockAlignSize = self.convWeightDimToSize(weightDim, self.operatorFuse)
				# DRAM_RD_MIGRATION (inActi): TASK_MIGRATION_DESTINATION (targetPe)
				inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:inActiTranSize, 
					TASK_ADDITION:DramSramDataMigraType.INACTIVE}
				# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:weightBlockAlignSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT}
				# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe)
				mlaTask = self.convMlaTaskGenerator(layerSplitInfo, poolStride, inActiBlockIndex, 
					weightBlockIndex, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, outActiDramAddr)
				convLayerTasks.append([inActiDramToSramMigraTask, weightDramToSramMigraTask, mlaTask])
				outActiDramAddr += 100
		self.peTasksLen = len(convLayerTasks[0])
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

	def fcBlockTasksGenerator(self, inActiDimBlocks, weightDimBlocks, outActiDimBlocks):
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
		outActiDramAddr = 0
		partsOfWeightAlongColumn = numOfWeightBlocks//numOfInActiBlocks
		for weightColumnBlockIndex in range(partsOfWeightAlongColumn):
			outActiDim = outActiDimBlocks[weightColumnBlockIndex]
			for weightRowBlockIndex in range(numOfInActiBlocks):
				inActiDim = inActiDimBlocks[weightRowBlockIndex]
				weightDim = weightDimBlocks[weightColumnBlockIndex*numOfInActiBlocks + weightRowBlockIndex]
				self.customAssert(inActiDim[0] == weightDim[1] and outActiDim[0] == weightDim[0], "fc unmatch 2")
				inActiTranSize = self.fcInActiDimToSize(inActiDim, self.operatorFuse)
				weightTranSize = self.fcWeightDimToSize(weightDim, self.operatorFuse)
				# DRAM_RD_MIGRATION (inActi): TASK_MIGRATION_DESTINATION (targetPe)
				inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:inActiTranSize, 
					TASK_ADDITION:DramSramDataMigraType.INACTIVE}
				# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:weightTranSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT}
				# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe)
				mlaTask = self.fcMlaTaskGenerator(weightDim, inActiBaseAddr, weightBaseAddr, 
					outActiBaseAddr, outActiDramAddr)
				fcLayerTasks.append([inActiDramToSramMigraTask, weightDramToSramMigraTask, mlaTask])
				# 
				outActiDramAddr += 100
		self.peTasksLen = len(fcLayerTasks[0])
		return fcLayerTasks

	def fcMlaTaskGenerator(self, weightDim, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, 
		outActiDramAddr):
		# matrixAWidth, matrixAHeight, matrixBWidth
		mlaParam = (MlaOperType.MM, (weightDim[1], 1, weightDim[0]))
		additionParam = (self.operatorFuse, 1) # (no operator fusion, poolSize=1)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR: weightBaseAddr, 
					TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
					TASK_OUTACTI_DRAM_ADDR: outActiDramAddr, TASK_ADDITION:additionParam}
		return mlaTask

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

	def mlaExeValidate(self):
		mlaParam = (MlaOperType.CONV, (224,22,3,3,3,4,1))
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR:48864, 
					TASK_OPER_B_ADDR:33024, TASK_OPER_C_ADDR:48976, TASK_OUTACTI_DRAM_ADDR:0, 
					TASK_DESTINATION:[self.qpeId[0],self.qpeId[1],0], 
					TASK_OPER_A_PEID:[self.qpeId[0],self.qpeId[1],1]}
		self.qpeDramAddTask(mlaTask)
		for _ in range(40000):
			rsp = self.qpeDramRunOneClock()
			if Task.TASK_NONE != rsp[TASK_NAME]:
				self.printInfo(str(rsp))
				if Task.DATA_MIGRATION_32_FINISH == rsp[TASK_NAME]:
					break

if __name__ == "__main__":
	ssd = QpeDramDistributorNoDataReuse(operatorFuse=True)
	ssd.vggModelSplitter()
	# ssd.resNetModelSplitter()
	ssd.distribute()
	# ssd.dramSramDataMigraValidation()
	# ssd.mlaExeValidate()