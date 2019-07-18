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
import math

class QpeDramDistributorFcFusion(DistributionGeneralClass):
	'''
	Step 1: Simulator --> Create an instance of simulator:SpiNNaker2Distributor
	Step 2: Splitter --> Generate splitting scheme of NN model
			Currently, there is only VGG
	Step 3: Distributor --> Distribute FC-layer into simulator
	'''
	def __init__(self, nocDoubleFreq=True, printFlag=True, operatorFuse=False, dataReuse=False):
		'''
		Step 1: Simulator
		'''
		DistributionGeneralClass.__init__(self, printFlag=printFlag)
		self.nocDoubleFreq = nocDoubleFreq
		self.operatorFuse = operatorFuse
		self.dataReuse = dataReuse
		self.customAssert(not (self.dataReuse and self.operatorFuse), 
			"operator fusion and data reuse could not be supported at same time")
		self.qpeId=[0,0]
		self.dramId=[4]
		self.qpeDram = QpeDram(qpeId=self.qpeId, dramId=self.dramId, nocDoubleFreq=self.nocDoubleFreq)
		self.qpePeStateArray = [PeState.IDLE] * NUM_PES_IN_QPE
		self.freePeCounter = NUM_PES_IN_QPE
		self.peTasksLen = None
		self.sendOutActiToDram = [False, False, False, False]

	def qpeDramRunOneClock(self):
		self.clockCounter += 1
		self.qpeDram.run()
		rsp = self.qpeDram.getNextOutTask()
		return rsp

	def qpeDramAddTask(self, task):
		self.qpeDram.addInTask(task)

	def vggModelSplitter(self):
		'''
		Step 2: Splitter (VGG)
		'''
		modelLayersSplitInfo, modelLayersParameter = QpeDramDistributorFcFusion.vgg(decreaseSize=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def resNetModelSplitter(self):
		'''
		Step 2: Splitter (VGG)
		'''
		modelLayersSplitInfo, modelLayersParameter = QpeDramDistributorFcFusion.resNet50(decreaseSize=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)
		
	def distribute(self):
		'''
		Step 3: Distributor
		'''
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "FC_52" not in layerName:
				continue
			self.layerName = layerName
			self.logBeginTime()
			self.singleFcDistribution()
			self.logEndTime()
			break

	def singleFcDistribution(self):
		# finishTaskCounter = 0		# For testing different with Data-Reuse
		# S1: Generate computation data (inputActi and weight) and validation data (outputActi)
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		layerTypeParameter = self.modelLayersParameter[self.layerName]
		if self.dataReuse:
			inActiDimBlocks, weightDimBlocks, outActiDimBlocks = fcSpittedDataDim(layerSplitInfo, widthHeightOrder=False)
		else:
			inActiDimBlocks, weightDimBlocks, outActiDimBlocks = fcSpittedDataDim(layerSplitInfo, widthHeightOrder=True)
		# S2: Distribute Data into DRAM
		# As parts of weight is always be times of 4, therefore, weights is equally divided into 4 DRAMs
		# InputActi are totally copied into 4 DRAMs
		# TODO: For No-Data-Tran, it is not necessary
		# S3: Prepare for Running SpiNNaker2
		if self.dataReuse:
			fcQpeTasks = self.fcBlockTasksGeneratorDataReuse(inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		else:
			fcQpeTasks = self.fcBlockTasksGeneratorFusion(inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		# for fcPeTasks in fcQpeTasks:
		# 	print("-"*30)
		# 	print("number of tasks in each pe: {}".format(len(fcPeTasks)))
		# 	for fcPeTask in fcPeTasks:
		# 		print(fcPeTask[0])
		# 		print(fcPeTask[1])
		# 		print(fcPeTask[2])
		# 	print("\n"*2)
		# input("")
		# S4: Running SpiNNaker2
		for peIndex in range(NUM_PES_IN_QPE):
			self.qpeDramAddTask(fcQpeTasks[peIndex][0].pop(0))
			self.qpeDramAddTask(fcQpeTasks[peIndex][0].pop(0))
			self.qpePeStateArray[peIndex] = PeState.READ_DRAM
		while True:
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
					peIndex = migraDest[2]
					peState = self.qpePeStateArray[peIndex]
					if PeState.READ_DRAM == peState:
						self.qpePeStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
					elif PeState.READ_DRAM_INACTI == peState:
						self.qpePeStateArray[peIndex] = PeState.MLA_EXE_DRAM
						fcpeTask = fcQpeTasks[peIndex].pop(0)
						if TASK_ADDITION2 not in fcpeTask[0]:
							self.sendOutActiToDram[peIndex] = True
						self.qpeDramAddTask(fcpeTask.pop(0))
					else:
						self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (inActi)")
				else:
					# targetPeId = self.pairedToTargetPeIdGenerator(migraDest, pePairShift=0)
					# peIndex = targetPeId[2]
					peIndex = migraDest[2]
					peState = self.qpePeStateArray[peIndex]
					if PeState.READ_DRAM == peState:
						self.qpePeStateArray[peIndex] = PeState.READ_DRAM_INACTI
					elif PeState.READ_DRAM_WEIGHT == peState:
						self.qpePeStateArray[peIndex] = PeState.MLA_EXE_DRAM
						fcpeTask = fcQpeTasks[peIndex].pop(0)
						if TASK_ADDITION2 not in fcpeTask[0]:
							self.sendOutActiToDram[peIndex] = True
						self.qpeDramAddTask(fcpeTask.pop(0))
					else:
						self.customAssert(False, "Not supoort PE state when DRAM_SRAM_DATA_MIGRA_FINISH (weight)")
			elif Task.DATA_MIGRATION_32_FINISH == rspName:
				self.printInfo(str(rsp))
				# PE_SRAM -> DRAM
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				# migraSourAddr = rsp[TASK_SRAM_ADDR]
				# migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# migraDestAddr = rsp[TASK_MIGRA_SRAM_ADDR]
				peIndex = migraSour[2]
				self.qpePeStateArray[peIndex] = PeState.IDLE
				# 
				if len(fcQpeTasks[peIndex]) > 0:
					# Add task into PE
					inActiMigraTask = fcQpeTasks[peIndex][0].pop(0)
					if Task.TASK_NONE != inActiMigraTask[TASK_NAME]:
						self.qpeDramAddTask(inActiMigraTask)
					self.qpeDramAddTask(fcQpeTasks[peIndex][0].pop(0))
					if Task.TASK_NONE != inActiMigraTask[TASK_NAME]:
						self.qpePeStateArray[peIndex] = PeState.READ_DRAM
					else:
						self.qpePeStateArray[peIndex] = PeState.READ_DRAM_WEIGHT
				else:			
					# Check if all tasks are finish
					if self.allTasksFinish(fcQpeTasks):
						self.printInfo("ALL FINISH")
						break
			elif Task.MLA_FINISH == rspName:
				self.printInfo(str(rsp))
				source = rsp[TASK_SOURCE]
				peIndex = source[2]
				if self.operatorFuse and self.sendOutActiToDram[peIndex] == False:
					self.qpePeStateArray[peIndex] = PeState.IDLE
					# Add task into PE
					if len(fcQpeTasks[peIndex]) > 0:
						self.qpeDramAddTask(fcQpeTasks[peIndex][0].pop(0))
						self.qpeDramAddTask(fcQpeTasks[peIndex][0].pop(0))
						self.qpePeStateArray[peIndex] = PeState.READ_DRAM
				else:
					if self.sendOutActiToDram[peIndex] == True:
						self.sendOutActiToDram[peIndex] = False
			else:
				self.printInfo(str(rsp))
				pass

	def allTasksFinish(self, fcQpeTasks):
		for fcPeTasks in fcQpeTasks:
			if len(fcPeTasks) > 0:
				return False
		for peState in self.qpePeStateArray:
			if PeState.IDLE != peState:
				return False
		return True

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
		fcLayerTasks = [[],[],[],[]]
		outActiDramAddr = 0
		partsOfWeightAlongColumn = numOfWeightBlocks//numOfInActiBlocks
		columnsEachPe = math.ceil(partsOfWeightAlongColumn / NUM_PES_IN_QPE)
		for weightColumnBlockIndex in range(partsOfWeightAlongColumn):
			outActiDim = outActiDimBlocks[weightColumnBlockIndex]
			peIndex = weightColumnBlockIndex // columnsEachPe
			peId = [self.qpeId[0], self.qpeId[1], peIndex]
			for weightRowBlockIndex in range(numOfInActiBlocks):
				inActiDim = inActiDimBlocks[weightRowBlockIndex]
				weightDim = weightDimBlocks[weightColumnBlockIndex*numOfInActiBlocks + weightRowBlockIndex]
				self.customAssert(inActiDim[0] == weightDim[1] and outActiDim[0] == weightDim[0], "fc unmatch 2")
				inActiTranSize = self.fcInActiDimToSize(inActiDim, self.operatorFuse)
				weightTranSize = self.fcWeightDimToSize(weightDim, self.operatorFuse)
				# DRAM_RD_MIGRATION (inActi): 
				inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:inActiTranSize, 
					TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION: peId.copy()}
				# DRAM_RD_MIGRATION (weight): 
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:weightTranSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT, TASK_MIGRATION_DESTINATION: peId.copy()}
				# MLA_DRAM_WR_MIGRATION:
				if (weightRowBlockIndex == numOfInActiBlocks - 1) or (self.operatorFuse == False):
					notSendResult = False
				else:
					notSendResult = True
				mlaTask = self.fcMlaTaskGenerator(peId, weightDim, inActiBaseAddr, weightBaseAddr, 
					outActiBaseAddr, outActiDramAddr, notSendResult)
				fcLayerTasks[peIndex].append([inActiDramToSramMigraTask, weightDramToSramMigraTask, mlaTask])
				# 
				outActiDramAddr += 100
		self.peTasksLen = len(fcLayerTasks[0])
		return fcLayerTasks

	def fcBlockTasksGeneratorDataReuse(self, inActiDimBlocks, weightDimBlocks, outActiDimBlocks):
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
		partsOfWeightAlongRow = numOfInActiBlocks
		partsOfWeightAlongColumn = numOfWeightBlocks//numOfInActiBlocks
		rowsEachPe = math.ceil(partsOfWeightAlongRow / NUM_PES_IN_QPE)
		for weightRowBlockIndex in range(numOfInActiBlocks):
			inActiDim = inActiDimBlocks[weightRowBlockIndex]
			peIndex = weightRowBlockIndex // rowsEachPe
			peId = [self.qpeId[0], self.qpeId[1], peIndex]
			for weightColumnBlockIndex in range(partsOfWeightAlongColumn):
				outActiDim = outActiDimBlocks[weightColumnBlockIndex]
				weightDim = weightDimBlocks[weightRowBlockIndex*partsOfWeightAlongColumn + weightColumnBlockIndex]
				self.customAssert(inActiDim[0] == weightDim[1] and outActiDim[0] == weightDim[0], "fc unmatch 2")
				inActiTranSize = self.fcInActiDimToSize(inActiDim, operatorFusion=False)
				weightTranSize = self.fcWeightDimToSize(weightDim, operatorFusion=False)
				if weightColumnBlockIndex == 0:
					# DRAM_RD_MIGRATION (inActi): 
					inActiDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
						TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:inActiTranSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION: peId.copy()}
				else:
					inActiDramToSramMigraTask = {TASK_NAME:Task.TASK_NONE}
				# DRAM_RD_MIGRATION (weight): 
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:self.dramId, TASK_MIGRATION_SIZE:weightTranSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT, TASK_MIGRATION_DESTINATION: peId.copy()}
				# MLA_DRAM_WR_MIGRATION:
				notSendResult = False
				mlaTask = self.fcMlaTaskGenerator(peId, weightDim, inActiBaseAddr, weightBaseAddr, 
					outActiBaseAddr, outActiDramAddr, notSendResult)
				fcLayerTasks[peIndex].append([inActiDramToSramMigraTask, weightDramToSramMigraTask, mlaTask])
				# 
				outActiDramAddr += 100
		self.peTasksLen = len(fcLayerTasks[0])
		return fcLayerTasks

	def fcMlaTaskGenerator(self, peId, weightDim, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, 
		outActiDramAddr, notSendResult):
		# matrixAWidth, matrixAHeight, matrixBWidth
		mlaParam = (MlaOperType.MM, (weightDim[1], 1, weightDim[0]))
		additionParam = (self.operatorFuse, 1) # (no operator fusion, poolSize=1)
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_MLA_PARAM:mlaParam, TASK_OPER_A_ADDR: weightBaseAddr, 
					TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
					TASK_OUTACTI_DRAM_ADDR: outActiDramAddr, TASK_ADDITION:additionParam, 
					TASK_DESTINATION: peId.copy(), TASK_OPER_A_PEID: peId.copy()}
		if notSendResult:
			mlaTask[TASK_ADDITION2] = True
		return mlaTask



if __name__ == "__main__":
	ssd = QpeDramDistributorFcFusion(operatorFuse=False, dataReuse=False)
	# ssd.vggModelSplitter()
	ssd.resNetModelSplitter()
	ssd.distribute()