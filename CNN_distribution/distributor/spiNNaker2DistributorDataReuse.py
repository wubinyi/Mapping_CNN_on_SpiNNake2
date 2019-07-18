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
parserSplitterPath = os.path.join(projectFolder, "parserSplitter")
if parserSplitterPath not in sys.path:
	sys.path.insert(0, parserSplitterPath)

from distributionGeneral import *
from spiNNaker2TriBlockSimulator import SpiNNaker2TriBlock
from spiNNakerSimulatorGeneral import *
from dataGenerator import convSpittedDataDim
from nnGeneral import PE_SRAM_LIMIT, BasicOperation
from convLayerMapper import ConvLayerMapper

class SpiNNaker2DistributorDataReuse(DistributionGeneralClass):
	'''
	Step 1: Simulator --> Create an instance of simulator:SpiNNaker2Distributor
	Step 2: Splitter --> Generate splitting scheme of NN model
			Currently, there is only VGG
	Step 3: Distributor --> Distribute CONV-layer into simulator
	'''
	def __init__(self, nocDoubleFreq=True, printFlag=True, operatorFusion=True):
		'''
		Step 1: Simulator
		'''
		DistributionGeneralClass.__init__(self, printFlag=printFlag)
		self.nocDoubleFreq = nocDoubleFreq
		self.operatorFusion = operatorFusion
		self.spiNNaker2 = SpiNNaker2TriBlock(nocDoubleFreq=self.nocDoubleFreq, noDataTran=True)
		# 
		self.inActiFromDram = None
		self.convInActiMigraTasks = None
		self.convWeightMigraTasks = None
		self.convMlaTask = None
		self.douBlockQpeList = None
		self.spiNNakerPeTasksContainer()
		self.nextLayerLoadFromDram = False

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

	def resNetModelSplitter(self):
		'''
		Step 2: Splitter (ResNet)
		'''
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorDataReuse.resNet50(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def vggModelSplitter(self):
		'''
		Step 2: Splitter (VGG)
		'''
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorDataReuse.vgg(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def distribute(self):
		'''
		Step 3: Distributor
		'''
		self.convLayersDistribution()
		# self.poolLayersDistribution()
		self.spiNNakerStop()
		
	def convLayersDistribution(self):
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "CONV_2" not in layerName:
				continue
			self.layerName = layerName
			self.printInfo("nextLayerLoadFromDram flag from previous layer: {}".format(self.nextLayerLoadFromDram))
			if "CONV_1" == layerName or self.nextLayerLoadFromDram == True:
				self.inActiFromDram = True
			else:
				self.inActiFromDram = False
			# # Determine if there is POOL for this CONV
			# layerNameSplitInfo = layerName.split("_")
			# poolLayerName = "POOL_"+str(int(layerNameSplitInfo[1])+1)
			# paddForPoolLayerName = "PADD_"+str(int(layerNameSplitInfo[1])+1)
			# if poolLayerName in layerNames and paddForPoolLayerName not in layerNames:
			# 	poolLayerTypeParam = self.modelLayersParameter[poolLayerName]
			# 	layerType, layerParam = poolLayerTypeParam
			# 	poolInActiDim, poolDim, poolStride, poolType, poolOutActiDim = layerParam
			# 	poolParam = (True, poolDim[0]*poolDim[1])
			# else:
			# 	poolParam = (False, 1)
			# self.printInfo("POOL for CONV: {}".format(poolParam))
			self.logBeginTime()
			self.singleConvDistribution()
			self.logEndTime()
			break

	def poolLayersDistribution(self):
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "POOL_2" not in layerName:
				continue
			self.layerName = layerName
			self.printInfo("nextLayerLoadFromDram flag from previous layer: {}".format(self.nextLayerLoadFromDram))
			if "CONV_1" == layerName or self.nextLayerLoadFromDram == True:
				self.inActiFromDram = True
			else:
				self.inActiFromDram = False
			self.logBeginTime()
			self.singlePoolDistribution()
			self.logEndTime()
			break			

	def singlePoolDistribution(self):
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		self.convInActiMigraTasks = self.poolInActiMigraTasksGenerator128Fit(layerSplitInfo)
		idlePes = self.loadInActisFromSram128Fit()
		print("idlePes: {}".format(idlePes))

	def singleConvDistribution(self):
		# S1: Generate computation data (inputActi and weight) and validation data (outputActi)
		layerSplitInfo = self.modelLayersSplitInfo[self.layerName]
		layerTypeParameter = self.modelLayersParameter[self.layerName]
		# inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, \
		# 	outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
		inActiDimBlocks, weightDimBlocks, outActiDimBlocks = convSpittedDataDim(layerSplitInfo, layerTypeParameter)
		# S2: Distribute Data into DRAM
		# Weights is equally divided into 4 DRAMs (As parts of weight is always be times of 4).
		# Only the first layer's inActi will be seperated into DRAM. Other Layer's inActi are stored
		# 		in HOST-QPE.
		# TODO: For No-Data-Tran, it is not necessary
		# S3: Prepare for Running SpiNNaker2
		self.convInActiMigraTasks, self.convWeightMigraTasks, self.convMlaTask = self.convBlockTasksGenerator(layerSplitInfo, 
			layerTypeParameter, inActiDimBlocks, weightDimBlocks, outActiDimBlocks)
		partsOfWeight = len(weightDimBlocks)
		partsOfInActi = len(inActiDimBlocks)
		partsOfInActiExpand =  self.roundToPowerOf2(partsOfInActi)
		weightBlockSize = self.align16(weightDimBlocks[0][0] * weightDimBlocks[0][1] * weightDimBlocks[0][2] * \
			self.alignMlaRow(weightDimBlocks[0][3]))
		for douBlockIndex in range(len(self.convInActiMigraTasks)):
			print("Double-Block: {}".format(douBlockIndex))
			for inActiMigraTask in self.convInActiMigraTasks[douBlockIndex]:
				print("inActi Task: {}".format(inActiMigraTask))
		print("")
		# for douBlockIndex in range(len(self.convWeightMigraTasks)):
		# 	print("Double-Block: {}".format(douBlockIndex))
		# 	for weightMigraTask in self.convWeightMigraTasks[douBlockIndex]:
		# 		print("weight Task: {}".format(weightMigraTask))
		# print("")
		print("pemla Task: {}".format(self.convMlaTask))
		print("")
		# S4: Running SpiNNaker2 -> Load Weight
		self.printInfo("Load weight from DRAM")
		if 16 == partsOfWeight:
			self.loadWeights16()
		elif 64 == partsOfWeight:
			self.loadWeights64()
		elif 32 == partsOfWeight:
			self.loadWeights32()
		elif 128 ==  partsOfWeight:
			self.loadWeights128()
		# S5: Running SpiNNaker2 -> Load InActi
		if self.inActiFromDram:
			self.printInfo("Load inActi from DRAM")
			if 16 == partsOfInActi:
				self.loadInActisFromDram16()
			else:
				if 64 == partsOfInActiExpand:
					idlePes = self.loadInActisFromDram64Fit()
				elif 16 == partsOfInActiExpand:
					idlePesIndex = self.loadInActisFromDram16Fit()
				elif 128 == partsOfInActiExpand:
					idlePes = self.loadInActisFromDram128Fit()
					self.printInfo("idlePes: {}".format(idlePes))
		else:
			self.printInfo("Load inActi from SRAM")
			if 8 == partsOfInActi:
				self.loadInActisFromSram8()
			elif 16 == partsOfInActi:
				self.loadInActisFromSram16()
			elif 4 == partsOfInActi:
				self.loadInActisFromSram4Fit()
			elif 2 == partsOfInActi or 1 == partsOfInActi:
				self.loadInActisFromSram4Fit()
			else:
				if 16 == partsOfInActiExpand:
					idlePesIndex = self.loadInActisFromSram16Fit()
					self.printInfo("idlePesIndex: {}".format(idlePesIndex))
				elif 8 == partsOfInActiExpand:
					idlePesIndex = self.loadInActisFromSram8Fit()
					self.printInfo("idlePesIndex: {}".format(idlePesIndex))
				elif 64 == partsOfInActiExpand:
					idlePes = self.loadInActisFromSram64Fit()
					self.printInfo("idlePes: {}".format(idlePes))
				elif 32 == partsOfInActiExpand:
					idlePes = self.loadInActisFromSram32Fit()
					self.printInfo("idlePes: {}".format(idlePes))
				elif 128 == partsOfInActiExpand:
					idlePes = self.loadInActisFromSram128Fit()
					self.printInfo("idlePes: {}".format(idlePes))
		# S6: Running SpiNNaker2 -> MLA
		self.printInfo("Begin MLA acceleration")
		if 16 == partsOfWeight and 16 == partsOfInActi:
			self.convMlaExe1616()
		elif 64 == partsOfWeight and 8 == partsOfInActi:
			self.convMlaExe_32_4_16_old(partsOfInActi=partsOfInActi, 
				partsOfWeight=partsOfWeight, weightSize=weightBlockSize)
		elif (32 <= partsOfWeight and partsOfWeight <= 128) and (4 <= partsOfInActi and partsOfInActi <= 16):
			self.convMlaExe_32_4_16(partsOfInActi=partsOfInActi, 
				partsOfWeight=partsOfWeight, weightSize=weightBlockSize, idlePeIdList=[])
		elif partsOfWeight * partsOfInActi == 128:
			self.convMlaExe_128(partsOfInActi, partsOfWeight)
		else:
			if 16 == partsOfInActiExpand and partsOfWeight >= 32:
				idlePeIdList = self.generateIdlePeList(idlePesIndex)
				self.convMlaExe_32_4_16(partsOfInActi=partsOfInActiExpand, 
					partsOfWeight=partsOfWeight, weightSize=weightBlockSize, idlePeIdList=idlePeIdList)
			elif 8 == partsOfInActiExpand and partsOfWeight >= 32:
				idlePeIdList = self.generateIdlePeList(idlePesIndex)
				self.convMlaExe_32_4_16(partsOfInActi=partsOfInActiExpand, 
					partsOfWeight=partsOfWeight, weightSize=weightBlockSize, idlePeIdList=idlePeIdList)
			elif 64 == partsOfInActiExpand and partsOfWeight == 32:
				idlePeIdList = self.generateIdlePeListFor64(idlePes)
				self.printInfo("idlePeIdList: {}".format(idlePeIdList))
				self.convMlaExe_32_4(partsOfInActi=partsOfInActiExpand, partsOfWeight=partsOfWeight, 
					weightSize=weightBlockSize, idlePeIdList=idlePeIdList)
				# self.weightMigrationInsideBlock(weightBlockSize)
			elif 16 == partsOfInActiExpand and 16 == partsOfWeight:
				idlePeIdList = self.generateIdlePeList(idlePesIndex)
				self.convMlaExe1616(idlePeIdList=idlePeIdList)
			elif 32 == partsOfInActiExpand and partsOfWeight == 32:
				idlePeIdList = self.generateIdlePeListFor32(idlePes)
				self.printInfo("idlePeIdList: {}".format(idlePeIdList))
				self.convMlaExe_32_4(partsOfInActi=partsOfInActiExpand, partsOfWeight=partsOfWeight, 
					weightSize=weightBlockSize, idlePeIdList=idlePeIdList)
			elif 128 == partsOfInActiExpand and partsOfWeight == 16:
				idlePeIdList = self.generateIdlePeListFor128(idlePes)
				self.printInfo("idlePeIdList: {}".format(idlePeIdList))
				self.convMlaExe_128_16(partsOfInActi=partsOfInActiExpand, partsOfWeight=partsOfWeight, 
					weightSize=weightBlockSize, idlePeIdList=idlePeIdList)

	def generateIdlePeList(self, idlePesIndex):
		'''
		For inActi <= 16
		'''
		idlePeIdList = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if len(idlePesIndex[douBlockIndex]) != 0:
				for qpeId in DOUBLOCK_QPE_ID_LIST[douBlockIndex]:
					for peIndex in idlePesIndex[douBlockIndex]:
						peId = qpeId.copy()
						peId.append(peIndex)
						idlePeIdList.append(peId)
		return idlePeIdList

	def generateIdlePeListFor128(self, idlePes):
		idlePeIdList = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			douBlockIdlePes = idlePes[douBlockIndex]
			for idlePe in douBlockIdlePes:
				idlePeIdList.append(idlePe)
		return idlePeIdList

	def generateIdlePeListFor64(self, idlePes):
		idlePeIdList = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			douBlockIdlePes = idlePes[douBlockIndex]
			for idlePe in douBlockIdlePes:
				if self.inActiFromDram:
					newIdlePeId = idlePe.copy()
					idleQpeId = idlePe[Y_AXIS_INDEX:Z_AXIS_INDEX]
					if idleQpeId in DOUBLOCK_QPE_ID_LIST[0]:
						newIdlePeId[Y_AXIS_INDEX] += 2
					elif idleQpeId in DOUBLOCK_QPE_ID_LIST[1]:
						newIdlePeId[X_AXIS_INDEX] -= 2
					elif idleQpeId in DOUBLOCK_QPE_ID_LIST[2]:
						newIdlePeId[X_AXIS_INDEX] += 2
					elif idleQpeId in DOUBLOCK_QPE_ID_LIST[3]:
						newIdlePeId[Y_AXIS_INDEX] -= 2
					else:
						self.customAssert(False, "Unknown idleQpeId: {}".format(idleQpeId))
					idlePeIdList.append(idlePe)
					idlePeIdList.append(newIdlePeId)
				else:
					newIdlePeId = idlePe.copy()
					idleQpeId = idlePe[Y_AXIS_INDEX:Z_AXIS_INDEX]
					if idleQpeId in DOUBLOCK_QPE_ID_LIST[0]:
						newIdlePeId[Y_AXIS_INDEX] -= 2
					elif idleQpeId in DOUBLOCK_QPE_ID_LIST[1]:
						newIdlePeId[X_AXIS_INDEX] += 2
					elif idleQpeId in DOUBLOCK_QPE_ID_LIST[2]:
						newIdlePeId[X_AXIS_INDEX] -= 2
					elif idleQpeId in DOUBLOCK_QPE_ID_LIST[3]:
						newIdlePeId[Y_AXIS_INDEX] += 2
					else:
						self.customAssert(False, "Unknown idleQpeId: {}".format(idleQpeId))
					idlePeIdList.append(idlePe)
					idlePeIdList.append(newIdlePeId)
		return idlePeIdList	

	def generateIdlePeListFor32(self, idlePes):
		idlePeIdList = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			douBlockIdlePes = idlePes[douBlockIndex]
			for idlePe in douBlockIdlePes:
				if self.inActiFromDram:
					idlePeIdList.append(idlePe)
					newIdlePeId = idlePe.copy()
					for _ in range(3):
						idleQpeId = idlePe[Y_AXIS_INDEX:Z_AXIS_INDEX]
						if idleQpeId in DOUBLOCK_QPE_ID_LIST[0]:
							newIdlePeId[Y_AXIS_INDEX] += 1
						elif idleQpeId in DOUBLOCK_QPE_ID_LIST[1]:
							newIdlePeId[X_AXIS_INDEX] -= 1
						elif idleQpeId in DOUBLOCK_QPE_ID_LIST[2]:
							newIdlePeId[X_AXIS_INDEX] += 1
						elif idleQpeId in DOUBLOCK_QPE_ID_LIST[3]:
							newIdlePeId[Y_AXIS_INDEX] -= 1
						else:
							self.customAssert(False, "Unknown idleQpeId: {}".format(idleQpeId))
						idlePeIdList.append(copy.deepcopy(newIdlePeId))
				else:
					idlePeIdList.append(idlePe)
					newIdlePeId = idlePe.copy()
					for _ in range(3):
						idleQpeId = idlePe[Y_AXIS_INDEX:Z_AXIS_INDEX]
						if idleQpeId in DOUBLOCK_QPE_ID_LIST[0]:
							newIdlePeId[Y_AXIS_INDEX] -= 1
						elif idleQpeId in DOUBLOCK_QPE_ID_LIST[1]:
							newIdlePeId[X_AXIS_INDEX] += 1
						elif idleQpeId in DOUBLOCK_QPE_ID_LIST[2]:
							newIdlePeId[X_AXIS_INDEX] -= 1
						elif idleQpeId in DOUBLOCK_QPE_ID_LIST[3]:
							newIdlePeId[Y_AXIS_INDEX] += 1
						else:
							self.customAssert(False, "Unknown idleQpeId: {}".format(idleQpeId))
						idlePeIdList.append(copy.deepcopy(newIdlePeId))
		return idlePeIdList		

	# =========================================================
	# ML Accelerator
	# =========================================================
	def convMlaExe1616(self, idlePeIdList=[]):
		# S1: Gnerate MLA tasks
		taskCounter = 0
		for yIndex in range(NUM_QPES_Y_AXIS):
			for xIndex in range(NUM_QPES_X_AXIS):
				for peIndex in range(NUM_PES_IN_QPE):
					taskDesti = [yIndex, xIndex, peIndex]
					if taskDesti in idlePeIdList:
						continue
					blockIndex = self.getBlockIndexInDouBlock(taskDesti)
					if blockIndex == None:
						continue
					if blockIndex == 0:
						mlaTask = copy.deepcopy(self.convMlaTask)
						mlaTask[TASK_DESTINATION] = taskDesti
						mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=0)
						mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
						self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
						mlaTask = copy.deepcopy(self.convMlaTask)
						mlaTask[TASK_DESTINATION] = taskDesti
						mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=1)
						mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
						self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
						taskCounter += 2
					else:
						mlaTask = copy.deepcopy(self.convMlaTask)
						mlaTask[TASK_DESTINATION] = taskDesti
						mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=2)
						mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
						self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
						mlaTask = copy.deepcopy(self.convMlaTask)
						mlaTask[TASK_DESTINATION] = taskDesti
						mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=3)
						mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
						self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
						taskCounter += 2
		# S2: Place MLA tasks
		for qpeIndex in range(len(FIRST_DOUBLOCK_QPE_IDS)):
			for peIndex in range(NUM_PES_IN_QPE):
				for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
					qpeId = DOUBLOCK_QPE_ID_LIST[douBlockIndex][qpeIndex]
					peTaskContainer = self.spiNNakerContainer[qpeId[Y_AXIS_INDEX]][qpeId[X_AXIS_INDEX]][peIndex]
					if len(peTaskContainer) > 0:
						task = peTaskContainer.pop(0)
						self.spiNNakerAddTask(task)
		# S3: Running SpiNNaker2
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_32_FINISH == rspName:
				self.printInfo(rsp)
				taskCounter -= 1
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				peTaskContainer = self.spiNNakerContainer[migraSour[Y_AXIS_INDEX]][migraSour[X_AXIS_INDEX]][migraSour[Z_AXIS_INDEX]]
				if len(peTaskContainer) > 0:
					self.spiNNakerAddTask(peTaskContainer.pop(0))
			else:
				self.printInfo(rsp)
		self.printInfo("MLA for 16x16 ALL FINISH")

	def convMlaExe_32_4_16_old(self, partsOfInActi, partsOfWeight, weightSize):
		self.customAssert((partsOfInActi >= 4) and (partsOfInActi <= 16), "Unsupport partsOfInActi: {}".format(partsOfInActi))
		self.customAssert(partsOfWeight >= 32, "Unsupport partsOfWeight: {}".format(partsOfWeight))
		# Determine how many data reuse outside QPE --> partsOfWeight >= 32 --> Move Weight
		dataReuseOutQpe = partsOfWeight // (NUM_PES_IN_QPE * NUM_OF_QPE_IN_DOUBLOCK)
		self.printInfo("dataReuseOutQpe: {}".format(dataReuseOutQpe))
		# Determine how many data reuse inside QPE --> partsOfInActi >= 4 and <= 16
		dataReuseInQpe = partsOfInActi // NUM_OF_DOUBLOCKS
		self.printInfo("dataReuseInQpe: {}".format(dataReuseInQpe))
		self.printInfo("Run MLA_EXE {} times".format(dataReuseInQpe * dataReuseOutQpe))
		migraRounds = NUM_OF_BLOCKS // dataReuseOutQpe
		self.printInfo("Migration Rounds: {}".format(migraRounds))
		# LOOP
		while dataReuseOutQpe > 0:
			dataReuseOutQpe -= 1
			# S1: Gnerate MLA tasks
			taskCounter = 0
			for yIndex in range(NUM_QPES_Y_AXIS):
				for xIndex in range(NUM_QPES_X_AXIS):
					for peIndex in range(NUM_PES_IN_QPE):
						taskDesti = [yIndex, xIndex, peIndex]
						blockIndex = self.getBlockIndexInDouBlock(taskDesti)
						if blockIndex == None:
							continue
						for dataReuseLoopIndex in range(0, dataReuseInQpe):
							pePairShift = (dataReuseLoopIndex + 1) % NUM_PES_IN_QPE
							mlaTask = copy.deepcopy(self.convMlaTask)
							mlaTask[TASK_DESTINATION] = taskDesti
							mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=pePairShift)
							mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
							self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
						taskCounter += dataReuseInQpe
			# S2: Place MLA tasks
			for qpeIndex in range(len(FIRST_DOUBLOCK_QPE_IDS)):
				for peIndex in range(NUM_PES_IN_QPE):
					for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
						qpeId = DOUBLOCK_QPE_ID_LIST[douBlockIndex][qpeIndex]
						task = self.spiNNakerContainer[qpeId[Y_AXIS_INDEX]][qpeId[X_AXIS_INDEX]][peIndex].pop(0)
						self.spiNNakerAddTask(task)
			# S3: Running SpiNNaker2
			while taskCounter > 0:
				rsp = self.spiNNakerRunOneClock()
				rspName = rsp[TASK_NAME]
				if Task.TASK_NONE == rspName:
					pass
				elif Task.DATA_MIGRATION_32_FINISH == rspName:
					self.printInfo(rsp)
					taskCounter -= 1
					migraSour = rsp[TASK_MIGRATION_SOURCE]
					peTaskContainer = self.spiNNakerContainer[migraSour[Y_AXIS_INDEX]][migraSour[X_AXIS_INDEX]][migraSour[Z_AXIS_INDEX]]
					if len(peTaskContainer) > 0:
						self.spiNNakerAddTask(peTaskContainer.pop(0))
				else:
					self.printInfo(rsp)
			# S4: weight Migration
			if dataReuseOutQpe > 0:
				self.weightMigrationBetweenBlocks(weightSize=weightSize, migraRounds=migraRounds)
		self.printInfo("MLA for 16x16 ALL FINISH")

	def convMlaExe_32_4_16(self, partsOfInActi, partsOfWeight, weightSize, idlePeIdList):
		self.customAssert((partsOfInActi >= 4) and (partsOfInActi <= 16), "Unsupport partsOfInActi: {}".format(partsOfInActi))
		self.customAssert(partsOfWeight >= 32, "Unsupport partsOfWeight: {}".format(partsOfWeight))
		# Determine how many data reuse outside QPE --> partsOfWeight >= 32 --> Move Weight
		dataReuseOutQpe = partsOfWeight // (NUM_PES_IN_QPE * NUM_OF_QPE_IN_DOUBLOCK)
		self.printInfo("dataReuseOutQpe: {}".format(dataReuseOutQpe))
		# Determine how many data reuse inside QPE --> partsOfInActi >= 4 and <= 16
		dataReuseInQpe = partsOfInActi // NUM_OF_DOUBLOCKS
		self.printInfo("dataReuseInQpe: {}".format(dataReuseInQpe))
		self.printInfo("Run MLA_EXE {} Loops".format(dataReuseInQpe * dataReuseOutQpe))
		migraRounds = NUM_OF_BLOCKS // dataReuseOutQpe
		self.printInfo("Migration Rounds: {}".format(migraRounds))
		# LOOP
		while dataReuseOutQpe > 0:
			dataReuseOutQpe -= 1
			# S1: Gnerate MLA tasks
			taskCounter = 0
			for yIndex in range(NUM_QPES_Y_AXIS):
				for xIndex in range(NUM_QPES_X_AXIS):
					for peIndex in range(NUM_PES_IN_QPE):
						taskDesti = [yIndex, xIndex, peIndex]
						if taskDesti not in idlePeIdList:
							blockIndex = self.getBlockIndexInDouBlock(taskDesti)
							if blockIndex == None:
								continue
							for dataReuseLoopIndex in range(0, dataReuseInQpe):
								pePairShift = (dataReuseLoopIndex + 1) % NUM_PES_IN_QPE
								mlaTask = copy.deepcopy(self.convMlaTask)
								mlaTask[TASK_DESTINATION] = taskDesti
								mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=pePairShift)
								mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
								self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
							taskCounter += dataReuseInQpe
			# S2: Place MLA tasks
			for qpeIndex in range(len(FIRST_DOUBLOCK_QPE_IDS)):
				for peIndex in range(NUM_PES_IN_QPE):
					for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
						qpeId = DOUBLOCK_QPE_ID_LIST[douBlockIndex][qpeIndex]
						peTaskContainer = self.spiNNakerContainer[qpeId[Y_AXIS_INDEX]][qpeId[X_AXIS_INDEX]][peIndex]
						if len(peTaskContainer) > 0:
							self.spiNNakerAddTask(peTaskContainer.pop(0))
			# S3: Running SpiNNaker2
			while taskCounter > 0:
				rsp = self.spiNNakerRunOneClock()
				rspName = rsp[TASK_NAME]
				if Task.TASK_NONE == rspName:
					pass
				elif Task.DATA_MIGRATION_32_FINISH == rspName:
					self.printInfo(rsp)
					taskCounter -= 1
					migraSour = rsp[TASK_MIGRATION_SOURCE]
					peTaskContainer = self.spiNNakerContainer[migraSour[Y_AXIS_INDEX]][migraSour[X_AXIS_INDEX]][migraSour[Z_AXIS_INDEX]]
					if len(peTaskContainer) > 0:
						self.spiNNakerAddTask(peTaskContainer.pop(0))
				else:
					self.printInfo(rsp)
			# S4: weight Migration
			if dataReuseOutQpe > 0:
				self.weightMigrationBetweenBlocks(weightSize=weightSize, migraRounds=migraRounds)
		self.printInfo("MLA ALL FINISH")

	def convMlaExe_128(self, partsOfInActi, partsOfWeight):
		self.customAssert(partsOfInActi * partsOfWeight == 128, "Unsupport tasks: {}-{}".format(partsOfInActi, partsOfWeight))
		# S1: Gnerate MLA tasks
		taskCounter = 0
		for yIndex in range(NUM_QPES_Y_AXIS):
			for xIndex in range(NUM_QPES_X_AXIS):
				for peIndex in range(NUM_PES_IN_QPE):
					taskDesti = [yIndex, xIndex, peIndex]
					blockIndex = self.getBlockIndexInDouBlock(taskDesti)
					if blockIndex == None:
						continue
					mlaTask = copy.deepcopy(self.convMlaTask)
					mlaTask[TASK_DESTINATION] = taskDesti
					mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=0)
					mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
					self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
					taskCounter += 1
		self.customAssert(taskCounter == 128, "taskCounter-[{}] should be 128".format(taskCounter))
		# S2: Place MLA tasks
		for qpeIndex in range(len(FIRST_DOUBLOCK_QPE_IDS)):
			for peIndex in range(NUM_PES_IN_QPE):
				for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
					qpeId = DOUBLOCK_QPE_ID_LIST[douBlockIndex][qpeIndex]
					peTaskContainer = self.spiNNakerContainer[qpeId[Y_AXIS_INDEX]][qpeId[X_AXIS_INDEX]][peIndex]
					self.spiNNakerAddTask(peTaskContainer.pop(0))
		# S3: Running SpiNNaker2
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_32_FINISH == rspName:
				self.printInfo(rsp)
				taskCounter -= 1
			else:
				self.printInfo(rsp)
		self.printInfo("MLA ALL FINISH")

	def convMlaExe_32_4(self, partsOfInActi, partsOfWeight, weightSize, idlePeIdList):
		self.customAssert(partsOfInActi >= 4, "Unsupport partsOfInActi: {}".format(partsOfInActi))
		self.customAssert(partsOfWeight >= 32, "Unsupport partsOfWeight: {}".format(partsOfWeight))
		# Determine how many data reuse outside Double-Block --> partsOfWeight >= 32 --> Move Weight
		dataReuseOutDouBlock = partsOfWeight // (NUM_PES_IN_QPE * NUM_OF_QPE_IN_DOUBLOCK)
		self.printInfo("dataReuseOutDouBlock: {}".format(dataReuseOutDouBlock))
		# Determine how many data reuse inside Double-Block --> partsOfInActi >= 4
		dataReuseInDouBlock = partsOfInActi // NUM_OF_DOUBLOCKS
		if dataReuseInDouBlock > NUM_PES_IN_QPE:
			dataReuseInQpe = NUM_PES_IN_QPE
			dataReuseInBlock = dataReuseInDouBlock // NUM_PES_IN_QPE
		else:
			dataReuseInQpe = dataReuseInDouBlock
			dataReuseInBlock = 1			
		self.printInfo("dataReuseInQpe: {}".format(dataReuseInQpe))
		self.printInfo("dataReuseInBlock: {}".format(dataReuseInBlock))
		self.printInfo("Run MLA_EXE {} Loops".format(dataReuseInDouBlock * dataReuseOutDouBlock))
		migraRounds = NUM_OF_BLOCKS // dataReuseOutDouBlock
		self.printInfo("Migration Rounds: {}".format(migraRounds))
		# LOOP out of double-block
		while dataReuseOutDouBlock > 0:
			dataReuseOutDouBlock -= 1
			# LOOP inside of between QPE of block
			dataReuseInBlockTemp = dataReuseInBlock
			while dataReuseInBlockTemp > 0:
				dataReuseInBlockTemp -= 1
				# S1: Gnerate MLA tasks
				taskCounter = 0
				for yIndex in range(NUM_QPES_Y_AXIS):
					for xIndex in range(NUM_QPES_X_AXIS):
						for peIndex in range(NUM_PES_IN_QPE):
							taskDesti = [yIndex, xIndex, peIndex]
							if taskDesti not in idlePeIdList:
								blockIndex = self.getBlockIndexInDouBlock(taskDesti)
								if blockIndex == None:
									continue
								for dataReuseLoopIndex in range(0, dataReuseInQpe):
									pePairShift = (dataReuseLoopIndex + 1) % NUM_PES_IN_QPE
									mlaTask = copy.deepcopy(self.convMlaTask)
									mlaTask[TASK_DESTINATION] = taskDesti
									mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=pePairShift)
									mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
									self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
								taskCounter += dataReuseInQpe
				# S2: Place MLA tasks
				for qpeIndex in range(len(FIRST_DOUBLOCK_QPE_IDS)):
					for peIndex in range(NUM_PES_IN_QPE):
						for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
							qpeId = DOUBLOCK_QPE_ID_LIST[douBlockIndex][qpeIndex]
							peTaskContainer = self.spiNNakerContainer[qpeId[Y_AXIS_INDEX]][qpeId[X_AXIS_INDEX]][peIndex]
							if len(peTaskContainer) > 0:
								self.spiNNakerAddTask(peTaskContainer.pop(0))
				# S3: Running SpiNNaker2
				while taskCounter > 0:
					rsp = self.spiNNakerRunOneClock()
					rspName = rsp[TASK_NAME]
					if Task.TASK_NONE == rspName:
						pass
					elif Task.DATA_MIGRATION_32_FINISH == rspName:
						self.printInfo(rsp)
						taskCounter -= 1
						migraSour = rsp[TASK_MIGRATION_SOURCE]
						peTaskContainer = self.spiNNakerContainer[migraSour[Y_AXIS_INDEX]][migraSour[X_AXIS_INDEX]][migraSour[Z_AXIS_INDEX]]
						if len(peTaskContainer) > 0:
							self.spiNNakerAddTask(peTaskContainer.pop(0))
					else:
						self.printInfo(rsp)
				# S4: weight Migration in Block
				if dataReuseInBlockTemp > 0:
					if 4 == dataReuseInBlock:
						self.weightMigrationInsideBlock(weightSize=weightSize)
					elif 2 == dataReuseInBlock:
						self.weightMigrationInsideBlock(weightSize=weightSize, halfFlag=True, shortSide=True)
						self.weightMigrationInsideBlock(weightSize=weightSize, halfFlag=True, shortSide=False)
					else:
						self.customAssert(False, "Unsupport")
			# weight Migration out of Block
			if dataReuseOutDouBlock > 0:
				self.weightMigrationBetweenBlocks(weightSize=weightSize, migraRounds=migraRounds)
		self.printInfo("MLA ALL FINISH")

	def convMlaExe_128_16(self, partsOfInActi, partsOfWeight, weightSize, idlePeIdList):
		self.customAssert(partsOfInActi == 128, "Unsupport partsOfInActi: {}".format(partsOfInActi))
		self.customAssert(partsOfWeight == 16, "Unsupport partsOfWeight: {}".format(partsOfWeight))
		# 
		dataReuseOutDouBlock = 1
		dataReuseInQpe = NUM_PES_IN_QPE
		dataReuseInBlock = 4
		self.printInfo("dataReuseOutDouBlock: {}".format(dataReuseOutDouBlock))
		self.printInfo("dataReuseInQpe: {}".format(dataReuseInQpe))
		self.printInfo("dataReuseInBlock: {}".format(dataReuseInBlock))
		# LOOP out of double-block
		while dataReuseOutDouBlock > 0:
			dataReuseOutDouBlock -= 1
			# LOOP inside of between QPE of block
			dataReuseInBlockTemp = dataReuseInBlock
			while dataReuseInBlockTemp > 0:
				dataReuseInBlockTemp -= 1
				# S1: Gnerate MLA tasks
				taskCounter = 0
				for yIndex in range(NUM_QPES_Y_AXIS):
					for xIndex in range(NUM_QPES_X_AXIS):
						for peIndex in range(NUM_PES_IN_QPE):
							taskDesti = [yIndex, xIndex, peIndex]
							if taskDesti not in idlePeIdList:
								blockIndex = self.getBlockIndexInDouBlock(taskDesti)
								if blockIndex == None:
									continue
								for dataReuseLoopIndex in range(0, dataReuseInQpe):
									pePairShift = (dataReuseLoopIndex + 1) % NUM_PES_IN_QPE
									mlaTask = copy.deepcopy(self.convMlaTask)
									mlaTask[TASK_DESTINATION] = taskDesti
									mlaTask[TASK_OPER_A_PEID] = self.targetToPairedPeIdGenerator(taskDesti, pePairShift=pePairShift)
									mlaTask[TASK_OUTACTI_DRAM_ADDR] = 0
									self.spiNNakerContainer[yIndex][xIndex][peIndex].append(mlaTask)
								taskCounter += dataReuseInQpe
				# S2: Place MLA tasks
				for qpeIndex in range(len(FIRST_DOUBLOCK_QPE_IDS)):
					for peIndex in range(NUM_PES_IN_QPE):
						for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
							qpeId = DOUBLOCK_QPE_ID_LIST[douBlockIndex][qpeIndex]
							peTaskContainer = self.spiNNakerContainer[qpeId[Y_AXIS_INDEX]][qpeId[X_AXIS_INDEX]][peIndex]
							if len(peTaskContainer) > 0:
								self.spiNNakerAddTask(peTaskContainer.pop(0))
				# S3: Running SpiNNaker2
				while taskCounter > 0:
					rsp = self.spiNNakerRunOneClock()
					rspName = rsp[TASK_NAME]
					if Task.TASK_NONE == rspName:
						pass
					elif Task.DATA_MIGRATION_32_FINISH == rspName:
						self.printInfo(rsp)
						taskCounter -= 1
						migraSour = rsp[TASK_MIGRATION_SOURCE]
						peTaskContainer = self.spiNNakerContainer[migraSour[Y_AXIS_INDEX]][migraSour[X_AXIS_INDEX]][migraSour[Z_AXIS_INDEX]]
						if len(peTaskContainer) > 0:
							self.spiNNakerAddTask(peTaskContainer.pop(0))
					else:
						self.printInfo(rsp)
				# S4: weight Migration in Block
				if dataReuseInBlockTemp > 0:
					if 4 == dataReuseInBlock:
						self.weightMigrationInsideBlock(weightSize=weightSize)
					elif 2 == dataReuseInBlock:
						self.weightMigrationInsideBlock(weightSize=weightSize, halfFlag=True, shortSide=True)
						self.weightMigrationInsideBlock(weightSize=weightSize, halfFlag=True, shortSide=False)
					else:
						self.customAssert(False, "Unsupport")
		self.printInfo("MLA ALL FINISH")

	def weightMigrationInsideBlock(self, weightSize, halfFlag=False, shortSide=True):
		if shortSide:
			migraFlag = [1, 1, 0, 0, 0, 0, 1, 1]
		else:
			migraFlag = [0, 0, 1, 1, 1, 1, 0, 0]
		# S1: Generate migration task
			# Create empty task container
		douBlockMigraTasks = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			douBlockTask = [[],[]]
			douBlockMigraTasks.append(douBlockTask)
			# Create migration task
		blockBasicQpeId = [[3,0], [1,0], [1,2], [1,4], [5,2], [5,0], [3,4], [5,4]]
		taskCounter = 0
		for blockIndex in range(NUM_OF_BLOCKS):
			basicQpeId = blockBasicQpeId[blockIndex]
			# Generate block QPE list
			blockQpeId = []
			blockQpeId.append(basicQpeId)
			nextQpeId = basicQpeId.copy()
			nextQpeId[Y_AXIS_INDEX] -= 1
			blockQpeId.append(nextQpeId)
			nextQpeId = nextQpeId.copy()
			nextQpeId[X_AXIS_INDEX] += 1
			blockQpeId.append(nextQpeId)
			nextQpeId = nextQpeId.copy()
			nextQpeId[Y_AXIS_INDEX] += 1
			blockQpeId.append(nextQpeId)
			# Create migration task inside block
			for qpeIndex in range(NUM_OF_QPE_IN_BLOCK):
				if halfFlag and (qpeIndex%2 != migraFlag[blockIndex]):
					continue
				nextQpeIndex = (qpeIndex+1) % NUM_OF_QPE_IN_BLOCK
				for peIndex in range(NUM_PES_IN_QPE):
					taskDest = blockQpeId[qpeIndex].copy()
					taskDest.append(peIndex)
					migraDest = blockQpeId[nextQpeIndex].copy()
					migraDest.append(peIndex)
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:taskDest,
						TASK_MIGRATION_SIZE:weightSize, TASK_MIGRATION_DESTINATION:migraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					douBlockIndex = blockIndex // NUM_OF_BLOCKS_IN_DOUBLOCKS
					blockIndexInDou = blockIndex % NUM_OF_BLOCKS_IN_DOUBLOCKS
					douBlockMigraTasks[douBlockIndex][blockIndexInDou].append(sramMigraTask)
					taskCounter += 1			
		# S2: Place migration tasks
		eachBlockTaskLen = len(douBlockMigraTasks[0][0])
		for taskIndex in range(eachBlockTaskLen):
			for blockIndex in range(NUM_OF_BLOCKS_IN_DOUBLOCKS):
				for douBlockIndex in range(NUM_OF_DOUBLOCKS):
					task = douBlockMigraTasks[douBlockIndex][blockIndex].pop(0)
					self.spiNNakerAddTask(task)		
		# S3: Run spiNNaker2 to migrate weight
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				taskCounter -= 1
			else:
				self.printInfo(rsp)
				pass
		self.printInfo("Finish Weight Migration inside block")

	def weightMigrationBetweenBlocks(self, weightSize, migraRounds=1):
		while migraRounds > 0:
			# S1: Generate migration task
			douBlockMigraTasks = []
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				douBlockTask = [[],[]]
				douBlockMigraTasks.append(douBlockTask)
			taskCounter = 0
			for yIndex in range(NUM_QPES_Y_AXIS):
				for xIndex in range(NUM_QPES_X_AXIS):
					for peIndex in range(NUM_PES_IN_QPE):
						taskDest = [yIndex, xIndex, peIndex]
						migraDest = taskDest.copy()
						douBlockIndex = self.getDouBlockIndex(taskDest)
						if douBlockIndex == None:
							continue
						blockIndex = self.getBlockIndexInDouBlock(taskDest)
						# 0-block
						if douBlockIndex == 0 and blockIndex == 0:
							migraDest[X_AXIS_INDEX] += 2
						# 1-block
						elif douBlockIndex == 0 and blockIndex == 1:
							migraDest[Y_AXIS_INDEX] -= 2
						# 2-block
						elif douBlockIndex == 1 and blockIndex == 0:
							migraDest[Y_AXIS_INDEX] += 2
						# 3-block
						elif douBlockIndex == 1 and blockIndex == 1:
							migraDest[X_AXIS_INDEX] += 2
						# 4-block
						elif douBlockIndex == 2 and blockIndex == 0:
							migraDest[Y_AXIS_INDEX] -= 2
						# 5-block
						elif douBlockIndex == 2 and blockIndex == 1:
							migraDest[X_AXIS_INDEX] -= 2
						# 6-block
						elif douBlockIndex == 3 and blockIndex == 0:
							migraDest[X_AXIS_INDEX] -= 2
						# 7-block
						elif douBlockIndex == 3 and blockIndex == 1:
							migraDest[Y_AXIS_INDEX] += 2
						else:
							self.customAssert(False, "Unknown douBlockIndex={}, blockIndex={}".format(douBlockIndex, blockIndex))
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:taskDest,
							TASK_MIGRATION_SIZE:weightSize, TASK_MIGRATION_DESTINATION:migraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						douBlockMigraTasks[douBlockIndex][blockIndex].append(sramMigraTask)
						taskCounter += 1
			# S2: Place migration tasks
			eachBlockTaskLen = len(douBlockMigraTasks[0][0])
			for taskIndex in range(eachBlockTaskLen):
				for blockIndex in range(NUM_OF_BLOCKS_IN_DOUBLOCKS):
					for douBlockIndex in range(NUM_OF_DOUBLOCKS):
						task = douBlockMigraTasks[douBlockIndex][blockIndex].pop(0)
						self.spiNNakerAddTask(task)
			# S3: Run spiNNaker2 to migrate weight
			while taskCounter > 0:
				rsp = self.spiNNakerRunOneClock()
				rspName = rsp[TASK_NAME]
				if Task.TASK_NONE == rspName:
					pass
				elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
					self.printInfo(rsp)
					taskCounter -= 1
				else:
					self.printInfo(rsp)
					pass
			# S3: Update loop time
			migraRounds -= 1
			self.printInfo("Finish One Migration Round")

	# =========================================================
	# Data Loading
	# =========================================================
	def loadWeights16(self):
		# Running SpiNNaker2 -> Load Weight
		# S1: Send 16 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convWeightMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convWeightMigraTasks[douBlockIndex].pop(0)
					self.spiNNakerAddTask(task)
					taskCounter += 1
			if tasksEmptyFlag:
				break
		# S2: Wait for 16 weight migration Tasks to be finish
		endQpeIdList = [[3,0], [3,5], [2,0], [2,5]]
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(rsp)
				# Add SRAM->SRAM Migration task
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				newMigraDest = migraDest.copy()
				if migraDest[X_AXIS_INDEX] < NUM_QPES_X_AXIS_HALF:
					newMigraDest[X_AXIS_INDEX] += 2
				else:
					newMigraDest[X_AXIS_INDEX] -= 2
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
				self.spiNNakerAddTask(sramMigraTask)
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in endQpeIdList:
					taskCounter -= 1
				# Along X Axis transmit data
				elif migraSour[Y_AXIS_INDEX] == migraDest[Y_AXIS_INDEX]:
					if migraSour[X_AXIS_INDEX] < migraDest[X_AXIS_INDEX]:
						if migraDest[X_AXIS_INDEX] <= NUM_QPES_X_AXIS_HALF:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] += 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
						else:
							newMigraDest = migraDest.copy()
							if migraDest[Y_AXIS_INDEX] <= NUM_QPES_Y_AXIS_HALF:
								newMigraDest[Y_AXIS_INDEX] += 2
							else:
								newMigraDest[Y_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					else:
						if migraDest[X_AXIS_INDEX] >= NUM_QPES_X_AXIS_HALF-1:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
						else:
							newMigraDest = migraDest.copy()
							if migraDest[Y_AXIS_INDEX] <= NUM_QPES_Y_AXIS_HALF:
								newMigraDest[Y_AXIS_INDEX] += 2
							else:
								newMigraDest[Y_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
				# Along Y Axis transmit data
				elif migraSour[X_AXIS_INDEX] == migraDest[X_AXIS_INDEX]:
					# Migrate data towards positive direction
					if migraSour[Y_AXIS_INDEX] < migraDest[Y_AXIS_INDEX]:
						if migraDest[Y_AXIS_INDEX] <= NUM_QPES_Y_AXIS_HALF:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] += 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
						else:
							newMigraDest = migraDest.copy()
							if migraDest[X_AXIS_INDEX] <= NUM_QPES_X_AXIS_HALF:
								newMigraDest[X_AXIS_INDEX] += 2
							else:
								newMigraDest[X_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# Migrate data towards negtive direction
					else:
						if migraDest[Y_AXIS_INDEX] >= NUM_QPES_Y_AXIS_HALF-1:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)							
						else:
							newMigraDest = migraDest.copy()
							if migraDest[X_AXIS_INDEX] <= NUM_QPES_X_AXIS_HALF:
								newMigraDest[X_AXIS_INDEX] += 2
							else:
								newMigraDest[X_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)							
				else:
					self.customAssert(False, "Not support data migration mode: {}->{}".format(migraSour, migraDest))
			else:
				self.printInfo(rsp)	
		self.printInfo("LOAD 16 weights FINISH")

	def loadWeights32(self):
		# Running SpiNNaker2 -> Load Weight
		# S1: Send 32 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convWeightMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convWeightMigraTasks[douBlockIndex].pop(0)
					taskMigraDestQpeId = task[TASK_MIGRATION_DESTINATION]
					if taskMigraDestQpeId[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[0,0], [0,5], [5,0], [5,5]]:
						task[TASK_ADDITION] = DramSramDataMigraType.WEIGHT2
					self.spiNNakerAddTask(task)
					taskCounter += 1
			if tasksEmptyFlag:
				break
		# S2: Wait for 16 weight migration Tasks to be finish
		endQpeIdList = [[[3,1], [1,2], [4,3], [2,4]],
						[[2,0], [0,3], [5,2], [3,5]]]
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(rsp)
				# Add SRAM->SRAM Migration task
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
				newMigraDest = migraDest.copy()
				if migraDestQpeId in DOUBLOCK_QPE_ID_LIST[0]:
					newMigraDest[X_AXIS_INDEX] += 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[1]:
					newMigraDest[Y_AXIS_INDEX] += 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[2]:
					newMigraDest[Y_AXIS_INDEX] -= 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[3]:
					newMigraDest[X_AXIS_INDEX] -= 2
				else:
					self.customAssert(False, "Unknown migraDestQpeId: {}".format(migraDestQpeId))
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_ADDITION:rsp[TASK_ADDITION]}
				self.spiNNakerAddTask(sramMigraTask)
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# 
				taskAddition = rsp[TASK_ADDITION]
				if (DramSramDataMigraType.WEIGHT == taskAddition) and (migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in endQpeIdList[0]):
					taskCounter -= 1
				elif (DramSramDataMigraType.WEIGHT2 == taskAddition) and (migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in endQpeIdList[1]):
					taskCounter -= 1
				# Along X Axis transmit data
				elif migraSour[Y_AXIS_INDEX] == migraDest[Y_AXIS_INDEX]:
					# Along positive X Axis direction
					if migraSour[X_AXIS_INDEX] < migraDest[X_AXIS_INDEX]:
						if migraDest[X_AXIS_INDEX] <= NUM_QPES_X_AXIS_HALF:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] += 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] += 2
							# if migraDest[Y_AXIS_INDEX] <= NUM_QPES_Y_AXIS_HALF:
							# 	newMigraDest[Y_AXIS_INDEX] += 2
							# else:
							# 	newMigraDest[Y_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)
					# Along negetive X Axis direction
					else:
						if migraDest[X_AXIS_INDEX] >= NUM_QPES_X_AXIS_HALF-1:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] -= 2
							# if migraDest[Y_AXIS_INDEX] <= NUM_QPES_Y_AXIS_HALF:
							# 	newMigraDest[Y_AXIS_INDEX] += 2
							# else:
							# 	newMigraDest[Y_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)
				# Along Y Axis transmit data
				elif migraSour[X_AXIS_INDEX] == migraDest[X_AXIS_INDEX]:
					# Migrate data towards positive direction
					if migraSour[Y_AXIS_INDEX] < migraDest[Y_AXIS_INDEX]:
						if migraDest[Y_AXIS_INDEX] <= NUM_QPES_Y_AXIS_HALF:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] += 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] -= 2
							# if migraDest[X_AXIS_INDEX] <= NUM_QPES_X_AXIS_HALF:
							# 	newMigraDest[X_AXIS_INDEX] += 2
							# else:
							# 	newMigraDest[X_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)
					# Migrate data towards negtive direction
					else:
						if migraDest[Y_AXIS_INDEX] >= NUM_QPES_Y_AXIS_HALF-1:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)							
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] += 2
							# if migraDest[X_AXIS_INDEX] <= NUM_QPES_X_AXIS_HALF:
							# 	newMigraDest[X_AXIS_INDEX] += 2
							# else:
							# 	newMigraDest[X_AXIS_INDEX] -= 2
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_ADDITION:rsp[TASK_ADDITION]}
							self.spiNNakerAddTask(sramMigraTask)							
				else:
					self.customAssert(False, "Not support data migration mode: {}->{}".format(migraSour, migraDest))
			else:
				self.printInfo(rsp)	
		self.printInfo("LOAD 32 weights FINISH")		

	def loadWeights64(self):
		# Running SpiNNaker2 -> Load Weight
		# S1: Send 64 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convWeightMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convWeightMigraTasks[douBlockIndex].pop(0)
					self.spiNNakerAddTask(task)
					taskCounter += 1
			if tasksEmptyFlag:
				break
		# S2: Wait for 16 weight migration Tasks to be finish
		# endQpeIdList = [[3,0], [3,1], [2,0], [2,1], 
		# 				[1,2], [1,3], [0,2], [0,3],
		# 				[3,4], [3,5], [2,4], [2,5],
		# 				[5,2], [5,3], [4,2], [4,3]]
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				# self.printInfo(rsp)
				# Add SRAM->SRAM Migration task
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
				newMigraDest = migraDest.copy()
				# First double block
				if migraDestQpeId in DOUBLOCK_QPE_ID_LIST[0]:
					newMigraDest[X_AXIS_INDEX] += 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[1]:
					newMigraDest[Y_AXIS_INDEX] += 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[2]:
					newMigraDest[Y_AXIS_INDEX] -= 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[3]:
					newMigraDest[X_AXIS_INDEX] -= 2
				else:
					self.customAssert(False, "Unknown migraDestQpeId: {}".format(migraDestQpeId))
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
				self.spiNNakerAddTask(sramMigraTask)
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				# self.printInfo(rsp)
				taskCounter -= 1
			else:
				# self.printInfo(rsp)
				pass
		self.printInfo("LOAD 64 weights FINISH")

	def loadWeights128(self):
		# Running SpiNNaker2 -> Load Weight
		# S1: Send 128 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convWeightMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convWeightMigraTasks[douBlockIndex].pop(0)
					self.spiNNakerAddTask(task)
					taskCounter += 1
			if tasksEmptyFlag:
				break
		# S2: Wait for 128 weight migration Tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				taskCounter -= 1
			else:
				self.printInfo(rsp)
				pass
		self.printInfo("LOAD 128 weights FINISH")

	def loadInActisFromDram16(self):
		# S5: Running SpiNNaker2 -> Load InActi
		taskCounter = 0
		# SS5-1: Send 16 inActi migration taskss
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					self.spiNNakerAddTask(task)
					taskCounter += 1
			if tasksEmptyFlag:
				break
		# SS5-2: Wait for 16 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(rsp)
				# Add SRAM->SRAM Migration task
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# Determine migration order
				if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[5,0], [0,5]]:
					migraOrder = [Y_AXIS_INDEX, X_AXIS_INDEX]
				elif migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[0,0], [5,5]]:
					migraOrder = [X_AXIS_INDEX, Y_AXIS_INDEX]
				else:
					self.customAssert(False, "Unknown migraDest: {}".format(migraDest))
				# Copy along X/Y axis
				newMigraDest = migraDest.copy()
				xyIndex = migraOrder[0]
				if migraDest[xyIndex] < NUM_QPES_XY_HALF:
					newMigraDest[xyIndex] += 1
				else:
					newMigraDest[xyIndex] -= 1
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
				self.spiNNakerAddTask(sramMigraTask)
				# Copy along X/Y axis
				newMigraDest = migraDest.copy()
				xyIndex = migraOrder[1]
				if migraDest[xyIndex] < NUM_QPES_XY_HALF:
					newMigraDest[xyIndex] += 1
				else:
					newMigraDest[xyIndex] -= 1
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
				self.spiNNakerAddTask(sramMigraTask)
				taskCounter += 1				
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
				# Second Double-Block
				if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[0,2], [1,2]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[X_AXIS_INDEX] -= 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				# Third Double-Block
				elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[4,3],[5,3]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[X_AXIS_INDEX] += 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				# First Double-Block
				elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[3,0],[3,1]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[Y_AXIS_INDEX] += 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				# Fourth Double-Block
				elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[2,4],[2,5]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[Y_AXIS_INDEX] -= 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				else:
					self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				self.printInfo(rsp)
		self.printInfo("LOAD 16 inActi FINISH")

	def loadInActisFromDram16Fit(self):
		idlePesIndex = [[],[],[],[]]
		# S1: Send 64 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						idlePesIndex[douBlockIndex].append(migraDest[Z_AXIS_INDEX])						
			if tasksEmptyFlag:
				break
		# SS5-2: Wait for 16 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(rsp)
				# Add SRAM->SRAM Migration task
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# Determine migration order
				if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[5,0], [0,5]]:
					migraOrder = [Y_AXIS_INDEX, X_AXIS_INDEX]
				elif migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[0,0], [5,5]]:
					migraOrder = [X_AXIS_INDEX, Y_AXIS_INDEX]
				else:
					self.customAssert(False, "Unknown migraDest: {}".format(migraDest))
				# Copy along X/Y axis
				newMigraDest = migraDest.copy()
				xyIndex = migraOrder[0]
				if migraDest[xyIndex] < NUM_QPES_XY_HALF:
					newMigraDest[xyIndex] += 1
				else:
					newMigraDest[xyIndex] -= 1
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
				self.spiNNakerAddTask(sramMigraTask)
				# Copy along X/Y axis
				newMigraDest = migraDest.copy()
				xyIndex = migraOrder[1]
				if migraDest[xyIndex] < NUM_QPES_XY_HALF:
					newMigraDest[xyIndex] += 1
				else:
					newMigraDest[xyIndex] -= 1
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
				self.spiNNakerAddTask(sramMigraTask)
				taskCounter += 1				
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
				# Second Double-Block
				if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[0,2], [1,2]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[X_AXIS_INDEX] -= 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				# Third Double-Block
				elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[4,3],[5,3]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[X_AXIS_INDEX] += 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				# First Double-Block
				elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[3,0],[3,1]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[Y_AXIS_INDEX] += 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				# Fourth Double-Block
				elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
					if migraDestQpeId in [[2,4],[2,5]]:
						taskCounter -= 1
					else:
						newMigraDest = migraDest.copy()
						newMigraDest[Y_AXIS_INDEX] -= 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
				else:
					self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				self.printInfo(rsp)
		self.printInfo("LOAD 16 inActi FINISH")
		return idlePesIndex

	def loadInActisFromDram64Fit(self):
		idlePes = [[],[],[],[]]
		# S1: Send 64 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						idlePes[douBlockIndex].append(migraDest)						
			if tasksEmptyFlag:
				break
		# S2: Wait for 64 inActi migration Tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(rsp)
				# Add SRAM->SRAM Migration task
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
				newMigraDest = migraDest.copy()
				# First double block
				if migraDestQpeId in DOUBLOCK_QPE_ID_LIST[0]:
					newMigraDest[Y_AXIS_INDEX] += 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[1]:
					newMigraDest[X_AXIS_INDEX] -= 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[2]:
					newMigraDest[X_AXIS_INDEX] += 2
				elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[3]:
					newMigraDest[Y_AXIS_INDEX] -= 2
				else:
					self.customAssert(False, "Unknown migraDestQpeId: {}".format(migraDestQpeId))
				sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
					TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
					TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
				self.spiNNakerAddTask(sramMigraTask)
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				taskCounter -= 1
			else:
				self.printInfo(rsp)
				pass
		self.printInfo("LOAD 64 inActi FINISH")
		self.printInfo("idlePes: {}".format(idlePes))
		return idlePes

	def loadInActisFromSram64Fit(self):
		idlePes = [[],[],[],[]]
		# S1: Send 64 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						idlePes[douBlockIndex].append(migraDest)						
			if tasksEmptyFlag:
				break
		# S2: Wait for 64 inActi migration Tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# --> Act like DRAM_SRAM_DATA_MIGRA_FINISH
				if migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX] in STORAGE_QPE_IDS:
					# Add SRAM->SRAM Migration task
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					newMigraDest = migraDest.copy()
					# First double block
					if migraDestQpeId in DOUBLOCK_QPE_ID_LIST[0]:
						newMigraDest[Y_AXIS_INDEX] -= 2
					elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[1]:
						newMigraDest[X_AXIS_INDEX] += 2
					elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[2]:
						newMigraDest[X_AXIS_INDEX] -= 2
					elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[3]:
						newMigraDest[Y_AXIS_INDEX] += 2
					else:
						self.customAssert(False, "Unknown migraDestQpeId: {}".format(migraDestQpeId))
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)					
				# --> Act like DATA_MIGRATION_ALL_FINISH
				else:
					taskCounter -= 1
			else:
				self.printInfo(rsp)
				pass
		self.printInfo("LOAD 64 inActi FINISH")
		self.printInfo("idlePes: {}".format(idlePes))
		return idlePes

	def loadInActisFromDram128Fit(self):
		idlePes = [[],[],[],[]]
		# S1: Send 64 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						idlePes[douBlockIndex].append(migraDest)						
			if tasksEmptyFlag:
				break
		# S2: Wait for 64 inActi migration Tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			# elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
			elif Task.DRAM_SRAM_DATA_MIGRA_FINISH == rspName:
				self.printInfo(rsp)
				taskCounter -= 1
			else:
				self.printInfo(rsp)
				pass
		self.printInfo("LOAD 128 inActi FINISH")
		self.printInfo("idlePes: {}".format(idlePes))
		return idlePes

	def loadInActisFromSram128Fit(self):
		idlePes = [[],[],[],[]]
		# S1: Send 64 weight migration Tasks
		taskCounter = 0
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						idlePes[douBlockIndex].append(migraDest)						
			if tasksEmptyFlag:
				break
		# S2: Wait for 64 inActi migration Tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				taskCounter -= 1
			else:
				self.printInfo(rsp)
				pass
		self.printInfo("LOAD 128 inActi FINISH")
		self.printInfo("idlePes: {}".format(idlePes))
		return idlePes

	def loadInActisFromSram8(self):
		# S5: Running SpiNNaker2 -> Load InActi
		taskCounter = 0
		# SS5-1: Send 8-(2) inActi migration taskss
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					self.spiNNakerAddTask(task)
					taskCounter += 1
			if tasksEmptyFlag:
				break
		# SS5-2: Wait for 16 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				# self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# --> Act like DRAM_SRAM_DATA_MIGRA_FINISH
				if migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX] in STORAGE_QPE_IDS:
					# Determine migration order
					if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[4,3], [1,2]]:
						migraOrder = [Y_AXIS_INDEX, X_AXIS_INDEX]
					elif migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[3,1], [2,4]]:
						migraOrder = [X_AXIS_INDEX, Y_AXIS_INDEX]
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[0]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] += 1
					else:
						newMigraDest[xyIndex] -= 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[1]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] -= 1
					else:
						newMigraDest[xyIndex] += 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy to self QPE but different PE
					newMigraDest = migraDest.copy()
					newMigraDest[Z_AXIS_INDEX] += 2
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					taskCounter += 1
				# --> Act like DATA_MIGRATION_ALL_FINISH
				else:
					migraSourQpeId = migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX]
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					# Second Double-Block
					if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,5], [1,5]]: 
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)							
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target
								newMigraDest = migraDest.copy()
								newMigraDest[X_AXIS_INDEX] += 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					# Third Double-Block
					elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[4,0],[5,0]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)								
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[X_AXIS_INDEX] -= 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)								
					# First Double-Block
					elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,0],[0,1]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[Y_AXIS_INDEX] -= 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					# Fourth Double-Block
					elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[5,4],[5,5]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[Y_AXIS_INDEX] += 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				# self.printInfo(rsp)
				pass
		self.printInfo("LOAD 8 inActi FINISH")

	def loadInActisFromSram16(self):
		# S5: Running SpiNNaker2 -> Load InActi
		taskCounter = 0
		# SS5-1: Send 16 inActi migration taskss
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					self.spiNNakerAddTask(task)
					taskCounter += 1
			if tasksEmptyFlag:
				break
		# SS5-2: Wait for 16 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# --> Act like DRAM_SRAM_DATA_MIGRA_FINISH
				if migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX] in STORAGE_QPE_IDS:
					# Determine migration order
					if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[4,3], [1,2]]:
						migraOrder = [Y_AXIS_INDEX, X_AXIS_INDEX]
					elif migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[3,1], [2,4]]:
						migraOrder = [X_AXIS_INDEX, Y_AXIS_INDEX]
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[0]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] += 1
					else:
						newMigraDest[xyIndex] -= 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[1]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] -= 1
					else:
						newMigraDest[xyIndex] += 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					taskCounter += 1
				# --> Act like DATA_MIGRATION_ALL_FINISH
				else:
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					# Second Double-Block
					if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,5], [1,5]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] += 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# Third Double-Block
					elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[4,0],[5,0]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] -= 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# First Double-Block
					elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,0],[0,1]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] -= 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# Fourth Double-Block
					elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[5,4],[5,5]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] += 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				self.printInfo(rsp)
		self.printInfo("LOAD 16 inActi from sram FINISH")

	def loadInActisFromSram32Fit(self):
		idlePes = [[],[],[],[]]
		# S5: Running SpiNNaker2 -> Load InActi
		taskCounter = 0
		# SS5-1: Send 32 inActi migration taskss
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						idlePes[douBlockIndex].append(migraDest)
			if tasksEmptyFlag:
				break
		self.printInfo("taskCounter: {}".format(taskCounter))
		# SS5-2: Wait for 16 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# --> Act like DRAM_SRAM_DATA_MIGRA_FINISH
				if migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX] in STORAGE_QPE_IDS:
					# Add SRAM->SRAM Migration task
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					newMigraDest = migraDest.copy()
					# First double block
					if migraDestQpeId in DOUBLOCK_QPE_ID_LIST[0]:
						newMigraDest[Y_AXIS_INDEX] -= 1
					elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[1]:
						newMigraDest[X_AXIS_INDEX] += 1
					elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[2]:
						newMigraDest[X_AXIS_INDEX] -= 1
					elif migraDestQpeId in DOUBLOCK_QPE_ID_LIST[3]:
						newMigraDest[Y_AXIS_INDEX] += 1
					else:
						self.customAssert(False, "Unknown migraDestQpeId: {}".format(migraDestQpeId))
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
				# --> Act like DATA_MIGRATION_ALL_FINISH
				else:
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					# Second Double-Block
					if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,5], [1,5]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] += 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# Third Double-Block
					elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[4,0],[5,0]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] -= 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# First Double-Block
					elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,0],[0,1]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] -= 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# Fourth Double-Block
					elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[5,4],[5,5]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] += 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				self.printInfo(rsp)
		self.printInfo("LOAD 32 inActi from sram FINISH")
		return idlePes

	def loadInActisFromSram16Fit(self):
		idlePeIndex = [[],[],[],[]]
		# S5: Running SpiNNaker2 -> Load InActi
		taskCounter = 0
		# SS5-1: Send 16 inActi migration taskss
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						douBlockIndex = self.whichDouBlock(migraDest)
						idlePeIndex[douBlockIndex].append(migraDest[Z_AXIS_INDEX])
			if tasksEmptyFlag:
				break
		self.printInfo("taskCounter: {}".format(taskCounter))
		# SS5-2: Wait for 16 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# --> Act like DRAM_SRAM_DATA_MIGRA_FINISH
				if migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX] in STORAGE_QPE_IDS:
					# Determine migration order
					if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[4,3], [1,2]]:
						migraOrder = [Y_AXIS_INDEX, X_AXIS_INDEX]
					elif migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[3,1], [2,4]]:
						migraOrder = [X_AXIS_INDEX, Y_AXIS_INDEX]
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[0]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] += 1
					else:
						newMigraDest[xyIndex] -= 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[1]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] -= 1
					else:
						newMigraDest[xyIndex] += 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					taskCounter += 1
				# --> Act like DATA_MIGRATION_ALL_FINISH
				else:
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					# Second Double-Block
					if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,5], [1,5]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] += 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# Third Double-Block
					elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[4,0],[5,0]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[X_AXIS_INDEX] -= 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# First Double-Block
					elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,0],[0,1]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] -= 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					# Fourth Double-Block
					elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[5,4],[5,5]]:
							taskCounter -= 1
						else:
							newMigraDest = migraDest.copy()
							newMigraDest[Y_AXIS_INDEX] += 1
							sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
								TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
								TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
							self.spiNNakerAddTask(sramMigraTask)
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				self.printInfo(rsp)
		self.printInfo("LOAD 16 inActi from sram FINISH")
		return idlePeIndex

	def loadInActisFromSram8Fit(self):
		idlePeIndex = [[],[],[],[]]
		# S5: Running SpiNNaker2 -> Load InActi
		taskCounter = 0
		# SS5-1: Send 8-(2) inActi migration taskss
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						douBlockIndex = self.whichDouBlock(migraDest)
						idlePeIndex[douBlockIndex].append(migraDest[Z_AXIS_INDEX])
			if tasksEmptyFlag:
				break
		# SS5-2: Wait for 16 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# --> Act like DRAM_SRAM_DATA_MIGRA_FINISH
				if migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX] in STORAGE_QPE_IDS:
					# Determine migration order
					if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[4,3], [1,2]]:
						migraOrder = [Y_AXIS_INDEX, X_AXIS_INDEX]
					elif migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[3,1], [2,4]]:
						migraOrder = [X_AXIS_INDEX, Y_AXIS_INDEX]
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[0]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] += 1
					else:
						newMigraDest[xyIndex] -= 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[1]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] -= 1
					else:
						newMigraDest[xyIndex] += 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy to self QPE but different PE
					newMigraDest = migraDest.copy()
					newMigraDest[Z_AXIS_INDEX] += 2
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					taskCounter += 1
				# --> Act like DATA_MIGRATION_ALL_FINISH
				else:
					migraSourQpeId = migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX]
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					# Second Double-Block
					if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,5], [1,5]]: 
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)							
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target
								newMigraDest = migraDest.copy()
								newMigraDest[X_AXIS_INDEX] += 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					# Third Double-Block
					elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[4,0],[5,0]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)								
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[X_AXIS_INDEX] -= 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)								
					# First Double-Block
					elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,0],[0,1]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[Y_AXIS_INDEX] -= 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					# Fourth Double-Block
					elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[5,4],[5,5]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[Y_AXIS_INDEX] += 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				# self.printInfo(rsp)
				pass
		self.printInfo("LOAD 8 inActi FINISH")
		return idlePeIndex

	def loadInActisFromSram4Fit(self):
		idlePeIndex = [[],[],[],[]]
		# S5: Running SpiNNaker2 -> Load InActi
		taskCounter = 0
		# SS5-1: Send 4-(1) inActi migration taskss
		while True:
			tasksEmptyFlag = True
			for douBlockIndex in range(NUM_OF_DOUBLOCKS):
				if len(self.convInActiMigraTasks[douBlockIndex]) > 0:
					tasksEmptyFlag = False
					task = self.convInActiMigraTasks[douBlockIndex].pop(0)
					if DramSramDataMigraType.EMPTY != task[TASK_ADDITION]:
						self.spiNNakerAddTask(task)
						taskCounter += 1
					else:
						migraDest = task[TASK_MIGRATION_DESTINATION]
						douBlockIndex = self.whichDouBlock(migraDest)
						idlePeIndex[douBlockIndex].append(migraDest[Z_AXIS_INDEX])
			if tasksEmptyFlag:
				break
		# SS5-2: Wait for 8 inActi migration tasks to be finish
		while taskCounter > 0:
			rsp = self.spiNNakerRunOneClock()
			rspName = rsp[TASK_NAME]
			if Task.TASK_NONE == rspName:
				pass
			elif Task.DATA_MIGRATION_ALL_FINISH == rspName:
				self.printInfo(rsp)
				migraSour = rsp[TASK_MIGRATION_SOURCE]
				migraDest = rsp[TASK_MIGRATION_DESTINATION]
				# --> Act like DRAM_SRAM_DATA_MIGRA_FINISH
				if (migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX] in STORAGE_QPE_IDS) or \
					(migraDest in [[3,1,1], [1,2,1], [2,4,1], [4,3,1]]):
					# Determine migration order
					if migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[4,3], [1,2]]:
						migraOrder = [Y_AXIS_INDEX, X_AXIS_INDEX]
					elif migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX] in [[3,1], [2,4]]:
						migraOrder = [X_AXIS_INDEX, Y_AXIS_INDEX]
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))
					# Copy to self QPE but different PE
					if migraDest not in [[3,1,1], [1,2,1], [2,4,1], [4,3,1]]:
						newMigraDest = migraDest.copy()
						newMigraDest[Z_AXIS_INDEX] += 1
						sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
							TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
							TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
						self.spiNNakerAddTask(sramMigraTask)
						taskCounter += 1
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[0]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] += 1
					else:
						newMigraDest[xyIndex] -= 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy along X/Y axis
					newMigraDest = migraDest.copy()
					xyIndex = migraOrder[1]
					if migraDest[xyIndex] >= NUM_QPES_XY_HALF:
						newMigraDest[xyIndex] -= 1
					else:
						newMigraDest[xyIndex] += 1
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					# Copy to self QPE but different PE
					newMigraDest = migraDest.copy()
					newMigraDest[Z_AXIS_INDEX] += 2
					sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
						TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
						TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
					self.spiNNakerAddTask(sramMigraTask)
					taskCounter += 1
				# --> Act like DATA_MIGRATION_ALL_FINISH
				else:
					migraSourQpeId = migraSour[Y_AXIS_INDEX:Z_AXIS_INDEX]
					migraDestQpeId = migraDest[Y_AXIS_INDEX:Z_AXIS_INDEX]
					# Second Double-Block
					if migraDestQpeId in SECOND_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,5], [1,5]]: 
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)							
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target
								newMigraDest = migraDest.copy()
								newMigraDest[X_AXIS_INDEX] += 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					# Third Double-Block
					elif migraDestQpeId in THIRD_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[4,0],[5,0]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)								
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[X_AXIS_INDEX] -= 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)								
					# First Double-Block
					elif migraDestQpeId in FIRST_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[0,0],[0,1]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[Y_AXIS_INDEX] -= 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					# Fourth Double-Block
					elif migraDestQpeId in FOURTH_DOUBLOCK_QPE_IDS:
						if migraDestQpeId in [[5,4],[5,5]]:
							if migraSourQpeId == migraDestQpeId:
								taskCounter -= 1
							else:
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
						else:
							if migraSourQpeId != migraDestQpeId:
								# --> To next target 
								newMigraDest = migraDest.copy()
								newMigraDest[Y_AXIS_INDEX] += 1
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
								# --> Copy to self
								newMigraDest = migraDest.copy()
								newMigraDest[Z_AXIS_INDEX] += 2
								sramMigraTask = {TASK_NAME:Task.DATA_MIGRATION, TASK_DESTINATION:migraDest,
									TASK_MIGRATION_SIZE:rsp[TASK_MIGRATION_SIZE], TASK_MIGRATION_DESTINATION:newMigraDest, 
									TASK_SRAM_ADDR:0, TASK_MIGRA_SRAM_ADDR:0}
								self.spiNNakerAddTask(sramMigraTask)
					else:
						self.customAssert(False, "Unknown migraDest: {}".format(migraDest))					
			else:
				# self.printInfo(rsp)
				pass
		self.printInfo("LOAD 4 inActi FINISH")
		return idlePeIndex

	# =========================================================
	# Migration and MLA_EXE Task Generation
	# =========================================================
	def poolInActiMigraTasksGenerator128Fit(self, layerSplitInfo):
		# S0: Number of inActi migration task (actual and expand)
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		inActiBlockSize = inActiSplitInfo[0][0] * inActiSplitInfo[1][0] * inActiSplitInfo[2][0]
		partsOfWidth = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		partsOfHeight = self.getTotalPartsFromSplitInfo(inActiSplitInfo[1])
		partsOfChannel = self.getTotalPartsFromSplitInfo(inActiSplitInfo[2])
		channelInDoublock = math.ceil(partsOfChannel / NUM_OF_DOUBLOCKS)
			# Get actual number of inActi migration task
		numOfTasksInDoublock = [0] * NUM_OF_DOUBLOCKS
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if partsOfChannel >= channelInDoublock:
				numOfTasksInDoublock[douBlockIndex] = channelInDoublock * partsOfWidth * partsOfHeight
				partsOfChannel = partsOfChannel - channelInDoublock
			else:
				numOfTasksInDoublock[douBlockIndex] = partsOfChannel * partsOfWidth * partsOfHeight
				partsOfChannel = partsOfChannel - partsOfChannel
			# Get expand number of inActi migration task
		numOfTasksInDouBlockExpand = self.roundToPowerOf2(numOfTasksInDoublock[0])
		self.printInfo("numOfTasksInDoublock: {}".format(numOfTasksInDoublock))
		self.printInfo("numOfTasksInDouBlockExpand: {}".format(numOfTasksInDouBlockExpand))
		# S1: double-block target destination
		if self.inActiFromDram:
			self.customAssert(False, "not support")
		else:
			destQpeIdList = [[[5,0], [5,1], [4,0], [4,1], [5,2], [4,2], [3,0], [3,1]],
							 [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [0,2], [1,2]],
							 [[5,5], [5,4], [4,5], [4,4], [3,5], [3,4], [5,3], [4,3]],
							 [[0,5], [0,4], [1,5], [1,4], [0,3], [1,3], [2,5], [2,4]]]
		# S1: Conv inActi migration task
		poolInActiMigraTasks = []
		inActiCounter = 0
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task migration source
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			sramMigrationTaskDest = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			if self.inActiFromDram:
				inActiMigraTaskSour = dramMigrationTaskDest
			else:
				inActiMigraTaskSour = sramMigrationTaskDest
			# SS1: Conv inActi migration task
			blockConvInActiMigraTasks = []
			actualNumOfInActiTask = numOfTasksInDoublock[douBlockIndex]
			for inActiBlockIndex in range(numOfTasksInDouBlockExpand):
				# Get migration destination
				qpeIndex = inActiBlockIndex // NUM_PES_IN_QPE
				peIndex = inActiBlockIndex % NUM_PES_IN_QPE
				taskDest = destQpeIdList[douBlockIndex][qpeIndex].copy()
				taskDest.append(peIndex)
				if inActiBlockIndex < actualNumOfInActiTask:
					# Get migration source
					if self.inActiFromDram:
						taskMigraSour = inActiMigraTaskSour
						taskName = Task.DRAM_SRAM_DATA_MIGRATION
					else:
						taskMigraSour = inActiMigraTaskSour.copy()
						taskMigraSour.append(peIndex)
						taskName = Task.DATA_MIGRATION
					# Get migration size
					inActiCounter += 1
					# DRAM_RD_MIGRATION (inActi)
					inActiRamToSramMigraTask = {TASK_NAME:taskName, 
						TASK_DESTINATION:taskMigraSour, TASK_MIGRATION_SIZE:inActiBlockSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
				else:
					inActiRamToSramMigraTask = {TASK_ADDITION:DramSramDataMigraType.EMPTY, 
						TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
			poolInActiMigraTasks.append(blockConvInActiMigraTasks)
		return poolInActiMigraTasks

	def convBlockTasksGenerator(self, layerSplitInfo, layerTypeParameter, inActiDimBlocks, 
		weightDimBlocks, outActiDimBlocks):
		# S1: Conv inActi migration task
		partsOfInActiOrigin = len(inActiDimBlocks)
		partsOfInActi =  self.roundToPowerOf2(partsOfInActiOrigin)
		if partsOfInActi != partsOfInActiOrigin:
			self.printInfo("Warning: partsOfInActi-{} -> {}".format(partsOfInActiOrigin, partsOfInActi))
		if 16 == partsOfInActi:
			convInActiMigraTasks = self.convInActiMigraTasksGenerator16Fit(layerSplitInfo, inActiDimBlocks)
		elif 8 == partsOfInActi:
			convInActiMigraTasks = self.convInActiMigraTasksGenerator16Fit(layerSplitInfo, inActiDimBlocks)
		elif 64 == partsOfInActi:
			convInActiMigraTasks = self.convInActiMigraTasksGenerator64Fit(layerSplitInfo, inActiDimBlocks)
		elif 4 == partsOfInActi:
			convInActiMigraTasks = self.convInActiMigraTasksGenerator16Fit(layerSplitInfo, inActiDimBlocks)
		elif 2 == partsOfInActi or 1 == partsOfInActi:
			convInActiMigraTasks = self.convInActiMigraTasksGeneratorLeastThan4Fit(layerSplitInfo, inActiDimBlocks)
		elif 32 == partsOfInActi:
			convInActiMigraTasks = self.convInActiMigraTasksGenerator32Fit(layerSplitInfo, inActiDimBlocks)
		elif 128 == partsOfInActi:
			convInActiMigraTasks = self.convInActiMigraTasksGenerator128Fit(layerSplitInfo, inActiDimBlocks)
		else:
			self.customAssert(False, "Not support parts of inActi: {}".format(partsOfInActi))
			# convInActiMigraTasks = None
			# pass
		# S2: Conv weight migration task
		partsOfWeight = len(weightDimBlocks)
		if 16 == partsOfWeight:
			convWeightMigraTasks = self.convWeightMigraTasksGenerator16(layerSplitInfo, weightDimBlocks)
		elif 64 == partsOfWeight:
			convWeightMigraTasks = self.convWeightMigraTasksGenerator64(layerSplitInfo, weightDimBlocks)
		elif 32 == partsOfWeight:
			convWeightMigraTasks = self.convWeightMigraTasksGenerator64(layerSplitInfo, weightDimBlocks)
		elif 128 == partsOfWeight:
			convWeightMigraTasks = self.convWeightMigraTasksGenerator128(layerSplitInfo, weightDimBlocks)
		else:
			self.customAssert(False, "Not support parts of weight: {}".format(partsOfWeight))
			# convWeightMigraTasks = None
		# S3: Conv execution task
		layerType, layerParameter = layerTypeParameter
		if len(layerParameter) == 2:
			(_, _, convStride, _), poolStride = layerParameter
		else:
			(_, _, convStride, _), poolStride, _ = layerParameter
		if isinstance(poolStride, tuple):
			poolDim, poolStride = poolStride
			if poolDim[0] != poolStride:
				poolStride = 1
		largestInActiBlockAlignSize = self.align16(inActiDimBlocks[0][0]) * inActiDimBlocks[0][1] * inActiDimBlocks[0][2]
		largestWeightBlockAlignSize = self.align16(weightDimBlocks[0][0] * weightDimBlocks[0][1] * weightDimBlocks[0][2] * \
			self.alignMlaRow(weightDimBlocks[0][3]))
		largestOutActiBlockAlignSize = self.align4(outActiDimBlocks[0][0]) * outActiDimBlocks[0][1] * outActiDimBlocks[0][2]
		inActiBaseAddr = SRAM_DATA_BEGIN_ADDR
		weightBaseAddr = inActiBaseAddr + largestInActiBlockAlignSize
		outActiBaseAddr = weightBaseAddr + largestWeightBlockAlignSize
		self.customAssert(outActiBaseAddr+largestOutActiBlockAlignSize < SRAM_END_ADDR, "sram overflow")
		mlaTask = self.convMlaTaskGenerator(layerSplitInfo, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, poolStride)
		return convInActiMigraTasks, convWeightMigraTasks, mlaTask

	def convInActiMigraTasksGenerator16(self, layerSplitInfo, inActiAlignBlocks):
		'''
		Fit for parts of inActi = 16, 8
		'''
		# S0: double-block target destination
		if self.inActiFromDram:
			destQpeIdList = [[0,0], [0,5], [5,0], [5,5]]
		else:
			destQpeIdList = [[3,1], [1,2], [4,3], [2,4]]
		# S1: Conv inActi migration task
		numOfInActiBlocksInDouBlock = len(inActiAlignBlocks) // NUM_OF_DOUBLOCKS # 4
		convInActiMigraTasks = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task destination
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			sramMigrationTaskDest = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			if self.inActiFromDram:
				inActiMigraTaskSour = dramMigrationTaskDest
			else:
				inActiMigraTaskSour = sramMigrationTaskDest
			# SS1: Conv inActi migration task
			blockConvInActiMigraTasks = []
			for inActiBlockIndexAlWidth in range(numOfInActiBlocksInDouBlock):
				inActiBlockIndex = numOfInActiBlocksInDouBlock*douBlockIndex + inActiBlockIndexAlWidth
				inActiBlockSize = len(inActiAlignBlocks[inActiBlockIndex])
				taskDest = destQpeIdList[douBlockIndex].copy()
				taskDest.append(inActiBlockIndexAlWidth)
				if self.inActiFromDram:
					taskMigraSour = inActiMigraTaskSour
					taskName = Task.DRAM_SRAM_DATA_MIGRATION
				else:
					taskMigraSour = inActiMigraTaskSour.copy()
					taskMigraSour.append(inActiBlockIndexAlWidth)
					taskName = Task.DATA_MIGRATION
				# DRAM_RD_MIGRATION (inActi)
				inActiRamToSramMigraTask = {TASK_NAME:taskName, 
					TASK_DESTINATION:taskMigraSour, TASK_MIGRATION_SIZE:inActiBlockSize, 
					TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION:taskDest}
				# # Dimension of inActi
				# inActiDim = self.getInActiDim(layerSplitInfo, inActiBlockIndex)
				# blockConvInActiMigraTasks.append((inActiDim, inActiRamToSramMigraTask))
				blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
			convInActiMigraTasks.append(blockConvInActiMigraTasks)
		return convInActiMigraTasks

	def convInActiMigraTasksGeneratorLeastThan4Fit(self, layerSplitInfo, inActiDimBlocks):
		'''
		Fit for parts of inActi: <=16, <=8
		'''
		# S0: Number of inActi migration task (actual and expand)
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		partsOfWidth = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		partsOfHeight = self.getTotalPartsFromSplitInfo(inActiSplitInfo[1])
		heightInDoublock = math.ceil(partsOfHeight / NUM_OF_DOUBLOCKS)
			# Get actual number of inActi migration task
		numOfTasksInDoublock = [0] * NUM_OF_DOUBLOCKS
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if partsOfHeight >= heightInDoublock:
				numOfTasksInDoublock[douBlockIndex] = heightInDoublock * partsOfWidth
				partsOfHeight = partsOfHeight - heightInDoublock
			else:
				numOfTasksInDoublock[douBlockIndex] = partsOfHeight * partsOfWidth
				partsOfHeight = partsOfHeight - partsOfHeight
			# Get expand number of inActi migration task
		numOfTasksInDouBlockExpand = self.roundToPowerOf2(numOfTasksInDoublock[0])
		self.printInfo("numOfTasksInDoublock: {}".format(numOfTasksInDoublock))
		self.printInfo("numOfTasksInDouBlockExpand: {}".format(numOfTasksInDouBlockExpand))
		# S1: Create migration destination in each double-block 
		if self.inActiFromDram:
			destQpeIdList = [[0,0], [0,5], [5,0], [5,5]]
		else:
			destQpeIdList = [[3,1], [1,2], [4,3], [2,4]]
		# S2: Generate conv inActi migration task
		convInActiMigraTasks = []
		inActiBlockCounter = 0
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task destination
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			sramMigrationTaskDest = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			if self.inActiFromDram:
				inActiMigraTaskSour = dramMigrationTaskDest
			else:
				inActiMigraTaskSour = sramMigrationTaskDest
			# SS1: Conv inActi migration task
			blockConvInActiMigraTasks = []
			for inActiBlockIndexAlWidth in range(numOfTasksInDouBlockExpand):
				taskDest = destQpeIdList[douBlockIndex].copy()
				peIndex = inActiBlockIndexAlWidth % NUM_PES_IN_QPE
				taskDest.append(peIndex)
				if inActiBlockIndexAlWidth < numOfTasksInDoublock[douBlockIndex]:
					inActiBlockDim = inActiDimBlocks[inActiBlockCounter]
					# if self.operatorFusion:
					# 	inActiBlockSize = inActiBlockDim[0] * inActiBlockDim[1] * inActiBlockDim[2]
					# else:
					# 	inActiBlockSize = self.align16(inActiBlockDim[0]) * inActiBlockDim[1] * inActiBlockDim[2]
					inActiBlockSize = self.convInActiDimToSize(inActiBlockDim, self.operatorFusion)
					if self.inActiFromDram:
						taskMigraSour = inActiMigraTaskSour
						taskName = Task.DRAM_SRAM_DATA_MIGRATION
					else:
						taskMigraSour = inActiMigraTaskSour.copy()
						taskMigraSour.append(peIndex)
						taskName = Task.DATA_MIGRATION
					# DRAM_RD_MIGRATION (inActi)
					inActiRamToSramMigraTask = {TASK_NAME:taskName, 
						TASK_DESTINATION:taskMigraSour, TASK_MIGRATION_SIZE:inActiBlockSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
					inActiBlockCounter += 1
				else:
					if douBlockIndex == 1 or douBlockIndex == 2:
						inActiRamToSramMigraTask = copy.deepcopy(convInActiMigraTasks[0][0])
					elif douBlockIndex == 3:
						inActiRamToSramMigraTask = copy.deepcopy(convInActiMigraTasks[1][0])
					else:
						self.customAssert(False, "Unsupport")
					if self.inActiFromDram:
						taskMigraSour = inActiMigraTaskSour
						taskName = Task.DRAM_SRAM_DATA_MIGRATION
					else:
						taskMigraSour = inActiMigraTaskSour.copy()
						taskMigraSour.append(peIndex)
						taskName = Task.DATA_MIGRATION
					inActiRamToSramMigraTask[TASK_DESTINATION] = taskMigraSour
					inActiRamToSramMigraTask[TASK_MIGRATION_DESTINATION] = taskDest
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)					
			convInActiMigraTasks.append(blockConvInActiMigraTasks)
		return convInActiMigraTasks

	def convInActiMigraTasksGenerator16Fit(self, layerSplitInfo, inActiDimBlocks):
		'''
		Fit for parts of inActi: <=16, <=8
		'''
		# S0: Number of inActi migration task (actual and expand)
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		partsOfWidth = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		partsOfHeight = self.getTotalPartsFromSplitInfo(inActiSplitInfo[1])
		heightInDoublock = math.ceil(partsOfHeight / NUM_OF_DOUBLOCKS)
			# Get actual number of inActi migration task
		numOfTasksInDoublock = [0] * NUM_OF_DOUBLOCKS
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if partsOfHeight >= heightInDoublock:
				numOfTasksInDoublock[douBlockIndex] = heightInDoublock * partsOfWidth
				partsOfHeight = partsOfHeight - heightInDoublock
			else:
				numOfTasksInDoublock[douBlockIndex] = partsOfHeight * partsOfWidth
				partsOfHeight = partsOfHeight - partsOfHeight
			# Get expand number of inActi migration task
		numOfTasksInDouBlockExpand = self.roundToPowerOf2(numOfTasksInDoublock[0])
		self.printInfo("numOfTasksInDoublock: {}".format(numOfTasksInDoublock))
		self.printInfo("numOfTasksInDouBlockExpand: {}".format(numOfTasksInDouBlockExpand))
		# S1: Create migration destination in each double-block 
		if self.inActiFromDram:
			destQpeIdList = [[0,0], [0,5], [5,0], [5,5]]
		else:
			destQpeIdList = [[3,1], [1,2], [4,3], [2,4]]
		# S2: Generate conv inActi migration task
		convInActiMigraTasks = []
		inActiBlockCounter = 0
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task destination
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			sramMigrationTaskDest = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			if self.inActiFromDram:
				inActiMigraTaskSour = dramMigrationTaskDest
			else:
				inActiMigraTaskSour = sramMigrationTaskDest
			# SS1: Conv inActi migration task
			blockConvInActiMigraTasks = []
			for inActiBlockIndexAlWidth in range(numOfTasksInDouBlockExpand):
				taskDest = destQpeIdList[douBlockIndex].copy()
				peIndex = inActiBlockIndexAlWidth % NUM_PES_IN_QPE
				taskDest.append(peIndex)
				if inActiBlockIndexAlWidth < numOfTasksInDoublock[douBlockIndex]:
					inActiBlockDim = inActiDimBlocks[inActiBlockCounter]
					# if self.operatorFusion:
					# 	inActiBlockSize = inActiBlockDim[0] * inActiBlockDim[1] * inActiBlockDim[2]
					# else:
					# 	inActiBlockSize = self.align16(inActiBlockDim[0]) * inActiBlockDim[1] * inActiBlockDim[2]
					inActiBlockSize = self.convInActiDimToSize(inActiBlockDim, self.operatorFusion)
					if self.inActiFromDram:
						taskMigraSour = inActiMigraTaskSour
						taskName = Task.DRAM_SRAM_DATA_MIGRATION
					else:
						taskMigraSour = inActiMigraTaskSour.copy()
						taskMigraSour.append(peIndex)
						taskName = Task.DATA_MIGRATION
					# DRAM_RD_MIGRATION (inActi)
					inActiRamToSramMigraTask = {TASK_NAME:taskName, 
						TASK_DESTINATION:taskMigraSour, TASK_MIGRATION_SIZE:inActiBlockSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
					inActiBlockCounter += 1
				else:
					inActiRamToSramMigraTask = {TASK_ADDITION:DramSramDataMigraType.EMPTY, 
						TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)					
			convInActiMigraTasks.append(blockConvInActiMigraTasks)
		return convInActiMigraTasks

	def convInActiMigraTasksGenerator32Fit(self, layerSplitInfo, inActiDimBlocks):
		'''
		Fit for parts of inActi: <=16, <=8
		'''
		# S0: Number of inActi migration task (actual and expand)
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		partsOfWidth = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		partsOfHeight = self.getTotalPartsFromSplitInfo(inActiSplitInfo[1])
		heightInDoublock = math.ceil(partsOfHeight / NUM_OF_DOUBLOCKS)
			# Get actual number of inActi migration task
		numOfTasksInDoublock = [0] * NUM_OF_DOUBLOCKS
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if partsOfHeight >= heightInDoublock:
				numOfTasksInDoublock[douBlockIndex] = heightInDoublock * partsOfWidth
				partsOfHeight = partsOfHeight - heightInDoublock
			else:
				numOfTasksInDoublock[douBlockIndex] = partsOfHeight * partsOfWidth
				partsOfHeight = partsOfHeight - partsOfHeight
			# Get expand number of inActi migration task
		numOfTasksInDouBlockExpand = self.roundToPowerOf2(numOfTasksInDoublock[0])
		self.printInfo("numOfTasksInDoublock: {}".format(numOfTasksInDoublock))
		self.printInfo("numOfTasksInDouBlockExpand: {}".format(numOfTasksInDouBlockExpand))
		# S1: Create migration destination in each double-block 
		if self.inActiFromDram:
			destQpeIdList = [[[0,1], [0,0]],
							 [[1,5], [0,5]],
							 [[4,0], [5,0]],
							 [[5,4], [5,5]]]
		else:
			destQpeIdList = [[[3,0], [3,1]],
							 [[0,2], [1,2]],
							 [[5,3], [4,3]],
							 [[2,5], [2,4]]]
		# S2: Conv inActi migration task
		convInActiMigraTasks = []
		inActiCounter = 0
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task migration source
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			sramMigrationTaskDest = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			if self.inActiFromDram:
				inActiMigraTaskSour = dramMigrationTaskDest
			else:
				inActiMigraTaskSour = sramMigrationTaskDest
			# SS1: Conv inActi migration task
			blockConvInActiMigraTasks = []
			actualNumOfInActiTask = numOfTasksInDoublock[douBlockIndex]
			for inActiBlockIndex in range(numOfTasksInDouBlockExpand):
				# Get migration destination
				qpeIndex = inActiBlockIndex // NUM_PES_IN_QPE
				peIndex = inActiBlockIndex % NUM_PES_IN_QPE
				taskDest = destQpeIdList[douBlockIndex][qpeIndex].copy()
				taskDest.append(peIndex)
				if inActiBlockIndex < actualNumOfInActiTask:
					# Get migration source
					if self.inActiFromDram:
						taskMigraSour = inActiMigraTaskSour
						taskName = Task.DRAM_SRAM_DATA_MIGRATION
					else:
						taskMigraSour = inActiMigraTaskSour.copy()
						taskMigraSour.append(peIndex)
						taskName = Task.DATA_MIGRATION
					# Get migration size
					inActiBlockDim = inActiDimBlocks[inActiCounter]
					# if self.operatorFusion:
					# 	inActiBlockSize = inActiBlockDim[0] * inActiBlockDim[1] * inActiBlockDim[2]
					# else:
					# 	inActiBlockSize = self.align16(inActiBlockDim[0]) * inActiBlockDim[1] * inActiBlockDim[2]
					inActiBlockSize = self.convInActiDimToSize(inActiBlockDim, self.operatorFusion)
					inActiCounter += 1
					# DRAM_RD_MIGRATION (inActi)
					inActiRamToSramMigraTask = {TASK_NAME:taskName, 
						TASK_DESTINATION:taskMigraSour, TASK_MIGRATION_SIZE:inActiBlockSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
				else:
					inActiRamToSramMigraTask = {TASK_ADDITION:DramSramDataMigraType.EMPTY, 
						TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
			convInActiMigraTasks.append(blockConvInActiMigraTasks)
		return convInActiMigraTasks

	def convInActiMigraTasksGenerator64Fit(self, layerSplitInfo, inActiDimBlocks):
		# S0: Number of inActi migration task (actual and expand)
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		partsOfWidth = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		partsOfHeight = self.getTotalPartsFromSplitInfo(inActiSplitInfo[1])
		heightInDoublock = math.ceil(partsOfHeight / NUM_OF_DOUBLOCKS)
			# Get actual number of inActi migration task
		numOfTasksInDoublock = [0] * NUM_OF_DOUBLOCKS
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if partsOfHeight >= heightInDoublock:
				numOfTasksInDoublock[douBlockIndex] = heightInDoublock * partsOfWidth
				partsOfHeight = partsOfHeight - heightInDoublock
			else:
				numOfTasksInDoublock[douBlockIndex] = partsOfHeight * partsOfWidth
				partsOfHeight = partsOfHeight - partsOfHeight
			# Get expand number of inActi migration task
		numOfTasksInDouBlockExpand = self.roundToPowerOf2(numOfTasksInDoublock[0])
		self.printInfo("numOfTasksInDoublock: {}".format(numOfTasksInDoublock))
		self.printInfo("numOfTasksInDouBlockExpand: {}".format(numOfTasksInDouBlockExpand))
		# S1: double-block target destination
		if self.inActiFromDram:
			destQpeIdList = [[[1,1], [1,0], [0,1], [0,0]],
							 [[1,4], [0,4], [1,5], [0,5]],
							 [[4,1], [5,1], [4,0], [5,0]],
							 [[4,4], [4,5], [5,4], [5,5]]]
		else:
			# destQpeIdList = [[[2,0], [2,1], [3,0], [3,1]],
			# 				 [[0,3], [1,3], [0,2], [1,2]],
			# 				 [[5,2], [4,2], [5,3], [4,3]],
			# 				 [[3,5], [3,4], [2,5], [2,4]]]
			destQpeIdList = [[[5,2], [4,2], [3,0], [3,1]],
							 [[2,0], [2,1], [0,2], [1,2]],
							 [[3,5], [3,4], [5,3], [4,3]],
							 [[0,3], [1,3], [2,5], [2,4]]]
		# S1: Conv inActi migration task
		convInActiMigraTasks = []
		inActiCounter = 0
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task migration source
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			sramMigrationTaskDest = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			if self.inActiFromDram:
				inActiMigraTaskSour = dramMigrationTaskDest
			else:
				inActiMigraTaskSour = sramMigrationTaskDest
			# SS1: Conv inActi migration task
			blockConvInActiMigraTasks = []
			actualNumOfInActiTask = numOfTasksInDoublock[douBlockIndex]
			for inActiBlockIndex in range(numOfTasksInDouBlockExpand):
				# Get migration destination
				qpeIndex = inActiBlockIndex // NUM_PES_IN_QPE
				peIndex = inActiBlockIndex % NUM_PES_IN_QPE
				taskDest = destQpeIdList[douBlockIndex][qpeIndex].copy()
				taskDest.append(peIndex)
				if inActiBlockIndex < actualNumOfInActiTask:
					# Get migration source
					if self.inActiFromDram:
						taskMigraSour = inActiMigraTaskSour
						taskName = Task.DRAM_SRAM_DATA_MIGRATION
					else:
						taskMigraSour = inActiMigraTaskSour.copy()
						taskMigraSour.append(peIndex)
						taskName = Task.DATA_MIGRATION
					# Get migration size
					inActiBlockDim = inActiDimBlocks[inActiCounter]
					# if self.operatorFusion:
					# 	inActiBlockSize = inActiBlockDim[0] * inActiBlockDim[1] * inActiBlockDim[2]
					# else:
					# 	inActiBlockSize = self.align16(inActiBlockDim[0]) * inActiBlockDim[1] * inActiBlockDim[2]
					inActiBlockSize = self.convInActiDimToSize(inActiBlockDim, self.operatorFusion)
					inActiCounter += 1
					# DRAM_RD_MIGRATION (inActi)
					inActiRamToSramMigraTask = {TASK_NAME:taskName, 
						TASK_DESTINATION:taskMigraSour, TASK_MIGRATION_SIZE:inActiBlockSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
				else:
					inActiRamToSramMigraTask = {TASK_ADDITION:DramSramDataMigraType.EMPTY, 
						TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
			convInActiMigraTasks.append(blockConvInActiMigraTasks)
		return convInActiMigraTasks

	def convInActiMigraTasksGenerator128Fit(self, layerSplitInfo, inActiDimBlocks):
		# S0: Number of inActi migration task (actual and expand)
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		partsOfWidth = self.getTotalPartsFromSplitInfo(inActiSplitInfo[0])
		partsOfHeight = self.getTotalPartsFromSplitInfo(inActiSplitInfo[1])
		heightInDoublock = math.ceil(partsOfHeight / NUM_OF_DOUBLOCKS)
			# Get actual number of inActi migration task
		numOfTasksInDoublock = [0] * NUM_OF_DOUBLOCKS
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			if partsOfHeight >= heightInDoublock:
				numOfTasksInDoublock[douBlockIndex] = heightInDoublock * partsOfWidth
				partsOfHeight = partsOfHeight - heightInDoublock
			else:
				numOfTasksInDoublock[douBlockIndex] = partsOfHeight * partsOfWidth
				partsOfHeight = partsOfHeight - partsOfHeight
			# Get expand number of inActi migration task
		numOfTasksInDouBlockExpand = self.roundToPowerOf2(numOfTasksInDoublock[0])
		self.printInfo("numOfTasksInDoublock: {}".format(numOfTasksInDoublock))
		self.printInfo("numOfTasksInDouBlockExpand: {}".format(numOfTasksInDouBlockExpand))
		# S1: double-block target destination
		if self.inActiFromDram:
			destQpeIdList = [[[3,1], [3,0], [2,1], [2,0], [1,1], [1,0], [0,1], [0,0]],
							 [[1,2], [0,2], [1,3], [0,3], [1,4], [0,4], [1,5], [0,5]],
							 [[4,3], [5,3], [4,2], [5,2], [4,1], [5,1], [4,0], [5,0]],
							 [[2,4], [2,5], [3,4], [3,5], [4,4], [4,5], [5,4], [5,5]]]
		else:
			destQpeIdList = [[[5,0], [5,1], [4,0], [4,1], [5,2], [4,2], [3,0], [3,1]],
							 [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [0,2], [1,2]],
							 [[5,5], [5,4], [4,5], [4,4], [3,5], [3,4], [5,3], [4,3]],
							 [[0,5], [0,4], [1,5], [1,4], [0,3], [1,3], [2,5], [2,4]]]
		# S1: Conv inActi migration task
		convInActiMigraTasks = []
		inActiCounter = 0
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task migration source
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			sramMigrationTaskDest = DOUBLOCK_ST_QPE_IDS[douBlockIndex]
			if self.inActiFromDram:
				inActiMigraTaskSour = dramMigrationTaskDest
			else:
				inActiMigraTaskSour = sramMigrationTaskDest
			# SS1: Conv inActi migration task
			blockConvInActiMigraTasks = []
			actualNumOfInActiTask = numOfTasksInDoublock[douBlockIndex]
			for inActiBlockIndex in range(numOfTasksInDouBlockExpand):
				# Get migration destination
				qpeIndex = inActiBlockIndex // NUM_PES_IN_QPE
				peIndex = inActiBlockIndex % NUM_PES_IN_QPE
				taskDest = destQpeIdList[douBlockIndex][qpeIndex].copy()
				taskDest.append(peIndex)
				if inActiBlockIndex < actualNumOfInActiTask:
					# Get migration source
					if self.inActiFromDram:
						taskMigraSour = inActiMigraTaskSour
						taskName = Task.DRAM_SRAM_DATA_MIGRATION
					else:
						taskMigraSour = inActiMigraTaskSour.copy()
						taskMigraSour.append(peIndex)
						taskName = Task.DATA_MIGRATION
					# Get migration size
					inActiBlockDim = inActiDimBlocks[inActiCounter]
					# if self.operatorFusion:
					# 	inActiBlockSize = inActiBlockDim[0] * inActiBlockDim[1] * inActiBlockDim[2]
					# else:
					# 	inActiBlockSize = self.align16(inActiBlockDim[0]) * inActiBlockDim[1] * inActiBlockDim[2]
					inActiBlockSize = self.convInActiDimToSize(inActiBlockDim, self.operatorFusion)
					inActiCounter += 1
					# DRAM_RD_MIGRATION (inActi)
					inActiRamToSramMigraTask = {TASK_NAME:taskName, 
						TASK_DESTINATION:taskMigraSour, TASK_MIGRATION_SIZE:inActiBlockSize, 
						TASK_ADDITION:DramSramDataMigraType.INACTIVE, TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
				else:
					inActiRamToSramMigraTask = {TASK_ADDITION:DramSramDataMigraType.EMPTY, 
						TASK_MIGRATION_DESTINATION:taskDest}
					blockConvInActiMigraTasks.append(inActiRamToSramMigraTask)
			convInActiMigraTasks.append(blockConvInActiMigraTasks)
		return convInActiMigraTasks

	def convWeightMigraTasksGenerator16(self, layerSplitInfo, weightDimBlocks):
		# double-block target destination
		destQpeIdList = [[0,0], [1,5], [5,0], [4,5]]
		# S1: Conv weight migration task
		numOfWeightBlocksInDouBlock = len(weightDimBlocks) // NUM_OF_DOUBLOCKS # 4
		convWeightMigraTasks = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task destination
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			# SS1: Conv weight migration task
			blockConvWeightMigraTasks = []
			for weightBlockIndexAlOutChannel in range(numOfWeightBlocksInDouBlock):
				weightBlockIndex = numOfWeightBlocksInDouBlock*douBlockIndex + weightBlockIndexAlOutChannel
				# weightBlockSize = len(weightAlignBlocks[weightBlockIndex])
				weightBlockDim = weightDimBlocks[weightBlockIndex]
				weightBlockSize = self.convWeightDimToSize(weightBlockDim, self.operatorFusion)
				taskDest = destQpeIdList[douBlockIndex].copy()
				taskDest.append(weightBlockIndexAlOutChannel%NUM_PES_IN_QPE)
				# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:dramMigrationTaskDest, TASK_MIGRATION_SIZE:weightBlockSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT, TASK_MIGRATION_DESTINATION:taskDest}
				# # Dimension of weight
				# weightDim = self.getWeightDim(layerSplitInfo, weightBlockIndex)
				# blockConvWeightMigraTasks.append((weightDim, weightDramToSramMigraTask))
				blockConvWeightMigraTasks.append(weightDramToSramMigraTask)
			convWeightMigraTasks.append(blockConvWeightMigraTasks)
		return convWeightMigraTasks

	def convWeightMigraTasksGenerator64(self, layerSplitInfo, weightDimBlocks):
		# double-block target destination
		destQpeIdList = [[[1,1], [0,0], [1,0], [0,1]], 
							[[1,4], [0,5], [0,4], [1,5]], 
							[[4,1], [5,0], [5,1], [4,0]], 
							[[4,4], [5,5], [4,5], [5,4]]]
		# S1: Conv weight migration task
		numOfWeightBlocksInDouBlock = len(weightDimBlocks) // NUM_OF_DOUBLOCKS # 4
		convWeightMigraTasks = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task destination
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			# SS1: Conv weight migration task
			blockConvWeightMigraTasks = []
			for weightBlockIndexAlOutChannel in range(numOfWeightBlocksInDouBlock):
				weightBlockIndex = numOfWeightBlocksInDouBlock*douBlockIndex + weightBlockIndexAlOutChannel
				# weightBlockSize = len(weightAlignBlocks[weightBlockIndex])
				weightBlockDim = weightDimBlocks[weightBlockIndex]
				weightBlockSize = self.convWeightDimToSize(weightBlockDim, self.operatorFusion)
				qpeIndex = weightBlockIndexAlOutChannel // 4
				taskDest = destQpeIdList[douBlockIndex][qpeIndex].copy()
				taskDest.append(weightBlockIndexAlOutChannel%NUM_PES_IN_QPE)
				# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:dramMigrationTaskDest, TASK_MIGRATION_SIZE:weightBlockSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT, TASK_MIGRATION_DESTINATION:taskDest}
				# # Dimension of weight
				# weightDim = self.getWeightDim(layerSplitInfo, weightBlockIndex)
				# blockConvWeightMigraTasks.append((weightDim, weightDramToSramMigraTask))
				blockConvWeightMigraTasks.append(weightDramToSramMigraTask)
			convWeightMigraTasks.append(blockConvWeightMigraTasks)
		return convWeightMigraTasks

	def convWeightMigraTasksGenerator128(self, layerSplitInfo, weightDimBlocks):
		# double-block target destination
		destQpeIdList = [[[3,1], [3,0], [2,1], [2,0], [1,1], [1,0], [0,1], [0,0]], 
							[[1,2], [0,2], [1,3], [0,3], [1,4], [0,4], [1,5], [0,5]], 
							[[4,3], [5,3], [4,2], [5,2], [4,1], [5,1], [4,0], [5,0]], 
							[[2,4], [2,5], [3,4], [3,5], [4,4], [4,5], [5,4], [5,5]]]
		# S1: Conv weight migration task
		numOfWeightBlocksInDouBlock = len(weightDimBlocks) // NUM_OF_DOUBLOCKS # 4
		convWeightMigraTasks = []
		for douBlockIndex in range(NUM_OF_DOUBLOCKS):
			# SS0: Get inActi RAM -> SRAM migration task destination
			dramMigrationTaskDest = [douBlockIndex+DRAM_ID_START]
			# SS1: Conv weight migration task
			blockConvWeightMigraTasks = []
			for weightBlockIndexAlOutChannel in range(numOfWeightBlocksInDouBlock):
				weightBlockIndex = numOfWeightBlocksInDouBlock*douBlockIndex + weightBlockIndexAlOutChannel
				# weightBlockSize = len(weightAlignBlocks[weightBlockIndex])
				weightBlockDim = weightDimBlocks[weightBlockIndex]
				weightBlockSize = self.convWeightDimToSize(weightBlockDim, self.operatorFusion)
				qpeIndex = weightBlockIndexAlOutChannel // NUM_PES_IN_QPE
				peIndex = weightBlockIndexAlOutChannel%NUM_PES_IN_QPE
				taskDest = destQpeIdList[douBlockIndex][qpeIndex].copy()
				taskDest.append(peIndex)
				# DRAM_RD_MIGRATION (weight): TASK_MIGRATION_DESTINATION (pairedPe)
				weightDramToSramMigraTask = {TASK_NAME:Task.DRAM_SRAM_DATA_MIGRATION, 
					TASK_DESTINATION:dramMigrationTaskDest, TASK_MIGRATION_SIZE:weightBlockSize, 
					TASK_ADDITION:DramSramDataMigraType.WEIGHT, TASK_MIGRATION_DESTINATION:taskDest}
				blockConvWeightMigraTasks.append(weightDramToSramMigraTask)
			convWeightMigraTasks.append(blockConvWeightMigraTasks)
		return convWeightMigraTasks

	def convMlaTaskGenerator(self, layerSplitInfo, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, poolStride):
		# poolFlag, poolSize = poolParam
		poolSize = poolStride * poolStride
		# S1: Conv execution task
		# MLA_DRAM_WR_MIGRATION: TASK_DESTINATION (targetPe), TASK_OPER_A_PEID (pairedPe), 
		# 						 TASK_OUTACTI_DRAM_ADDR
		inActiDim = self.getInActiDim(layerSplitInfo, 0)
		weightDim = self.getWeightDim(layerSplitInfo, 0)
		mlaParam = (MlaOperType.CONV, inActiDim+weightDim)
		# Determine the outActi storage place: SRAM/DRAM
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		outActiWidthSplitInfo = outActiSplitInfo[0]
		outActiHeightSplitInfo = outActiSplitInfo[1]
		outActiChannelSplitInfo = outActiSplitInfo[2]
		outActiWidth = ConvLayerMapper.convOutActivationWidthAlign(outActiWidthSplitInfo[0]) * outActiWidthSplitInfo[1] + \
			ConvLayerMapper.convOutActivationWidthAlign(outActiWidthSplitInfo[2]) * outActiWidthSplitInfo[3]
		outActiHeight = BasicOperation.splitInfoIntegration(outActiHeightSplitInfo)
		outChannel = BasicOperation.splitInfoIntegration(outActiChannelSplitInfo)
		# operator fusion decrease the size
		outActiSize = outActiWidth * outActiHeight * outChannel
		if self.operatorFusion:
			outActiSize = outActiSize / 4 # quantization Layer decrease size by 4
			outActiSize = outActiSize / poolSize # pooling layer
		if outActiSize > PE_SRAM_LIMIT * NUM_PES_IN_QPE * NUM_OF_QPE_IN_BLOCK or (self.operatorFusion == False):
			taskName = Task.MLA_EXE
			self.nextLayerLoadFromDram = True
			if self.operatorFusion == False:
				self.printInfo("outActi store in DRAM not SRAM: No operatorFusion")
			else:
				self.printInfo("outActi store in DRAM not SRAM: {} - {}".format(outActiSize, PE_SRAM_LIMIT * NUM_PES_IN_QPE * NUM_OF_QPE_IN_BLOCK))
		else:
			taskName = Task.MLA_EXE_SRAM
			self.nextLayerLoadFromDram = False
		additionParam = (self.operatorFusion, poolSize)
		mlaTask = {TASK_NAME:taskName, TASK_OPER_A_ADDR: weightBaseAddr, TASK_MLA_PARAM:mlaParam,
			TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, TASK_ADDITION:additionParam}
		return mlaTask	

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
	# 
	# =========================================================
	def getDouBlockIndex(self, qpeId):
		qpeId = qpeId[Y_AXIS_INDEX:Z_AXIS_INDEX]
		for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
			if qpeId in DOUBLOCK_QPE_ID_LIST[douBlockIndex]:
				return douBlockIndex
		return None

	def getBlockIndexInDouBlock(self, qpeId):
		qpeId = qpeId[Y_AXIS_INDEX:Z_AXIS_INDEX]
		for douBlockIndex in range(len(DOUBLOCK_QPE_ID_LIST)):
			if qpeId in DOUBLOCK_QPE_ID_LIST[douBlockIndex]:
				blockIndex = DOUBLOCK_QPE_ID_LIST[douBlockIndex].index(qpeId) // NUM_OF_QPE_IN_BLOCK
				return blockIndex
		return None

	def roundToPowerOf2(self, value):
		return int(math.pow(2, math.ceil(math.log(value, 2))))

	# =========================================================
	# Simple Validation
	# =========================================================
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


if __name__ == "__main__":
	ssd = SpiNNaker2DistributorDataReuse(operatorFusion=True)
	# ssd.resNetModelSplitter()
	ssd.vggModelSplitter()
	ssd.distribute()
	# ssd.dramSramDataMigraValidation()
	# ssd.mlaExeValidate()
	# npa = np.array(ssd.spiNNakerContainer)
	# print("ssd.spiNNakerContainer[5][5][1]: {}".format(ssd.spiNNakerContainer[5][5][1]))
	# print("{}".format(npa.shape))