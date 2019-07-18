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
from qpeSimulator import QuadPE
from spiNNakerSimulatorGeneral import *
from dataGenerator import convDataSplitter
from collections import deque
import copy
import datetime


class QpeDistributorNoDataReuse(DistributionGeneralClass):
	def __init__(self, nocDoubleFreq=True, noDataTran=True, printFlag=True, qpeId=[0,0], convDistri=True):
		DistributionGeneralClass.__init__(self, printFlag=printFlag)
		self.nocDoubleFreq = nocDoubleFreq
		self.noDataTran = noDataTran
		self.qpeId = qpeId
		self.convDistri = convDistri
		self.qpe = QuadPE(self.qpeId, nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)

	def updateModelSplitInfoParameter(self, modelLayersSplitInfo, modelLayersParameter):
		self.modelLayersSplitInfo = modelLayersSplitInfo
		self.modelLayersParameter = modelLayersParameter

	def clockGenerate(self):
		self.clockCounter += 1
		rsp = self.qpe.run()
		return rsp

	def addTaskToQpe(self, task):
		self.qpe.commandFrom(task)

	def targetPairedPeIdGenerator(self, peIndex, pePairShift=0):
		targetPeId = copy.deepcopy(self.qpeId)
		pairedPeId = copy.deepcopy(self.qpeId)
		targetPeId.append(peIndex%NUM_PES_IN_QPE)
		pairedPeId.append((peIndex+pePairShift)%NUM_PES_IN_QPE)
		return targetPeId, pairedPeId

	def run(self):
		modelLayersSplitInfo, modelLayersParameter = QpeDistributorNoDataReuse.vgg()
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)
		if self.convDistri:
			self.convLayersDistribution()
		else:
			self.fcLayerDistribution()

	def convLayersDistribution(self):
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "CONV_" not in layerName:
				continue
			self.logBeginTime()
			self.singleConvDistribution(layerName)
			self.logEndTime()
			self.recordLayerDistriResult()
			self.layerDistriResultReset()
			break

	def fcLayerDistribution(self):
		layerNames = list(self.modelLayersSplitInfo.keys())
		for layerName in layerNames:
			if "FC_" not in layerName:
				continue
			self.logBeginTime()
			# TODO
			self.logEndTime()
			self.recordLayerDistriResult()
			self.layerDistriResultReset()
			break

	def singleFcDistribution(self, layerName):
		pass

	def singleConvDistribution(self, layerName):
		# Generate data for inActi, weight and outActi 
		# Generate number of blocks of inActi and weight
		self.layerName = layerName
		layerSplitInfo = self.modelLayersSplitInfo[layerName]
		layerTypeParameter = self.modelLayersParameter[layerName]
		inActiAlignBlocks, inActiBaseAddr, weightAlignBlocks, weightBaseAddr, outActiAlignBlocks, \
			outActiBaseAddr = convDataSplitter(layerSplitInfo, layerTypeParameter)
		numOfInActiBlocks = len(inActiAlignBlocks)
		numOfWeightBlocks = len(weightAlignBlocks)
		outActiAlignBlockSize = len(outActiAlignBlocks[0])
		# Generate tasks queue, which 
		peFreeFlag = [True]*NUM_PES_IN_QPE
		inActiBlockIndex = 0
		weightBlockIndex = 0
		outActiDramAddr = 0
		allTaskIntoQpeFlag = False
		qpeTasks = [[],[],[],[]]
		qpeTasksPointer = 0
		enablePause = False
		while True:
			# Write inActi and weight into PE
			if True in peFreeFlag and allTaskIntoQpeFlag == False:
				# Write Qpe inQueue
				peIndex = peFreeFlag.index(True)
				inActiBlock = inActiAlignBlocks[inActiBlockIndex]
				weightBlock = weightAlignBlocks[weightBlockIndex]
				targetPeId, pairedPeId = self.targetPairedPeIdGenerator(peIndex=peIndex)
				peTasks = self.hostTaskGenerator(targetPeId, pairedPeId, inActiBlock, weightBlock, inActiBaseAddr, 
					weightBaseAddr, layerSplitInfo, inActiBlockIndex, weightBlockIndex, outActiBaseAddr, outActiDramAddr)
				qpeTasks[peIndex] = peTasks
				# Update loop parameter
				outActiDramAddr += outActiAlignBlockSize
				peFreeFlag[peIndex] = False
				weightBlockIndex += 1
				if weightBlockIndex == numOfWeightBlocks:
					weightBlockIndex = 0
					inActiBlockIndex += 1
					if inActiBlockIndex == numOfInActiBlocks:
						allTaskIntoQpeFlag = True
			# Run QPE
			# while (not (True in peFreeFlag)) or (allTaskIntoQpeFlag == True):
			if (False in peFreeFlag) or (allTaskIntoQpeFlag == True):
				# Get single task
				for peTasksIndex in list(range(qpeTasksPointer, NUM_PES_IN_QPE))+list(range(0, qpeTasksPointer)):
					if len(qpeTasks[peTasksIndex]) > 0:
					 	singleTask = qpeTasks[qpeTasksPointer].popleft()
					 	if Task.MLA_EXE == singleTask[TASK_NAME]:
					 		# self.printInfo("MLA_EXE")
					 		enablePause = True
					 	self.addTaskToQpe(singleTask)
					 	qpeTasksPointer = (peTasksIndex+1) % NUM_PES_IN_QPE
					 	break
					else:
						qpeTasksPointer = (peTasksIndex+1) % NUM_PES_IN_QPE
				# Run QPE
				qpeRsps = self.clockGenerate()
				if not self.nocDoubleFreq:
					qpeRsps = [qpeRsps]
				for qpeRsp in qpeRsps:
					if qpeRsp[TASK_NAME] == Task.MLA_FINISH:
						# self.printInfo("MLA_FINISH: {}".format(qpeRsp))
						# self.compClockCounter += qpeRsp[TASK_MLA_COMP_CLKS]
						# self.wrOutActiIntoSramClockCounter += qpeRsp[TASK_MLA_OUTACTI_WR_CLKS]
						# self.mlaRunClks += qpeRsp[TASK_MLA_TOTAL_CLKS]
						pass
					elif qpeRsp[TASK_NAME] == Task.SRAM_DATA_32:
						pass
					elif qpeRsp[TASK_NAME] == Task.DATA_MIGRATION_32_FINISH:
						# self.printInfo("DATA_MIGRATION_32_FINISH: {}".format(qpeRsp))
						peId = qpeRsp[TASK_MIGRATION_SOURCE]
						peFreeFlag[peId[2]] = True
						self.finishedPeCounter += 1
						if self.finishedPeCounter%16 == 0:
							self.printInfo("Finished Parts: {}".format(self.finishedPeCounter))
				if (not (False in peFreeFlag)) and (allTaskIntoQpeFlag == True):
					break
				# if enablePause:
				# 	input("-----")

	def recordLayerDistriResult(self):
		self.printInfo("Total Clocks: {}".format(self.clockCounter))
		self.printInfo("MLA Computation Clocks: {}".format(self.compClockCounter))
		self.printInfo("MLA Write outActi into SRAM Clocks: {}".format(self.wrOutActiIntoSramClockCounter))
		self.printInfo("Simulation Time (Second): {}".format(self.simulationTimeSecond))
		self.distriResults.append({DIS_LAYER_NAME:self.layerName, DIS_LAYER_TOTAL_CLKS:self.clockCounter, 
			DIS_LAYER_USED_PES:self.finishedPeCounter, DIS_LAYER_MLA_CLKS:self.mlaRunClks,
			DIS_LAYER_MLA_COMPUTE_CLKS:self.compClockCounter,
			DIS_LAYER_MLA_WR_OUTACTI_SRAM_CLKS:self.wrOutActiIntoSramClockCounter, 
			DIS_LAYER_SIMULATION_SECOND:self.simulationTimeSecond})

	def hostTaskGenerator(self, targetPeId, pairedPeId, inActiBlock, weightBlock, inActiBaseAddr, 
		weightBaseAddr, layerSplitInfo, inActiBlockIndex, weightBlockIndex, outActiBaseAddr, outActiDramAddr):
		peTasks = deque([])
		if self.noDataTran:
			wrInActiTasks = self.writeDataBlockIntoPeWithoutData(inActiBlock, targetPeId)
			wrWeightTasks = self.writeDataBlockIntoPeWithoutData(weightBlock, pairedPeId)
		else:
			wrInActiTasks = self.writeDataBlockIntoPe(inActiBlock, inActiBaseAddr, targetPeId)
			wrWeightTasks = self.writeDataBlockIntoPe(weightBlock, weightBaseAddr, pairedPeId)
		mlaTask = self.generateMlaConvTaskIntoPe(layerSplitInfo, inActiBlockIndex, weightBlockIndex, targetPeId, 
			pairedPeId, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, outActiDramAddr)
		peTasks.extend(wrInActiTasks)
		peTasks.extend(wrWeightTasks)
		peTasks.append(mlaTask)
		return peTasks

	def writeDataBlockIntoPe(self, dataBlock, baseAddr, targetPeId, taskDataLen=NOC_SRAM_BW_BYTES):
		wrDataQueue = deque([])
		baseIndex = 0
		address = baseAddr
		task = {TASK_NAME:Task.SRAM_WR, TASK_DESTINATION:targetPeId, TASK_DATA_LEN:taskDataLen}
		while baseIndex < len(dataBlock):
			task[TASK_DATA] = dataBlock[baseIndex: baseIndex+taskDataLen]
			task[TASK_SRAM_ADDR] = address
			wrDataQueue.append(copy.deepcopy(task))
			baseIndex += taskDataLen
			address += taskDataLen
		return wrDataQueue

	def writeDataBlockIntoPeWithoutData(self, dataBlock, targetPeId, taskDataLen=NOC_SRAM_BW_BYTES):
		wrDataQueue = deque([])
		numOfTasks = len(dataBlock) // taskDataLen
		# self.printInfo("numOfTasks: {}".format(numOfTasks))
		task = {TASK_NAME:Task.SRAM_WR, TASK_DESTINATION:targetPeId}
		for _ in range(numOfTasks):
			wrDataQueue.append(copy.deepcopy(task))
		return wrDataQueue

	def generateMlaConvTaskIntoPe(self, layerSplitInfo, inActiBlockIndex, weightBlockIndex, targetPeId, 
		pairedPeId, inActiBaseAddr, weightBaseAddr, outActiBaseAddr, outActiDramAddr):
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
		mlaTask = {TASK_NAME:Task.MLA_EXE, TASK_DESTINATION:targetPeId, TASK_MLA_PARAM:mlaParam, 
					TASK_OPER_A_PEID:pairedPeId, TASK_OPER_A_ADDR: weightBaseAddr, 
					TASK_OPER_B_ADDR:inActiBaseAddr, TASK_OPER_C_ADDR:outActiBaseAddr, 
					TASK_OUTACTI_DRAM_ADDR: outActiDramAddr, TASK_ADDITION: (False, 1)}
		return mlaTask

if __name__ == "__main__":
	# CONV_1: [Thread]
	# ---> 1143114: Total Clocks: 1143114
	# ---> 1143114: MLA Computation Clocks: 1354752
	# ---> 1143114: MLA Write outActi into SRAM Clocks: 802816
	# ---> 1143114: Simulation Time (Second): 154
	# CONV_1: [No thread/process]
	# ---> 1143114: CONV_1-Total Clocks: 1143114
	# ---> 1143114: CONV_1-MLA Computation Clocks: 0
	# ---> 1143114: CONV_1-MLA Write outActi into SRAM Clocks: 0
	# ---> 1143114: CONV_1-Simulation Time (Second): 64
	qpeDistributor = QpeDistributorNoDataReuse(nocDoubleFreq=True)
	qpeDistributor.run()
	print(qpeDistributor.getDistriResults())