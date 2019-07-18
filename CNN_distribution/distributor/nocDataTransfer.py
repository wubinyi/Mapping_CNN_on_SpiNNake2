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
	
from spiNNakerSimulatorGeneral import *
from spiNNaker2Simulator import SpiNNaker2Simulator
from spiNNaker2SimulatorProcess import SpiNNaker2SimulatorProcess
from spiNNaker2TriBlockSimulator import SpiNNaker2TriBlock
import queue
import multiprocessing
from nnGeneral import PE_SRAM_LIMIT
import datetime


class NocDataTransferSimulation():
	def __init__(self, nocDoubleFreq=True, noDataTran=False, threadFlag=False, triBlockFlag=False):
		self.nocDoubleFreq = nocDoubleFreq
		self.noDataTran = noDataTran
		self.threadFlag = threadFlag
		self.triBlockFlag = triBlockFlag
		if self.threadFlag:
			self.inQueue = queue.Queue()
			self.outQueue = queue.Queue()
			self.clockQueue = queue.Queue()
			self.spiNNaker2 = SpiNNaker2Simulator(self.clockQueue, self.inQueue, self.outQueue, 
				nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)
			self.spiNNaker2.start()
		else:
			if self.triBlockFlag:
				self.spiNNaker2 = SpiNNaker2TriBlock(nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)
			else:
				self.inQueue = multiprocessing.Queue()
				self.outQueue = multiprocessing.Queue()
				self.clockQueue = multiprocessing.Queue()
				self.spiNNaker2 = SpiNNaker2SimulatorProcess(self.clockQueue, self.inQueue, self.outQueue, 
					nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)			
				self.spiNNaker2.start()
		self.clockCounter = 0

	def spiNNakerThreadStop(self):
		if self.triBlockFlag:
			self.spiNNaker2.spiNNaker2Stop()
		else:
			self.clockQueue.put(Task.TASK_NO_CLOCK)
			self.spiNNaker2.join()	

	def spiNNaker2ClockGenerator(self):
		self.clockCounter += 1
		if self.triBlockFlag and (not self.threadFlag):
			self.spiNNaker2.spiNNaker2Run()
			rsp = self.spiNNaker2.getNextOutTask()
			return rsp
		else:
			self.clockQueue.put(Task.TASK_CLOCK)
			rsp = self.outQueue.get()
			return rsp

	def taskToInQueue(self, task):
		if self.triBlockFlag:
			self.spiNNaker2.addInTask(task)
		else:
			self.inQueue.put(task)	

	def printInfo(self, content):
		print("----> {:<8}: {}".format(self.clockCounter, content))

	def errorInfo(self, content):
		print("xxxx> {:<8}: ERROR -> {}".format(self.clockCounter, content))

	def customAssert(self, condition, content):
		funcName = sys._getframe().f_code.co_name
		lineNumber = sys._getframe().f_lineno
		fileName = sys._getframe().f_code.co_filename
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def areThreadsAlive(self):
		self.printInfo("spiNNaker2 is alive: {}".format(self.spiNNaker2.is_alive()))
		for qpeThreadRow in self.spiNNaker2.qpeThreadArray:
			for qpeThread in qpeThreadRow:
				self.printInfo("Qpe thread-{} is alive: {}".format(qpeThread.threadId, qpeThread.is_alive()))

	def listLoopShift(self, elements, left=True, offset=1):
		if left:
			for _ in range(offset):
				leftElem = elements.pop(0)
				elements.append(leftElem)
		else:
			for _ in range(offset):
				rightElem = elements.pop(-1)
				elements.insert(0, rightElem)

	# =====================================================
	# 				Noc Simple Data Migration
 	# ===================================================== 
	def writeSramOrValidate(self, writeFlag=True, originalWeightAddr=0x8000, weightSize=0x900):
		ylist = list(range(NUM_QPES_Y_AXIS))
		xlist = list(range(NUM_QPES_X_AXIS))
		weight = 0
		lastY = 0
		for x in xlist:
			for y in ylist:
				for peIndex in range(NUM_PES_IN_QPE):
					# Write
					if writeFlag:
						counter = 0
						while counter < weightSize:
							self.taskToInQueue({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION: [y,x,peIndex], 
											TASK_SRAM_ADDR: originalWeightAddr+counter, 
											TASK_DATA_LEN: NOC_SRAM_BW_BYTES, 
											TASK_DATA: [weight]*NOC_SRAM_BW_BYTES})
							self.spiNNaker2ClockGenerator()
							counter += NOC_SRAM_BW_BYTES
					# Read
					else:
						sramData = self.spiNNaker2.qpeThreadArray[y][x].qpe.peArray[peIndex].sram.sram[0:weightSize]
						if sramData == [weight]*weightSize:
							pass
							# print("QPE[{}][{}]-PE[{}]: PASS\n".format(y,x,peIndex))
						else:
							# print("QPE[{}][{}]-PE[{}]: FAIL\n".format(y,x,peIndex))
							self.errorInfo("QPE[{}][{}]-PE[{}]: FAIL".format(y,x,peIndex))
					# Update sram value
					weight += 1
			lastY = ylist[-1]
			self.listLoopShift(ylist)
		# Wait Last PQE command to be finish
		if writeFlag:
			clocksToWait = (lastY + xlist[-1]) * 4 + 3
			for _ in range(clocksToWait):
				self.spiNNaker2ClockGenerator()
			# Print info 
			totalPe = NUM_QPES_X_AXIS * NUM_QPES_Y_AXIS * NUM_PES_IN_QPE
			self.printInfo("Finish writing {} bytes data into {} PE".format(weightSize*totalPe, totalPe))

	def dataMigrationProcess(self, sourcePeId=[0,0,1], destPeId=[1,1,1], originalWeightAddr=0x8000, weightSize=0x900):
		copyWeightAddr = originalWeightAddr + weightSize
		self.taskToInQueue({TASK_NAME:Task.DATA_MIGRATION, TASK_SRAM_ADDR: originalWeightAddr,
							TASK_MIGRATION_SIZE:weightSize, TASK_DESTINATION:sourcePeId, 
							TASK_MIGRA_SRAM_ADDR:copyWeightAddr, TASK_MIGRATION_DESTINATION:destPeId})	
		while True:
			rsp = self.spiNNaker2ClockGenerator()
			if rsp[TASK_NAME] == Task.DATA_MIGRATION_FINISH:
				break
		# After receiving "DATA_MIGRATION_FINISH", still need to wait some clocks, until all task are finished
		waitClocks = (abs(sourcePeId[0]-destPeId[0]) + abs(sourcePeId[1]-destPeId[1])) * 4 + 3 - 4
		for _ in range(waitClocks):
			self.spiNNaker2ClockGenerator()
		# Print info 
		self.printInfo("Finish migrate {} bytes data".format(weightSize))		

	def dataMigrationValidate(self, sourcePeId=[0,0,1], destPeId=[1,1,1], originalWeightAddr=0x8000, weightSize=0x900):
		beginIndex = originalWeightAddr + weightSize - (TOTAL_SRAM_SIZE-PE_SRAM_LIMIT)
		endIndex = beginIndex + weightSize
		data = sourcePeId[0] * NUM_PES_IN_QPE * NUM_QPES_X_AXIS + sourcePeId[1] * NUM_PES_IN_QPE + sourcePeId[2]
		sramData = self.spiNNaker2.qpeThreadArray[destPeId[0]][destPeId[1]].qpe.peArray[destPeId[2]%4].sram.sram[beginIndex:endIndex]
		if sramData == [data]*weightSize:
			pass
			# print("Data-Migration-> QPE[{}][{}]-PE[{}]: PASS\n".format(1,1,1))
		else:
			# print("Data-Migration-> QPE[{}][{}]-PE[{}]: FAIL\n".format(1,1,1))
			self.errorInfo("Data-Migration-> QPE[{}][{}]-PE[{}]: FAILED".format(1,1,1))	

	def simpleDataMigrationSimulation(self):
		self.printInfo("SpiNNaker2 START")
		# Write weight into QPE
		self.writeSramOrValidate()
		# Data migration
		self.dataMigrationProcess()
		# Stop SpiNNaker2
		self.spiNNakerThreadStop()
		self.printInfo("SpiNNaker2 STOP")
		# Validate SRAM data
		self.writeSramOrValidate(writeFlag=False)
		# Validate migrated data
		self.dataMigrationValidate()
		self.printInfo("----> Finish Validation: PASS if no FAILED/ERROR Infos <----")

	# =========================================================================
	# 				Noc Data Tri-Migration (Data reuse inside block)
 	# =========================================================================
	def getNearestBlockBasic(self, qpeId):
		# Basic X
		if qpeId[1] < NUM_QPES_X_AXIS_HALF:
			basicX = 0
		else:
			basicX = 3
		# Basic Y
		if qpeId[0] < NUM_QPES_Y_AXIS_HALF:
			basicY = 0
		else:
			basicY = 3
		return basicY, basicX

	def triBlockQpeSequenceGenerate(self, rightShift=False, downShift=False, 
		basicX=TRIBLOCK_BASIC[0][1], basicY=TRIBLOCK_BASIC[0][0]):
		self.customAssert(NUM_QPES_X_AXIS_TRIBLOCK==NUM_QPES_Y_AXIS_TRIBLOCK, "X and Y not equal")
		self.customAssert( not (rightShift==True and downShift==True), "rightShift and downShift cannot be both true")
		yPosList = list(range(NUM_QPES_Y_AXIS_TRIBLOCK))
		xPosList = list(range(NUM_QPES_X_AXIS_TRIBLOCK))
		if rightShift:
			self.listLoopShift(xPosList)
		if downShift:
			self.listLoopShift(yPosList, left=False)
		qpeSequence = []
		for loop in range(NUM_QPES_X_AXIS_TRIBLOCK):
			for index in range(NUM_QPES_Y_AXIS_TRIBLOCK):
				qpeSequence.append([yPosList[index]+basicY, xPosList[index]+basicX])
			self.listLoopShift(yPosList)
		return qpeSequence

	def writeTriBlockSram(self, validateFlag=False, originalBlockAddr=0x8000, dataSize=0x900, 
		basicX=TRIBLOCK_BASIC[0][1], basicY=TRIBLOCK_BASIC[0][0], data=0):
		if self.noDataTran and validateFlag:
			return
		sourQpeSequence = self.triBlockQpeSequenceGenerate(basicX=basicX, basicY=basicY)
		# self.printInfo("sourQpeSequence: {}".format(sourQpeSequence))
		# data = 0
		lastQpeId = None
		sramBaseAddr = TOTAL_SRAM_SIZE-PE_SRAM_LIMIT
		peData = [0] * NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		for qpeId in sourQpeSequence:
			lastQpeId = qpeId.copy()
			for peIndex in range(NUM_PES_IN_QPE):
				peId = qpeId.copy()
				peId.append(peIndex)
				peDataIndex = ((peId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK + (peId[1]-basicX)) * NUM_PES_IN_QPE + peIndex
				# print("======== {}, {}".format(peDataIndex, data))
				try:
					peData[peDataIndex] = data
				except IndexError as ie:
					print("peDataIndex: {}".format(peDataIndex))
					print("peData: {}".format(peData))
					print("peId: {}".format(peId))
					assert(False)
				# Write
				if not validateFlag:
					counter = 0
					while counter < dataSize:
						if self.noDataTran:
							self.taskToInQueue({TASK_NAME: Task.SRAM_WR, TASK_DESTINATION: peId})
						else:
							self.taskToInQueue({TASK_NAME: Task.SRAM_WR, TASK_DESTINATION: peId, 
												TASK_SRAM_ADDR: originalBlockAddr+counter, 
												TASK_DATA_LEN: NOC_SRAM_BW_BYTES, 
												TASK_DATA: [data]*NOC_SRAM_BW_BYTES})
						self.spiNNaker2ClockGenerator()
						counter += NOC_SRAM_BW_BYTES
				# Validate
				else:
					sramData = self.spiNNaker2.qpeThreadArray[peId[0]][peId[1]].qpe.peArray[peId[2]].\
							sram.sram[originalBlockAddr-sramBaseAddr:originalBlockAddr-sramBaseAddr+dataSize]
					if sramData == [data]*dataSize:
						pass
					else:
						self.errorInfo("QPE[{}][{}]-PE[{}]: FAIL".format(peId[0],peId[1],peId[2]))
				# Update data
				data += 1 
		# Wait Last PE command to be finish
		if not validateFlag:
			clocksToWait = (lastQpeId[0]-basicY + lastQpeId[1]-basicX) * 4 + 3
			for _ in range(clocksToWait):
				self.spiNNaker2ClockGenerator()
			# Print info 
			totalPe = NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
			self.printInfo("Finish writing {} bytes data into {} PE".format(dataSize*totalPe, totalPe))
		return peData

	def dataTriMigrationTasksGenerate(self, originalBlockAddr=0x8000, dataSize=0x900, copiedBlockAddr=None, 
		basicX=TRIBLOCK_BASIC[0][1], basicY=TRIBLOCK_BASIC[0][0]):
		if copiedBlockAddr is None:
			copiedBlockAddr = originalBlockAddr + dataSize
		# Generate migration tasks
		migrationTasks = []
		for yAxisMigrationIndex in range(NUM_QPES_Y_AXIS_TRIBLOCK):
			for xAxisMigrationIndex in range(NUM_QPES_X_AXIS_TRIBLOCK):
				# For the first data reuse, there is no need to migrate data
				if xAxisMigrationIndex == 0 and yAxisMigrationIndex == 0:
					continue
				# Determine Migrate original block or copied block
				loopTime = yAxisMigrationIndex * NUM_QPES_X_AXIS_TRIBLOCK + xAxisMigrationIndex
				# migrateOriginalBlock = True if loopTime % 2 == 1 else False
					# Migrate original block to copied block
				if loopTime % 2 == 1:
					sourceAddr = originalBlockAddr
					destAddr = copiedBlockAddr
					# Migrate copied block to original block
				else:
					sourceAddr = copiedBlockAddr
					destAddr = originalBlockAddr 					
				# Generate migration command
				sourQpeSequence = self.triBlockQpeSequenceGenerate(basicX=basicX, basicY=basicY)
				if xAxisMigrationIndex == 0:
					destQpeSequence = self.triBlockQpeSequenceGenerate(downShift=True, basicX=basicX, basicY=basicY)
				else:
					destQpeSequence = self.triBlockQpeSequenceGenerate(rightShift=True, basicX=basicX, basicY=basicY)
				for qpeIndex in range(len(sourQpeSequence)):
					for peIndex in range(NUM_PES_IN_QPE):
						destination = (sourQpeSequence[qpeIndex]).copy()
						destination.append(peIndex)
						migrateDest = (destQpeSequence[qpeIndex]).copy()
						migrateDest.append(peIndex)
						migrationTask = {TASK_NAME:Task.DATA_MIGRATION, 
											TASK_SRAM_ADDR:sourceAddr,
											TASK_MIGRATION_SIZE:dataSize, 
											TASK_DESTINATION:destination, 
											TASK_MIGRA_SRAM_ADDR:destAddr, 
											TASK_MIGRATION_DESTINATION:migrateDest}
						migrationTasks.append(migrationTask)
		return migrationTasks

	def dataTriMigration(self, originalPeData, originalBlockAddr=0x8000, dataSize=0x900, copiedBlockAddr=None,
		basicX=TRIBLOCK_BASIC[0][1], basicY=TRIBLOCK_BASIC[0][0]):
		if copiedBlockAddr is None:
			copiedBlockAddr = originalBlockAddr + dataSize
		migrationTasks = self.dataTriMigrationTasksGenerate(originalBlockAddr=originalBlockAddr, dataSize=dataSize, 
			basicX=basicX, basicY=basicY)
		numOfTasks = NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		originalPeData = originalPeData.copy()
		copiedPeData = [0] * numOfTasks
		for index in range(NUM_QPES_X_AXIS_TRIBLOCK*NUM_QPES_Y_AXIS_TRIBLOCK-1):
		# for index in range(3):
			self.printInfo("Round {}".format(index))
			# Send 9*4 migration tasks
			migrationSize = 0
			for migrationTask in migrationTasks[index*numOfTasks:(index+1)*numOfTasks]:
				if migrationTask[TASK_SRAM_ADDR] == originalBlockAddr:
					# Copy data from original block to copied block
					copiedPeId = migrationTask[TASK_MIGRATION_DESTINATION]
					originalPeId = migrationTask[TASK_DESTINATION]
					copiedPeIndex = ((copiedPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+copiedPeId[1]-basicX)*NUM_PES_IN_QPE+copiedPeId[2]
					originalPeIndex = ((originalPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+originalPeId[1]-basicX)*NUM_PES_IN_QPE+originalPeId[2]
					try:
						copiedPeData[copiedPeIndex] = originalPeData[originalPeIndex]
					except IndexError as ie:
						print("copiedPeId: {}".format(copiedPeId))
						print("originalPeId: {}".format(originalPeId))
						print("copiedPeIndex: {}".format(copiedPeIndex))
						print("originalPeIndex: {}".format(originalPeIndex))
						print("originalPeData: {}".format(originalPeData))
						print("copiedPeData: {}".format(copiedPeData))
						assert(False)
				else:
					# Copy data from copied block to original block
					copiedPeId = migrationTask[TASK_DESTINATION]
					originalPeId = migrationTask[TASK_MIGRATION_DESTINATION]
					copiedPeIndex = ((copiedPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+copiedPeId[1]-basicX)*NUM_PES_IN_QPE+copiedPeId[2]
					originalPeIndex = ((originalPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+originalPeId[1]-basicX)*NUM_PES_IN_QPE+originalPeId[2]
					try:
						originalPeData[originalPeIndex] = copiedPeData[copiedPeIndex]
					except IndexError as ie:
						print("copiedPeId: {}".format(copiedPeId))
						print("originalPeId: {}".format(originalPeId))
						print("copiedPeIndex: {}".format(copiedPeIndex))
						print("originalPeIndex: {}".format(originalPeIndex))
						print("originalPeData: {}".format(originalPeData))
						print("copiedPeData: {}".format(copiedPeData))
						assert(False)
				migrationSize += migrationTask[TASK_MIGRATION_SIZE]
				self.taskToInQueue(copy.deepcopy(migrationTask))
				self.spiNNaker2ClockGenerator()
			# Wait 9 migration tasks to be finish
			migrationFinishFlags = [False]*numOfTasks
			migrateSourceId = None
			migrateDestId = None
			while True:
				rsp = self.spiNNaker2ClockGenerator()
				if rsp[TASK_NAME] == Task.DATA_MIGRATION_ALL_FINISH:
					migrateSourceId = rsp[TASK_MIGRATION_SOURCE]
					migrateDestId = rsp[TASK_MIGRATION_DESTINATION]
					finishFlagIndex = ((migrateSourceId[0]-basicY) * NUM_QPES_X_AXIS_TRIBLOCK + migrateSourceId[1]-basicX) * \
										NUM_PES_IN_QPE + migrateSourceId[2]
					migrationFinishFlags[finishFlagIndex] = True
					self.printInfo("Finish migrate data from {} to {}".format(migrateSourceId, migrateDestId))
				if not (False in migrationFinishFlags):
					break
			# # After receiving "DATA_MIGRATION_FINISH", still need to wait some clocks, until all task are finished
			# waitClocks = (abs(migrateSourceId[0]-migrateDestId[0]) + abs(migrateSourceId[1]-migrateDestId[1])) * 4 + 3 - 4
			# for _ in range(waitClocks+500):
			# 	self.spiNNaker2ClockGenerator()
			# Print info
			self.printInfo("Finish migrate {} bytes data".format(migrationSize))
		return originalPeData, copiedPeData

	def dataTriMigrationSingleTask(self, originalPeData, originalBlockAddr=0x8000, 
		basicX=TRIBLOCK_BASIC[0][1], basicY=TRIBLOCK_BASIC[0][0]):
		migrationTasks = self.dataTriMigrationTasksGenerate(originalBlockAddr=originalBlockAddr, dataSize=dataSize, 
			basicX=basicX, basicY=basicY)
		numOfTasks = NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		originalPeData = originalPeData.copy()
		copiedPeData = [0] * numOfTasks
		for index in range(NUM_QPES_X_AXIS_TRIBLOCK*NUM_QPES_Y_AXIS_TRIBLOCK-1):
		# for index in range(3):
			self.printInfo("Round {}".format(index))
			# Seperate 9*4 into 3 groups, each row/column as a groups
			taskGroups = []
			for index in range(NUM_QPES_Y_AXIS_TRIBLOCK):
				taskGroups.append([])
			migrateSourceId = migrationTasks[index*numOfTasks][TASK_DESTINATION]
			migrateDestId = migrationTasks[index*numOfTasks][TASK_MIGRATION_DESTINATION]
			if migrateSourceId[0] == migrateDestId[0]:
				rowGroupFlag = True
			else:
				rowGroupFlag = False			
			for migrationTask in migrationTasks[index*numOfTasks:(index+1)*numOfTasks]:
				# Arrange them as groups
				migrateSourceId = migrationTask[TASK_DESTINATION]
				migrateDestId = migrationTask[TASK_MIGRATION_DESTINATION]
				if migrateSourceId[0] == migrateDestId[0]:
					self.customAssert(rowGroupFlag==True, "It should be row group")
					taskGroups[migrateSourceId[0]-basicY].append(migrationTask)
				else:
					self.customAssert(rowGroupFlag==False, "It should not be row group")
					taskGroups[migrateSourceId[1]-basicX].append(migrationTask)
				# Simulate the data change of each block
				if migrationTask[TASK_SRAM_ADDR] == originalBlockAddr:
					# Copy data from original block to copied block
					copiedPeId = migrationTask[TASK_MIGRATION_DESTINATION]
					originalPeId = migrationTask[TASK_DESTINATION]
					copiedPeIndex = ((copiedPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+copiedPeId[1]-basicX)*NUM_PES_IN_QPE+copiedPeId[2]
					originalPeIndex = ((originalPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+originalPeId[1]-basicX)*NUM_PES_IN_QPE+originalPeId[2]
					copiedPeData[copiedPeIndex] = originalPeData[originalPeIndex]
				else:
					# Copy data from copied block to original block
					copiedPeId = migrationTask[TASK_DESTINATION]
					originalPeId = migrationTask[TASK_MIGRATION_DESTINATION]
					copiedPeIndex = ((copiedPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+copiedPeId[1]-basicX)*NUM_PES_IN_QPE+copiedPeId[2]
					originalPeIndex = ((originalPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+originalPeId[1]-basicX)*NUM_PES_IN_QPE+originalPeId[2]
					originalPeData[originalPeIndex] = copiedPeData[copiedPeIndex]
			# Send migration tasks and receive finish signal
			migrationFinishFlags = [False]*numOfTasks
			taskGroupsOutFlags = [True]*NUM_QPES_X_AXIS_TRIBLOCK
			migrationSize = 0
			migrateSourceId = None
			migrateDestId = None
			while True:
				# Put task
				while True in taskGroupsOutFlags:
					groupIndex = taskGroupsOutFlags.index(True)
					taskGroupsOutFlags[groupIndex] = False
					if len(taskGroups[groupIndex]) > 0:
						migrationTask = taskGroups[groupIndex].pop(0)
						migrationSize += migrationTask[TASK_MIGRATION_SIZE]
						self.taskToInQueue(copy.deepcopy(migrationTask))
				# Clock and receive finish signal
				rsp = self.spiNNaker2ClockGenerator()
				if rsp[TASK_NAME] == Task.DATA_MIGRATION_ALL_FINISH:
					migrateSourceId = rsp[TASK_MIGRATION_SOURCE]
					migrateDestId = rsp[TASK_MIGRATION_DESTINATION]
					finishFlagIndex = ((migrateSourceId[0]-basicY) * NUM_QPES_X_AXIS_TRIBLOCK + migrateSourceId[1]-basicX) * \
										NUM_PES_IN_QPE + migrateSourceId[2]
					migrationFinishFlags[finishFlagIndex] = True
					if rowGroupFlag:
						taskGroupsOutFlags[migrateSourceId[0]-basicY] = True
					else:
						taskGroupsOutFlags[migrateSourceId[1]-basicX] = True
					self.printInfo("Finish migrate data from {} to {}".format(migrateSourceId, migrateDestId))
				if not (False in migrationFinishFlags):
					break
			# # After receiving "DATA_MIGRATION_FINISH", still need to wait some clocks, until all task are finished
			# waitClocks = (abs(migrateSourceId[0]-migrateDestId[0]) + abs(migrateSourceId[1]-migrateDestId[1])) * 4 + 3 - 4
			# for _ in range(waitClocks):
			# 	self.spiNNaker2ClockGenerator()
			# Print info
			self.printInfo("Finish migrate {} bytes data".format(migrationSize))
		return originalPeData, copiedPeData

	def dataTriMigrationValidate(self, originalPeData, copiedPeData, blockIndex,
		originalBlockAddr=0x8000, dataSize=0x900, copiedBlockAddr=None, 
		basicX=TRIBLOCK_BASIC[0][1], basicY=TRIBLOCK_BASIC[0][0]):
		if self.noDataTran:
			return
		if copiedBlockAddr is None:
			copiedBlockAddr = originalBlockAddr + dataSize
		sramBaseAddr = TOTAL_SRAM_SIZE-PE_SRAM_LIMIT
		# Validate block data
		for yAxisIndex in range(NUM_QPES_Y_AXIS_TRIBLOCK):
			for xAxisIndex in range(NUM_QPES_X_AXIS_TRIBLOCK):
				for peIndex in range(NUM_PES_IN_QPE):
					peDataIndex = (yAxisIndex * NUM_QPES_X_AXIS_TRIBLOCK + xAxisIndex) * NUM_PES_IN_QPE + peIndex
					if self.triBlockFlag:
						originalData = self.spiNNaker2.triBlockQpeProcessArray[blockIndex].qpeArray[yAxisIndex*NUM_QPES_X_AXIS_TRIBLOCK+xAxisIndex].\
							peArray[peIndex].sram.sram[originalBlockAddr-sramBaseAddr:originalBlockAddr-sramBaseAddr+dataSize]
					else:
						originalData = self.spiNNaker2.qpeThreadArray[yAxisIndex+basicY][xAxisIndex+basicX].qpe.peArray[peIndex].\
							sram.sram[originalBlockAddr-sramBaseAddr:originalBlockAddr-sramBaseAddr+dataSize]
					if originalData == [originalPeData[peDataIndex]]*dataSize:
						pass
					else:
						self.errorInfo("QPE[{}][{}]-PE[{}] original: FAIL".format(yAxisIndex+basicY,xAxisIndex+basicX,peIndex))
						self.errorInfo("Expect originalData in SRAM: {}".format(originalPeData[peDataIndex]))
						self.errorInfo("originalData in SRAM: {}".format(originalData))	
					if self.triBlockFlag:
						copiedData = self.spiNNaker2.triBlockQpeProcessArray[blockIndex].qpeArray[yAxisIndex*NUM_QPES_X_AXIS_TRIBLOCK+xAxisIndex].\
							peArray[peIndex].sram.sram[copiedBlockAddr-sramBaseAddr:copiedBlockAddr-sramBaseAddr+dataSize]
					else:
						copiedData = self.spiNNaker2.qpeThreadArray[yAxisIndex+basicY][xAxisIndex+basicX].qpe.peArray[peIndex].\
							sram.sram[copiedBlockAddr-sramBaseAddr:copiedBlockAddr-sramBaseAddr+dataSize]
					if copiedData == [copiedPeData[peDataIndex]]*dataSize:
						pass
					else:
						self.errorInfo("QPE[{}][{}]-PE[{}] copied: FAIL".format(yAxisIndex+basicY,xAxisIndex+basicX,peIndex))
						self.errorInfo("Expect copiedData in SRAM: {}".format(copiedPeData[peDataIndex]))
						self.errorInfo("copiedData in SRAM: {}".format(copiedData))					

	def dataTriMigrationSimulation(self, basicX, basicY):
		# basicX = TRIBLOCK_BASIC[3][1]
		# basicY = TRIBLOCK_BASIC[3][0]
		self.printInfo("SpiNNaker2 START")
		# Write data into QPE
		peData = self.writeTriBlockSram(basicX=basicX, basicY=basicY)
		# Data tri-migration
		originalPeData, copiedPeData = self.dataTriMigration(peData, basicX=basicX, basicY=basicY)
		# originalPeData, copiedPeData = self.dataTriMigrationSingleTask(peData, basicX=basicX, basicY=basicY)
		# Stop SpiNNaker2
		self.spiNNakerThreadStop()
		self.printInfo("SpiNNaker2 STOP")
		# Vlidate tri-migrated data
		self.dataTriMigrationValidate(originalPeData, copiedPeData, blockIndex=0, basicX=basicX, basicY=basicY)
		# self.writeTriBlockSram(validateFlag=True, basicX=basicX, basicY=basicY)
		if self.noDataTran:
			self.printInfo("----> Finish Validation: Validation result not avaliable <----")
		else:
			self.printInfo("----> Finish Validation: PASS if no FAILED/ERROR Infos <----")

	# =========================================================================
	# 				Noc Data Tri-Migration (Data reuse outside block)
 	# =========================================================================
	def qpePairBetweenTriBlocks(self, up=False, right=False):
		basicX = 0
		basicY = 0
		if right:
			basicX = 3
		if up:
			basicY = 3
		yPosList = list(range(NUM_QPES_Y_AXIS_TRIBLOCK))
		xPosList = list(range(NUM_QPES_X_AXIS_TRIBLOCK))
		qpeSequence = []
		for yPos in yPosList:
			for xPos in xPosList:
				qpeSequence.append([yPos+basicY,xPos+basicX])
		return qpeSequence

	def dataTriMigraTasksBetweenBlocksGenerator(self, up=False, down=False, left=False, right=False,
		originalBlockAddr=0x8000, dataSize=0x900, copiedBlockAddr=None):
		'''
		up: 	the two upper blocks of spiNNaker
		down:	the two under blocks of spiNNaker
		left:	the two left side blocks of spiNNaker
		right:	the two right side blocks of spiNNaker
		'''
		if copiedBlockAddr is None:
			copiedBlockAddr = originalBlockAddr + dataSize
		self.customAssert([up,down,left,right].count(True)==1, "Only one of them could be True")
		# Generate source and destination qpe sequence
		if up:
			sourQpeSequence = self.qpePairBetweenTriBlocks(up=True)
			destQpeSequence = self.qpePairBetweenTriBlocks(up=True, right=True)
		if down:
			sourQpeSequence = self.qpePairBetweenTriBlocks()
			destQpeSequence = self.qpePairBetweenTriBlocks(right=True)
		if left:
			sourQpeSequence = self.qpePairBetweenTriBlocks()
			destQpeSequence = self.qpePairBetweenTriBlocks(up=True)
		if right:
			sourQpeSequence = self.qpePairBetweenTriBlocks(right=True)
			destQpeSequence = self.qpePairBetweenTriBlocks(up=True, right=True)
		# 
		sourMigrationTasks = []
		destMigrationTasks = []
		for qpeIndex in range(len(sourQpeSequence)):
			for peIndex in range(NUM_PES_IN_QPE):
				destination = (sourQpeSequence[qpeIndex]).copy()
				destination.append(peIndex)
				migrateDest = (destQpeSequence[qpeIndex]).copy()
				migrateDest.append(peIndex)
				migrationTask = {TASK_NAME:Task.DATA_MIGRATION, 
									TASK_SRAM_ADDR:copiedBlockAddr,
									TASK_MIGRATION_SIZE:dataSize, 
									TASK_DESTINATION:destination, 
									TASK_MIGRA_SRAM_ADDR:originalBlockAddr, 
									TASK_MIGRATION_DESTINATION:migrateDest}
				sourMigrationTasks.append(copy.deepcopy(migrationTask))
				migrationTask[TASK_DESTINATION] = migrateDest
				migrationTask[TASK_MIGRATION_DESTINATION] = destination
				destMigrationTasks.append(copy.deepcopy(migrationTask))
		return sourMigrationTasks, destMigrationTasks

	def dataTriMigraBetweenBlocks(self, originalPeData, copiedPeData, upDown=False, leftRight=False, 
		originalBlockAddr=0x8000, dataSize=0x900, copiedBlockAddr=None):
		'''
		upDown: data migration direction between blocks -> up and down
		leftRight: data migration direction between blocks -> left and right
		'''
		if copiedBlockAddr is None:
			copiedBlockAddr = originalBlockAddr + dataSize
		self.customAssert(False in [upDown, leftRight] and True in [upDown, leftRight], "Not support")
		if upDown:
			sourMigrationTasks1, destMigrationTasks1 = self.dataTriMigraTasksBetweenBlocksGenerator(left=True, 
				originalBlockAddr=originalBlockAddr, dataSize=dataSize, copiedBlockAddr=copiedBlockAddr)
			sourMigrationTasks2, destMigrationTasks2 = self.dataTriMigraTasksBetweenBlocksGenerator(right=True, 
				originalBlockAddr=originalBlockAddr, dataSize=dataSize, copiedBlockAddr=copiedBlockAddr)
			originalPeData[0] = copiedPeData[2].copy()
			originalPeData[1] = copiedPeData[3].copy()
			originalPeData[2] = copiedPeData[0].copy()
			originalPeData[3] = copiedPeData[1].copy()
		else:
			sourMigrationTasks1, destMigrationTasks1 = self.dataTriMigraTasksBetweenBlocksGenerator(up=True, 
				originalBlockAddr=originalBlockAddr, dataSize=dataSize, copiedBlockAddr=copiedBlockAddr)
			sourMigrationTasks2, destMigrationTasks2 = self.dataTriMigraTasksBetweenBlocksGenerator(down=True, 
				originalBlockAddr=originalBlockAddr, dataSize=dataSize, copiedBlockAddr=copiedBlockAddr)
			originalPeData[0] = copiedPeData[1].copy()
			originalPeData[1] = copiedPeData[0].copy()
			originalPeData[2] = copiedPeData[3].copy()
			originalPeData[3] = copiedPeData[2].copy()
		# Send source migration task
		for index in range(len(sourMigrationTasks1)):
			# print("sourMigrationTasks1[index]: {}".format(sourMigrationTasks1[index]))
			# print("sourMigrationTasks2[index]: {}".format(sourMigrationTasks2[index]))
			self.taskToInQueue(sourMigrationTasks1[index])
			self.taskToInQueue(sourMigrationTasks2[index])
		task1FinishFlag = [False] * NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		task2FinishFlag = [False] * NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		# Wait source migration task finish
		while True:
			rsp = self.spiNNaker2ClockGenerator()
			if rsp[TASK_NAME] == Task.DATA_MIGRATION_ALL_FINISH:
				migrateSourceId = rsp[TASK_MIGRATION_SOURCE]
				migrateDestId = rsp[TASK_MIGRATION_DESTINATION]
				self.printInfo("Finish source migrate data between blocks from {} to {}".format(migrateSourceId, migrateDestId))
				if upDown:
					# Block 3
					if migrateDestId[1] < NUM_QPES_X_AXIS_HALF:
						finishIndex = (migrateDestId[0] - NUM_QPES_Y_AXIS_HALF) * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1]
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task1FinishFlag[finishIndex] = True
					# Block 4
					else:
						finishIndex = (migrateDestId[0] - NUM_QPES_Y_AXIS_HALF) * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1]
						finishIndex = finishIndex - NUM_QPES_X_AXIS_HALF
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task2FinishFlag[finishIndex] = True				
				else:
					# Block 2
					if migrateDestId[0] < NUM_QPES_Y_AXIS_HALF:
						finishIndex = migrateDestId[0] * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1] - NUM_QPES_X_AXIS_HALF
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task1FinishFlag[finishIndex] = True						
					# Block 4
					else:
						finishIndex = (migrateDestId[0] - NUM_QPES_Y_AXIS_HALF) * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1]
						finishIndex = finishIndex - NUM_QPES_X_AXIS_HALF
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task2FinishFlag[finishIndex] = True	
			if not (False in task1FinishFlag) and not (False in task2FinishFlag):
				break	
		# Send destination migration task
		for index in range(len(destMigrationTasks1)):
			self.taskToInQueue(destMigrationTasks1[index])
			self.taskToInQueue(destMigrationTasks2[index])
		task1FinishFlag = [False] * NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		task2FinishFlag = [False] * NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		# Wait destination migration task finish
		while True:
			rsp = self.spiNNaker2ClockGenerator()
			if rsp[TASK_NAME] == Task.DATA_MIGRATION_ALL_FINISH:
				migrateSourceId = rsp[TASK_MIGRATION_SOURCE]
				migrateDestId = rsp[TASK_MIGRATION_DESTINATION]
				self.printInfo("Finish destination migrate data between blocks from {} to {}".format(migrateSourceId, migrateDestId))
				if upDown:
					# Block 1
					if migrateDestId[1] < NUM_QPES_X_AXIS_HALF:
						finishIndex = migrateDestId[0] * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1]
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task1FinishFlag[finishIndex] = True
					# Block 2
					else:
						finishIndex = migrateDestId[0] * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1] - NUM_QPES_X_AXIS_HALF
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task2FinishFlag[finishIndex] = True				
				else:
					# Block 1
					if migrateDestId[0] < NUM_QPES_Y_AXIS_HALF:
						finishIndex = migrateDestId[0] * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1]
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task1FinishFlag[finishIndex] = True					
					# Block 3
					else:
						finishIndex = (migrateDestId[0] - NUM_QPES_Y_AXIS_HALF) * NUM_QPES_X_AXIS_TRIBLOCK + migrateDestId[1]
						finishIndex = finishIndex * NUM_PES_IN_QPE + migrateDestId[2]
						task2FinishFlag[finishIndex] = True	
			if not (False in task1FinishFlag) and not (False in task2FinishFlag):
				break

	def dataTriMigrationInAllBlocks(self, originalPeData, originalBlockAddr=0x8000, dataSize=0x900, copiedBlockAddr=None):
		originalPeData = copy.deepcopy(originalPeData)
		copiedPeData = copy.deepcopy(originalPeData)
		if copiedBlockAddr is None:
			copiedBlockAddr = originalBlockAddr + dataSize
		migrationTasks = []
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			migrationTasksSingleBlock = self.dataTriMigrationTasksGenerate(originalBlockAddr=originalBlockAddr, dataSize=dataSize, 
				basicX=TRIBLOCK_BASIC[blockIndex][1], basicY=TRIBLOCK_BASIC[blockIndex][0])
			migrationTasks.append(migrationTasksSingleBlock)

		numOfTasks = NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE
		for index in range(NUM_QPES_X_AXIS_TRIBLOCK*NUM_QPES_Y_AXIS_TRIBLOCK-1):
			self.printInfo("Round {} for 4 Blocks".format(index))
			# Send 9*4 migration tasks * 4 Blocks
			migrationSize = 0
			for blockIndex in range(NUM_OF_TRIBLOCKS):
				basicX = TRIBLOCK_BASIC[blockIndex][1]
				basicY = TRIBLOCK_BASIC[blockIndex][0]
				for migrationTask in migrationTasks[blockIndex][index*numOfTasks:(index+1)*numOfTasks]:
					if migrationTask[TASK_SRAM_ADDR] == originalBlockAddr:
						# Copy data from original block to copied block
						copiedPeId = migrationTask[TASK_MIGRATION_DESTINATION]
						originalPeId = migrationTask[TASK_DESTINATION]
						# print("originalPeId: {}".format(originalPeId))
						copiedPeIndex = ((copiedPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+copiedPeId[1]-basicX)*NUM_PES_IN_QPE+copiedPeId[2]
						originalPeIndex = ((originalPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+originalPeId[1]-basicX)*NUM_PES_IN_QPE+originalPeId[2]
						try:
							copiedPeData[blockIndex][copiedPeIndex] = originalPeData[blockIndex][originalPeIndex]
						except IndexError as ie:
							print("copiedPeId: {}".format(copiedPeId))
							print("originalPeId: {}".format(originalPeId))
							print("copiedPeIndex: {}".format(copiedPeIndex))
							print("originalPeIndex: {}".format(originalPeIndex))
							print("originalPeData: {}".format(originalPeData))
							print("copiedPeData: {}".format(copiedPeData))
							assert(False)
					else:
						# Copy data from copied block to original block
						copiedPeId = migrationTask[TASK_DESTINATION]
						originalPeId = migrationTask[TASK_MIGRATION_DESTINATION]
						copiedPeIndex = ((copiedPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+copiedPeId[1]-basicX)*NUM_PES_IN_QPE+copiedPeId[2]
						originalPeIndex = ((originalPeId[0]-basicY)*NUM_QPES_X_AXIS_TRIBLOCK+originalPeId[1]-basicX)*NUM_PES_IN_QPE+originalPeId[2]
						try:
							originalPeData[blockIndex][originalPeIndex] = copiedPeData[blockIndex][copiedPeIndex]
						except IndexError as ie:
							print("copiedPeId: {}".format(copiedPeId))
							print("originalPeId: {}".format(originalPeId))
							print("copiedPeIndex: {}".format(copiedPeIndex))
							print("originalPeIndex: {}".format(originalPeIndex))
							print("originalPeData: {}".format(originalPeData))
							print("copiedPeData: {}".format(copiedPeData))
							assert(False)
					migrationSize += migrationTask[TASK_MIGRATION_SIZE]
					self.taskToInQueue(copy.deepcopy(migrationTask))
					# self.spiNNaker2ClockGenerator()
			# Wait 9*4 migration tasks to be finish * 4 Blocks
			self.printInfo("----> migrationSize: {}".format(migrationSize))
			migrationFinishFlags = [False]*numOfTasks*NUM_OF_TRIBLOCKS
			while True:
				rsp = self.spiNNaker2ClockGenerator()
				if rsp[TASK_NAME] == Task.DATA_MIGRATION_ALL_FINISH:
					migrateSourceId = rsp[TASK_MIGRATION_SOURCE]
					migrateDestId = rsp[TASK_MIGRATION_DESTINATION]
					finishFlagIndex = (migrateSourceId[0] * NUM_QPES_X_AXIS + migrateSourceId[1]) * \
										NUM_PES_IN_QPE + migrateSourceId[2]
					migrationFinishFlags[finishFlagIndex] = True
					# print("finishFlagIndex-{}: {}".format(migrateSourceId, finishFlagIndex))
					self.printInfo("Finish migrate data from {} to {}".format(migrateSourceId, migrateDestId))
					# print("migrationFinishFlags: {}".format(migrationFinishFlags))
				if not (False in migrationFinishFlags):
					break
			self.printInfo("Finish migrate {} bytes data".format(migrationSize))
		return originalPeData, copiedPeData

	def getBlockIndex(self, peID):
		yId = peID[0]
		xId = peID[1]
		if xId < NUM_QPES_X_AXIS_HALF and yId < NUM_QPES_Y_AXIS_HALF:
			return 0
		elif xId >= NUM_QPES_X_AXIS_HALF and yId < NUM_QPES_Y_AXIS_HALF:
			return 1
		elif xId < NUM_QPES_X_AXIS_HALF and yId >= NUM_QPES_Y_AXIS_HALF:
			return 2
		elif xId >= NUM_QPES_X_AXIS_HALF and yId >= NUM_QPES_Y_AXIS_HALF:
			return 3
		else:
			self.customAssert(False, "Unkown X and Y axis of qpeId")

	def dataTriMigrationBetweenBlocksSimulation(self, originalBlockAddr, dataSize):
		self.printInfo("SpiNNaker2 START")

		self.printInfo("Write data into All PEs - 4 blocks")
		data = 0
		peData = []
		for blockIndex in range(NUM_OF_TRIBLOCKS):
			self.printInfo("Write data into {}th block".format(blockIndex))
			blockPeData = self.writeTriBlockSram(originalBlockAddr=originalBlockAddr, dataSize=dataSize, 
				basicX=TRIBLOCK_BASIC[blockIndex][1], basicY=TRIBLOCK_BASIC[blockIndex][0], data=data)
			peData.append(blockPeData)
			data += NUM_QPES_X_AXIS_TRIBLOCK * NUM_QPES_Y_AXIS_TRIBLOCK * NUM_PES_IN_QPE

		originalPeData = copy.deepcopy(peData)
		for loopIndex in range(NUM_OF_TRIBLOCKS):
			self.printInfo("Data Migration inside 4 blocks")
			originalPeData, copiedPeData = self.dataTriMigrationInAllBlocks(originalPeData, 
				originalBlockAddr=originalBlockAddr, dataSize=dataSize, copiedBlockAddr=None)
			self.printInfo("4 Blocks-originalData: {}".format(originalPeData))
			self.printInfo("4 Blocks-copiedData: {}".format(copiedPeData))

			if loopIndex < NUM_OF_TRIBLOCKS-1:
				self.printInfo("Data migration between blocks")
				if loopIndex % 2 == 0:
					self.dataTriMigraBetweenBlocks(originalPeData, copiedPeData, upDown=True, leftRight=False,
						originalBlockAddr=originalBlockAddr, dataSize=dataSize)
				else:
					self.dataTriMigraBetweenBlocks(originalPeData, copiedPeData, upDown=False, leftRight=True,
						originalBlockAddr=originalBlockAddr, dataSize=dataSize)

		self.printInfo("SpiNNaker2 STOP")
		self.spiNNakerThreadStop()

		self.printInfo("Data Validation")
		if self.threadFlag:
			for blockIndex in range(NUM_OF_TRIBLOCKS):
				self.printInfo("Data Validation for {}th block".format(blockIndex))		
				self.dataTriMigrationValidate(originalPeData[blockIndex], copiedPeData[blockIndex], blockIndex=blockIndex,
					originalBlockAddr=originalBlockAddr, dataSize=dataSize, 
					basicX=TRIBLOCK_BASIC[blockIndex][1], basicY=TRIBLOCK_BASIC[blockIndex][0])

			if self.noDataTran:
				self.printInfo("----> Finish Validation: Validation result not avaliable <----")
			else:
				self.printInfo("----> Finish Validation: PASS if no FAILED/ERROR Infos <----")
		else:
			self.printInfo("----> Validation: Validation is not supported for process <----")

def timeConverToSecond(currentTime):
	hours = currentTime.hour
	minutes = currentTime.minute
	second = currentTime.second
	totalSecond = (hours * 60 + minutes) * 60 + second
	return totalSecond	

if __name__ == "__main__":	
	nocDataTransferSimulation = NocDataTransferSimulation(noDataTran=True, nocDoubleFreq=True, 
		threadFlag=False, triBlockFlag=True)
	beginTime = timeConverToSecond(datetime.datetime.now())
	nocDataTransferSimulation.dataTriMigrationBetweenBlocksSimulation(originalBlockAddr=0x8000, dataSize=0x900)
	# nocDataTransferSimulation.dataTriMigrationSimulation(0, 0)
	endTime = timeConverToSecond(datetime.datetime.now())
	print("Simulation time: {}".format(endTime - beginTime))
	nocDataTransferSimulation.spiNNakerThreadStop()

	# =============================================================================
	# Note:
	# When using process, it is not supported that to read the data from other 
	# 	process even it is joined.
	# =============================================================================

	# =============================================================================
	# triBlockFlag=False
	# =============================================================================
	# noDataTran=True, nocDoubleFreq=False, threadFlag=True -> simulation time: 234
	# dataSize = 0x900 -> trimigration takes 20796 -> 94717 = 73921
	# Without data migration -> 5199 * 140 = 727860

	# noDataTran=True, nocDoubleFreq=True, threadFlag=True -> simulation time: 158
	# dataSize = 0x900 -> trimigration takes 20796 -> 61274 = 40478  (20796 -> 61179 = 40383)
	# Without data migration -> 5199 * 140 = 727860

	# noDataTran=True, nocDoubleFreq=True, threadFlag=False -> simulation time: 408
	# dataSize = 0x900 -> trimigration takes 20796 -> 68188 = 47392
	# Without data migration -> 5199 * 140 = 727860	

	# noDataTran=False, nocDoubleFreq=True, threadFlag=True -> simulation time: 228
	# dataSize = 0x900 -> trimigration takes 20796 -> 61599 = 40803
	# Without data migration -> 5199 * 140 = 727860	

	# =============================================================================
	# triBlockFlag=True
	# =============================================================================
	# noDataTran=True, nocDoubleFreq=True, threadFlag=False -> simulation time: 71
	# dataSize = 0x900 -> trimigration takes 20796 -> 59343 = 38547
	# Without data migration -> 5199 * 140 = 727860	

	# noDataTran=False, nocDoubleFreq=True, threadFlag=False -> simulation time: 102
	# dataSize = 0x900 -> trimigration takes 20796 -> 59346 = 38550
	# Without data migration -> 5199 * 140 = 727860

	# noDataTran=True, nocDoubleFreq=False, threadFlag=True -> simulation time: 93
	# dataSize = 0x900 -> trimigration takes 20796 -> 93884 = 73088
	# Without data migration -> 5199 * 140 = 727860

	# =============================================================================
	# triBlockFlag=True, with DRAM, 4 Host - No latency
	# ============================================================================= 
	# noDataTran=True, nocDoubleFreq=True, threadFlag=False -> simulation time: 77
	# dataSize = 0x900 -> trimigration takes 20796 -> 59354 = 38558
	# Without data migration -> 5199 * 140 = 727860	

	# =============================================================================
	# triBlockFlag=True, with DRAM, 4 Host - With latency
	# ============================================================================= 
	# noDataTran=True, nocDoubleFreq=True, threadFlag=False -> simulation time: 70
	# dataSize = 0x900 -> trimigration takes 20796 -> 59354 = 40018
	# Without data migration -> 5199 * 140 = 727860	

	# =============================================================================
	# triBlockFlag=True, with DRAM, 4 Host - With latency, SpiNNaker2-Host-Interface
	# ============================================================================= 
	# noDataTran=True, nocDoubleFreq=True, threadFlag=False -> simulation time: 70
	# dataSize = 0x900 -> trimigration takes 20796 -> 60873/60858 = 40077/40062
	# Without data migration -> 5199 * 140 = 727860	

	# =============================================================================
	# triBlockFlag=True, with DRAM, 4 Host - With latency, SpiNNaker2-Host-Interface with latency
	# ============================================================================= 
	# noDataTran=True, nocDoubleFreq=True, threadFlag=False -> simulation time: 69
	# dataSize = 0x900 -> trimigration takes 20796 -> 61555 = 40759
	# Without data migration -> 5199 * 140 = 727860	