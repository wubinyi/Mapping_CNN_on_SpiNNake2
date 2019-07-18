import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
from spiNNakerSimulatorGeneral import *
from qpeSimulator import QuadPE
from dramControllerSimulator import DramController
from spiNNakerHostInterface import SpiNNakerHostInterface

class QpeDram(GeneralClass):
	def __init__(self, qpeId=[0,0], dramId=[0], nocDoubleFreq=True, printFlag=True, sramHalfFreq=False,
		icproValidateNoti=False, hostInterfaceLatency=10):
		self.nocDoubleFreq = nocDoubleFreq
		self.printFlag = printFlag
		self.sramHalfFreq = sramHalfFreq
		self.icproValidateNoti = icproValidateNoti
		self.hostInterfaceLatency = hostInterfaceLatency
		self.qpe = QuadPE(componentId=qpeId, nocDoubleFreq=self.nocDoubleFreq, noDataTran=True, 
			sramHalfFreq=self.sramHalfFreq, icproValidateNoti=self.icproValidateNoti)
		self.dramController = DramController(componentId=dramId, noDataTran=True)
		self.qdInQueue = deque([])
		self.qdOutQueue = deque([])
		self.qdHostInterface = SpiNNakerHostInterface(HOST_INTERFACE_ID, self.qdInQueue, self.qdOutQueue, 
			latency=self.hostInterfaceLatency)
		self.clockCounter = 0

	def customAssert(self, condition, content):
		# funcName = sys._getframe().f_code.co_name
		# lineNumber = sys._getframe().f_lineno
		# fileName = sys._getframe().f_code.co_filename
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def printInfo(self, content):
		if self.printFlag:
			print("---> {:<7}: {}-{}".format(self.clockCounter, "QpeDram", content))

	def addInTask(self, task):
		'''
		External
		'''
		self.qdHostInterface.addTask(task)

	def getNextOutTask(self):
		'''
		External
		'''
		if len(self.qdOutQueue) == 0:
			return {TASK_NAME: Task.TASK_NONE}
		nextTask = self.qdOutQueue.popleft()
		return nextTask

	def getNextInTask(self):
		'''
		Internal
		'''
		if len(self.qdInQueue) == 0:
			return None
		nextTask = self.qdInQueue.popleft()
		# self.printInfo(nextTask)
		return nextTask

	def addOutTask(self, task):
		'''
		Internal
		'''
		self.qdHostInterface.addTask(task)

	def run(self):
		self.clockCounter += 1
		# Run QpeDram Host Interface
		self.qdHostInterface.transfer()
		# Run Qpe + Dram
		self.qpeDramRun()
		# Processing task from outside of QpeDram
		self.hostTaskDistributor()


	# ==========================================================
	# Distribute HOST task into 
	# ==========================================================
	def hostTaskDistributor(self):
		task = self.getNextInTask()
		if task == None:
			return
		self.qpe.commandFrom(task)

	# ==========================================================
	# Run Qpe + Dram
	# ==========================================================
	def qpeDramRun(self):
		# Run QPE
		rspList = self.qpe.run()
		if not self.nocDoubleFreq:
			rspList = [rspList]
		# Run DRAM
		rspList.append(self.dramController.run())
		# Response Process
		for rsp in rspList:
			if Task.TASK_NONE != rsp[TASK_NAME] and Task.TASK_DOWN != rsp[TASK_NAME]:
				rspDest = rsp[TASK_DESTINATION]
				# -> DRAM/HOST
				if len(rspDest) == 1:
					if rspDest == HOST_ID:
						self.addOutTask(rsp)
					else:
						self.dramController.addTask(rsp)
				# -> QPE
				else:
					self.qpe.commandFrom(rsp)

