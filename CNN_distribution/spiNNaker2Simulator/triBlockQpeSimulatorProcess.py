import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
# print("System path: {}".format(sys.path))
from spiNNakerSimulatorGeneral import *
from qpeSimulator import QuadPE
import queue
import multiprocessing
import time
from icproValidate.dataGenerator import convDataSplitter
import numpy as np
import datetime
from dramControllerSimulator import DramController

class TriBlockQpeProcess(multiprocessing.Process):
	def __init__(self, triBlockIndex, clockQueue, inQueue, outQueue, nocDoubleFreq=False, noDataTran=True):
		multiprocessing.Process.__init__(self)
		self.triBlockIndex = triBlockIndex
		self.nocDoubleFreq = nocDoubleFreq
		self.noDataTran = noDataTran
		self.clockQueue = clockQueue
		self.inQueue = inQueue
		self.outQueue = outQueue
		self.clockCounter = 0
		self.triBlockQpeGenerator()
		self.dramController = DramController([self.triBlockIndex+DRAM_ID_START], noDataTran=self.noDataTran)

	def triBlockQpeGenerator(self):
		self.qpeArray = []
		self.triBlockBaseId = TRIBLOCK_BASIC[self.triBlockIndex]
		for yIndex in range(self.triBlockBaseId[0], self.triBlockBaseId[0]+NUM_QPES_Y_AXIS_TRIBLOCK):
			for xIndex in range(self.triBlockBaseId[1], self.triBlockBaseId[1]+NUM_QPES_X_AXIS_TRIBLOCK):
				qpeId = [yIndex, xIndex]
				qpe = QuadPE(qpeId, nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)
				self.qpeArray.append(qpe)

	def customAssert(self, condition, content):
		# funcName = sys._getframe().f_code.co_name
		# lineNumber = sys._getframe().f_lineno
		# fileName = sys._getframe().f_code.co_filename
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def run(self):
		while True:
			clockTask = self.clockQueue.get()
			if Task.TASK_CLOCK == clockTask:
				self.qpeArraySingleStep()
			else:
				self.customAssert(Task.TASK_NO_CLOCK==clockTask, "Clock task-{} should be TASK_NO_CLOCK".format(clockTask))
				break		

	def qpeArraySingleStep(self):
		# Get out all tasks out of inQueue and put them into corresponding QPE
		self.processInQueue()
		# Run All QuadPE one time and get all response
		qpeArrayDramRsp = self.runQpeArray()
		# Run Dram one clock and get response
		self.runDramController(qpeArrayDramRsp)
		# Process responses from QPE-Array
		self.processOutTaskFromQpeArray(qpeArrayDramRsp)


	def processInQueue(self):
		try:
			while True:
				task = self.inQueue.get(block=False)
				taskDest = task[TASK_NEXT_DESTINATION]
				qpeIndex = (taskDest[0]-self.triBlockBaseId[0])*NUM_QPES_X_AXIS_TRIBLOCK + taskDest[1]-self.triBlockBaseId[1]
				self.qpeArray[qpeIndex].commandFrom(task)
		except queue.Empty as emptyExce:
			pass
		except IndexError as ie:
			print("triBlockIndex: {}".format(self.triBlockIndex))
			print("task: {}".format(task))
			print("qpeIndex: {}".format(qpeIndex))
			self.customAssert(False, ie)

	def isForLocalBlockQpe(self, nextDest):
		if len(nextDest) < 2:
			if nextDest == HOST_ID:
				return False
			elif nextDest[0] == self.triBlockIndex+DRAM_ID_START:
				return True
			else:
				self.customAssert(False, "Unsupport next Destination: {}".format(nextDest))
		if nextDest[0] >= self.triBlockBaseId[0] and nextDest[0] < (self.triBlockBaseId[0]+NUM_QPES_Y_AXIS_TRIBLOCK):
			if nextDest[1] >= self.triBlockBaseId[1] and nextDest[1] < (self.triBlockBaseId[1]+NUM_QPES_X_AXIS_TRIBLOCK):
				return True
		return False	

	def runQpeArray(self):
		qpeArrayRsp = []
		for qpeIndex in range(len(self.qpeArray)):
			rspList = self.qpeArray[qpeIndex].run()
			if not self.nocDoubleFreq:
				rspList = [rspList]
			for rsp in rspList:
				if Task.TASK_NONE == rsp[TASK_NAME]:
					continue
				# rspNextDest = rsp[TASK_NEXT_DESTINATION]
				# if self.isForLocalBlockQpe(rspNextDest):
				# 	targetQpeIndex = (rspNextDest[0]-self.triBlockBaseId[0])*NUM_QPES_X_AXIS_TRIBLOCK + \
				# 		rspNextDest[1]-self.triBlockBaseId[1]
				# 	self.qpeArray[targetQpeIndex].commandFrom(rsp)
				# else:
				# 	self.outQueue.put(rsp)
				qpeArrayRsp.append(rsp)
		return qpeArrayRsp

	def runDramController(self, qpeArrayDramRsp):
		dramOutTask = self.dramController.run()
		if Task.TASK_NONE != dramOutTask[TASK_NAME] and Task.TASK_DOWN != dramOutTask[TASK_NAME]:
			qpeArrayDramRsp.append(dramOutTask)

	def processOutTaskFromQpeArray(self, qpeArrayRsp):
		for qpeRsp in qpeArrayRsp:
			rspNextDest = qpeRsp[TASK_NEXT_DESTINATION]
			if self.isForLocalBlockQpe(rspNextDest):
				if len(rspNextDest) < 2:
					self.dramController.addTask(qpeRsp)
				else:
					targetQpeIndex = (rspNextDest[0]-self.triBlockBaseId[0])*NUM_QPES_X_AXIS_TRIBLOCK + \
						rspNextDest[1]-self.triBlockBaseId[1]
					self.qpeArray[targetQpeIndex].commandFrom(qpeRsp)
			else:
				self.outQueue.put(qpeRsp)	
		self.outQueue.put({TASK_NAME:Task.PROCESS_END})		



if __name__ == "__main__":
	for index in range(NUM_OF_TRIBLOCKS):
		triBlockQpe = TriBlockQpeProcess(index, None, None, None, None)
		print("triBlockQpe.triBlockBaseId: {}".format(triBlockQpe.triBlockBaseId))
		for qpe in triBlockQpe.qpeArray:
			print("qpeId: {}".format(qpe.componentId))
		print("------------------------")
