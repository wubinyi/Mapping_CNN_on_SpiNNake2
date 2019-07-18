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


class QpeProcess(multiprocessing.Process):
	def __init__(self, clockQueue, inQueue, outQueue, threadIdQpeId, nocDoubleFreq=False, noDataTran=True):
		'''
		threadIdQpeId = [y,x]
		'''
		multiprocessing.Process.__init__(self)
		self.nocDoubleFreq = nocDoubleFreq
		self.noDataTran = noDataTran
		self.qpe = QuadPE(threadIdQpeId, nocDoubleFreq=self.nocDoubleFreq, noDataTran=self.noDataTran)
		self.clockQueue = clockQueue
		self.inQueue = inQueue
		self.outQueue = outQueue
		self.threadId = threadIdQpeId
		self.clockCounter = 0

	def customAssert(self, condition, content):
		funcName = sys._getframe().f_code.co_name
		lineNumber = sys._getframe().f_lineno
		fileName = sys._getframe().f_code.co_filename
		callerName = sys._getframe().f_back.f_code.co_name
		assert(condition), "{}-{}(): {}.".format(type(self).__name__, callerName, content)

	def run(self):
		while True:
			clockTask = self.clockQueue.get()
			if Task.TASK_CLOCK == clockTask:
				self.qpeSingleStep()
			else:
				self.customAssert(Task.TASK_NO_CLOCK==clockTask, "Clock task-{} should be TASK_NO_CLOCK".format(clockTask))
				break

	def qpeSingleStep(self):
		# Get out all tasks out of inQueue
		try:
			while True:
				task = self.inQueue.get(block=False)
				self.qpe.commandFrom(task)
		except queue.Empty as emptyExce:
			pass
		# Run QuadPE one time and put task into outQueue if necessary
		rsp = self.qpe.run()
		self.outQueue.put(rsp)
		# if self.nocDoubleFreq:
		# 	for singleRsp in rsp:
		# 		self.outQueue.put(singleRsp)
		# else:
		# 	self.outQueue.put(rsp)