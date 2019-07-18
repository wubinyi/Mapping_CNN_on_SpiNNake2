from spiNNakerSimulatorGeneral import *
from collections import deque

class SpiNNakerHostInterface(GeneralClass):
	def __init__(self, componentId, ssInQueue, ssOutQueue, queueSize=None, latency=10):
		GeneralClass.__init__(self, componentId, queueSize)
		self.ssInQueue = ssInQueue
		self.ssOutQueue = ssOutQueue
		self.latency = latency
		self.latencyStageGenerator(self.latency)

	def latencyStageGenerator(self, latency):
		for index in range(latency-1):
			exec('self.taskStage{} = deque([])'.format(index+1))
		# names = self.__dict__
		# for index in range(latency):
		# 	names['taskStage{}'.format(index+1)] = deque([])

	def transfer(self):
		task = self.stagePipeline()
		if task == None:
			return
		else:
			destination = task[TASK_DESTINATION]
			if self.isHostId(destination):
				self.ssOutQueue.append(task)
			else:
				self.ssInQueue.append(task)

	def stagePipeline(self):
		# print("names: {}".format(names))
		if self.latency > 1:
			variables = self.__dict__
			# Last stage
			if len(variables["taskStage"+str(self.latency-1)]) > 0:
				outTask = variables["taskStage"+str(self.latency-1)].popleft()
			else:
				outTask = None
			# 
			for index in range(self.latency-2, 0, -1):
				if len(variables["taskStage"+str(index)]) > 0:
					variables["taskStage"+str(index+1)].append(variables["taskStage"+str(index)].popleft())
			# First stage
			task = self.getNextTask()
			if task != None:
				self.taskStage1.append(task)
			return outTask
		else:
			return self.getNextTask()
		


if __name__ == "__main__":
	ssIn = deque([])
	ssOut = deque([])
	shInter = SpiNNakerHostInterface(HOST_INTERFACE_ID, ssIn, ssOut)
	# shInter.taskStage1.append(12)
	# print("taskStage1: {}".format(shInter.taskStage1))
	# shInter.stagePipeline()
	shInter.addTask({TASK_NAME:Task.TASK_DOWN, TASK_DESTINATION:HOST_ID})
	shInter.addTask({TASK_NAME:Task.SRAM_WR, TASK_DESTINATION:[3,3,4]})
	for clock in range(20):
		shInter.transfer()
		if len(ssIn) > 0:
			print("--->{}-ssIn: {}".format(clock, ssIn.popleft()))
		if len(ssOut) > 0:
			print("--->{}-ssOut: {}".format(clock, ssOut.popleft()))