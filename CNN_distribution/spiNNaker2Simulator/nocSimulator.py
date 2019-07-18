from spiNNakerSimulatorGeneral import *


class NoC(GeneralClass):
	'''
	ARM Proccessor
	'''
	def __init__(self, componentId, queueSize=None):
		GeneralClass.__init__(self, componentId, queueSize)
		self.noc1stStage = deque([])
		self.noc2ndStage = deque([])
		self.noc3rdStage = deque([])
		self.noc1stStage.append({TASK_NAME:Task.TASK_NONE})
		self.noc2ndStage.append({TASK_NAME:Task.TASK_NONE})
		self.noc3rdStage.append({TASK_NAME:Task.TASK_NONE})

	def transfer(self):
		# print("===== >>> taskQueue of NoC: {}".format(self.taskQueue))
		task = self.getNextTask()
		return self.fourStagePipeline(task)

	def isForLocalQpe(self, compId):
		if len(compId) < 2:
			return False
		return compId[0:2] == self.componentId[0:2]

	def fourStagePipeline(self, task):
		returnTask = self.noc3rdStage.popleft()
		if Task.TASK_NONE != returnTask[TASK_NAME]:
			self.router(returnTask)
		self.noc3rdStage.append(self.noc2ndStage.popleft())
		self.noc2ndStage.append(self.noc1stStage.popleft())
		if task != None:
			self.noc1stStage.append(task)
		else:
			self.noc1stStage.append({TASK_NAME:Task.TASK_NONE})
		return returnTask

	def router(self, task):
		'''
		Router will only process the task, whose destination is inside local QPE.
		Router will add/update the next destination:
			1. Non-DRAM/Non-HOST: 
			2. HOST:
				next destination is still HOST
			3. DRAM:
				A. route the task to the QPE nearest to DRAM
				B. After arriving the nearest DRAM QPE, set next destionation to be DRAM destination
		'''
		destination = task[TASK_DESTINATION]
		# Add next destination for non-localQpe task
		if not self.isForLocalQpe(destination):
			if len(destination) == 1:
				if not (destination[0] in [0,4,5,6,7]):
					self.customAssert(False, 
						"Destination{} should be one of [0,4,5,6,7]".format(destination))
				# HOST
				# if destination == HOST_ID:
				# 	task[TASK_NEXT_DESTINATION] = HOST_ID
				if destination == HOST_ID:
					if self.hostQpeId() != self.componentId[0:2]:
						task[TASK_NEXT_DESTINATION] = self.generateNextDestination(self.hostQpeId())
					else:
						task[TASK_NEXT_DESTINATION] = destination
				# DRAM
				else:
					if self.nearestDramQpeID(destination) != self.componentId[0:2]:
						task[TASK_NEXT_DESTINATION] = self.generateNextDestination(self.nearestDramQpeID(destination))
					else:
						task[TASK_NEXT_DESTINATION] = destination
			else:
				if not (len(destination)==3):
					self.customAssert(False, "Unsupport destination: {}-{}".format(destination, task))
				# TODO
				# self.customAssert(False, "Unsupport router task for QPE simulation: {}".format(task))
				if destination[0:2] != self.componentId[0:2]:
					task[TASK_NEXT_DESTINATION] = self.generateNextDestination(destination[0:2])
				# else:
				# 	task[TASK_NEXT_DESTINATION] = destination

	def generateNextDestination(self, destination):
		'''
		Follow X axis firstly, then Y axis.
		'''
		source = self.componentId[0:2]
		if source[1] != destination[1]:
			if destination[1] > source[1]:
				return [source[0], source[1]+1]
			else:
				return [source[0], source[1]-1]
		if source[0] != destination[0]:
			if destination[0] > source[0]:
				return [source[0]+1, source[1]]
			else:
				return [source[0]-1, source[1]]
		self.customAssert(False, "Destination and Source should be different.")

	def isIdle(self):
		return len(self.taskQueue) == 0 and len(self.noc1stStage) == 0 and \
				len(self.noc2ndStage) == 0 and len(self.noc3rdStage) == 0


# def generateNextDestination(destination, localqpe):
# 	'''
# 	Follow X axis firstly, then Y axis.
# 	'''
# 	source = localqpe
# 	if source[1] != destination[1]:
# 		if destination[1] > source[1]:
# 			return [source[0], source[1]+1]
# 		else:
# 			return [source[0], source[1]-1]
# 	if source[0] != destination[0]:
# 		if destination[0] > source[0]:
# 			return [source[0]+1, source[1]]
# 		else:
# 			return [source[0]-1, source[1]]
# 	assert(False), "Destination and Source should be different."

# if __name__ == "__main__":
# 	nextDes = [0, 0]
# 	while True:
# 		nextDes = generateNextDestination([5, 5], nextDes)
# 		print("nextDes: {}".format(nextDes))