from spiNNakerSimulatorGeneral import *

class DramInterface(GeneralClass):
	'''
	ARM Proccessor
	'''
	def __init__(self, componentId, queueSize=None):
		GeneralClass.__init__(self, componentId, queueSize)
		self.noc1stStage = deque([])
		self.noc2ndStage = deque([])
		self.noc3rdStage = deque([])
		self.noc4thStage = deque([])
		self.noc5thStage = deque([])
		self.noc6thStage = deque([])

	def transfer(self):
		returnTask = {TASK_NAME:Task.TASK_NONE}
		if len(self.noc6thStage) > 0:
			returnTask = self.noc6thStage.popleft()
		if len(self.noc5thStage) > 0:
			self.noc6thStage.append(self.noc5thStage.popleft())
		if len(self.noc4thStage) > 0:
			self.noc5thStage.append(self.noc4thStage.popleft())
		if len(self.noc3rdStage) > 0:
			self.noc4thStage.append(self.noc3rdStage.popleft())
		if len(self.noc2ndStage) > 0:
			self.noc3rdStage.append(self.noc2ndStage.popleft())
		if len(self.noc1stStage) > 0:
			self.noc2ndStage.append(self.noc1stStage.popleft())
		task = self.getNextTask()
		if task != None:
			self.noc1stStage.append(task)
		return returnTask