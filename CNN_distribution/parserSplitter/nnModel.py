class NNModel(object):
	def __init__(self):
		self.nnModel = {}
		self.layerCounter = 0

	def addLayer(self, layerType, layerParameter, name=None):
		self.layerCounter += 1
		if name == None:
			layerName = str(self.layerCounter)
		else:
			layerName = name
		self.nnModel[layerName] = (layerType, layerParameter)

	def getNNModel(self):
		return self.nnModel