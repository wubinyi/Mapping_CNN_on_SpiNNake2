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

PADDING_32_BIT_CLKS = 2
INT32_RELU_PIXEL_CLKS = 8
INT8_RELU_PIXEL_CLKS = 2.5
INT32_MAXPOOL_PIXEL_CLKS = 75 / 4
INT8_MAXPOOL_PIXEL_CLKS = 48 / 4
INT32_MATRIX_ELE_PIXEL_CLKS = 8
QUAN_PIXEL_CLKS = 8

PADDING = "PADD"
POOLING = "POOL"
ACTIVATION = "ACTI"
QUANTIZATION = "QUAN"
MATRIX_ELEMENT = "MAT_ELE"

class SpiNNaker2DistributorNoncConv(DistributionGeneralClass):
	def __init__(self, reorderFlag=False, printFlag=True, vggFlag=True):
		DistributionGeneralClass.__init__(self, printFlag=printFlag)
		self.reorderFlag = reorderFlag
		self.distriResultsEachLayer = {}
		if vggFlag:
			self.vggModelSplitter()
		else:
			self.resNetModelSplitter()

	def getDistriResultEachLayer(self):
		return self.distriResultsEachLayer

	def vggModelSplitter(self):
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorNoncConv.vgg(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def resNetModelSplitter(self):
		modelLayersSplitInfo, modelLayersParameter = SpiNNaker2DistributorNoncConv.resNet50(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitInfoParameter(modelLayersSplitInfo, modelLayersParameter)

	def distribute(self):
		layerNames = list(self.modelLayersSplitInfo.keys())
		numOfLayers = len(layerNames)
		for layerNameIndex in range(numOfLayers):
			layerName = layerNames[layerNameIndex]
			# Search CONV layer
			if "CONV_" in layerName:
				convSplitInfo = self.modelLayersSplitInfo[layerName]
				convBlockLayerNames = []
				for nextLayerIndex in range(layerNameIndex+1, numOfLayers):
					nextLayerName = layerNames[nextLayerIndex]
					if "PADD_" in nextLayerName or "CONV_" in nextLayerName or "FC_" in nextLayerName:
						break
					if "POOL_" in nextLayerName:
						poolSplitInfo = self.modelLayersSplitInfo[nextLayerName]
						layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = poolSplitInfo
						if weightStrideSplitInfo[0][0] != weightStrideSplitInfo[1]:
							break
					convBlockLayerNames.append(nextLayerName)
				# input("convBlockLayerNames: {}".format(convBlockLayerNames))
				self.getQuanActiPoolDistriResult(convSplitInfo, convBlockLayerNames, layerName)
			# Search POOL layer
			if "POOL_" in layerName:
				poolSplitInfo = self.modelLayersSplitInfo[layerName]
				layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = poolSplitInfo
				if weightStrideSplitInfo[0][0] != weightStrideSplitInfo[1]:
					poolBlockLayerNames = [layerName]
					self.getQuanActiPoolDistriResult(poolSplitInfo, poolBlockLayerNames, layerName)

	def getQuanActiPoolDistriResult(self, convSplitInfo, convBlockLayerNames, convlayerName):
		# print(convSplitInfo)
		# print(convBlockLayerNames)
		# input("")
		layerType, inActiSplitInfo, weightStrideSplitInfo, outActiSplitInfo, clocks, requiredPEs = convSplitInfo
		outActiWidth = outActiSplitInfo[0][0]
		outActiHeight = outActiSplitInfo[1][0]
		outActiChannel = outActiSplitInfo[2][0]
		self.distributeScale = int(math.pow(2, math.ceil(math.log((requiredPEs / 128), 2))))
		# print('{}: {}'.format(convlayerName, distributeScale))
		# Padding layer
		layerNames = list(self.modelLayersSplitInfo.keys())
		paddLayerName = "PADD_"+str(int(convlayerName.split("_")[-1]))
		if paddLayerName in layerNames:
			if isinstance(inActiSplitInfo[2], list):
				inputSize_32bits = self.align16(inActiSplitInfo[0][0]) * inActiSplitInfo[1][0] * inActiSplitInfo[2][0] // 4
			else:
				inputSize_32bits = self.align16(inActiSplitInfo[0][0]) * inActiSplitInfo[1][0] * inActiSplitInfo[2] // 4
			computeClocks = int(inputSize_32bits * PADDING_32_BIT_CLKS)
			self.distriResults[convlayerName] = computeClocks
			self.distriResultsEachLayer[convlayerName] = {}
			self.distriResultsEachLayer[convlayerName][PADDING] = computeClocks
		else:
			self.distriResults[convlayerName] = 0
			self.distriResultsEachLayer[convlayerName] = {}		
		# 
		if self.reorderFlag:
			self.getClockWithReorder(outActiWidth, outActiHeight, outActiChannel, convlayerName, convBlockLayerNames)
		else:
			# Get MAT_ELE_ layer name, 32 bits
			for layerName in convBlockLayerNames:
				if "MAT_ELE_" in layerName:
					self.getClocks(outActiWidth, outActiHeight, outActiChannel, INT32_MATRIX_ELE_PIXEL_CLKS, convlayerName, MATRIX_ELEMENT)
					break
			# Get ACTI_ layer name, 32 bits
			for layerName in convBlockLayerNames:
				if "ACTI_" in layerName:
					self.getClocks(outActiWidth, outActiHeight, outActiChannel, INT32_RELU_PIXEL_CLKS, convlayerName, ACTIVATION)	
					break
			# Get QUAN_ layer name, 32 bits
			for layerName in convBlockLayerNames:
				if "QUAN_" in layerName:
					self.getClocks(outActiWidth, outActiHeight, outActiChannel, QUAN_PIXEL_CLKS, convlayerName, QUANTIZATION)
					break
			# Get POOL_ layer name, 8 bits
			for layerName in convBlockLayerNames:
				if "POOL_" in layerName:
					self.getClocks(outActiWidth, outActiHeight, outActiChannel, INT8_MAXPOOL_PIXEL_CLKS, convlayerName, POOLING)
					outActiWidth = outActiWidth // 2
					outActiHeight = outActiHeight // 2			
					break

	def getClockWithReorder(self, outActiWidth, outActiHeight, outActiChannel, blockName, convBlockLayerNames):
		reorderClockResults1 = 0
		reorderClockResultsEachLayer1 = {}
		outActiWidth1 = outActiWidth
		outActiHeight1 = outActiHeight
		int8Flag = False
		# Get MAT_ELE_ layer name, 32 bits
		for layerName in convBlockLayerNames:
			if "MAT_ELE_" in layerName:
				inputPixels = outActiWidth1 * outActiHeight1 * outActiChannel
				computeClocks = int(inputPixels * INT32_MATRIX_ELE_PIXEL_CLKS) * self.distributeScale
				reorderClockResults1 += computeClocks
				reorderClockResultsEachLayer1[MATRIX_ELEMENT] = computeClocks
				break
		# Get QUAN_ layer name, 32 bit
		for layerName in convBlockLayerNames:
			if "QUAN_" in layerName:
				inputPixels = outActiWidth1 * outActiHeight1 * outActiChannel
				computeClocks = int(inputPixels * QUAN_PIXEL_CLKS) * self.distributeScale
				reorderClockResults1 += computeClocks
				reorderClockResultsEachLayer1[QUANTIZATION] = computeClocks
				int8Flag = True
				break
		# Get POOL_ layer name, 32 bit
		for layerName in convBlockLayerNames:
			if "POOL_" in layerName:
				inputPixels = outActiWidth1 * outActiHeight1 * outActiChannel
				if int8Flag:
					computeClocks = int(inputPixels * INT8_MAXPOOL_PIXEL_CLKS) * self.distributeScale
				else:
					computeClocks = int(inputPixels * INT32_MAXPOOL_PIXEL_CLKS) * self.distributeScale
				reorderClockResults1 += computeClocks
				reorderClockResultsEachLayer1[POOLING] = computeClocks
				outActiWidth1 = outActiWidth1 // 2
				outActiHeight1 = outActiHeight1 // 2
				break			
		# Get ACTI_ layer name
		for layerName in convBlockLayerNames:
			if "ACTI_" in layerName:
				inputPixels = outActiWidth1 * outActiHeight1 * outActiChannel
				if int8Flag:
					computeClocks = int(inputPixels * INT8_RELU_PIXEL_CLKS) * self.distributeScale
				else:
					computeClocks = int(inputPixels * INT32_RELU_PIXEL_CLKS) * self.distributeScale
				reorderClockResults1 += computeClocks
				reorderClockResultsEachLayer1[ACTIVATION] = computeClocks
				break
		# if "POOL_2" == blockName:
		# 	input("POOL_2: {}".format(reorderClockResults1))
		# 	input("POOL_2: {}".format(reorderClockResultsEachLayer1))
		# 	input("POOL_2: {}-{}".format(outActiWidth1, outActiHeight1))
		#
		reorderClockResults2 = 0
		reorderClockResultsEachLayer2 = {}
		outActiWidth2 = outActiWidth
		outActiHeight2 = outActiHeight
		int8Flag = False
		# Get MAT_ELE_ layer name, 32 bits
		for layerName in convBlockLayerNames:
			if "MAT_ELE_" in layerName:
				inputPixels = outActiWidth2 * outActiHeight2 * outActiChannel
				computeClocks = int(inputPixels * INT32_MATRIX_ELE_PIXEL_CLKS) * self.distributeScale
				reorderClockResults1 += computeClocks
				reorderClockResultsEachLayer1[MATRIX_ELEMENT] = computeClocks
				break
		# Get POOL_ layer name, 32 bit
		for layerName in convBlockLayerNames:
			if "POOL_" in layerName:
				inputPixels = outActiWidth2 * outActiHeight2 * outActiChannel
				computeClocks = int(inputPixels * INT32_MAXPOOL_PIXEL_CLKS) * self.distributeScale
				reorderClockResults2 += computeClocks
				reorderClockResultsEachLayer2[POOLING] = computeClocks
				outActiWidth2 = outActiWidth2 // 2
				outActiHeight2 = outActiHeight2 // 2
				break
		# Get QUAN_ layer name, 32 bit
		for layerName in convBlockLayerNames:
			if "QUAN_" in layerName:
				inputPixels = outActiWidth2 * outActiHeight2 * outActiChannel
				computeClocks = int(inputPixels * QUAN_PIXEL_CLKS) * self.distributeScale
				reorderClockResults2 += computeClocks
				reorderClockResultsEachLayer2[QUANTIZATION] = computeClocks
				int8Flag = True
				break			
		# Get ACTI_ layer name
		for layerName in convBlockLayerNames:
			if "ACTI_" in layerName:
				inputPixels = outActiWidth2 * outActiHeight2 * outActiChannel
				if int8Flag:
					computeClocks = int(inputPixels * INT8_RELU_PIXEL_CLKS) * self.distributeScale
				else:
					computeClocks = int(inputPixels * INT32_RELU_PIXEL_CLKS) * self.distributeScale
				reorderClockResults2 += computeClocks
				reorderClockResultsEachLayer2[ACTIVATION] = computeClocks
				break
		# if "POOL_2" == blockName:
		# 	input("POOL_2: {}".format(reorderClockResults2))
		# 	input("POOL_2: {}".format(reorderClockResultsEachLayer2))
		# 	input("POOL_2: {}-{}".format(outActiWidth2, outActiHeight2))
		# 
		if reorderClockResults1 >= reorderClockResults2:
			self.distriResults[blockName] += reorderClockResults2
			self.distriResultsEachLayer[blockName] = {**self.distriResultsEachLayer[blockName], **reorderClockResultsEachLayer2}
		else:
			self.distriResults[blockName] += reorderClockResults1
			self.distriResultsEachLayer[blockName] = {**self.distriResultsEachLayer[blockName], **reorderClockResultsEachLayer1}		

	def getClocks(self, outActiWidth, outActiHeight, outActiChannel, clocksPerPixel, blockName, operation):
		inputPixels = outActiWidth * outActiHeight * outActiChannel
		computeClocks = int(inputPixels * clocksPerPixel) * self.distributeScale
		if blockName in self.distriResults:
			self.distriResults[blockName] += computeClocks
			self.distriResultsEachLayer[blockName][operation] = computeClocks
		else:
			self.distriResults[blockName] = computeClocks
			self.distriResultsEachLayer[blockName] = {}
			self.distriResultsEachLayer[blockName][operation] = computeClocks

if __name__ == "__main__":
	sdnc = SpiNNaker2DistributorNoncConv(reorderFlag=False, vggFlag=False)
	sdnc.distribute()
	resultsWoReorder = sdnc.getDistriResults()
	resultsEachLayerWoReorder = sdnc.getDistriResultEachLayer()

	# sdnc = SpiNNaker2DistributorNoncConv(reorderFlag=True, vggFlag=True)
	# sdnc.distribute()
	# resultsReorder = sdnc.getDistriResults()
	# resultsEachLayerReorder = sdnc.getDistriResultEachLayer()

	print("Distribution result without reorder: \n{}\n".format(resultsWoReorder))
	print("Distribution result without reorder (Each layer): \n{}\n".format(resultsEachLayerWoReorder))
	# print("Distribution result with reorder: \n{}\n".format(resultsReorder))
	# print("Distribution result with reorder (Each layer): \n{}\n".format(resultsEachLayerReorder))