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

from nnGeneral import BasicOperation
from spiNNakerSimulatorGeneral import *
from convLayerMapper import ConvLayerMapper
from rooflineModel import AcceleratorType, ROOFLINE_FORMAT, RooflineModel, LINE_NAME1
from distributionGeneral import DistributionGeneralClass
from fcLayerMapper import FcLayerMapper

VGG_CONV_NO_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 2294223,
	# "CONV_4": 5828888,
	"CONV_5": 16207531,
	# "CONV_7": 6247798,
	"CONV_9": 12616201,
	"CONV_11": 6369574,
	"CONV_15": 3261186
}

VGG_CONV_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 835667,
	# "CONV_4": 5572513,
	"CONV_5": 11050716,
	# "CONV_7": 6128823,
	"CONV_9": 12417980,
	"CONV_11": 6262524,
	"CONV_15": 3254902
}

VGG_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 1065837,
	# "CONV_4": 4990816,
	"CONV_5": 9697264,
	# "CONV_7": 5449664,
	"CONV_9": 10838832,
	"CONV_11": 5419584,
	"CONV_15": 2918496
}

VGG_CONV_DIFF_DISTRI_SCHEME_RESULTS = [VGG_CONV_NO_FUSION_NO_DATA_REUSE_RESULT, VGG_CONV_FUSION_NO_DATA_REUSE_RESULT, 
											VGG_CONV_FUSION_DATA_REUSE_RESULT]


VGG_FC_NO_FUSION_RESULT = {
	# "FC_19": 13252510,
	"FC_21": 546718
}

VGG_FC_FUSION_RESULT = {
	# "FC_19": 12901950,
	"FC_21": 532670
}

VGG_FC_DIFF_DISTRI_RESULTS = [VGG_FC_NO_FUSION_RESULT, VGG_FC_FUSION_RESULT]




RESNET_CONV_NO_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1" : 	2402170,
	"CONV_5":	2246963,
	"CONV_7":	813317,
	"CONV_SC_14":	15341673,
	"CONV_19": 	806278,
	"CONV_25": 	761451,
	"CONV_26":	422420,
	"CONV_42":	4145252,
	"CONV_43": 	1644412
}

RESNET_CONV_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 	2304196,
	"CONV_5":	1717913,
	"CONV_7":	785374,
	"CONV_SC_14":	13356221,
	"CONV_19": 	788695,
	"CONV_25": 	753440,
	"CONV_26":	383365,
	"CONV_42":	3675771,
	"CONV_43": 	1527487
}

RESNET_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1":  	2320116,
	"CONV_5":	471968,
	"CONV_7":	716836,
	"CONV_SC_14":	2722496,
	"CONV_19": 	700808,
	"CONV_25": 	739465,
	"CONV_26":	369810,
	"CONV_42":	879624,
	"CONV_43": 	1598560
}
RESNET_CONV_DIFF_DISTRI_SCHEME_RESULTS = [RESNET_CONV_NO_FUSION_NO_DATA_REUSE_RESULT, RESNET_CONV_FUSION_NO_DATA_REUSE_RESULT, 
											RESNET_CONV_FUSION_DATA_REUSE_RESULT]

RESNET_FC_NO_FUSION_RESULT = {
	"FC_52": 276382
}

RESNET_FC_FUSION_RESULT = {
	"FC_52": 269502
}
RESNET_FC_DIFF_DISTRI_RESULTS = [RESNET_FC_NO_FUSION_RESULT, RESNET_FC_FUSION_RESULT]

class QpeDramRoofline():
	def __init__(self, printFlag=True, considerOutActi=False, isConv=True, vggFlag=True):
		self.printFlag = printFlag
		self.considerOutActi = considerOutActi
		self.isConv = isConv
		self.vggFlag = vggFlag

	def customAssert(self, condition, content):
		assert(condition), "{}: {}.".format(type(self).__name__, content)

	def updateModelSplitParameter(self, modelLayerSplitInfo):
		self.modelLayerSplitInfo = modelLayerSplitInfo

	def writeLogFile(self, context):
		if self.printFlag:
			BasicOperation.customPrintT(context)

	# ==================================================================================================
	# 									CONV Layer Distribution
	# ==================================================================================================
	def getConvLayersRfParamDiffScheme(self):
		# Get "operation per Byte" from all layers
		# self.layersOpesPerByte = self.getConvLayersOpsPerByte()
		# Generate roofline parameter "operation per Byte" and "operation per second"
		print("\nCONV roofline parameter")
		diffSchemeRfInfo = []
		if self.vggFlag:
			convDistributionResults = VGG_CONV_DIFF_DISTRI_SCHEME_RESULTS
		else:
			convDistributionResults = RESNET_CONV_DIFF_DISTRI_SCHEME_RESULTS
		for distriResult in convDistributionResults:
			rfInfo = {}
			for layerName, layerTotalClks in distriResult.items():
				opsPerByte, operations = self.layersOpesPerByte[layerName]
				scale = PE_FREQ / layerTotalClks
				opsPerSec = operations * scale / MEGA_TO_GIGA
				rfInfo[layerName] = (opsPerByte, opsPerSec)
			diffSchemeRfInfo.append(rfInfo)
			print("rfInfo: {}".format(rfInfo))
		return diffSchemeRfInfo

	def getConvLayersOpsPerByte(self):
		for layerName, layerSplitInfo in self.modelLayerSplitInfo.items():
			if "CONV_" not in layerName:
				continue
			self.layersOpesPerByte[layerName] = self.getSingleConvLayerOpsPerByte(layerSplitInfo)

	def getSingleConvLayerOpsPerByte(self, layerSplitInfo):
		layerType, inActiDim, weightDim, outActiDim, clocks, requiredPEs = layerSplitInfo
		outActiWidth = BasicOperation.splitInfoIntegration(outActiDim[0])
		outActiHeight = BasicOperation.splitInfoIntegration(outActiDim[1])
		filterWidth = BasicOperation.splitInfoIntegration(weightDim[0][0])
		filterHeight = BasicOperation.splitInfoIntegration(weightDim[0][1])
		filterInChannel = BasicOperation.splitInfoIntegration(weightDim[0][2])
		filterOutChannel = BasicOperation.splitInfoIntegration(weightDim[0][3])
		stride = weightDim[1]
		inActiWidth = ConvLayerMapper.outToIn(outActiWidth, stride, filterWidth)
		inActiHeight = ConvLayerMapper.outToIn(outActiHeight, stride, filterHeight)
		inActiChannel = BasicOperation.splitInfoIntegration(inActiDim[2])
		self.customAssert(inActiChannel==filterInChannel, "Input Channel unmatch: {}".format(layerSplitInfo))
		dataSize = inActiWidth * inActiHeight * inActiChannel + filterWidth * filterHeight * filterInChannel * filterOutChannel
		if self.considerOutActi:
			dataSize += outActiWidth * outActiHeight * filterOutChannel
		operations = ConvLayerMapper.convTheoreticalClocks(inActiWidth, inActiHeight, inActiChannel, filterWidth, 
			filterHeight, stride, filterOutChannel) * 2
		opsPerByte = operations / dataSize
		return (opsPerByte, operations)

	# ==================================================================================================
	# 									FC Layer Distribution
	# ==================================================================================================
	def getFcLayersRfParamDiffScheme(self):
		# Get "operation per Byte" from all layers
		# self.layersOpesPerByte = self.getFcLayersOpsPerByte()
		# Generate roofline parameter "operation per Byte" and "operation per second"
		print("\nFC roofline parameter")
		diffSchemeRfInfo = []
		if self.vggFlag:
			fcDistributionResults = VGG_FC_DIFF_DISTRI_RESULTS
		else:
			fcDistributionResults = RESNET_FC_DIFF_DISTRI_RESULTS
		for distriResult in fcDistributionResults:
			rfInfo = {}
			for layerName, layerTotalClks in distriResult.items():
				opsPerByte, operations = self.layersOpesPerByte[layerName]
				scale = PE_FREQ / layerTotalClks
				opsPerSec = operations * scale / MEGA_TO_GIGA
				rfInfo[layerName] = (opsPerByte, opsPerSec)
			diffSchemeRfInfo.append(rfInfo)
			print("rfInfo: {}".format(rfInfo))
		return diffSchemeRfInfo

	def getFcLayersOpsPerByte(self):
		for layerName, layerSplitInfo in self.modelLayerSplitInfo.items():
			if "FC_" not in layerName:
				continue
			self.layersOpesPerByte[layerName] = self.getSingleFcLayerOpsPerByte(layerSplitInfo)

	def getSingleFcLayerOpsPerByte(self, layerSplitInfo):
		layerType, inActiSplitInfo, weightSplitInfo, outActiSplitInfo, clocks, requiredPEs = layerSplitInfo
		inActiWidth = BasicOperation.splitInfoIntegration(inActiSplitInfo[0])
		inActiHeight = BasicOperation.splitInfoIntegration(inActiSplitInfo[1])
		weightWidth = BasicOperation.splitInfoIntegration(weightSplitInfo[0])
		weightHeight = BasicOperation.splitInfoIntegration(weightSplitInfo[1])
		outActiWidth = BasicOperation.splitInfoIntegration(outActiSplitInfo[0])
		outActiHeight = BasicOperation.splitInfoIntegration(outActiSplitInfo[1])		
		self.customAssert(inActiWidth == weightHeight and weightWidth == outActiWidth and inActiHeight == outActiHeight, \
			"dimension unmatch")
		dataSize = inActiWidth * inActiHeight + weightWidth * weightHeight
		if self.considerOutActi:
			dataSize += outActiWidth * outActiHeight
		operations = FcLayerMapper.denseTheoreticalClocks(weightHeight, weightWidth) * 2
		opsPerByte = operations / dataSize
		return (opsPerByte, operations)

	# ==================================================================================================
	# 									Roofline Model
	# ==================================================================================================
	def roofline(self):
		roofline = RooflineModel(title="Roofline of QPE-DRAM")
		# CONV+FC without decreasing input-activation-overlap-size
		if self.vggFlag:
			modelLayersSplitInfo, _ = DistributionGeneralClass.vgg(decreaseSize=True)
		else:
			modelLayersSplitInfo, _ = DistributionGeneralClass.resNet50(decreaseSize=True)
		self.updateModelSplitParameter(modelLayersSplitInfo)
		# 
		self.layersOpesPerByte = {}
		self.getConvLayersOpsPerByte()
		self.getFcLayersOpsPerByte()
		for key, value in self.layersOpesPerByte.items():
			print("{}: {}".format(key, value))
		# CONV: Getting Roofline parameter
		rooflineInfoDiffScheme = self.getConvLayersRfParamDiffScheme()
		layerNames = list(rooflineInfoDiffScheme[0].keys())
		opsPerByteList, opsPerSecondDiffSchemeList, improveLineFormat = \
			self.getScattDataForRooflineModel(rooflineInfoDiffScheme)
		roofline.addScatter(opsPerByteList, opsPerSecondDiffSchemeList, 
							ROOFLINE_FORMAT[0:len(opsPerSecondDiffSchemeList)], improveLineFormat, layerNames=layerNames)
		# FC: Getting Roofline parameter
		rooflineInfoDiffScheme = self.getFcLayersRfParamDiffScheme()
		layerNames = list(rooflineInfoDiffScheme[0].keys())
		opsPerByteList, opsPerSecondDiffSchemeList, improveLineFormat = \
			self.getScattDataForRooflineModel(rooflineInfoDiffScheme)
		roofline.addScatter(opsPerByteList, opsPerSecondDiffSchemeList, 
							ROOFLINE_FORMAT[0:len(opsPerSecondDiffSchemeList)], improveLineFormat, labelFlag=False, layerNames=layerNames)
		# Plot Roofline
		roofline.plot()

	def getScattDataForRooflineModel(self, rooflineInfoDiffScheme):
		drawLayerNameList = []
		opsPerByteList = []
		opsPerSecondDiffSchemeList = []
		improveLineFormat = []
		for schemeIndex in range(len(rooflineInfoDiffScheme)):
			opsPerSecondDiffSchemeList.append([])
		layerNameKeys = list(rooflineInfoDiffScheme[0].keys())
		# Loop For different layers
		for index in range(len(layerNameKeys)):
			layerName = layerNameKeys[index]
			# Get opsPerByte and eliminate the layer with same opsPerByte
			layerRooflineInfo = rooflineInfoDiffScheme[0][layerName]
			opsPerByte, opsPerSecond = layerRooflineInfo
			if len(opsPerByteList) > 0:
				if math.ceil(comparedOpsPerByte) != math.ceil(opsPerByte):
					opsPerByteList.append(opsPerByte)
					drawLayerNameList.append(layerName)
					comparedOpsPerByte = opsPerByte
					improveLineFormat.append("k--")
				else:
					opsPerByteList.append(opsPerByteList[-1]*1.01)
					drawLayerNameList.append(layerName)
					improveLineFormat.append("k-")
			else:
				opsPerByteList.append(opsPerByte)
				drawLayerNameList.append(layerName)
				comparedOpsPerByte = opsPerByte
				improveLineFormat.append("k--")
			# Get opsPerSecond along different distributor scheme
			for schemeIndex in range(len(rooflineInfoDiffScheme)):
				layerRooflineInfo = rooflineInfoDiffScheme[schemeIndex][layerName]
				opsPerByte, opsPerSecond = layerRooflineInfo
				if improveLineFormat[-1] == "k--":
					self.customAssert(opsPerByteList[-1]==opsPerByte, "opsPerByte unmatch")
				opsPerSecondDiffSchemeList[schemeIndex].append(opsPerSecond)
		BasicOperation.customPrintF("\n-----drawLayerNameList: {}".format(drawLayerNameList))
		BasicOperation.customPrintF("\n-----opsPerByteList: {}".format(opsPerByteList))
		BasicOperation.customPrintF("\n-----opsPerSecondDiffSchemeList: {}".format(opsPerSecondDiffSchemeList))
		return opsPerByteList, opsPerSecondDiffSchemeList, improveLineFormat


def barPlot(vggFlag=True):
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pylab
	          # 'figure.figsize': (15, 5),
	params = {'legend.fontsize': 'xx-large',
	         'axes.labelsize': 'xx-large',
	         'axes.titlesize':'xx-large',
	         'xtick.labelsize':'xx-large',
	         'ytick.labelsize':'xx-large'}
	pylab.rcParams.update(params)

	textSize = 'larger'
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	if vggFlag:
		fcDistributionResults = VGG_FC_DIFF_DISTRI_RESULTS
		fcDistributionResults.append({"FC_21":0})
		distributionResults = VGG_CONV_DIFF_DISTRI_SCHEME_RESULTS
	else:
		fcDistributionResults = RESNET_FC_DIFF_DISTRI_RESULTS
		fcDistributionResults.append({"FC_52":0})
		distributionResults = RESNET_CONV_DIFF_DISTRI_SCHEME_RESULTS
	for index in range(len(distributionResults)):
		fcResult = fcDistributionResults[index]
		convResult = distributionResults[index]
		distributionResults[index] = {**convResult, **fcResult}
	# print(distributionResults)
	# 
	keys = list(distributionResults[0].keys())
	# print(keys)
	values = []
	for index in range(len(distributionResults)):
		values.append(list(distributionResults[index].values()))
	# print(values)
	elementsInGroup = len(values)
	N = len(keys)
	fig, ax = plt.subplots()
	ind = np.arange(N)*1.5    # the x locations for the groups
	width = 0.35         # the width of the bars
	p = []
	for index in range(elementsInGroup):
		pTemp = ax.bar(ind+width*index, values[index], width, color=colors[index], bottom=0)
		p.append(pTemp)
	# 
	# ax.set_title('Scores by group and gender')
	ax.set_xlabel("Operations", fontsize=20)
	ax.set_ylabel("Clocks", fontsize=20)
	ax.set_xticks(ind + width*(elementsInGroup-1) / 2)
	ax.set_xticklabels(keys)
	ax.legend(p, LINE_NAME1)
	ax.autoscale_view()
	plt.show()


if __name__ == "__main__":
	barPlot(vggFlag=False)

	# qpeSimulator = QpeDramRoofline(isConv=False, considerOutActi=True, vggFlag=False)
	# qpeSimulator.roofline()
	# qpeSimulator = QpeDramRoofline(considerOutActi=True, isConv=False)
	# qpeSimulator.roofline()

# No Data Reuse
# 'CONV_1': (1119.080616433052, 18.58631700616342)
# 'CONV_3': (2042.788804071247, 78.93256571198768)
# 'CONV_4': (2042.788804071247, 57.05140285298951)
# 'CONV_6': (2549.5215243472126, 73.86796283008174)
# 'CONV_10': (1311.7908496732025, 72.53381720371918)

# Data Reuse
# 'CONV_1': (1119.080616433052, 18.994835052937642)
# 'CONV_3': (2042.788804071247, 81.970185828058)
# 'CONV_4': (2042.788804071247, 86.54214274431834)
# 'CONV_6': (2549.5215243472126, 80.04945657592981)
# 'CONV_10': (1311.7908496732025, 82.76406206150021)