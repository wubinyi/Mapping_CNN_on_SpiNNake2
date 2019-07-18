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
from rooflineModel import AcceleratorType, ROOFLINE_FORMAT, RooflineModel, LINE_NAME2
from distributionGeneral import DistributionGeneralClass
from fcLayerMapper import FcLayerMapper

MLA_16_4_VGG_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 119254,
	"CONV_7": 214575,
	"CONV_9": 413733
}

MLA_16_2_VGG_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 124267,
	"CONV_7": 327914,
	"CONV_9": 639999
}

MLA_8_4_VGG_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 125476,
	"CONV_7": 370876,
	"CONV_9": 729882
}

VGG_CONV_DIFF_DISTRI_SCHEME_RESULTS = [MLA_8_4_VGG_CONV_FUSION_DATA_REUSE_RESULT, MLA_16_2_VGG_CONV_FUSION_DATA_REUSE_RESULT, 
										MLA_16_4_VGG_CONV_FUSION_DATA_REUSE_RESULT]

MLA_16_4_RESNET_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 99623,
	"CONV_SC_14": 115207,
	"CONV_19": 44987,
	"CONV_42": 47116
}

MLA_16_2_RESNET_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 168451,
	"CONV_SC_14": 180906,
	"CONV_19": 59216,
	"CONV_42": 55317
}

MLA_8_4_RESNET_CONV_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 168583,
	"CONV_SC_14": 177153,
	"CONV_19": 64335,
	"CONV_42": 57354
}

RESNET_CONV_DIFF_DISTRI_SCHEME_RESULTS = [MLA_8_4_RESNET_CONV_FUSION_DATA_REUSE_RESULT, MLA_16_2_RESNET_CONV_FUSION_DATA_REUSE_RESULT, 
											MLA_16_4_RESNET_CONV_FUSION_DATA_REUSE_RESULT]

VGG_MLA_16_4_FC_NO_FUSION_RESULT = {
	"FC_21": 132044
}

VGG_MLA_16_2_FC_FUSION_RESULT = {
	"FC_21": 131864
}

VGG_MLA_8_4_FC_FUSION_RESULT = {
	"FC_21": 132816
}

VGG_FC_DIFF_DISTRI_RESULTS = [VGG_MLA_8_4_FC_FUSION_RESULT, VGG_MLA_16_2_FC_FUSION_RESULT, VGG_MLA_16_4_FC_NO_FUSION_RESULT]

RESNET_FC_DIFF_DISTRI_RESULTS = [{}]

class SpiNNaker2HwCompareRoofline():
	def __init__(self, printFlag=True, considerOutActi=False, isConv=True, vggFlag=True):
		self.printFlag = printFlag
		self.considerOutActi = considerOutActi
		self.isConv = isConv
		self.vggFlag = vggFlag

	def customAssert(self, condition, content):
		assert(condition), "{}: {}.".format(type(self).__name__, content)

	def updateModelSplitParameter(self, modelLayersSplitInfo, modelLayersParameter):
		self.modelLayersSplitInfo = modelLayersSplitInfo
		self.modelLayersParameter = modelLayersParameter

	def writeLogFile(self, context):
		if self.printFlag:
			BasicOperation.customPrintT(context)

	# ==================================================================================================
	# 									CONV Layer Distribution
	# ==================================================================================================
	def getConvLayersRfParamDiffScheme(self):
		# Get "operation per Byte" from all layers
		# convLayersOpesPerByte = self.getConvLayersOpsPerByte()
		# for key, value in convLayersOpesPerByte.items():
		# 	print("{}: {}".format(key, value))
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
		for layerName, layerSplitInfo in self.modelLayersSplitInfo.items():
			if "CONV_" not in layerName:
				continue
			# Determine if there is POOL for this CONV
			layerTypeParameter = self.modelLayersParameter[layerName]
			layerType, convParameter = layerTypeParameter
			poolParam = convParameter[1]
			if isinstance(poolParam, tuple):
				poolDim, poolStride = poolParam
				if poolDim[0] != poolStride or poolDim[1] != poolStride:
					poolSize = 1
				else:
					poolSize = poolDim[0] * poolDim[1]
			else:
				poolSize = poolParam * poolParam
			self.layersOpesPerByte[layerName] = self.getSingleConvLayerOpsPerByte(layerSplitInfo, poolSize)

	def getSingleConvLayerOpsPerByte(self, layerSplitInfo, poolSize):
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
			outActiSize = outActiWidth * outActiHeight * filterOutChannel
			# outActiSize = outActiSize // poolSize
			dataSize += outActiSize
		operations = ConvLayerMapper.convTheoreticalClocks(inActiWidth, inActiHeight, inActiChannel, filterWidth, 
			filterHeight, stride, filterOutChannel) * 2
		opsPerByte = operations / dataSize
		return (opsPerByte, operations)

	# ==================================================================================================
	# 									FC Layer Distribution
	# ==================================================================================================
	def getFcLayersRfParamDiffScheme(self):
		# Get "operation per Byte" from all layers
		# convLayersOpesPerByte = self.getFcLayersOpsPerByte()
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
		for layerName, layerSplitInfo in self.modelLayersSplitInfo.items():
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
		roofline = RooflineModel(AcceleratorType.SpiNNaker2HwCompare)
		# CONV+FC without decreasing input-activation-overlap-size
		if self.vggFlag:
			modelLayersSplitInfo, modelLayersParameter = DistributionGeneralClass.vgg(decreaseSize=True, forSpiNNaker2=True)
		else:
			modelLayersSplitInfo, modelLayersParameter = DistributionGeneralClass.resNet50(decreaseSize=True, forSpiNNaker2=True)
		self.updateModelSplitParameter(modelLayersSplitInfo, modelLayersParameter)
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
		roofline.addScatter(opsPerByteList, opsPerSecondDiffSchemeList, ROOFLINE_FORMAT[0:len(opsPerSecondDiffSchemeList)], 
			improveLineFormat, layerNames=layerNames, lineName=LINE_NAME2)
		# FC: Getting Roofline parameter
		rooflineInfoDiffScheme = self.getFcLayersRfParamDiffScheme()
		layerNames = list(rooflineInfoDiffScheme[0].keys())
		opsPerByteList, opsPerSecondDiffSchemeList, improveLineFormat = \
			self.getScattDataForRooflineModel(rooflineInfoDiffScheme)
		roofline.addScatter(opsPerByteList, opsPerSecondDiffSchemeList, ROOFLINE_FORMAT[0:len(opsPerSecondDiffSchemeList)], 
			improveLineFormat, labelFlag=False, layerNames=layerNames, lineName=LINE_NAME2)					
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
		# fcDistributionResults.append({"FC_52":0})
		distributionResults = RESNET_CONV_DIFF_DISTRI_SCHEME_RESULTS
	if vggFlag:
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
	ax.legend(p, LINE_NAME2)
	ax.autoscale_view()
	plt.show()

if __name__ == "__main__":
	barPlot(vggFlag=False)
	# srf = SpiNNaker2HwCompareRoofline(isConv=False, considerOutActi=True, vggFlag=False)
	# srf.roofline()
	# srf = SpiNNaker2HwCompareRoofline(considerOutActi=True, isConv=False)
	# srf.roofline()