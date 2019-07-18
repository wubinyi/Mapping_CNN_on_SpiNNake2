'''
1. Quantization
2. Source of MAC array (Alignment) --> A, B from AHB(ARM), SRAM or NoC
3. Destination of MAC array (Alignment) --> C to AHB(ARM), SRAM or NoC
4. Alignment of C, must be 16*4*4 bytes ??
5. the data arrangment of C
'''
import sys, os
projectFolder = os.path.dirname(os.getcwd())
if projectFolder not in sys.path:
	sys.path.insert(0, projectFolder)
s2sFolderPath = os.path.join(projectFolder, "spiNNaker2Simulator")
if s2sFolderPath not in sys.path:
	sys.path.insert(0, s2sFolderPath)
parserSplitterPath = os.path.join(projectFolder, "parserSplitter")
if parserSplitterPath not in sys.path:
	sys.path.insert(0, parserSplitterPath)

from nnModel import NNModel
from nnMemParser import NNMemParser
from nnParserUpgrade import NNParserUpgrade
from nnParser import NNParser
from nnGeneral import *
from fcLayerMapper import *
from convLayerMapper import *
from poolLayerMapper import *
from actiLayerMapper import *
from quanLayerMapper import *
from paddLayerMapper import *
from convLayerMapperDecrease import *
from convLayerMapperForSpiNNaker import *
from matEleLayerMapper import *
import math
from datetime import datetime


class NNModelToSpiNNaker2:
	def __init__(self, nnHyperparameter, sramBlocks=SRAM_BLOCKS, logEn=False, printFlag=True, upgrade=False):
		self.nnHyperparameter = nnHyperparameter
		self.printFlag = printFlag
		self.layerParameters = None
		self.layerMems = {}
		self.convLayerMaps = {}
		self.poolLayerMaps = {}
		self.fcLayerMaps = {}
		self.layerSplitParameters = {}
		self.logEn = logEn
		self.logfile = self.createFile()
		self.sramBlocks = sramBlocks
		self.upgrade = upgrade

	def createFile(self):
		if self.logEn:
			time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
			filename = "NNModelToSpiNNaker2"+time+".nn"
			file = open(filename,'a')
			return file
		else:
			return None

	def writeLogFile(self, context):
		if self.printFlag:
			BasicOperation.customPrintT(context)
		if self.logEn:
			self.logfile.write(context+"\n")

	def closeFile(self):
		if self.logEn:
			self.logfile.close()

	def getLayerParameters(self):
		return self.layerParameters

	# ===================================================================================================================
	# 											   Model parser
	# ===================================================================================================================
	def layerParameterFormatPrinter(self, arg0, arg1, title=False):
		self.writeLogFile("{:^20}|{:^80}".format(arg0, arg1))
		if title:
			self.writeLogFile("{:^20}|{:^80}".format("-"*20, "-"*80))

	def printLayerParameters(self):
		self.layerParameterFormatPrinter("Layer Name", "Layer Parameter", title=True)
		for layerName, layerTypeParameter in self.layerParameters.items():
			self.layerParameterFormatPrinter(layerName, str(layerTypeParameter))
		self.writeLogFile("\n")

	def hyperParaParser(self):
		'''
		nnHyperparameter = {layerType_0:layerParameter_0, layerType_1:layerParameter_1, layerType_2:layerParameter_2, ...}
			|---> INPUT: ([Width, Height, Channels])
			|---> CONV:  ([F_Width, F_Height, Channels, Output_Channels], stride, paddType)
			|---> POOL:  ([P_Width, P_Height], stride, poolType)
			|---> FC: 	 (output)
		layerParameters = {key_0:layerParameter_0, key_1:layerParameter_1, key_2:layerParameter_2, ... }
			|---> INPUT: ("input activation dimension", "output activation dimension")
			|---> PADD:  ("input activation DIM", "output activation DIM")
			|---> CONV:  ("input activation DIM", "filter DIM", stride, "output activation DIM")
			|---> POOL:  ("input activation DIM", "pooling filter DIM", stride, poolType, "output activation DIM")
			|---> FC:    (inNeuron, outNeuron)
			|---> QUAN:  ("input DIM", "output DIM")
		'''
		if self.upgrade:
			nnParser = NNParserUpgrade(self.nnHyperparameter)
		else:
			nnParser = NNParser(self.nnHyperparameter)
		self.layerParameters = nnParser.nnParser()
		self.printLayerParameters()

	# ===================================================================================================================
	# 											   Layer memory calculation
	# ===================================================================================================================
	def layerMemFormatPrinter(self, arg0, arg1, title=False):
		self.writeLogFile("{0:^20}|{1[0]:^30}|{1[1]:^30}|{1[2]:^30}".format(arg0, arg1))
		if title:
			self.writeLogFile("{:^20}|{:^30}|{:^30}|{:^30}".format("-"*20, "-"*30, "-"*30, "-"*30))

	def printlayerMems(self):
		self.layerMemFormatPrinter("Layer Name", 
			["Input Activation Size (Bytes)", "Weight Size (Bytes)", "Output Activation Size (Bytes)"], title=True)
		for layerName, layerMem in self.layerMems.items():
			self.layerMemFormatPrinter(layerName, layerMem)
		self.writeLogFile("\n")

	def nnModelMem(self):
		for layerName, layerTypeParameter in self.layerParameters.items():
			layerMem = NNMemParser.layerMemParser(layerTypeParameter)
			if layerMem != None:
				self.layerMems[layerName] = layerMem
		self.printlayerMems()

	# ===================================================================================================================
	# 											    Convolutional Layer Mapping
	# ===================================================================================================================
	def convLayerMapFormatPrinter(self, arg0, arg1, title=False):
		self.writeLogFile("{0:^15}|{1[0]:^10}|{1[1]:^30}|{1[2]:^30}|{1[3]:^30}|{1[4]:^30}|{1[5]:^10}|{1[6]:^15}".\
			format(arg0, arg1))
		if title:
			self.writeLogFile("{:^15}|{:^10}|{:^30}|{:^30}|{:^30}|{:^30}|{:^10}|{:^15}".\
				format("-"*15, "-"*10, "-"*30, "-"*30, "-"*30, "-"*30, "-"*10, "-"*15))

	def printConvLayersMaps(self):
		self.convLayerMapFormatPrinter("Layer Name", 
			["PEs", "Input Activation Size / PE", "Filter Size / PE", "Output Activation Size / PE", \
			"Computation Clocks / PE", "inActiUtil", "SramUtil"], title=True)
		for layerName, convLayerMapParameter in self.convLayerMaps.items():
			self.convLayerMapFormatPrinter(layerName, convLayerMapParameter)
		self.writeLogFile("\n")

	def convLayersMapping(self):
		'''
		As each MAC array has 4*16 MACs and are running at the same time, in order to fully utilizate all computation resource, 
			4 filters are needed.
		As weight are fetch from SRAM and because of memory alignment(32-bit), we prefer to use 4 filters.
		Base on the two reason illustrated above, we split all filters in one layer into sub-group, which compose of 4 filters.
		'''
		for layerName, layerTypeParameter in self.layerParameters.items():
			self.singleConvLayerMapping(layerName, layerTypeParameter)
		self.printConvLayersMaps()

	def singleConvLayerMapping(self, layerName, layerTypeParameter, decreaseSize=False, forSpiNNaker2=False):
		if "CONV_" in layerName:
			layerType, layerParameter = layerTypeParameter
			assert (LayerType.CONV == layerType or LayerType.CONV_SC == layerType), \
				"This Layer is not CONV, rather {}".format(layerType)
			for memoryBlocks in range(self.sramBlocks, 0, -1):
				if forSpiNNaker2:
					convMapper = ConvLayerMapperForSpiNNaker(layerName, layerParameter, 
							sramAvailable=math.floor(PE_SRAM_LIMIT/memoryBlocks))
				else:
					if decreaseSize:
						convMapper = ConvLayerMapperDecrease(layerName, layerParameter, 
							sramAvailable=math.floor(PE_SRAM_LIMIT/memoryBlocks))
					else:
						convMapper = ConvLayerMapper(layerName, layerParameter, 
							sramAvailable=math.floor(PE_SRAM_LIMIT/memoryBlocks))
				layerMapInfo = convMapper.map()
				if convMapper.isMeet():
					break
			self.convLayerMaps = {**self.convLayerMaps, **layerMapInfo}
			layerSplitInfo = convMapper.getSplitInfo()
			clocks = convMapper.getClocks()
			self.addConvLayerSplitInfo(layerName, layerSplitInfo, clocks)
			extraLayerInfo = convMapper.getExtraLayer()
			if extraLayerInfo != None:
				extraLayerName, extraLayerParameter = extraLayerInfo
				requiredPEs = 1
				extraLayerParameter = extraLayerParameter + (requiredPEs, )
				self.layerSplitParameters[extraLayerName] = extraLayerParameter		
			return layerSplitInfo

	def addConvLayerSplitInfo(self, layerName, layerSplitInfo, clocks):
		inWidth = layerSplitInfo[CONV_IN_WIDTH]
		inHeight = layerSplitInfo[CONV_IN_HEIGHT]
		inChannel = layerSplitInfo[CONV_IN_CHANNEL]
		filterWidth = layerSplitInfo[CONV_FILTER_WIDTH]
		filterHeight = layerSplitInfo[CONV_FILTER_HEIGHT]
		stride = layerSplitInfo[CONV_FILTER_STRIDE]
		outWidth = layerSplitInfo[CONV_OUT_WIDTH]
		outHeight = layerSplitInfo[CONV_OUT_HEIGHT]
		outChannel = layerSplitInfo[CONV_OUT_CHANNEL]
		inActivationDim = [inWidth, inHeight, inChannel]
		filterDim = [[filterWidth, filterHeight, inChannel, outChannel], stride]
		outActivationDim = [outWidth, outHeight, outChannel]
		requiredPEs = NNModelToSpiNNaker2.requiredPEsCalculation([inWidth, inHeight, inChannel, outChannel])
		self.layerSplitParameters[layerName] = (LayerType.CONV, inActivationDim, filterDim, outActivationDim, clocks, requiredPEs)

	# ===================================================================================================================
	# 											     Pooling Layer Mapping
	# ===================================================================================================================
	def poolLayerMapFormatPrinter(self, arg0, arg1, title=False):
		self.writeLogFile("{0:^15}|{1[0]:^10}|{1[1]:^30}|{1[2]:^30}|{1[3]:^30}|{1[4]:^30}|{1[5]:^10}|{1[6]:^10}".\
			format(arg0, arg1))
		if title:
			self.writeLogFile("{:^15}|{:^10}|{:^30}|{:^30}|{:^30}|{:^30}|{:^10}|{:^10}".\
				format("-"*15, "-"*10, "-"*30, "-"*30, "-"*30, "-"*30, "-"*10, "-"*10))

	def printPoolLayersMaps(self):
		self.poolLayerMapFormatPrinter("Layer Name", 
			["PEs", "Input Activation Size / PE", "Filter Size / PE", "Output Activation Size / PE", \
			"Computation Clocks / PE", "inActiUtil", "SramUtil"], title=True)
		for layerName, poolLayerMapParameter in self.poolLayerMaps.items():
			self.poolLayerMapFormatPrinter(layerName, poolLayerMapParameter)
		self.writeLogFile("\n")

	def poolLayersMapping(self):
		for layerName, layerTypeParameter in self.layerParameters.items():
			self.singlePoolLayerMapping(layerName, layerTypeParameter)
		self.printPoolLayersMaps()

	def singlePoolLayerMapping(self, layerName, layerTypeParameter, prelayerSplitInfo=None, forSpiNNaker2=False):
		if "POOL_" in layerName:
			layerType, layerParameter = layerTypeParameter
			assert (LayerType.POOL == layerType), "This Layer is not POOL, rather {}".format(layerType)
			poolMapper = PoolLayerMapper(layerName, layerParameter, 
				prelayerSplitInfo=prelayerSplitInfo, forSpiNNaker2=forSpiNNaker2)
			layerMapInfo = poolMapper.map()
			self.poolLayerMaps = {**self.poolLayerMaps, **layerMapInfo}
			layerSplitInfo = poolMapper.getSplitInfo()
			clocks = poolMapper.getClocks()
			self.addPoolLayerSplitInfo(layerName, layerSplitInfo, clocks)
			return layerSplitInfo

	def addPoolLayerSplitInfo(self, layerName, layerSplitInfo, clocks):
		inWidth = layerSplitInfo[POOL_IN_WIDTH]
		inHeight = layerSplitInfo[POOL_IN_HEIGHT]
		channel = layerSplitInfo[POOL_CHANNEL]
		filterWidth = layerSplitInfo[POOL_FILTER_WIDTH]
		filterHeight = layerSplitInfo[POOL_FILTER_HEIGHT]
		poolStride = layerSplitInfo[POOL_FILTER_STRIDE]
		outWidth = layerSplitInfo[POOL_OUT_WIDTH]
		outHeight = layerSplitInfo[POOL_OUT_HEIGHT]
		inActivationDim = [inWidth, inHeight, channel]
		filterDim = [[filterWidth, filterHeight], poolStride]
		outActivationDim = [outWidth, outHeight, channel]
		requiredPEs = NNModelToSpiNNaker2.requiredPEsCalculation([inWidth, inHeight, channel])
		self.layerSplitParameters[layerName] = (LayerType.POOL, inActivationDim, filterDim, outActivationDim, clocks, requiredPEs)
	# ===================================================================================================================
	# 											Fully-Connected Layer Mapping
	# ===================================================================================================================
	def fcLayerMapFormatPrinter(self, arg0, arg1, title=False):
		self.writeLogFile("{0:^15}|{1[0]:^10}|{1[1]:^30}|{1[2]:^30}|{1[3]:^30}|{1[4]:^30}|{1[5]:^15}".format(arg0, arg1))
		if title:
			self.writeLogFile("{:^15}|{:^10}|{:^30}|{:^30}|{:^30}|{:^30}|{:^15}".\
				format("-"*15, "-"*10, "-"*30, "-"*30, "-"*30, "-"*30, "-"*15))

	def printFcLayersMaps(self):
		self.fcLayerMapFormatPrinter("Layer Name", 
			["PEs", "Input Activation Size / PE", "Weight Size / PE", "Output Activation Size / PE", \
			"Computation Clocks / PE", "SramUtil"], title=True)
		for layerName, fcLayerMapParameter in self.fcLayerMaps.items():
			self.fcLayerMapFormatPrinter(layerName, fcLayerMapParameter)
		self.writeLogFile("\n")

	def fcLayersMapping(self):
		for layerName, layerTypeParameter in self.layerParameters.items():
			self.singleFcLayerMapping(layerName, layerTypeParameter)
		self.printFcLayersMaps()

	def singleFcLayerMapping(self, layerName, layerTypeParameter, forSpiNNaker2):
		if "FC_" in layerName:
			layerType, layerParameter = layerTypeParameter
			assert (LayerType.FC == layerType), "This Layer is not FC, rather {}".format(layerType)
			for memoryBlocks in range(self.sramBlocks, 0, -1):
				fcMapper = FcLayerMapper(layerName, layerParameter, sramAvailable=math.floor(PE_SRAM_LIMIT/memoryBlocks), 
					forSpiNNaker2=forSpiNNaker2)
				layerMapInfo = fcMapper.map()
				if fcMapper.isMeet():
					break
			self.fcLayerMaps = {**self.fcLayerMaps, **layerMapInfo}
			layerSplitInfo = fcMapper.getSplitInfo()
			clocks = fcMapper.getClocks()
			self.addFcLayerSplitInfo(layerName, layerSplitInfo, clocks)
			fcExtraLayerInfo = fcMapper.getExtraLayer()
			if fcExtraLayerInfo != None:
				extraLayerName, extraLayerParameter = fcExtraLayerInfo
				self.layerSplitParameters[extraLayerName] = extraLayerParameter	
			return layerSplitInfo

	def addFcLayerSplitInfo(self, layerName, layerSplitInfo, clocks):
		outNeuons = layerSplitInfo[FC_OUT]
		inNeurons = layerSplitInfo[FC_IN]
		inActivationDim = [inNeurons, 1]
		weightDim = [outNeuons, inNeurons]
		outActivationDim = [outNeuons, 1]
		requiredPEs = NNModelToSpiNNaker2.requiredPEsCalculation([inNeurons, outNeuons])
		self.layerSplitParameters[layerName] = (LayerType.FC, inActivationDim, weightDim, outActivationDim, clocks, requiredPEs)

	# ===================================================================================================================
	# 											Activation Layer Mapping
	# ===================================================================================================================
	def singleActiLayerMapping(self, layerName, layerTypeParameter, preLayerName, prelayerSplitInfo):
		if "ACTI_" in layerName:
			layerType, layerParameter = layerTypeParameter
			assert (LayerType.ACTI == layerType), "This Layer is not ACTI, rather {}".format(layerType)
			actiMapper = ActiLayerMapper(preLayerName, prelayerSplitInfo)
			actiMapper.map()
			layerSplitInfo = actiMapper.getSplitInfo()
			clocks = actiMapper.getClocks()
			self.addActiLayerSplitInfo(layerName, layerSplitInfo, clocks)
			return layerSplitInfo

	def addActiLayerSplitInfo(self, layerName, layerSplitInfo, clocks):
		widthSplitInfo = layerSplitInfo[ACTI_WIDTH]
		heightSplitInfo = layerSplitInfo[ACTI_HEIGHT]
		if ACTI_CHANNEL in layerSplitInfo:
			# CONV
			channelSplitInfo = layerSplitInfo[ACTI_CHANNEL]
			inoutDim = [widthSplitInfo, heightSplitInfo, channelSplitInfo]
		else:
			# FC
			inoutDim = [widthSplitInfo, heightSplitInfo]
		requiredPEs = NNModelToSpiNNaker2.requiredPEsCalculation(inoutDim)		
		self.layerSplitParameters[layerName] = (LayerType.ACTI, inoutDim, [0], inoutDim, clocks, requiredPEs)	

	# ===================================================================================================================
	# 											Quantization Layer Mapping
	# ===================================================================================================================
	def singleQuanLayerMapping(self, layerName, layerTypeParameter, preLayerName, prelayerSplitInfo):
		if "QUAN_" in layerName:
			layerType, layerParameter = layerTypeParameter
			assert (LayerType.QUAN == layerType or LayerType.QUAN_SC == layerType), \
				"This Layer is not QUAN, rather {}".format(layerType)
			quanMapper = QuanLayerMapper(preLayerName, prelayerSplitInfo)
			quanMapper.map()
			layerSplitInfo = quanMapper.getSplitInfo()
			clocks = quanMapper.getClocks()
			self.addQuanLayerSplitInfo(layerName, layerSplitInfo, clocks)
			return layerSplitInfo

	def addQuanLayerSplitInfo(self, layerName, layerSplitInfo, clocks):
		widthSplitInfo = layerSplitInfo[QUAN_WIDTH]
		heightSplitInfo = layerSplitInfo[QUAN_HEIGHT]
		if QUAN_CHANNEL in layerSplitInfo:
			# POOL and CONV-ACTI
			channelSplitInfo = layerSplitInfo[QUAN_CHANNEL]
			inoutDim = [widthSplitInfo, heightSplitInfo, channelSplitInfo]
		else:
			# FC-ACTI
			inoutDim = [widthSplitInfo, heightSplitInfo]
		requiredPEs = NNModelToSpiNNaker2.requiredPEsCalculation(inoutDim)		
		self.layerSplitParameters[layerName] = (LayerType.QUAN, inoutDim, [0], inoutDim, clocks, requiredPEs)	

	# ===================================================================================================================
	# 											    Padding Layer Mapping
	# ===================================================================================================================
	def singlePaddLayerMapping(self, layerName, layerTypeParameter, postLayerSplitInfo=None):
		assert("PADD" in layerName), "This Layer's name-{} should contain PADD".format(layerName)
		layerType, layerParameter = layerTypeParameter
		assert(LayerType.PADD == layerType or LayerType.PADD_SC == layerType), \
			"This Layer is not PADD, rather {}".format(layerType)
		paddMapper = PaddLayerMapper(layerParameter, postLayerSplitInfo)
		try:
			paddMapper.map()
		except:
			print("layerName: {}".format(layerName))
			print("postLayerSplitInfo: {}".format(postLayerSplitInfo))
			assert(False)
		layerSplitInfo = paddMapper.getSplitInfo()
		self.addPaddLayerSplitInfo(layerName, layerSplitInfo)
		
	def addPaddLayerSplitInfo(self, layerName, layerSplitInfo):
		paddInWidth = layerSplitInfo[PADD_IN_WIDTH]
		paddInHeight = layerSplitInfo[PADD_IN_HEIGHT]
		paddInChannel = layerSplitInfo[PADD_IN_CHANNEL]
		paddDimOverlap = layerSplitInfo[PADD_DIM_OVERLAP]
		paddOutWidth = layerSplitInfo[PADD_OUT_WIDTH] 
		paddOutHeight = layerSplitInfo[PADD_OUT_HEIGHT] 
		paddOutChannel = layerSplitInfo[PADD_OUT_CHANNEL] 
		inActiDim = [paddInWidth, paddInHeight, paddInChannel]
		outActiDim = [paddOutWidth, paddOutHeight, paddOutChannel]
		self.layerSplitParameters[layerName] = (LayerType.PADD, inActiDim, paddDimOverlap, outActiDim, "-----", "-----")

	# ===================================================================================================================
	# 									Matrix Element-wise operation Layer Mapping
	# ===================================================================================================================
	def singleMatEleLayerMapping(self, layerName, layerTypeParameter, preLayerName, prelayerSplitInfo):
		assert("MAT_ELE" in layerName), "This Layer's name-{} should contain MAT_ELE".format(layerName)
		layerType, layerParameter = layerTypeParameter
		assert (LayerType.MAT_ELE == layerType), "This Layer is not MAT_ELE, rather {}".format(layerType)
		matEleMapper = MatEleLayerMapper(preLayerName, prelayerSplitInfo)
		matEleMapper.map()
		layerSplitInfo = matEleMapper.getSplitInfo()
		clocks = matEleMapper.getClocks()
		self.addMatEleLayerSplitInfo(layerName, layerSplitInfo, clocks)
		return layerSplitInfo

	def addMatEleLayerSplitInfo(self, layerName, layerSplitInfo, clocks):
		widthSplitInfo = layerSplitInfo[MAT_ELE_WIDTH]
		heightSplitInfo = layerSplitInfo[MAT_ELE_HEIGHT]
		if MAT_ELE_CHANNEL in layerSplitInfo:
			channelSplitInfo = layerSplitInfo[MAT_ELE_CHANNEL]
			inoutDim = [widthSplitInfo, heightSplitInfo, channelSplitInfo]
		else:
			inoutDim = [widthSplitInfo, heightSplitInfo]
		requiredPEs = NNModelToSpiNNaker2.requiredPEsCalculation(inoutDim)
		self.layerSplitParameters[layerName] = (LayerType.MAT_ELE, inoutDim, [0], inoutDim, clocks, requiredPEs)		

	# ===================================================================================================================
	# 											   NN Model split info
	# ===================================================================================================================
	def layerSplitParametersFormatPrinter(self, arg0, arg1, title=False):
		self.writeLogFile("{0:^15}|{1:^66}|{2:^40}|{3:^50}|{4:^20}|{5:^10}".\
			format(arg0, str(arg1[1]), str(arg1[2]), str(arg1[3]), str(arg1[4]), arg1[5]))
		if title:
			self.writeLogFile("{:^15}|{:^66}|{:^40}|{:^50}|{:^20}|{:^10}".\
				format("-"*15, "-"*66, "-"*40, "-"*50, "-"*20, "-"*10))
	# def layerSplitParametersFormatPrinter(self, arg0, arg1, title=False):
	# 	self.writeLogFile("{0:^15}|{1:^20}|{2:^64}|{3:^35}|{4:^50}|{5:^20}".\
	# 		format(arg0, str(arg1[0]), str(arg1[1]), str(arg1[2]), str(arg1[3]), str(arg1[4])))
	# 	if title:
	# 		self.writeLogFile("{:^15}|{:^20}|{:^64}|{:^35}|{:^50}|{:^20}".\
	# 			format("-"*15, "-"*20, "-"*64, "-"*35, "-"*50, "-"*20))

	def printLayerSplitParameters(self):
		if self.printFlag:
			self.layerSplitParametersFormatPrinter("Layer Name", 
				["Layer Type", "Input Activation Dimension", "Weight Dimension", "Output Activation Dimension", \
				"Computation clocks", "PEs"], title=True)
			for layerName, layerSplitParameter in self.layerSplitParameters.items():
				self.layerSplitParametersFormatPrinter(layerName, layerSplitParameter)
			self.writeLogFile("\n")		

	# ===================================================================================================================
	# 											  		 Main
	# ===================================================================================================================
	def seperateRun(self):
		self.hyperParaParser()
		self.nnModelMem()
		self.convLayersMapping()
		self.poolLayersMapping()
		self.fcLayersMapping()
		self.printLayerSplitParameters()
		self.closeFile()

	def run(self, decreaseSize=False, forSpiNNaker2=False):
		self.hyperParaParser()
		self.nnModelMem()
		prelayerSplitInfo = None
		preLayerName = None
		# Record padd layer, after splitting CONV/FC/POOL, come back to resplit them
		paddLayerName = None
		paddLayerTypeParameter = None
		for layerName, layerTypeParameter in self.layerParameters.items():
			layerType, layerParameter = layerTypeParameter
			if LayerType.INPUT == layerType:
				prelayerSplitInfo = None
			elif LayerType.CONV == layerType:
				prelayerSplitInfo = self.singleConvLayerMapping(layerName, layerTypeParameter, decreaseSize, forSpiNNaker2)
				if preLayerName == paddLayerName:
					self.singlePaddLayerMapping(paddLayerName, paddLayerTypeParameter, postLayerSplitInfo=prelayerSplitInfo)
					paddLayerName = None
					paddLayerTypeParameter = None
			elif LayerType.ACTI == layerType:
				prelayerSplitInfo = self.singleActiLayerMapping(layerName, layerTypeParameter, preLayerName, prelayerSplitInfo)
			elif LayerType.PADD == layerType:
				paddLayerName = layerName
				paddLayerTypeParameter = layerTypeParameter
				prelayerSplitInfo = None
				self.singlePaddLayerMapping(layerName, layerTypeParameter)
			elif LayerType.POOL == layerType:
				if "QUAN" in preLayerName:
					assert(QUAN_CHANNEL in prelayerSplitInfo), "prelayerSplitInfo should have key: {}".format(QUAN_CHANNEL)
					prelayerSplitInfo = self.singlePoolLayerMapping(layerName, layerTypeParameter, prelayerSplitInfo)
				else:
					prelayerSplitInfo = self.singlePoolLayerMapping(layerName, layerTypeParameter, forSpiNNaker2=forSpiNNaker2)
				if preLayerName == paddLayerName:
					self.singlePaddLayerMapping(paddLayerName, paddLayerTypeParameter, postLayerSplitInfo=prelayerSplitInfo)
					paddLayerName = None
					paddLayerTypeParameter = None
			elif LayerType.FC == layerType:
				prelayerSplitInfo = self.singleFcLayerMapping(layerName, layerTypeParameter, forSpiNNaker2)
			elif LayerType.QUAN == layerType:
				prelayerSplitInfo = self.singleQuanLayerMapping(layerName, layerTypeParameter, preLayerName, prelayerSplitInfo)
			elif LayerType.MAT_ELE == layerType:
				prelayerSplitInfo = self.singleMatEleLayerMapping(layerName, layerTypeParameter, preLayerName, prelayerSplitInfo)
			elif LayerType.PADD_SC == layerType:
				paddLayerName = layerName
				layerType, layerParameter = layerTypeParameter
				(inActiDim, outActiDim, (paddDim, overlapSize), shortcutSource) = layerParameter
				paddLayerTypeParameterTemp = (layerType, (inActiDim, outActiDim, (paddDim, overlapSize)))
				paddLayerTypeParameter = paddLayerTypeParameterTemp
				prelayerSplitInfo = None
				self.singlePaddLayerMapping(layerName, paddLayerTypeParameterTemp)
			elif LayerType.CONV_SC == layerType:
				layerType, layerParameter = layerTypeParameter
				((inActiDim, filterDim, stride, outActiDim), poolStride, shortcutSource) = layerParameter
				convLayerParameterTemp = (layerType, ((inActiDim, filterDim, stride, outActiDim), poolStride))
				prelayerSplitInfo = self.singleConvLayerMapping(layerName, convLayerParameterTemp, decreaseSize, forSpiNNaker2)
				if preLayerName == paddLayerName:
					self.singlePaddLayerMapping(paddLayerName, paddLayerTypeParameter, postLayerSplitInfo=prelayerSplitInfo)
					paddLayerName = None
					paddLayerTypeParameter = None
			elif LayerType.QUAN_SC == layerType:
				layerType, layerParameter = layerTypeParameter
				(inActivationDim, inActivationDim, shortcutSource) = layerParameter
				quanLayerParameterTemp = (layerType, (inActivationDim, inActivationDim))
				prelayerSplitInfo = self.singleQuanLayerMapping(layerName, quanLayerParameterTemp, preLayerName, prelayerSplitInfo)
			else:
				assert(False), "Unknown layer type: {}".format(layerType)
			preLayerName = layerName
		self.printConvLayersMaps()
		self.printPoolLayersMaps()
		self.printFcLayersMaps()
		self.printLayerSplitParameters()
		self.closeFile()
		return self.layerSplitParameters

	# ===================================================================================================================
	# 											  		 Main
	# ===================================================================================================================
	@staticmethod
	def requiredPEsCalculation(dimensions):
		requiredPEs = 1
		for element in dimensions:
			if isinstance(element, list):
				requiredPEs = requiredPEs * (element[1]+element[3])
		return requiredPEs

if __name__ == "__main__":
	# VGG-19
	# https://www.quora.com/What-is-the-VGG-neural-network
	nnModel = NNModel()
	nnModel.addLayer(LayerType.INPUT, ([224, 224, 3]))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 3, 64], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 64, 128], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 128], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 128, 256], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 256], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 256, 512], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.CONV, ([3, 3, 512, 512], 1, PaddType.SAME, ActivationType.ReLU))
	nnModel.addLayer(LayerType.POOL, ([2, 2], 2, PoolType.MAXPOOL))
	nnModel.addLayer(LayerType.FC, (4096, ActivationType.ReLU))
	nnModel.addLayer(LayerType.FC, (1024, ActivationType.ReLU))
	# print("nnHyperparameter: ", nnModel.getNNModel())
	nnModelToSpiNNaker2 = NNModelToSpiNNaker2(nnModel.getNNModel())
	nnModelToSpiNNaker2.run()