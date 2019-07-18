from enum import Enum, auto
import math


FC_IN = "fcInNeuron"
FC_OUT = "fcOutNeuron"


CONV_IN_WIDTH = "convInWidth"
CONV_IN_HEIGHT = "convInHeight"
CONV_IN_CHANNEL = "convInChannel"
CONV_FILTER_WIDTH = "convFilterWidth"
CONV_FILTER_HEIGHT = "convFilterHeight"
CONV_FILTER_STRIDE = "convFilterStride"
CONV_OUT_WIDTH = "convOutWidth"
CONV_OUT_HEIGHT = "convOutHeight"
CONV_OUT_CHANNEL = "convOutChannel"


POOL_IN_WIDTH = "poolInWidth"
POOL_IN_HEIGHT = "poolInHeight"
POOL_CHANNEL = "poolInChannel"
POOL_FILTER_WIDTH = "poolFilterWidth"
POOL_FILTER_HEIGHT = "poolFIlterHeight"
POOL_FILTER_STRIDE = "poolStride"
POOL_OUT_WIDTH = "poolOutWidth"
POOL_OUT_HEIGHT = "poolOutHeight"


# CONV: width
# FC： outNeuron
ACTI_WIDTH = "actiWidth"
# CONV: height
# FC： inNeuron
ACTI_HEIGHT = "actiHeight"
# CONV: channel
# FC： ----
ACTI_CHANNEL = "actiChannel"


QUAN_WIDTH = "quanWidth"
QUAN_HEIGHT = "quanHeight"
QUAN_CHANNEL = "quanChannel"


PADD_IN_WIDTH = "paddInWidth"
PADD_IN_HEIGHT = "paddInHeight"
PADD_IN_CHANNEL = "paddInChannel"
PADD_DIM_OVERLAP = "paddDimOverlap"
PADD_OUT_CHANNEL = "paddOutChannel"
PADD_OUT_WIDTH = "paddOutWidth"
PADD_OUT_HEIGHT = "paddOutHeight"



MAT_ELE_WIDTH = "matEleWidth"
MAT_ELE_HEIGHT = "matEleHeight"
MAT_ELE_CHANNEL = "matEleChannel"

# ========================================================================



SRAM_BYTES_ALIGN = 16
MAC_ARRAY_ROW = 4
MAC_ARRAY_COLUMN = 16
MAC_OUT_BYTES = 4
PE_SRAM_LIMIT = 96 * 1024
SRAM_BLOCKS = 2

PEs = "numOfPe"
INACTI = "inputActivationSize"
INACTIALIG = "inputActivationSizeAlign"
WEIGHT = "weightSize"
WEIGHTALIG = "weightSizeAlign"
OUTACTI = "outputActivationSize"
OUTACTIALIG = "outputActivationSizeAlign"

class PaddType(Enum):
	VALID = auto()
	SAME = auto()


class LayerType(Enum):
	INPUT = auto()
	CONV = auto()
	ACTI = auto()
	CONVE = auto()
	PADD = auto()
	POOL = auto()
	FC = auto()
	FCE = auto()
	QUAN = auto()
	CONV_BLOCK = auto()
	PADD_SC = auto()
	CONV_SC = auto()
	QUAN_SC = auto()
	ACTI_SC = auto()
	MAT_ELE = auto()
	# FC
	FCTL = auto()
	FCTR = auto()
	FCBL = auto()
	FCBR = auto()
	# CONV sufficient
	CONVSU = auto()
	CONVSUTL = auto()
	CONVSUTLP = auto()
	CONVSUTLM = auto()
	CONVSUTR = auto()
	CONVSUTRP = auto()
	CONVSUTRM = auto()
	CONVSUBL = auto()
	CONVSUBLP = auto()
	CONVSUBLM = auto()
	CONVSUBR = auto()
	CONVSUBRP = auto()
	CONVSUBRM = auto()
	# CONV insufficient
	CONVIS = auto()
	CONVISTL = auto()
	CONVISTLP = auto()
	CONVISTLM = auto()
	CONVISTR = auto()
	CONVISTRP = auto()
	CONVISTRM = auto()
	CONVISBL = auto()
	CONVISBLP = auto()
	CONVISBLM = auto()
	CONVISBR = auto()
	CONVISBRP = auto()
	CONVISBRM = auto()	
	# POOL
	POOLTL = auto()
	POOLTLF = auto()
	POOLTLB = auto()
	POOLTR = auto()
	POOLTRF = auto()
	POOLTRB = auto()
	POOLBL = auto()
	POOLBLF = auto()
	POOLBLB = auto()
	POOLBR = auto()
	POOLBRF = auto()
	POOLBRB = auto()
	POOLEE = auto()


class BasicOperation():

	@staticmethod
	def listInnerProduct(elements):
		product = 1
		if isinstance(elements, list):
			for element in elements:
				product = product * element
		else:
			product = elements
		return product

	@staticmethod
	def ceilOf(num, factor):
		return int(math.ceil(num/factor) * factor)

	@staticmethod
	def listCeilOf(elements, factor):
		assert(isinstance(elements, list)), "BasicOperation.listCeilOf: type(elements): {}".format(type(elements))
		assert(len(elements) != 0), "BasicOperation.listCeilOf: elements contains 0 element"
		assert(len(elements)%2 == 0), "BasicOperation.listCeilOf: elements contains {} elements".format(len(elements))
		lengthAlign = 0
		for pairIndex in range(0, int(len(elements)/2)):
			index = pairIndex * 2
			lengthAlign += BasicOperation.ceilOf(elements[index], factor) * elements[index+1]
		return lengthAlign

	@staticmethod
	def splitInfoIntegration(splitInfo):
		if isinstance(splitInfo, list):
			assert(len(splitInfo) != 0), "BasicOperation.listCeilOf: splitInfo contains 0 element"
			assert(len(splitInfo)%2 == 0), "BasicOperation.listCeilOf: splitInfo contains {} elements".format(len(splitInfo))
			length = 0
			for pairIndex in range(0, int(len(splitInfo)/2)):
				index = pairIndex * 2
				length += splitInfo[index] * splitInfo[index+1]
			return length	
		elif isinstance(splitInfo, int):
			return splitInfo
		else:
			assert(False),"splitInfoIntegration: Unknown type!!"

	@staticmethod
	def getSplitDimFromSplitInfo(splitInfo):
		assert(isinstance(splitInfo, list)), "BasicOperation.listCeilOf: type(splitInfo): {}".format(type(splitInfo))
		assert(len(splitInfo) != 0), "BasicOperation.listCeilOf: splitInfo contains 0 element"
		assert(len(splitInfo)%2 == 0), "BasicOperation.listCeilOf: splitInfo contains {} elements".format(len(splitInfo))
		return [splitInfo[0], splitInfo[2]]

	@staticmethod
	def getSplitPartsFromSplitInfo(splitInfo):
		assert(isinstance(splitInfo, list)), "BasicOperation.listCeilOf: type(splitInfo): {}".format(type(splitInfo))
		assert(len(splitInfo) != 0), "BasicOperation.listCeilOf: splitInfo contains 0 element"
		assert(len(splitInfo)%2 == 0), "BasicOperation.listCeilOf: splitInfo contains {} elements".format(len(splitInfo))
		return [splitInfo[1], splitInfo[3]]		

	@staticmethod
	def getTotalPartsFromSplitInfo(splitInfo):
		# assert(isinstance(splitInfo, list)), "BasicOperation.listCeilOf: type(splitInfo): {}".format(type(splitInfo))
		if isinstance(splitInfo, list):
			assert(len(splitInfo) != 0), "BasicOperation.listCeilOf: splitInfo contains 0 element"
			assert(len(splitInfo)%2 == 0), "BasicOperation.listCeilOf: splitInfo contains {} elements".format(len(splitInfo))
			return splitInfo[1]+splitInfo[3]		
		else:
			return 1

	@staticmethod
	def floorOf(num, factor):
		return int(math.floor(num/factor) * factor)

	@staticmethod
	def oneDimSplitting(length, parts):
		'''
		There two ways to split "length", dependent on whether "length" can be equally splited by "parts"
		1. M-----M-----M-----M-----M-----M-----M...-----M
		2. M+1---M+1---M+1---M+1---M+1---M+1---M+1...---M
		splittingResult = [largePart, numOfLargePart, smallPart, numOfSmallPart]
		'''
		assert(isinstance(length, int)), "BasicOperation.oneDimSplitting: type(length): {}".format(type(length))
		assert(isinstance(parts, int)), "BasicOperation.oneDimSplitting: type(parts): {}".format(type(parts))
		splittingResult = [0, 0, 0, 0]
		if length % parts == 0:
			splittingResult[0] = int(length / parts)
			splittingResult[1] = parts
		else:
			largePart = int(length / parts) + 1
			smallPart = largePart - 1
			numOfLargePart = length % parts
			numOfSmallPart = parts - numOfLargePart
			splittingResult[0] = largePart
			splittingResult[1] = numOfLargePart
			splittingResult[2] = smallPart
			splittingResult[3] = numOfSmallPart
		return splittingResult

	@staticmethod
	def oneDimSplittingWithAlign(length, parts, factor):
		'''
		This function is only used to splitting the channel for POOL, when oneDimSplitting() failed because of alignment.
		It is not recommend to apply this function in CONV and FC for splitting the width. As the aligment unit for
			the width is MAC_ARRAY_COLUMN (16), which will cause large difference between each PEs.
		Moreover, it is also not recommend to apply this function in CONV for splitting the output-channel of filters, 
			because it will cause difference (4 filters) between each PEs. For CONV, 4 filters size is prretty large,
			causing large different of SRAM
		However, it is suitable for POOL, even it cause 4 channel difference, because each channel's size is small.
		'''
		lengthAlign = BasicOperation.ceilOf(length, factor)
		splittingResult = BasicOperation.oneDimSplitting(length, parts)
		largePart = splittingResult[0]
		numOfLargePart = splittingResult[1]
		largePartAlign = BasicOperation.ceilOf(largePart, factor)
		splittingResultAlign = None
		for numOfLargePartTemp in range(parts-1, 0, -1):
			rest = length - largePartAlign * numOfLargePartTemp
			# Check value of rest
			if rest <= 0: 
				continue
			# Check equally division
			if rest % (parts - numOfLargePartTemp) != 0:
				continue
			# generate "splittingResultAlignTemp"
			restPartSize =  rest / (parts - numOfLargePartTemp)
			splittingResultAlignTemp = [0] * 4
			splittingResultAlignTemp[0] = largePartAlign
			splittingResultAlignTemp[1] = numOfLargePartTemp
			splittingResultAlignTemp[2] = int(restPartSize)
			splittingResultAlignTemp[3] = parts - numOfLargePartTemp
			if lengthAlign != BasicOperation.listCeilOf(splittingResultAlignTemp, factor):
				continue
			if length != BasicOperation.listCeilOf(splittingResultAlignTemp, 1):
				continue
			splittingResultAlign = splittingResultAlignTemp
			assert(largePartAlign >= restPartSize), "oneDimSplittingWithAlign 3"
		return splittingResultAlign


	@staticmethod
	def printDict(dict, printFlag=True):
		if printFlag:
			for key, value in dict.items():
				print("{:<10}: {}".format(key, value))
			print("\n")

	@staticmethod
	def customPrintF(string, printFlag=False):
		if printFlag:
			print(string)

	@staticmethod
	def customPrintT(string, printFlag=True):
		if printFlag:
			print(string)

	@staticmethod
	def splitInfoToTotalLength(splitInfo):
		length = 0
		for index in range(2):
			index2 = index * 2
			length += splitInfo[index2] * splitInfo[index2+1]
		return length

	def customPause(printContent="Press any key to continue"):
		input(printContent)