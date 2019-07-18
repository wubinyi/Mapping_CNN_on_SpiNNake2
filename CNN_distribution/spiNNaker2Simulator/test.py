import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
from spiNNakerSimulatorGeneral import *
import math
import numpy as np

# =======================================================================================
# 	Test
# =======================================================================================
def getOutColumnRowRound(width, height, channelBatch):
	outWidthRound = []
	while width > 0:
		if width > MLA_MAC_COLUMN:
			outWidthRound.append(MLA_MAC_COLUMN)
		else:
			outWidthRound.append(width)
		width -= MLA_MAC_COLUMN
	outHeightRound = [1]*height
	outChannelRound = []
	while channelBatch > 0:
		if channelBatch >= MLA_MAC_ROW:
			outChannelRound.append(MLA_MAC_ROW)
		else:
			outChannelRound.append(channelBatch)
		channelBatch -= MLA_MAC_ROW			
	return outWidthRound, outHeightRound, outChannelRound

def align16(length):
	return math.ceil(length/MLA_MAC_COLUMN) * MLA_MAC_COLUMN

def getAddr(outWidthRound, outwidthRoundCounter, outheightRoundCounter, inActiStartAddr, addrHeightOffset):
	addr = 0
	for index in range(outwidthRoundCounter):
		addr += outWidthRound[index]
	addr += outheightRoundCounter * addrHeightOffset
	addr += inActiStartAddr
	return addr

def convRun(inWidth, inHeight, inChannel, filterWidth, filterHeight, outChannel, stride,
			inActiStartAddr):
	addrChannelOffset = align16(inWidth) * inHeight
	addrHeightOffset = align16(inWidth)
	outWidth = math.floor((inWidth - filterWidth) / stride) + 1
	outHeight = math.floor((inHeight - filterHeight) / stride) + 1
	outWidthRound, outHeightRound, outChannelRound = getOutColumnRowRound(outWidth, outHeight, outChannel)
	outWidthRoundLen = len(outWidthRound)
	outHeightRoundLen = len(outHeightRound)
	outChannelRoundLen = len(outChannelRound)
	print("OutWidth Len: {}".format(outWidthRoundLen))
	print("OutHeight Len: {}".format(outHeightRoundLen))
	print("OutChannel Len: {}".format(outChannelRoundLen))
	inChannelRound = inChannel
	filterHeightRound = filterHeight
	outwidthRoundCounter = 0
	outheightRoundCounter = 0
	outChannelRoundCounter = 0
	inChannelRoundCounter = 0
	filterHeightRoundCounter = 0
	inActiLen = 0
	while True:
		print("-"*66)
		print("OutWidth Index: {}".format(outwidthRoundCounter))
		print("OutHeight Index: {}".format(outheightRoundCounter))
		print("OutChannel Index: {}".format(outChannelRoundCounter))
		outWidth =  outWidthRound[outwidthRoundCounter]
		inWidth = (outWidth - 1) * stride + filterWidth
		inActiAddr = getAddr(outWidthRound, outwidthRoundCounter, outheightRoundCounter, inActiStartAddr, addrHeightOffset)

		# Single Round
		channelAddr = inActiAddr
		while inChannelRoundCounter < inChannelRound:
			heightAddr = 0
			while filterHeightRoundCounter < filterHeightRound:
				widthAddr = 0
				# Fetch one row of input-activation
				print("New Row Addr: {}".format(hex(channelAddr + heightAddr + widthAddr)))
				widthAddr += MLA_MAC_COLUMN
				inActiLen += MLA_MAC_COLUMN 
				while inActiLen < inWidth:
					print("Addr: {}".format(hex(channelAddr + heightAddr + widthAddr)))
					widthAddr += 1
					inActiLen += 1
				# Prepare for next Row
				inActiLen = 0
				filterHeightRoundCounter += 1
				heightAddr += addrHeightOffset
			# Prepare for next Channel
			filterHeightRoundCounter = 0
			inChannelRoundCounter += 1
			channelAddr += addrChannelOffset
		# Prepare for next Round
		inChannelRoundCounter = 0

		# Next Round
		print("-"*66+"\n")
		input("Press any key to continue")
		outwidthRoundCounter += 1
		if outwidthRoundCounter == outWidthRoundLen:
			outwidthRoundCounter = 0
			outheightRoundCounter += 1
			if outheightRoundCounter == outHeightRoundLen:
				outheightRoundCounter = 0
				outChannelRoundCounter += 1
				if outChannelRoundCounter ==outChannelRoundLen:
					outChannelRoundCounter = 0
					break

def compareData():
	outActiAlign = np.loadtxt("outActiAlign.txt", delimiter=' ')
	outActiArray = np.loadtxt("outActiArray.txt", delimiter=' ')
	print("Size : {}-{}".format(len(outActiAlign), len(outActiArray)))
	print("Type : {}-{}".format(type(outActiAlign), type(outActiArray)))
	for index in range(len(outActiAlign)):
		if outActiAlign[index] != outActiArray[index]:
			print("Position[{}] unequal: {}-{}, difference: {}".format(index, outActiAlign[index], outActiArray[index], outActiAlign[index]-outActiArray[index]))

if __name__ == "__main__":
	# convRun(inWidth=33, inHeight=4, inChannel=3, filterWidth=3, filterHeight=3, outChannel=5, stride=1,
	# 		inActiStartAddr=0x8000)
	compareData()