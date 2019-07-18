import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from nnGeneral import *
from spiNNaker2Simulator import *
import math
import copy
from enum import Enum, auto

ROOFLINE_FORMAT = ['bx', 'gh', 'r^', 'c*', 'm^', 'y^', 'k^']

LINE_NAME1 = ["Without data reuse and operator fusion", "Without data reuse and with operator fusion", "With data reuse and operator fusion"]

LINE_NAME2 = ["MAC array size: 8*4", "MAC array size: 16*2", "MAC array size: 16*4"]

class AcceleratorType(Enum):
	QPE = auto()
	SpiNNaker2 = auto()
	SpiNNaker2HwCompare = auto()

def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

class RooflineModel():
	def __init__(self, 
				acceleratorType=AcceleratorType.QPE,
				opIntensityLimited=4000, 
				title='Roofline of SpiNNaker2', 
				xlabel='Ops/Byte (log scale)', 
				ylabel='Attainable Gops (log scale)'):
		'''
		peakMemoryBw: unit GBytes
		maxOps: G
		'''
		self.figure, self.axes= plt.subplots()
		self.acceleratorType = acceleratorType
		# if AcceleratorType.QPE != acceleratorType:
		# 	opIntensityLimited = int(opIntensityLimited * 100)
		self.opIntensityLimited = opIntensityLimited
		self.halfPerformance = False
		self.dramPeakMemoryBw, self.nocPeakMemoryBw, self.maxOps = self.performanceEstimate(acceleratorType)
		self.fontsize = 'xx-large'
		self.rooflineFigure(title, xlabel, ylabel, opIntensityLimited)
		import matplotlib.pylab as pylab
		          # 'figure.figsize': (15, 5),
		params = {'legend.fontsize': 'xx-large',
		         'axes.labelsize': 'xx-large',
		         'axes.titlesize':'xx-large',
		         'xtick.labelsize':'xx-large',
		         'ytick.labelsize':'xx-large'}
		pylab.rcParams.update(params)

	def customAssert(self, condition, content, frame=None):
		if frame == None:
			assert(condition), "{}: {}.".format(type(self).__name__, content)
		else:
			funcName = frame.f_code.co_name
			lineNumber = frame.f_lineno
			fileName = frame.f_code.co_filename
			assert(condition), "{}-{}(): {}.".format(type(self).__name__, funcName, content)

	def performanceEstimate(self, acceleratorType):
		if AcceleratorType.QPE == acceleratorType:
			dramPeakMemoryBw = DRAM_FREQ*DRAM_MAX_DATA_WIDTH_BYTES/MEGA_TO_GIGA/DRAM_CLOCKS_PER_OPER
			nocPeakMemoryBw = NOC_FREQ*NOC_MAX_DATA_WIDTH_BYTES/MEGA_TO_GIGA
			maxOps = PE_FREQ*MAC_ARRAY_SIZE*MAC_OPER_PER_CLOCK*NUM_PES_IN_QPE/MEGA_TO_GIGA
		elif AcceleratorType.SpiNNaker2 == acceleratorType or AcceleratorType.SpiNNaker2HwCompare == acceleratorType:
			dramPeakMemoryBw = DRAM_FREQ*DRAM_MAX_DATA_WIDTH_BYTES*NUM_DRAMS/MEGA_TO_GIGA/DRAM_CLOCKS_PER_OPER
			nocPeakMemoryBw = NOC_FREQ*NOC_MAX_DATA_WIDTH_BYTES*8/MEGA_TO_GIGA
			maxOps = PE_FREQ*MAC_ARRAY_SIZE*MAC_OPER_PER_CLOCK*NUM_PES_IN_QPE*NUM_QPES_X_AXIS*NUM_QPES_Y_AXIS/MEGA_TO_GIGA
			# self.customAssert(False, content="Unsupport AcceleratorType {}".format(acceleratorType), frame=sys._getframe())
			if AcceleratorType.SpiNNaker2HwCompare == acceleratorType:
				self.halfPerformance = True
		else:
			self.customAssert(False, content="Unknown AcceleratorType {}".format(acceleratorType), frame=sys._getframe())
		return dramPeakMemoryBw, nocPeakMemoryBw, maxOps

	def addHalfPerformance(self, opIntensityLimited):
		xticks = list(range(1, opIntensityLimited))
		for x in xticks:
			if min(self.dramPeakMemoryBw*x, self.maxOps/2) == self.maxOps/2:
				xDramTickBoundary = x
				break
		xticksDram = list(range(1, xDramTickBoundary+1))
		for x in xticks:
			if min(self.nocPeakMemoryBw*x, self.maxOps/2) == self.maxOps/2:
				xNocTickBoundary = x
				break
		xticksNoC = list(range(1, xNocTickBoundary+1))
		if xDramTickBoundary <= xNocTickBoundary:
			xMaxOpsTickBoundary = xDramTickBoundary
		else:
			xMaxOpsTickBoundary = xNocTickBoundary
		xticksOps = list(range(xMaxOpsTickBoundary, opIntensityLimited))
		yticksMaxOps = [self.maxOps/2 for x in xticksOps]
		return xticksOps, yticksMaxOps

	def rooflineFigure(self, title, xlabel, ylabel, opIntensityLimited):
		# self.figure.suptitle(title, fontsize=16)
		self.axes.set_xlabel(xlabel, fontsize=self.fontsize)
		self.axes.set_ylabel(ylabel, fontsize=self.fontsize)
		xticks = list(range(1, opIntensityLimited))
		for x in xticks:
			if min(self.dramPeakMemoryBw*x, self.maxOps) == self.maxOps:
				xDramTickBoundary = x
				break
		xticksDram = list(range(1, xDramTickBoundary+1))
		for x in xticks:
			if min(self.nocPeakMemoryBw*x, self.maxOps) == self.maxOps:
				xNocTickBoundary = x
				break
		xticksNoC = list(range(1, xNocTickBoundary+1))
		if xDramTickBoundary <= xNocTickBoundary:
			xMaxOpsTickBoundary = xDramTickBoundary
		else:
			xMaxOpsTickBoundary = xNocTickBoundary
		xticksOps = list(range(xMaxOpsTickBoundary, opIntensityLimited))
		# yticksDram = [min(self.dramPeakMemoryBw*x, self.maxOps) for x in xticks]
		# yticksNoc = [min(self.nocPeakMemoryBw*x, self.maxOps) for x in xticks]
		yticksDram = [self.dramPeakMemoryBw*x for x in xticksDram]
		yticksNoc = [self.nocPeakMemoryBw*x for x in xticksNoC]
		yticksMaxOps = [self.maxOps for x in xticksOps]
		self.axes.plot(xticksDram, yticksDram, 'b')
		self.axes.plot(xticksNoC, yticksNoc, 'r')
		self.axes.plot(xticksOps, yticksMaxOps, 'k')
		if self.halfPerformance:
			xticksHalfOps, yticksMaxHalfOps = self.addHalfPerformance(opIntensityLimited)
			self.axes.plot(xticksHalfOps, yticksMaxHalfOps, 'k')
			performanceText = "Maximum computational performance 2.3TOPS"		
			self.axes.text((xMaxOpsTickBoundary+opIntensityLimited)/10, self.maxOps/2*1.1, performanceText, horizontalalignment='center', 
				verticalalignment='center', fontsize=self.fontsize)
		# 
		xDramTickLen = math.log(xDramTickBoundary) - math.log(xDramTickBoundary/2)
		yDramTickLen = math.log(self.dramPeakMemoryBw*xDramTickBoundary) - math.log(self.dramPeakMemoryBw*xDramTickBoundary/2)
		dramLineAngle = math.degrees(math.atan(yDramTickLen/xDramTickLen))
		if AcceleratorType.QPE == self.acceleratorType:
			dramText = "DRAM_bandwidth"
		else:
			dramText = "4 * DRAM_bandwidth"
		self.axes.text(xDramTickBoundary/10, self.dramPeakMemoryBw*xDramTickBoundary/8.5, dramText, horizontalalignment='center', 
			verticalalignment='center', rotation=dramLineAngle, fontsize=self.fontsize)
		xNocTickLen = math.log(xNocTickBoundary) - math.log(xNocTickBoundary/2)
		yNocTickLen = math.log(self.nocPeakMemoryBw*xNocTickBoundary) - math.log(self.nocPeakMemoryBw*xNocTickBoundary/2)
		nocLineAngle = math.degrees(math.atan(yNocTickLen/xNocTickLen))
		if AcceleratorType.QPE == self.acceleratorType:
			nocText = "NoC_bandwidth"
		else:
			nocText = "4 * NoC_bandwidth"
		self.axes.text(xNocTickBoundary/7, self.nocPeakMemoryBw*xNocTickBoundary/6, nocText, horizontalalignment='center', 
			verticalalignment='center', rotation=nocLineAngle, fontsize=self.fontsize)
		if AcceleratorType.QPE == self.acceleratorType:
			performanceText = "Maximum computational performance 127.8GOPS"
		else:
			performanceText = "Maximum computational performance 4.6TOPS"		
		self.axes.text((xMaxOpsTickBoundary+opIntensityLimited)/10, self.maxOps*1.1, performanceText, horizontalalignment='center', 
			verticalalignment='center', fontsize=self.fontsize)
		self.axes.set_xscale("log")
		self.axes.set_yscale("log")
		# # 1st method
		# self.axes.set_yticks([10, 20, 30, 40 ,100, 200, 300, 400, 1000, 2000, 3000, 4000])
		# self.axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
		# # self.axes.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x}:{pos}"))
		# 2nd method
		subs = [2.0, 3.0, 4.0, 6.0]  # ticks to show per decade
		self.axes.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=subs)) #set the ticks position
		self.axes.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(ticks_format))  #add the custom ticks
		for tick in self.axes.xaxis.get_major_ticks():
			tick.label.set_fontsize(16) 
		for tick in self.axes.yaxis.get_major_ticks():
			tick.label.set_fontsize(16) 	

	def addScatter(self, xticks, yticks, fmt, improveLineFmt, labelFlag=True, layerNames=None, lineName=LINE_NAME1):
		'''
		https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot

		Args:
			xticks: one-dimension list --> shape (M)
			yticks: two-dimension list --> shape (N,M) unit G
					[[fisrt scheme result], [second scheme result], [third scheme result]]
			fmt: one-dimension list, format,  --> shape (N)
		'''
		dislog = math.log(self.opIntensityLimited) / 300
		# Check dimension
		# pList = []
		M = len(xticks)
		N = len(yticks)
		for index in range(N):
			m = len(yticks[index])
			assert(M == m), "dimension of yticks[{}]-{} should be equal to {}".format(index, m, M)
		assert(N == len(fmt)), \
			"dimension of yticks-{} and fmt-{} should be equal".format(N, len(fmt))
		yticksTranspose = list(map(list, zip(*yticks)))
		# yTextTicks = self.tickSeperate(yticks)
		yTextTicks = self.getTextTicks(yticks)
		for step in range(len(yticks)):
			if labelFlag:
				p = self.axes.plot(xticks, yticks[step], fmt[step], label=lineName[step], linewidth=2, markersize=8)
			else:
				p = self.axes.plot(xticks, yticks[step], fmt[step], linewidth=2, markersize=8)
			# pList.append(p)
			self.axes.legend()
			# for i in range(len(xticks)):
			# 	self.axes.text(math.exp(math.log(xticks[i])+dislog), yTextTicks[step][i], 
			# 		"({:.2f}, {:.2f})".format(xticks[i], yticks[step][i]), color='blue', fontsize="medium")
		if layerNames is not None:
			for i in range(len(xticks)):
				self.axes.text(math.exp(math.log(xticks[i])+dislog), yticks[0][i], layerNames[i], color='blue', fontsize="large")			
		for step in range(len(yticksTranspose)):
			self.axes.plot([xticks[step]]*N, yticksTranspose[step], improveLineFmt[step])
		# self.axes.legend(tuple(pList), tuple(LINE_NAME[0:len(pList)]))

	def tickSeperate(self, ticks):
		textDis = math.log(self.maxOps) / 50
		# input("textDis: {}".format(textDis))
		seperatedTick = copy.deepcopy(ticks)
		# return seperatedTick
		midPosition = int(len(ticks) / 2)
		# 
		for x in range(len(seperatedTick[0])):
			for y in range(midPosition-1, -1, -1):
				distance = math.fabs(math.log(seperatedTick[y+1][x]) - math.log(seperatedTick[y][x]))
				# print("seperatedTick[{}][{}]: {}".format(y+1, x, seperatedTick[y+1][x]))
				# print("seperatedTick[{}][{}]: {}".format(y, x, seperatedTick[y][x]))
				# input("distance: {}".format(distance))
				if distance < textDis:
					if math.log(seperatedTick[y+1][x]) > math.log(seperatedTick[y][x]):
						seperatedTick[y][x] = math.exp((math.log(seperatedTick[y+1][x]) - textDis))
					else:
						seperatedTick[y][x] = math.exp((math.log(seperatedTick[y+1][x]) + textDis))
		# 
		# input("1>>> {}-{}".format(midPosition, len(ticks)-1))
		if midPosition <= len(ticks) - 1:
			# input("2>>> {}-{}".format(midPosition, len(ticks)-1))
			for x in range(len(seperatedTick[0])):
				for y in range(midPosition+1, len(ticks), 1):
					distance = math.fabs(math.log(seperatedTick[y][x]) - math.log(seperatedTick[y-1][x]))
					# print("seperatedTick[{}][{}]: {}".format(y+1, x, seperatedTick[y+1][x]))
					# print("seperatedTick[{}][{}]: {}".format(y, x, seperatedTick[y][x]))
					# input("distance: {}".format(distance))
					if distance < textDis:
						if seperatedTick[y-1][x] > seperatedTick[y][x]:
							seperatedTick[y][x] = math.exp(math.log(seperatedTick[y-1][x]) - textDis)
						else:
							seperatedTick[y][x] = math.exp(math.log(seperatedTick[y-1][x]) + textDis)
		return seperatedTick

	def getTextTicks(self, ticks):
		textDis = math.log(self.maxOps) / 52
		# input("textDis: {}".format(textDis))
		textTicks = np.tile(np.amin(np.asarray(ticks), axis=0),(len(ticks),1))
		textTicks = np.log(textTicks)
		print("textTicks: \n{}".format(textTicks))
		textTicksShape = textTicks.shape
		distanceMat = np.zeros((1, textTicksShape[1]))
		for index in range(1, textTicksShape[0]):
			distanceMat = np.concatenate((distanceMat, np.asarray([textDis*index]*textTicksShape[1]).reshape((1, textTicksShape[1]))), axis=0)
		print("distanceMat: \n{}".format(distanceMat))
		textTicks = np.add(textTicks, distanceMat)
		textTicks = np.exp(textTicks)
		print("textTicks: \n{}".format(textTicks))
		return textTicks.tolist()


	def plot(self):
		plt.show()

if __name__ == "__main__":
	roofline = RooflineModel()
	roofline.addScatter([18.4,30.,80.], [[50, 150, 200], [100, 200, 250], [150, 250, 300]], ['bx', 'g^', 'r^'])
	roofline.plot()