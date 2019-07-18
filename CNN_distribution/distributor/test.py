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
# from spiNNaker2TriBlockSimulator import SpiNNaker2TriBlock
# from spiNNakerSimulatorGeneral import *
# from dataGenerator import convDataSplitter
# import numpy as np
# from nnGeneral import PE_SRAM_LIMIT, BasicOperation
# from convLayerMapper import ConvLayerMapper

DistributionGeneralClass.resNet50(decreaseSize=True, forSpiNNaker2=True)