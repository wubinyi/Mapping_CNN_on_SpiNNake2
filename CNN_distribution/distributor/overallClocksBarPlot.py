import numpy as np
import matplotlib.pyplot as plt


# N = 5
# menMeans = (150, 160, 146, 172, 155)
# menStd = (20, 30, 32, 10, 20)

# fig, ax = plt.subplots()

# ind = np.arange(N)    # the x locations for the groups
# width = 0.35         # the width of the bars
# p1 = ax.bar(ind, menMeans, width, color='r', bottom=0, yerr=menStd)


# womenMeans = (145, 149, 172, 165, 200)
# womenStd = (30, 25, 20, 31, 22)
# p2 = ax.bar(ind + width, womenMeans, width,
#             color='y', bottom=0, yerr=womenStd)

# ax.set_title('Scores by group and gender')
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

# ax.legend((p1[0], p2[0]), ('Men', 'Women'))
# # ax.yaxis.set_units(inch)
# ax.autoscale_view()

# plt.show()

VGG_ARM_CLK = {'CONV_1': {'PADD': 5760, 'ACTI': 200704, 'QUAN': 200704}, 
				'CONV_2': {'PADD': 27648, 'ACTI': 262144, 'QUAN': 262144, 'POOL': 393216}, 
				'CONV_4': {'PADD': 36864, 'ACTI': 100352, 'QUAN': 100352}, 
				'CONV_5': {'PADD': 32768, 'ACTI': 114688, 'QUAN': 114688, 'POOL': 172032}, 
				'CONV_7': {'PADD': 32768, 'ACTI': 50176, 'QUAN': 50176}, 
				'CONV_8': {'PADD': 36864, 'ACTI': 50176, 'QUAN': 50176}, 
				'CONV_9': {'PADD': 32768, 'ACTI': 50176, 'QUAN': 50176, 'POOL': 75264}, 
				'CONV_11': {'PADD': 36864, 'ACTI': 25088, 'QUAN': 25088}, 
				'CONV_12': {'PADD': 24576, 'ACTI': 28672, 'QUAN': 28672}, 
				'CONV_13': {'PADD': 24576, 'ACTI': 28672, 'QUAN': 28672, 'POOL': 43008}, 
				'CONV_15': {'PADD': 16384, 'ACTI': 7168, 'QUAN': 7168}, 
				'CONV_16': {'PADD': 16384, 'ACTI': 7168, 'QUAN': 7168}, 
				'CONV_17': {'PADD': 16384, 'ACTI': 7168, 'QUAN': 7168, 'POOL': 10752}}

VGG_MLA_CLK = {"CONV_1":	119254,
				"CONV_2":	527999,
				"CONV_4": 	220028,
				"CONV_5": 	381753,
				"CONV_7":	214575,
				"CONV_8": 	421137,
				"CONV_9": 	413733,
				"CONV_11": 	263877,
				"CONV_12": 	447849,
				"CONV_13": 	446890,
				"CONV_15": 	189908,
				"CONV_16": 	189910,
				"CONV_17": 	189819,
				"FC_19":	3235078,
				"FC_20":	527622,
				"FC_21":	132044}

RESNET_ARM_CLK = {'CONV_1': {'PADD': 7560, 'ACTI': 57344, 'QUAN': 57344}, 
					'POOL_2': {'PADD': 3616, 'POOL': 18816}, 
					'CONV_3': {'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_4': {'PADD': 16384, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_5': {}, 
					'CONV_SC_5': {'MAT_ELE': 50176, 'ACTI': 50176, 'QUAN': 50176}, 
					'CONV_6': {'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_7': {'PADD': 16384, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_8': {'MAT_ELE': 50176, 'ACTI': 50176, 'QUAN': 50176}, 
					'CONV_9': {'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_10': {'PADD': 16384, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_11': {'MAT_ELE': 50176, 'ACTI': 50176, 'QUAN': 50176}, 
					'CONV_12': {'ACTI': 7168, 'QUAN': 7168}, 
					'CONV_13': {'PADD': 18432, 'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_14': {}, 'CONV_SC_14': {'MAT_ELE': 28672, 'ACTI': 28672, 'QUAN': 28672}, 
					'CONV_15': {'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_16': {'PADD': 18432, 'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_17': {'MAT_ELE': 25088, 'ACTI': 25088, 'QUAN': 25088}, 
					'CONV_18': {'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_19': {'PADD': 18432, 'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_20': {'MAT_ELE': 25088, 'ACTI': 25088, 'QUAN': 25088}, 
					'CONV_21': {'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_22': {'PADD': 18432, 'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_23': {'MAT_ELE': 25088, 'ACTI': 25088, 'QUAN': 25088}, 
					'CONV_24': {'ACTI': 3584, 'QUAN': 3584}, 
					'CONV_25': {'PADD': 18432, 'ACTI': 3136, 'QUAN': 3136}, 
					'CONV_26': {}, 
					'CONV_SC_26': {'MAT_ELE': 14336, 'ACTI': 14336, 'QUAN': 14336}, 
					'CONV_27': {'ACTI': 3584, 'QUAN': 3584}, 
					'CONV_28': {'PADD': 18432, 'ACTI': 3136, 'QUAN': 3136}, 
					'CONV_29': {'MAT_ELE': 12544, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_30': {'ACTI': 3584, 'QUAN': 3584}, 
					'CONV_31': {'PADD': 18432, 'ACTI': 3136, 'QUAN': 3136}, 
					'CONV_32': {'MAT_ELE': 12544, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_33': {'ACTI': 3584, 'QUAN': 3584}, 
					'CONV_34': {'PADD': 18432, 'ACTI': 3136, 'QUAN': 3136}, 
					'CONV_35': {'MAT_ELE': 12544, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_36': {'ACTI': 3584, 'QUAN': 3584}, 
					'CONV_37': {'PADD': 18432, 'ACTI': 3136, 'QUAN': 3136}, 
					'CONV_38': {'MAT_ELE': 12544, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_39': {'ACTI': 3584, 'QUAN': 3584}, 
					'CONV_40': {'PADD': 18432, 'ACTI': 3136, 'QUAN': 3136}, 
					'CONV_41': {'MAT_ELE': 12544, 'ACTI': 12544, 'QUAN': 12544}, 
					'CONV_42': {'ACTI': 1792, 'QUAN': 1792}, 
					'CONV_43': {'PADD': 36864, 'ACTI': 1568, 'QUAN': 1568}, 
					'CONV_44': {}, 
					'CONV_SC_44': {'MAT_ELE': 7168, 'ACTI': 7168, 'QUAN': 7168}, 
					'CONV_45': {'ACTI': 1792, 'QUAN': 1792}, 
					'CONV_46': {'PADD': 36864, 'ACTI': 1568, 'QUAN': 1568}, 
					'CONV_47': {'MAT_ELE': 6272, 'ACTI': 6272, 'QUAN': 6272}, 
					'CONV_48': {'ACTI': 1792, 'QUAN': 1792}, 
					'CONV_49': {'PADD': 36864, 'ACTI': 1568, 'QUAN': 1568}, 
					'CONV_50': {'MAT_ELE': 6272, 'ACTI': 6272, 'QUAN': 6272, 'POOL': 9408}}

RESNET_MLA_CLK = { "CONV_1": 	99623,
				"POOL_2": 	0,
				"CONV_3": 	15892,
				"CONV_4": 	37868,
				"CONV_5": 	45312,
				"CONV_SC_5": 	45294,
				"CONV_6": 	51087,
				"CONV_7": 	37874,
				"CONV_8": 	45297,
				"CONV_9": 	51082,
				"CONV_10": 	37869,
				"CONV_11": 	45300,
				"CONV_12": 	43659,
				"CONV_13": 	45029,
				"CONV_14": 	30382,
				"CONV_SC_14": 	115207,
				"CONV_15": 	33273,
				"CONV_16": 	45022,
				"CONV_17": 	30366,
				"CONV_18": 	33268,
				"CONV_19": 	44987,
				"CONV_20": 	30382,
				"CONV_21": 	33272,
				"CONV_22": 	45036,
				"CONV_23": 	30361,
				"CONV_24": 	45089,
				"CONV_25": 	56688,
				"CONV_26": 	32182,
				"CONV_SC_26": 	112026,
				"CONV_27": 	32740 ,
				"CONV_28": 	56687,
				"CONV_29": 	32183,
				"CONV_30": 	32773,
				"CONV_31": 	56689,
				"CONV_32": 	32184,
				"CONV_33": 	32759,
				"CONV_34": 	56689,
				"CONV_35": 	32180,
				"CONV_36": 	32724,
				"CONV_37": 	56688,
				"CONV_38": 	32185,
				"CONV_39": 	32757,
				"CONV_40": 	56687,
				"CONV_41": 	32181,
				"CONV_42": 	47116,
				"CONV_43": 	133136,
				"CONV_44": 	63808,
				"CONV_SC_44": 	169362,
				"CONV_45": 	83733,
				"CONV_46": 	133139,
				"CONV_47": 	63808,
				"CONV_48": 	83734,
				"CONV_49": 	133138,
				"CONV_50": 	63807,
				"FC_52" :  66513}

def clockCalculation(vggFlag=True):
	clockSummary = {"CONV":0, "FC":0, 'PADD': 0, 'MAT_ELE': 0, 'ACTI': 0, 'QUAN': 0, 'POOL': 0}
	if vggFlag:
		print("VGG-16:")
		mlaClocks = VGG_MLA_CLK
		armClocks = VGG_ARM_CLK
	else:
		print("RESNET-50:")
		mlaClocks = RESNET_MLA_CLK
		armClocks = RESNET_ARM_CLK

	blockNames = list(mlaClocks.keys())
	for blockName in blockNames:
		if "FC" in blockName:
			clockSummary["FC"] += mlaClocks[blockName]
		elif "CONV" in blockName:
			clockSummary["CONV"] += mlaClocks[blockName]
		elif "POOL" in blockName:
			clockSummary["POOL"] += mlaClocks[blockName]
		else:
			assert(False), "Unsupported"

		if blockName in armClocks:
			armOpers = armClocks[blockName]
			armOperNames = list(armOpers.keys())
			for armOperName in armOperNames:
				clockSummary[armOperName] += armOpers[armOperName]
	print("clockSummary: {}".format(clockSummary))
	return clockSummary

if __name__ == "__main__":
	vggClockSummary = clockCalculation(vggFlag=True)
	resnetClockSummary = clockCalculation(vggFlag=False)
	operationNames = list(vggClockSummary.keys())
	vggClocks = list(vggClockSummary.values())
	resnetClocks = list(resnetClockSummary.values())
	print("vgg sum clocks: {}".format(sum(vggClocks)))
	print("resnet sum clocks: {}".format(sum(resnetClocks)))
	# 
	import matplotlib.pylab as pylab
	          # 'figure.figsize': (15, 5),
	params = {'legend.fontsize': 'xx-large',
	         'axes.labelsize': 'xx-large',
	         'axes.titlesize':'xx-large',
	         'xtick.labelsize':'xx-large',
	         'ytick.labelsize':'xx-large'}
	pylab.rcParams.update(params)
	# 
	fig, ax = plt.subplots()
	N = 2
	ind = np.arange(N)    # the x locations for the groups
	width = 0.3      # the width of the bars: can also be len(x) sequence
	p = []
	for index in range(len(operationNames)):
		clockList = [vggClocks[index], resnetClocks[index]]
		bottomClocks = [sum(vggClocks[:index]), sum(resnetClocks[:index])]
		tempPlot = ax.bar(ind, clockList, width, bottom=bottomClocks)
		p.append(tempPlot)

	ax.set_xlabel("CNN model", fontsize=20)
	ax.set_ylabel("Clocks", fontsize=20)
	ax.set_xticks(ind)
	ax.set_xticklabels(["VGG-16", "RESNET-50"])
	ax.legend(p, operationNames)
	ax.autoscale_view()
	plt.show()


	# p1 = plt.bar(ind, menMeans, width, yerr=menStd)
	# p2 = plt.bar(ind, womenMeans, width,bottom=menMeans, yerr=womenStd)