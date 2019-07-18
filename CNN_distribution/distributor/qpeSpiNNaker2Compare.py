
QPE_VGG_NO_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 2294223,
	# "CONV_4": 5828888,
	"CONV_5": 16207531,
	# "CONV_7": 6247798,
	"CONV_9": 12616201,
	"CONV_11": 6369574,
	"CONV_15": 3261186,
	"FC_21": 546718
}

QPE_VGG_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 835667,
	# "CONV_4": 5572513,
	"CONV_5": 11050716,
	# "CONV_7": 6128823,
	"CONV_9": 12417980,
	"CONV_11": 6262524,
	"CONV_15": 3254902,
	"FC_21": 532670
}

QPE_VGG_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 1065837,
	# "CONV_4": 4990816,
	"CONV_5": 9697264,
	# "CONV_7": 5449664,
	"CONV_9": 10838832,
	"CONV_11": 5419584,
	"CONV_15": 2918496
}

# ==================================================================================

QPE_RESNET_NO_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1" : 	2402170,
	"CONV_5":	2246963,
	"CONV_7":	813317,
	"CONV_SC_14":	15341673,
	"CONV_19": 	806278,
	"CONV_25": 	761451,
	"CONV_26":	422420,
	"CONV_42":	4145252,
	"CONV_43": 	1644412,
	"FC_52": 276382
}

QPE_RESNET_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 	2304196,
	"CONV_5":	1717913,
	"CONV_7":	785374,
	"CONV_SC_14":	13356221,
	"CONV_19": 	788695,
	"CONV_25": 	753440,
	"CONV_26":	383365,
	"CONV_42":	3675771,
	"CONV_43": 	1527487,
	"FC_52": 269502
}

QPE_RESNET_FUSION_DATA_REUSE_RESULT = {
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

# ===================================================================================

SPI_VGG_NO_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 494729,
	# "CONV_4": 1426347,
	"CONV_5": 4139243,
	# "CONV_7": 1247556,
	"CONV_9": 2538019,
	"CONV_11": 1409122,
	"CONV_15": 1457186,
	"FC_21": 150355
}

SPI_VGG_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 114249,
	# "CONV_4": 1165281,
	"CONV_5": 3557807,
	# "CONV_7": 1117243,
	"CONV_9": 2437194,
	"CONV_11": 1302970,
	"CONV_15": 1446968,
	"FC_21": 132044
}

SPI_VGG_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 119254,
	# "CONV_4": 220028,
	"CONV_5": 381753,
	# "CONV_7": 214575,
	"CONV_9": 413733,
	"CONV_11": 263877,
	"CONV_15": 189908
}

# ===================================================================================

SPI_RESNET_NO_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1":	216184,
	"CONV_5": 	332016,
	"CONV_7": 	169620,
	"CONV_SC_14":	1088654,
	"CONV_19":	190469,
	"CONV_25":	206821,
	"CONV_26": 	159985,
	"CONV_42": 	356132,
	"CONV_43":	408563,
	"FC_52": 75575
}

SPI_RESNET_FUSION_NO_DATA_REUSE_RESULT = {
	"CONV_1": 	157468,
	"CONV_5":	228080,
	"CONV_7":	154441,
	"CONV_SC_14":	939526,
	"CONV_19": 	177575,
	"CONV_25": 	204440,
	"CONV_26":	128559,
	"CONV_42":	325505,
	"CONV_43": 	278969,
	"FC_52": 66513
}

SPI_RESNET_FUSION_DATA_REUSE_RESULT = {
	"CONV_1": 99623,
	"CONV_5": 45312,
	"CONV_7": 37874,
	"CONV_SC_14": 115207,
	"CONV_19": 44987,
	"CONV_25": 56688,
	"CONV_26": 32182,
	"CONV_42": 47116,
	"CONV_43": 133136
}



QPE_VGG_DIFF_DISTRI_SCHEME_RESULTS = [QPE_VGG_NO_FUSION_NO_DATA_REUSE_RESULT, QPE_VGG_FUSION_NO_DATA_REUSE_RESULT, 
											QPE_VGG_FUSION_DATA_REUSE_RESULT]

QPE_RESNET_DIFF_DISTRI_SCHEME_RESULTS = [QPE_RESNET_NO_FUSION_NO_DATA_REUSE_RESULT, QPE_RESNET_FUSION_NO_DATA_REUSE_RESULT, 
											QPE_RESNET_FUSION_DATA_REUSE_RESULT]

SPI_VGG_DIFF_DISTRI_SCHEME_RESULTS = [SPI_VGG_NO_FUSION_NO_DATA_REUSE_RESULT, SPI_VGG_FUSION_NO_DATA_REUSE_RESULT, 
											SPI_VGG_FUSION_DATA_REUSE_RESULT]

SPI_RESNET_DIFF_DISTRI_SCHEME_RESULTS = [SPI_RESNET_NO_FUSION_NO_DATA_REUSE_RESULT, SPI_RESNET_FUSION_NO_DATA_REUSE_RESULT, 
											SPI_RESNET_FUSION_DATA_REUSE_RESULT]

DISTRI_SCHEME_RESULTS = [[QPE_VGG_NO_FUSION_NO_DATA_REUSE_RESULT, SPI_VGG_NO_FUSION_NO_DATA_REUSE_RESULT], 
							[QPE_VGG_FUSION_NO_DATA_REUSE_RESULT, SPI_VGG_FUSION_NO_DATA_REUSE_RESULT], 
							[QPE_VGG_FUSION_DATA_REUSE_RESULT, SPI_VGG_FUSION_DATA_REUSE_RESULT],
							[QPE_RESNET_NO_FUSION_NO_DATA_REUSE_RESULT, SPI_RESNET_NO_FUSION_NO_DATA_REUSE_RESULT],
							[QPE_RESNET_FUSION_NO_DATA_REUSE_RESULT, SPI_RESNET_FUSION_NO_DATA_REUSE_RESULT],
							[QPE_RESNET_FUSION_DATA_REUSE_RESULT, SPI_RESNET_FUSION_DATA_REUSE_RESULT]]

LINE_NAME = ["QPE", "SpiNNaker2"]

def barPlot():
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

	for distri_result in DISTRI_SCHEME_RESULTS:
		# 
		keys = list(distri_result[0].keys())
		# 
		values = []
		for index in range(len(distri_result)):
			values.append(list(distri_result[index].values()))
			# 
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
		ax.legend(p, LINE_NAME)
		ax.autoscale_view()
		plt.show()


	# if vggFlag:
	# 	fcDistributionResults = VGG_FC_DIFF_DISTRI_RESULTS
	# 	fcDistributionResults.append({"FC_21":0})
	# 	distributionResults = VGG_CONV_DIFF_DISTRI_SCHEME_RESULTS
	# else:
	# 	fcDistributionResults = RESNET_FC_DIFF_DISTRI_RESULTS
	# 	fcDistributionResults.append({"FC_52":0})
	# 	distributionResults = RESNET_CONV_DIFF_DISTRI_SCHEME_RESULTS

	# for index in range(len(distributionResults)):
	# 	fcResult = fcDistributionResults[index]
	# 	convResult = distributionResults[index]
	# 	distributionResults[index] = {**convResult, **fcResult}
	# # print(distributionResults)
	# # 
	# keys = list(distributionResults[0].keys())
	# # print(keys)
	# values = []
	# for index in range(len(distributionResults)):
	# 	values.append(list(distributionResults[index].values()))
	# # print(values)
	# elementsInGroup = len(values)
	# N = len(keys)
	# fig, ax = plt.subplots()
	# ind = np.arange(N)*1.5    # the x locations for the groups
	# width = 0.35         # the width of the bars
	# p = []
	# for index in range(elementsInGroup):
	# 	pTemp = ax.bar(ind+width*index, values[index], width, color=colors[index], bottom=0)
	# 	p.append(pTemp)
	# # 
	# # ax.set_title('Scores by group and gender')
	# ax.set_xlabel("Operations", fontsize=20)
	# ax.set_ylabel("Clocks", fontsize=20)
	# ax.set_xticks(ind + width*(elementsInGroup-1) / 2)
	# ax.set_xticklabels(keys)
	# ax.legend(p, LINE_NAME)
	# ax.autoscale_view()
	# plt.show()

if __name__ == "__main__":
	barPlot()