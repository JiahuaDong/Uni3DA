import torch
import numpy as np
import torch.nn as nn

output = np.array([0.9, 0.8, 0.6, 0.5, 0.4, 0.78, 0.63, 0.66, 0.45, 0.19])
label = np.array([0, 1, 2, 3, 9, 4, 5, 4, 8, 6])
pred = np.array([0, 2, 2, 4, 5, 4, 5, 3, 3, 5])


def testtongji(output, pred, label, class_list, class_num, class_correct, w=-1):
	#output =nn.Softmax(-1)(output)
	source_class = class_list
	w0 = w
	for each_output, each_pred, each_label in zip(output, pred, label):
		if each_output < w0:
			if each_label not in source_class:
				class_num[15] += 1
				class_correct[15] += 1
			else:
				class_num[each_label] += 1
		else:
			if each_pred == each_label:
				class_num[each_label] += 1
				class_correct[each_label] += 1
			else:
				if each_label not in source_class:
					class_num[15] += 1
				else:
					class_num[each_label] += 1
	return class_num, class_correct

	# for i in range(7):
	# 	class_acc[i] = class_correct[i]/class_num[i]
	# print(class_acc)




output = torch.from_numpy(output)
label = torch.from_numpy(label)
pred = torch.from_numpy(pred)
class_correct = [0, 0, 0, 0, 0, 0, 0]
class_num = [0, 0, 0, 0, 0, 0, 0]
source_class = [0,1,2,3,4,5]
#testtongji(output=output, label= label, pred=pred, class_list=source_class, class_num=class_num, class_correct=class_correct)

