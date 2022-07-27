import torch
import numpy as np
import torch.nn as nn


def testtongji(output, pred, label, class_list, class_num, class_correct, w=-1):
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








