import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import pdb
import pred_var
import train

def confident_samples(model, target_samples, k, class_num, bs, filter=10):
	target_out = []
	k = k * bs
	confidence = torch.zeros((target_samples.size(0),class_num))
	with torch.no_grad():
		model.eval()
		for i in range(filter):
			features, outputs = model(target_samples)
			predictions = nn.Softmax(dim=1)(outputs)
			target_out.append(torch.unsqueeze(predictions, 2))
			target_out += target_out
			confidence += predictions.to('cpu')
		all_predictions = torch.cat(target_out, dim=2)
		predictive_variance = pred_var.predictiveVariance(all_predictions)
		confidence /= (filter*1.0)
		max_values = confidence.max(dim=1)
		max_class = confidence.argmax(dim=1)
		rank_samples = torch.argsort(predictive_variance)
		samples_to_select = rank_samples[:int(k)]
	model.train()
	return samples_to_select
