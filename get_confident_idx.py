import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import pdb
import pred_var
import train
import network

def confident_samples(model, target_samples, k, class_num, bs, filter=10):
    target_out = []
    k = k * bs
    with torch.no_grad():
        model.eval()
        for i in range(filter):
            aug_target = network.Augmenter(target_samples.clone().detach().numpy())
            aug_target = torch.from_numpy(aug_target).float().cuda()
            features, outputs = model(aug_target)
            predictions = nn.Softmax(dim=1)(outputs)
            target_out.append(torch.unsqueeze(predictions, 2))
        all_predictions = torch.cat(target_out, dim=2)
        predictive_variance = pred_var.predictiveVariance(all_predictions)
        rank_samples = torch.argsort(predictive_variance)
        samples_to_select = rank_samples[:int(k)]
    model.train()
    return samples_to_select
