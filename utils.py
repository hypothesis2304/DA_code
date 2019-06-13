import numpy as np
import torch
import torch.nn as nn
import network

def get_copies(inputs_source, inputs_target, labels_source):
    inputs_source1, inputs_source2 = inputs_source, network.Augmenter(inputs_source.clone().detach().numpy())
    inputs_target1, inputs_target2 = inputs_target, network.Augmenter(inputs_target.clone().detach().numpy())
    inputs_source1, labels_source = inputs_source1.float().cuda(), labels_source.cuda()
    inputs_target1 = inputs_target1.float().cuda()
    inputs_source2 = torch.from_numpy(inputs_source2).float().cuda()
    inputs_target2 = torch.from_numpy(inputs_target2).float().cuda()
    return inputs_source1, inputs_source1, inputs_target1, inputs_target1, labels_source

def sigmoid_rampup(current, rampup_length):
    if rampup_length * 3 == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
