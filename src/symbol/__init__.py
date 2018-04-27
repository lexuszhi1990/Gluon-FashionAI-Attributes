# -*- coding: utf-8 -*-

from .densenet import get_pretrained_densenet, get_densenet121_net, get_densenet201_net

def get_symbol(network, num_classes, ctx):
    return eval('get_' + network + '_net')(num_classes, ctx)
