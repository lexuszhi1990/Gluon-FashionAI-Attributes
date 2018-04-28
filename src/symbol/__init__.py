# -*- coding: utf-8 -*-

from .densenet import get_densenet121_symbol, get_densenet201_symbol, get_pretrained_densenet121_symbol, get_pretrained_densenet201_symbol

def get_symbol(network, num_classes, ctx):
    net = eval('get_' + network + '_symbol')(num_classes, ctx)

    return net
