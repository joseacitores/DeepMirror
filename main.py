#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:33:57 2022

@author: josemiguelacitores
"""

from models import *


if __name__ == '__main__':
    res = benchmark.benchmark_models([Tree(),GCN(),Transformer()])
    print(res)