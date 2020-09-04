import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def jma_filter(series_last:float, e0_last:float, e1_last:float, e2_last:float, jma_last:float, 
    length=7, phase=50, power=2):
    if phase < -100:
        phase_ratio = 0.5
    elif phase > 100:
        phase_ratio = 2.5
    else:    
        phase_ratio = phase / (100 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = pow(beta, power)
    e0 = (1 - alpha) * series_last + alpha * e0_last
    e1 = (series_last - e0) * (1 - beta) + beta * e1_last
    e2 = (e0 + phase_ratio * e1 - jma_last) * pow(1 - alpha, 2) + pow(alpha, 2) * e2_last
    jma = e2 + jma_last
    return jma, e0, e1, e2, 


def appy_jma_filter(series, length:int, phase:int, power:int):
    jma = [series[0]]
    e0 = [0]
    e1 = [0]
    e2 = [0]
    for value in series:
        jma_next, e0_next, e1_next, e2_next  = jma_filter(
            series_last=value, e0_last=e0[-1], e1_last=e1[-1],
            e2_last=e2[-1], jma_last=jma[-1], length=length,
            phase=phase, power=power
        )
        jma.append(jma_next)
        e0.append(e0_next)
        e1.append(e1_next)
        e2.append(e2_next)
    jma.pop(0)
    jma[0:length] = [None] * length
    return jma

