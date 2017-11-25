import os
import numpy as np
import math
import pandas as pd

from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# Quality functions
def qualitySSE(x,y):
    # Sum squared error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return ((x-y)**2).sum(), (x-y)**2

def qualityMSE(x,y):
    # Mean squared error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return ((x-y)**2).mean() , (x-y)**2

def qualityRMSE(x,y):
    # Root mean squared error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return (((x-y)**2).mean())**(0.5) , (x-y)**2

def qualityMAE(x,y):
    # Mean absolute error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return (x-y).abs().mean(), (x-y).abs()

def qualityMAPE(x,y):
    # Mean absolute percentage error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    qlt = ((x-y).abs()/x).replace([np.inf, -np.inf], np.nan)
    return qlt.mean() , (x-y).abs()

def qualityMACAPE(x,y):
    # Mean average corrected absolute percentage error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    qlt = (2*(x-y).abs()/(x+y)).replace([np.inf, -np.inf], np.nan)
    return qlt.mean() , (x-y).abs()

def qualityMedianAE(x,y):
    # Median absolute error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
    return ((x-y).abs()).median(), (x-y).abs()

def BuildForecast(h, ts, AlgName, AlgTitle, ParamsArray, step='D'):
	FRC_TS = dict()
	for p in ParamsArray:
		frc_horizon = pd.date_range(ts.index[-1], periods=h+1, freq=step)[1:]
		frc_ts = pd.DataFrame(index = ts.index.append(frc_horizon), columns = ts.columns)
		
		for cntr in ts.columns:
			frc_ts[cntr] = eval(AlgName)(ts[cntr], h, p)
		
#         frc_ts.columns = frc_ts.columns+('%s %s' % (AlgTitle, p))
		FRC_TS['%s %s' % (AlgTitle, p)] = frc_ts
	return FRC_TS

def plotTSForecast(ts, frc_ts, ts_num=0, alg_title=''):
	frc_ts.columns = ts.columns+'; '+alg_title
	ts[ts.columns[0]].plot(style='b', linewidth=1.0, marker='o')
	ax = frc_ts[frc_ts.columns[0]].plot(style='r-^', figsize=(15,5), linewidth=1.0)
	plt.xlabel('Time ticks')
	plt.ylabel('TS values')
	plt.legend()
	return ax
	
def InitExponentialSmoothing(x, h, Params):

    T = len(x)
    alpha = Params['alpha']
    AdaptationPeriod=Params['AdaptationPeriod']
    FORECAST = [np.NaN]*(T+h)
    if alpha>1:
        w.warn('Alpha can not be more than 1')
        #alpha = 1
        return FORECAST
    if alpha<0:
        w.warn('Alpha can not be less than 0')
        #alpha = 0
        return FORECAST
    y = x[0]
    t0=0
    for t in range(0, T):
        if not math.isnan(x[t]):
            if math.isnan(y):
                y=x[t]
                t0=t
            if (t-t0+1)<AdaptationPeriod:
                y = y*(1-alpha*(t-t0+1)/(AdaptationPeriod)) + alpha*(t-t0+1)/(AdaptationPeriod)*x[t]
            y = y*(1-alpha) + alpha*x[t]
            #else do not nothing
        FORECAST[t+h] = y
    return FORECAST	

def WintersExponentialSmoothing(x, h, Params):
    T = len(x)
    alpha = Params['alpha']
    delta = Params['delta']
    p = Params['seasonality_period']
    N = T//p
    
    FORECAST = [np.NaN]*(T+h)

    if alpha>1:
        w.warn('Alpha can not be more than 1')
        #alpha = 1
        return FORECAST
    if alpha<0:
        w.warn('Alpha can not be less than 0')
        #alpha = 0
        return FORECAST
    if delta>1:
        w.warn('beta can not be more than 1')
        #beta = 1
        return FORECAST
    if delta<0:
        w.warn('beta can not be less than 0')
        #beta = 0
        return FORECAST
    
    l = np.nan # initialize ts level 
    s = np.zeros(T) # initalize seasonality values (it must be vector of lenth p)
    A = np.zeros(N)
    
    for t in range(T):
        if not math.isnan(x[t]):
            if math.isnan(l):
                l= x[t]
            if np.all(s == 0):
                for j in range(N):
                        A[j] = np.sum([x[p * j + k] for k in range(p)])/p
                for i in range(p):
                    s[i] = np.sum([x[p * j + i]/A[j] for j in range(N)])/N
            
            if t >= p:
                l_prev = l
                l = alpha * (x[t] - s[t - p]) + (1 - alpha) * l_prev # recurrent smoothing of level
                s[t] = delta * (x[t] - l_prev) + (1 - delta) * s[t - p] # recurrent smoothing of seasonality
            
        FORECAST[t + h] = l + s[t - p + 1 + (h - 1) % p]
    return FORECAST 

def TheilWageExponentialSmoothing(x, h, Params):
    T = len(x)
    alpha = Params['alpha']
    beta = Params['beta']
    delta = Params['delta']
    p = Params['seasonality_period']
    N = T//p
    
    FORECAST = [np.NaN]*(T+h)

    if alpha>1:
        w.warn('Alpha can not be more than 1')
        #alpha = 1
        return FORECAST
    if alpha<0:
        w.warn('Alpha can not be less than 0')
        #alpha = 0
        return FORECAST
    if delta>1:
        w.warn('beta can not be more than 1')
        #beta = 1
        return FORECAST
    if delta<0:
        w.warn('beta can not be less than 0')
        #beta = 0
        return FORECAST
    
    l = np.nan # initialize ts level 
    b = np.nan 
    s = np.zeros(T) # initalize seasonality values (it must be vector of lenth p)
    A = np.zeros(N)
    
    for t in range(T):
        if not math.isnan(x[t]):
            if math.isnan(l):
                l = x[t]
            if math.isnan(b):
                b = np.sum([(x[p + i] - x[i])/p for i in range(1, p + 1)])/p
            if np.all(s == 0):
                for j in range(N):
                        A[j] = np.sum([x[p * j + k] for k in range(p)])/p
                for i in range(p):
                    s[i] = np.sum([x[p * j + i]/A[j] for j in range(N)])/N
            
            if t >=p:
                l_prev = l
                b_prev = b
                l = alpha * (x[t] - s[t - p]) + (1 - alpha) * (l_prev + b_prev) # recurrent smoothing of level
                b = beta * (l - l_prev) + (1 - beta) * b_prev # reccurent smoothing of trend
                s[t] = delta * (x[t] - l_prev - b_prev) + (1 - delta) * s[t - p] # recurrent smoothing of seasonality
            
        FORECAST[t + h] = l + h * b + s[t - p + 1 + (h - 1) % p]
    return FORECAST

def MultiplicativeExponentialSmoothing(x, h, Params):
    T = len(x)
    alpha = Params['alpha']
    beta = Params['beta']
    delta = Params['delta']
    p = Params['seasonality_period']
    N = T//p
    
    FORECAST = [np.NaN]*(T+h)

    if alpha>1:
        w.warn('Alpha can not be more than 1')
        #alpha = 1
        return FORECAST
    if alpha<0:
        w.warn('Alpha can not be less than 0')
        #alpha = 0
        return FORECAST
    if delta>1:
        w.warn('beta can not be more than 1')
        #beta = 1
        return FORECAST
    if delta<0:
        w.warn('beta can not be less than 0')
        #beta = 0
        return FORECAST
    
    l = np.nan # initialize ts level 
    b = np.nan 
    s = np.zeros(T) # initalize seasonality values (it must be vector of lenth p)
    A = np.zeros(N)
    
    for t in range(T):
        if not math.isnan(x[t]):
            if math.isnan(l):
                l = x[t]
            if math.isnan(b):
                b = np.sum([(x[p + i] - x[i])/p for i in range(1, p + 1)])/p
            if np.all(s == 0):
                for j in range(N):
                        A[j] = np.sum([x[p * j + k] for k in range(p)])/p
                for i in range(p):
                    s[i] = np.sum([x[p * j + i]/A[j] for j in range(N)])/N
            
            if t >= p:
                l_prev = l
                b_prev = b
                l = alpha * (x[t] / s[t - p]) + (1 - alpha) * (l_prev + b_prev) # recurrent smoothing of level
                b = beta * (l - l_prev) + (1 - beta) * b_prev # reccurent smoothing of trend
                s[t] = delta * (x[t] / l) + (1 - delta) * s[t - p] # recurrent smoothing of seasonality
            
        FORECAST[t + h] = (l + h * b) * s[t - p + 1 + (h - 1) % p]
    return FORECAST