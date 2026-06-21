import numpy as np

def minmax(series):
    if len(series) == 0:
        return series
    series = series.fillna(0)
    mn, mx = series.min(), series.max()
    if mn == mx:
        return series * 0
    return (series - mn) / (mx - mn)