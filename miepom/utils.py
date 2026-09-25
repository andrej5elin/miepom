"""Utility functions for the MIEPOM library."""

import time

def bar(i,count):
    i = int(i/count*50)
    return "|"+ "="*i + "-"*(50-i)+ "|"

def print_progress(i, count = 100, text = "Progress..."):
    def _t(s):
        m = s/60.
        h = m/60.
        if h> 1.:
            return h,"h"
        elif m > 1.:
            return m, "m"
        else:
            return s,"s"      
    if i == 0:
        print_progress.t0 = time.time()
        print(bar(i,count), f"{100*i/count:.1f} %", end = "\r")                      
    else:
        t1 = time.time() 
        if count == i:
            dt = t1-print_progress.t0
            dt,unit = _t(dt)
            print(bar(i,count), f"{100*i/count:02.1f} % Done in {dt:>4.1f} {unit}")
        else:
            ETA = (t1 - print_progress.t0)*(count - i)/i
            ETA,unit = _t(ETA)
            print(bar(i,count), f"{100*i/count:>4.1f} % ETA: {ETA:>4.1f} {unit}", end = "\r")
    
def print_line(symbol = "-", count = 72):
    print(symbol * count)
    
