import numpy as np
import math
import time
from scipy.optimize import fsolve
import Tkinter as tk

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

print screen_width, 'x', screen_height


x1, y1, theta1 = 100, 100, 46
x2, y2, theta2 = 120, 200, 30
x3, y3, theta3 = 200, 120, 62

def equations(p):
    theta,x,y = p
    return ( theta - math.atan((x1 + x)/(y1 + y)) + theta1, \
              x     - (y2 + y)*math.tan(theta2 + theta) + x2,  \
              y     - (x3 + x)/math.tan(theta3 + theta) + y3)

print time.time()
theta, x, y  =fsolve(equations, (0,0,0))
print time.time()

print equations((theta,x, y))


