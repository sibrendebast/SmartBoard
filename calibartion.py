import numpy as np
import math
import time
from scipy.optimize import fsolve
import pygame


pygame.init()
screen = pygame.display.set_mode((0, 0),pygame.FULLSCREEN)
width, height = screen.get_size()
pygame.display.set_caption('Space Game')
pygame.mouse.set_visible(True)


color = (255,100,100)
pygame.draw.circle(screen, color, (1*width/4,height/2), 5)
pygame.draw.circle(screen, color, (2*width/4,height/2), 5)
pygame.draw.circle(screen, color, (3*width/4,height/2), 5)
pygame.display.update()



time.sleep(2)

pygame.quit()

##root = tk.Tk()
##
##
##
##
### make it cover the entire screen
##w, h = root.winfo_screenwidth(), root.winfo_screenheight()
##root.overrideredirect(1)
##root.geometry("%dx%d+0+0" % (w, h))
##root.focus_set() # <-- move focus to this widget
##
####canvas  = tk.Canvas(root)
##root.mainloop()
##root.destroy()



##
##x1, y1, theta1 = 100, 100, 45
##x2, y2, theta2 = 120, 200, 30
##x3, y3, theta3 = 200, 120, 62
##
##def equations(p):
##    theta,x,y = p
##    return ( theta - math.atan((x1 + x)/(y1 + y)) + theta1, \
##              x     - (y2 + y)*math.tan(theta2 + theta) + x2,  \
##              y     - (x3 + x)/math.tan(theta3 + theta) + y3)
##
##print time.time()
##theta, x, y  =fsolve(equations, (0,0,0))
##print time.time()
##
##print equations((theta,x, y))
##

