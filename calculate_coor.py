
import math

width = 395
height = 280

theta = 28/180.0*3.14159265358
phi = (90 - 30)/180.0*3.14159265358
alpha = 60.0/180.0*3.14159265358
beta = 15.0/180.0*3.14159265358


    ## analytical
x = width/(1+math.tan(phi)/math.tan(theta))
y = x/math.tan(theta)
print x,y
##
##    ## set of equations using 2 angles
##    a = np.array([[1,-math.tan(theta)],[1,math.tan(phi)]])
##    b = np.array([0,width])
##    x = np.linalg.solve(a,b)
##    print x

    ## set of equations using 3 angles, solved using least squares
##
##    A = np.array([[1,-math.tan(theta)],[1,math.tan(phi)],[1,math.tan(alpha)],[-1, math.tan(beta)]])
##    y = np.array([0,width,math.tan(alpha)*height,math.tan(beta)*height-width])
##
##    B = np.linalg.lstsq(A,y)
##    return (B[0][0],B[0][1])


