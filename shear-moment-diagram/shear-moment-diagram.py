#
# Python-Based Shear and Bending Moment Diagram generator
# =============================================================
#
#    Author: Robert Grandin
#
#    Date:   November 2014  (Creation)
#
#
# PURPOSE:
#    This code calculates shear and bending moment diagrams for beams with
#    pin/roller connections.  Cantilevered beams are not suported!
#
#
# INSTRUCTIONS & NOTES:
#    - Specify the non-support loads using the dictionary structures like the
#      provided examples.  Reaction forces will be calculated automatically.
#    - For ease of use with multiple problems, create a python module which 
#      defines the necessary input and import the module into this script.
#      An example module file has been provided to demonstrate this method.
#
#
# HOMEWORK DISCLAIMER:
#   This tool is intended to be a learning aid.  Feel free to use it to check your
#   work, but do not use it in place of learning how to find the solution yourself.
#
#
#



#
# ========================
#
#    DATA FILE SAMPLE INPUT
#
#import exampleShearMoment                              # Name of python file, no extension
#reload(exampleShearMoment)                             # Force reload to catch any updates/revisions
#beam,units,makePlots,pointLoads,distLoads,reactionPositions = exampleShearMoment.DefineInputs()     # Call input-definition function
#
#beamStart = beam[0]
#beamEnd = beam[1]
#npoints = beam[2]
#
#unitsDist = units[0]
#unitsShear = units[1]
#unitsMoment = units[2]




# ========================
#
#    SAMPLE INPUT
#

beamStart = 0.0
beamEnd = 1.0
npoints = 10001

unitsDist = 'm'
unitsShear = 'N'
unitsMoment = 'N-m'

makePlots = True

pointLoads = [{ 'position': 0.5e0, 'type' : 'v', 'value' : -2.0e3}]
distLoads = [{'start' : 0.0e0, 'end' : 0.5e0, 'startVal' : -5.0e3, 'endVal' : -5.0e3, 'type' : 'v', 'shape' : 'linear'}]

reactionPositions = [0.0e0, 1.0e0]





















# =============================================================================================
#
#
#
#             NO EDITS REQUIRED BELOW HERE
#
#
#
# =============================================================================================






# ========================
#
#    IMPORT PYTHON MODULES REQUIRED FOR SOLUTION
#

import numpy                        # General numerical capability
import matplotlib.pyplot as plt     # 2D plotting






# ========================
#
#    DEFINE FUNCTIONS TO CALCULATE VALUES FOR DISTRIBUTED LOADS
#

def IntegrateDistLoad(load, x):
    
    retval = 0.0
    
    if(load['shape'] == 'linear'):
        a = load['startVal']
        b = load['endVal']
        x1 = load['start']
        x2 = load['end']
        distVal = lambda xx: (b - a)/(x2 - x1)*(xx - x1)
        retval = a*x + distVal(x)

    
    return retval





def CentroidDistLoad(load):
    
    retval = 0.0
    
    if(load['shape'] == 'linear'):
        a = load['startVal']
        b = load['endVal']
        x1 = load['start']
        x2 = load['end']
        Aval = 0.5*(a + b)*(x2 - x1)
        Axval = 1.0/6.0*(x2 - x1)*(a*(2*x1 + x2) + b*(x1 + 2*x2))
        retval = Axval/Aval

    
    return retval









# ========================
#
#    BEGIN ACTUAL CALCULATIONS
#

# Solve for support reactions

A = numpy.zeros((2,2))
b = numpy.zeros((2,1))

sumForce = 0.0
sumMoments = 0.0


# Sum applied forces
for i in range(len(pointLoads)):
    if(pointLoads[i]['type'] == 'v'):
            sumForce += pointLoads[i]['value']

for i in range(len(distLoads)):
    if(distLoads[i]['type'] == 'v'):
        sumForce += IntegrateDistLoad(distLoads[i], distLoads[i]['end'])


# Sum applied moments about 0.0
for i in range(len(pointLoads)):
    if(pointLoads[i]['type'] == 'v'):
        sumMoments += pointLoads[i]['value']*pointLoads[i]['position']
    if(pointLoads[i]['type'] == 'm'):
        sumMoments += pointLoads[i]['value']

for i in range(len(distLoads)):
    if(distLoads[i]['type'] == 'v'):
        force = IntegrateDistLoad(distLoads[i], distLoads[i]['end'])
        sumMoments += force*CentroidDistLoad(distLoads[i])


# Setup first equation: R1 + R2 = sumForce
A[0][0] = 1.0
A[0][1] = 1.0
b[0] = -sumForce


# Setup second equation: M1 + M2 = sumMoments
A[1][0] = reactionPositions[0]
A[1][1] = reactionPositions[1]
b[1] = -sumMoments


# Solve equations
x,res,rank,singularvals = numpy.linalg.lstsq(A,b)


# Add reactions as point loads
pointLoads.append({ 'position': reactionPositions[0], 'type' : 'v', 'value' : x[0]})
pointLoads.append({ 'position': reactionPositions[1], 'type' : 'v', 'value' : x[1]})





# Create arrays to store calculation results
x = numpy.linspace(beamStart, beamEnd, npoints)
v = numpy.zeros(x.shape)
m = numpy.zeros(x.shape)




# Loop through beam length to fill shear and moment values
dx = x[1] - x[0]
for i in range(x.shape[0]):
    
    # Loop through point loads.  When analyzing the beam from left-to-right, point
    # loads are included *after* they have been passed (e.g., x > loadPosition).
    for j in range(len(pointLoads)):
        
        # Check for point-forces
        if(pointLoads[j]['type'] == 'v' and pointLoads[j]['position'] <= x[i]):
            v[i] += pointLoads[j]['value']
            
        # Check for point-moments
        if(pointLoads[j]['type'] == 'm' and pointLoads[j]['position'] <= x[i]):
            m[i] += pointLoads[j]['value']


    # Loop through distributed loads.
    for j in range(len(distLoads)):
        
        # Find integral of distributed load from the load's start to the current
        # position and add its contribution to the shear force array.
        if(distLoads[j]['type'] == 'v' and distLoads[j]['start'] <= x[i] and distLoads[j]['end'] >= x[i]):
            v[i] += IntegrateDistLoad(distLoads[j], x[i])
            
        # If we're past the distributed load, add its total (i.e., integrated) 
        # contribution
        if(distLoads[j]['type'] == 'v' and distLoads[j]['end'] < x[i]):
            v[i] += IntegrateDistLoad(distLoads[j], distLoads[j]['end'])


        # Distributed moments are not supported





    # Perform simple integration of moment from shear force.  For a sufficiently-large
    # number of points this simple backward-difference approach should be acceptable.
    if(i > 0):
        m[i] += m[i-1] + v[i-1]*dx

    







# ========================
#
#    PRINT INFORMATION ABOUT APPLIED LOADS AND LOCATIONS OF MIN/MAX VALUES
#


print(' ')
print('Applied Loads')
print('--------------------------------------')
print(' ')
print('  Point Loads')
for i in range(len(pointLoads)):
    print('    x: %8.3f     Load: %8.3f     Type: %s' % (pointLoads[i]['position'], 
                                                           pointLoads[i]['value'], 
                                                           pointLoads[i]['type']))    

if(len(distLoads) > 0):
    print(' ')
    print('  Distributed Loads -- Only Shear Supported')
    for i in range(len(distLoads)):
        print('    x: ( %8.3f, %8.3f )   Loads:  ( %8.3f, %8.3f )   Shape: %s' % 
                                                        (distLoads[i]['start'], 
                                                         distLoads[i]['end'], 
                                                         distLoads[i]['startVal'],
                                                         distLoads[i]['endVal'],
                                                         distLoads[i]['shape']))  
print(' ')
print(' ')
print('Extreme Values')
print('--------------------------------------')
print(' ')
print('  Shear')
print('    Min: %8.3f at x = %8.3f' % (numpy.amin(v), x[numpy.argmin(v)]))
print('    Max: %8.3f at x = %8.3f' % (numpy.amax(v), x[numpy.argmax(v)]))
print(' ')
print('  Bending Moment')
print('    Min: %8.3f at x = %8.3f' % (numpy.amin(m), x[numpy.argmin(m)]))
print('    Max: %8.3f at x = %8.3f' % (numpy.amax(m), x[numpy.argmax(m)]))
print(' ')




# ========================
#
#    GENERATE PLOTS
#  

if(makePlots):    
    minx = numpy.min(x)
    maxx = numpy.max(x)
    minv = numpy.min(v)
    maxv = numpy.max(v)
    minm = numpy.min(m)
    maxm = numpy.max(m)
    
    xRange = maxx - minx
    vRange = maxv - minv
    mRange = maxm - minm
    
    bufferFraction = 0.02
    
    
    
    plt.figure()
    plt.plot(x, v, '-b')
    plt.hold(True)
    plt.plot([beamStart, beamEnd], [0.0, 0.0], '-k')
    for i in range(len(pointLoads)):
        xx = pointLoads[i]['position']
        plt.plot([xx,xx],[minv,maxv], '--k')
    for i in range(len(distLoads)):
        xx = distLoads[i]['start']
        plt.plot([xx,xx],[minv,maxv], '--k')
        xx = distLoads[i]['end']
        plt.plot([xx,xx],[minv,maxv], '--k')
    plt.title('Shear Force')
    plt.grid(True)
    plt.xlabel('Position [' + str(unitsDist) + ']')
    plt.ylabel('Shear Force [' + str(unitsShear) + ']')
    plt.xlim([minx - xRange*bufferFraction, maxx + xRange*bufferFraction])
    plt.ylim([minv - vRange*bufferFraction, maxv + vRange*bufferFraction])
    
    
    
    plt.figure()
    plt.plot(x, m, '-b')
    plt.hold(True)
    plt.plot([beamStart, beamEnd], [0.0, 0.0], '-k')
    for i in range(len(pointLoads)):
        xx = pointLoads[i]['position']
        plt.plot([xx,xx],[minv,maxv], '--k')
    for i in range(len(distLoads)):
        xx = distLoads[i]['start']
        plt.plot([xx,xx],[minv,maxv], '--k')
        xx = distLoads[i]['end']
        plt.plot([xx,xx],[minv,maxv], '--k')
    plt.title('Bending Moment')
    plt.grid(True)
    plt.xlabel('Position [' + str(unitsDist) + ']')
    plt.ylabel('Bending Moment [' + str(unitsMoment) + ']')
    plt.xlim([minx - xRange*bufferFraction, maxx + xRange*bufferFraction])
    plt.ylim([minm - mRange*bufferFraction, maxm + mRange*bufferFraction])