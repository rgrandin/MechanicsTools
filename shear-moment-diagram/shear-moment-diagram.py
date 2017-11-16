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
#    pin/roller connections.
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
#    SAMPLE INPUT - Cantilevered Beam
#

beamStart = 0.0
beamEnd = 14.0
npoints = int(2000.0 * (beamEnd - beamStart) + 1.0)

unitsDist = 'm'
unitsShear = 'N'
unitsMoment = 'N-m'

makePlots = True

pointLoads = [{ 'position': 10.0e0, 'type' : 'v', 'value' : -7.5e3},
              { 'position': 14.0e0, 'type' : 'v', 'value' : -6.0e3},
              { 'position': 14.0e0, 'type' : 'm', 'value' : -40.0e3}]
distLoads = [{'start' : 0.0e0, 'end' : 10.0e0, 'startVal' : -2.0e3, 'endVal' : -2.0e3, 'type' : 'v', 'shape' : 'linear'},
             {'start' : 10.0e0, 'end' : 14.0e0, 'startVal' : -1.0e3, 'endVal' : -1.0e3, 'type' : 'v', 'shape' : 'linear'}]

reactionPositions = [0.0e0]




# ========================
#
#    SAMPLE INPUT - Cantilevered Beam 2
#

beamStart = 0.0
beamEnd = 3.0
npoints = int(2000.0 * (beamEnd - beamStart) + 1.0)

unitsDist = 'm'
unitsShear = 'N'
unitsMoment = 'N-m'

makePlots = True

pointLoads = [{ 'position': 3.0e0, 'type' : 'v', 'value' : -6.0e3}]
distLoads = [{'start' : 0.0e0, 'end' : 1.5e0, 'startVal' : -8.0e3, 'endVal' : -8.0e3, 'type' : 'v', 'shape' : 'linear'}]

reactionPositions = [0.0e0]




# ========================
#
#    SAMPLE INPUT - Cantilevered Beam 3
#

beamStart = 0.0
beamEnd = 10.0
npoints = int(2000.0 * (beamEnd - beamStart) + 1.0)

unitsDist = 'ft'
unitsShear = 'lb'
unitsMoment = 'lb-ft'

makePlots = True

pointLoads = [{ 'position': 10.0e0, 'type' : 'v', 'value' : -100.0e0},
              { 'position': 5.0e0,  'type' : 'm', 'value' : -800.0e0}]
distLoads = []

reactionPositions = [0.0e0]





# ========================
#
#    SAMPLE INPUT - Simply-supported beam
#

# beamStart = 0.0
# beamEnd = 1.0
# npoints = 10001
#
# unitsDist = 'm'
# unitsShear = 'N'
# unitsMoment = 'N-m'
#
# makePlots = True
#
# pointLoads = [{ 'position': 0.5e0, 'type' : 'v', 'value' : -2.0e3}]
# distLoads = [{'start' : 0.0e0, 'end' : 0.5e0, 'startVal' : -5.0e3, 'endVal' : -5.0e3, 'type' : 'v', 'shape' : 'linear'}]
#
# reactionPositions = [0.0e0, 1.0e0]



# ========================
#
#    SAMPLE INPUT - Simply-supported beam 2
#

# beamStart = 0.0
# beamEnd = 6.0
# npoints = int(2000.0 * (beamEnd - beamStart) + 1.0)
#
# unitsDist = 'm'
# unitsShear = 'kN'
# unitsMoment = 'kN-m'
#
# makePlots = True
#
# pointLoads = []
# distLoads = [{'start' : 0.0e0, 'end' : 1.5e0, 'startVal' : -6.0e0, 'endVal' : -6.0e0, 'type' : 'v', 'shape' : 'linear'},
#              {'start' : 4.5, 'end' : 6.0e0, 'startVal' : -6.0e0, 'endVal' : -6.0e0, 'type' : 'v', 'shape' : 'linear'}]
#
# reactionPositions = [1.5e0, 4.5e0]












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
        retval = a*(x - x1) + distVal(x)


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


if(len(reactionPositions) > 1):

    A = numpy.zeros((2,2))
    b = numpy.zeros((2,1))


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

    pass   # if(len(reactionPositions) > 1)

else:

    pointLoads.append({ 'position': reactionPositions[0], 'type' : 'v', 'value' : -sumForce})
    pointLoads.append({ 'position': reactionPositions[0], 'type' : 'm', 'value' : sumMoments})

    pass   # if(len(reactionPositions) == 1)



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
        m[i] = m[i-1] + v[i-1]*dx   # Negative sign due to right-hand-rule ?


        # Loop through point loads.  When analyzing the beam from left-to-right, point
        # loads are included *after* they have been passed (e.g., x > loadPosition).
        for j in range(len(pointLoads)):

            # Check for point-moments
            if(pointLoads[j]['type'] == 'm' and numpy.abs(pointLoads[j]['position'] - x[i]) < dx*0.5):
                m[i] -= pointLoads[j]['value']








# ========================
#
#    PRINT INFORMATION ABOUT APPLIED LOADS AND LOCATIONS OF MIN/MAX VALUES
#

keypoints = []
for i in range(len(pointLoads)):
    keypoints.append(pointLoads[i]['position'])
for i in range(len(distLoads)):
    keypoints.append(distLoads[i]['start'])
    keypoints.append(distLoads[i]['end'])

keypoints = list(set(keypoints))
keypoints.sort()
keypointsIdx = []
for i in range(len(keypoints)):
    keypointsIdx.append(int(keypoints[i] / dx + 0.5))



print(' ')
print('Applied Loads')
print('--------------------------------------')
print(' ')
print('  Point Loads')
for i in range(len(pointLoads)):
    unit = unitsShear
    if(pointLoads[i]['type'] == 'm'):
        unit = unitsMoment
    print('    x: %10.3f [%s]    Load: %10.3f [%s]    Type: %s' % (
                                                           pointLoads[i]['position'],
                                                           str(unitsDist),
                                                           pointLoads[i]['value'],
                                                           str(unit),
                                                           pointLoads[i]['type']))

if(len(distLoads) > 0):
    print(' ')
    print('  Distributed Loads -- Only Shear Enabled')
    for i in range(len(distLoads)):
        unit = unitsShear
        if(distLoads[i]['type'] == 'm'):
            unit = unitsMoment
        print('    x: ( %10.3f, %10.3f ) [%s]    Loads:  ( %10.3f, %10.3f ) [%s]    Shape: %s' %
                                                        (distLoads[i]['start'],
                                                         distLoads[i]['end'],
                                                         str(unitsDist),
                                                         distLoads[i]['startVal'],
                                                         distLoads[i]['endVal'],
                                                         str(unit),
                                                         distLoads[i]['shape']))
print(' ')
print(' ')
print('Key-Point Values')
print('--------------------------------------')
print(' ')
for i in range(len(keypoints)):
    print('    x: %10.3f [%s]    Shear: %10.3f [%s]    Moment: %10.3f [%s]' % (
                                                           keypoints[i],
                                                           str(unitsDist),
                                                           v[keypointsIdx[i]],
                                                           str(unitsShear),
                                                           m[keypointsIdx[i]],
                                                           str(unitsMoment)))
print(' ')
print(' ')
print('Extreme Values')
print('--------------------------------------')
print(' ')
print('  Shear')
print('    Min: %10.3f [%s]  at x = %10.3f [%s]' % (numpy.amin(v), str(unitsShear), x[numpy.argmin(v)], str(unitsDist)))
print('    Max: %10.3f [%s]  at x = %10.3f [%s]' % (numpy.amax(v), str(unitsShear), x[numpy.argmax(v)], str(unitsDist)))
print(' ')
print('  Bending Moment')
print('    Min: %10.3f [%s]  at x = %10.3f [%s]' % (numpy.amin(m), str(unitsMoment), x[numpy.argmin(m)], str(unitsDist)))
print('    Max: %10.3f [%s]  at x = %10.3f [%s]' % (numpy.amax(m), str(unitsMoment), x[numpy.argmax(m)], str(unitsDist)))
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
    plt.savefig('ShearDiagram.png')



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
    plt.savefig('MomentDiagram.png')
