#
# Example file for defining shear and bending moment
# =============================================================
#
#    Author: Robert Grandin
#
#    Date:   November 2014  (Creation)
#
#
# 
#


#def  DefineInputs():
#    beamStart = 0.0
#    beamEnd = 50.0
#    npoints = 10001
#    beam = [beamStart, beamEnd, npoints]
#    
#    unitsDist = 'ft'
#    unitsShear = 'kip'
#    unitsMoment = 'kip-ft'
#    units = [unitsDist, unitsShear, unitsMoment]
#    
#    makePlots = True
#    
#    pointLoads = [{ 'position': 0.0e0, 'type' : 'v', 'value' : -10.0e0}]
#    pointLoads.append({ 'position': 37.5e0, 'type' : 'm', 'value' : -50.0e0})
#    
#    distLoads = [{'start' : 10.0e0, 'end' : 25.e0, 'startVal' : -1.0e0, 'endVal' : -1.0e0, 'type' : 'v', 'shape' : 'linear'}]
#    
#    reactionPositions = [10.0e0, 25.0e0]
#    
#    return beam,units,makePlots,pointLoads,distLoads,reactionPositions