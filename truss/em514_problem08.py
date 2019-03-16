# Generate input data for EM 514, Homework Problem 8

def  DefineInputs():
    area = 200.0e-6    
    nodes =     [{'x': 0.0e0, 'y': 0.0e0,  'z': 0.0e0, 'xflag': 'd', 'xbcval': 0.0, 'yflag': 'd', 'ybcval': 0.0e0,   'zflag': 'd', 'zbcval': 0.0e0}]
    nodes.append({'x': 3.0e0, 'y': 0.0e0,  'z': 0.0e0, 'xflag': 'f', 'xbcval': 0.0, 'yflag': 'f', 'ybcval': 0.0e0,   'zflag': 'd', 'zbcval': 0.0e0})
    nodes.append({'x': 6.0e0, 'y': 0.0e0,  'z': 0.0e0, 'xflag': 'f', 'xbcval': 0.0, 'yflag': 'f', 'ybcval': 0.0e0,   'zflag': 'd', 'zbcval': 0.0e0}) 
    nodes.append({'x': 3.0e0, 'y': -4.0e0, 'z': 0.0e0, 'xflag': 'f', 'xbcval': 0.0, 'yflag': 'f', 'ybcval': -9.0e3,   'zflag': 'd', 'zbcval': 0.0e0})
    nodes.append({'x': 6.0e0, 'y': -4.0e0, 'z': 0.0e0, 'xflag': 'f', 'xbcval': 0.0, 'yflag': 'f', 'ybcval': -15.0e3,   'zflag': 'd', 'zbcval': 0.0e0})
    nodes.append({'x': 9.0e0, 'y': -4.0e0, 'z': 0.0e0, 'xflag': 'f', 'xbcval': 0.0, 'yflag': 'd', 'ybcval': 0.0e0,   'zflag': 'd', 'zbcval': 0.0e0})
    
    members =     [{'start': 0, 'end': 1, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6}]
    members.append({'start': 1, 'end': 2, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    members.append({'start': 0, 'end': 3, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    members.append({'start': 1, 'end': 3, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    members.append({'start': 2, 'end': 3, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    members.append({'start': 3, 'end': 4, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    members.append({'start': 2, 'end': 4, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    members.append({'start': 2, 'end': 5, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    members.append({'start': 4, 'end': 5, 'E': 200.0e9, 'A': area, 'sigma_yield': 36.0e6, 'sigma_ult': 66.0e6})
    
    return nodes, members