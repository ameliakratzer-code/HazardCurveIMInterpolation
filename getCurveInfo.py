import pymysql
import argparse
import matplotlib.pyplot as plt
import pyproj
import os
import math

parser = argparse.ArgumentParser('Allow user to input site name, period')
# User enter sitenames with spaces
parser.add_argument('--sitenames', nargs='+')
parser.add_argument('--interpsitename')
parser.add_argument('--period', default=2)
parser.add_argument('--output', default='Sites',help='Enter name of folder you want to store photos in')
args = parser.parse_args()
# Check arguments
if args.sitenames == None:
    print('Sitename is required.')
    exit()
# Connect to the database
connection = pymysql.connect(host = 'moment.usc.edu',
                            user = 'cybershk_ro',
                            password = 'CyberShake2007',
                            database = 'CyberShake')

def downloadHazardCurve(nameSite):
    with connection.cursor() as cursor:
        # Queries to get hazard curve information
        query1 = '''SELECT CyberShake_Runs.Run_ID FROM CyberShake_Sites
                    INNER JOIN CyberShake_Runs
                    ON CyberShake_Sites.CS_Site_ID = CyberShake_Runs.Site_ID
                    INNER JOIN Studies
                    ON CyberShake_Runs.Study_ID = Studies.Study_ID
                    WHERE CyberShake_Sites.CS_Short_Name = %s AND Studies.Study_Name = 'Study 22.12 LF';
                    '''
        cursor.execute(query1, (nameSite))
        runID = cursor.fetchall()
        # Use query1 value - the run_Id as WHERE Hazard_Curves.Run_ID = query1
        period = float(args.period)
        query2 = '''SELECT * FROM Hazard_Curve_Points
            INNER JOIN Hazard_Curves
            ON Hazard_Curve_Points.Hazard_Curve_ID = Hazard_Curves.Hazard_Curve_ID
            INNER JOIN IM_Types
            ON Hazard_Curves.IM_Type_ID = IM_Types.IM_Type_ID
            WHERE Hazard_Curves.Run_ID = %s AND IM_Types.IM_Type_Value = %s AND IM_Types.IM_Type_Component='RotD50';
            '''
        cursor.execute(query2, (runID, period))
        result = cursor.fetchall()
    #get list of x and y coordinates from result tuple
    xCoords = []
    yCoords = []
    for row in result:
        xCoords.append(row[2])
        yCoords.append(row[3])
    plotHazardCurve(xCoords, yCoords, nameSite)
    return xCoords, yCoords
    
def plotHazardCurve(xVals, yVals, nameSite):
    # plot of hazard curve using matplotlib
    plotFeatures()
    plt.plot(xVals, yVals, marker='^')
    # If any output argument is provided, store the image under the site name, period
    if args.output != None:
        # Store photos in specific directory
        directory = f"/Users/ameliakratzer/Desktop/LinInterpolation/{args.output}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        fileName = f'{nameSite}' + 'per' + str(args.period) + '.png'
        plt.title(f'{nameSite}, 2 sec RotD50')
        path = os.path.join(directory, fileName)
        plt.savefig(path)
    else:
        plt.show(block=False)
        plt.pause(5)
        plt.close()

def plotFeatures():
    plt.figure()
    plt.xscale('linear')
    plt.xlim(0, 2)
    plt.yscale('log')
    plt.ylim(1e-6,1)
    plt.xlabel('Accel (g)')
    plt.ylabel('Prob')
    plt.grid(axis = 'y')

def getUTM(sitename):
    #get lat lon of site
    with connection.cursor() as cursor:
        query3 = '''SELECT CS_Site_Lat, CS_Site_Lon FROM CyberShake_Sites
                    WHERE CS_Short_Name = %s
        '''
        cursor.execute(query3, (sitename))
        location = cursor.fetchall()
        lat, lon = location[0][0], location[0][1]
    myProj = pyproj.Proj(proj ='utm', zone = 11, ellps = 'WGS84')
    x, y = myProj(lon, lat)
    return x, y

def getDistance(point1x, point1y, point2x, point2y, SIx, SIy):
    # Used in bilinear interpolation
    # Find where line point1 to point2 and line interpSite intersect
    m1 = (point2y-point1y) / (point2x-point1x)
    b1 = point2y-m1*point2x
    m2 = -(1/m1)
    b2 = SIy-m2*SIx
    xIntersection = (b2-b1)/(m1-m2)
    yIntersection = m2*xIntersection+b2
    d = (SIx-xIntersection)**2 + (SIy-yIntersection)**2
    return d**0.5

# Used when checking if input sites form a square
def disFormula(x0, y0, x1, y1):
    dsquared = (x0-x1)**2 + (y0-y1)**2
    d = dsquared**0.5
    return d

# Plot with the interpolated curve and actual curve them overlayed
# Used in linear and bilinear interpolation
def plotInterpolated(xCoords, sI, interpolatedProbs):
    xActual, yActual = downloadHazardCurve(sI)
    # Describing the quality of the fit
    print('\nPercent difference')
    listDifferences = []
    for i in range(len(xCoords)):
        # Only calculate percent difference if simulated >= 1E-6
        if yActual[i] >= 1e-6:
            avg = (yActual[i] + interpolatedProbs[i]) / 2
            percentDifference = (abs(yActual[i] - interpolatedProbs[i]) / avg) * 100 if yActual[i] != 0 else 0
            print(percentDifference)
            listDifferences.append(percentDifference)
    maxDiff = max(listDifferences)
    avgDiff = sum(listDifferences) / len(listDifferences)
    print(f'\nMaxdiff: {round(maxDiff, 1)}%, avgDiff: {round(avgDiff,1)}%\n')
    # Plotting of overlayed curve
    plotHazardCurve(xCoords,interpolatedProbs, sI+' Interpolated')
    plotFeatures()
    plt.title(f'Overlayed {sI}, 2 sec RotD50')
    plt.plot(xActual, yActual, color='green', linewidth = 2, label = "Actual", marker='^')
    plt.plot(xActual, interpolatedProbs, color='pink', linewidth = 2, label = 'Interpolated', marker='^')
    plt.legend()
    path = os.path.join(f"/Users/ameliakratzer/Desktop/LinInterpolation/{args.output}", 'Overlayed' + '.png')
    plt.savefig(path)

def linearinterpolation(s0, s1, sI):
    # Prob values for two known sites
    xCoords, probCoords0 = (downloadHazardCurve(s0))
    probCoords1 = (downloadHazardCurve(s1))[1]
    interpolatedProbs = []
    # Convert from lat/lon to UTM
    x0, y0 = getUTM(s0)
    x1, y1 = getUTM(s1)
    x, y = getUTM(sI)
    # Check if sI is in between input sites
    if (x0 <= x<= x1 or x1 <= x <= x0) and (y0 <= y <= y1 or y1 <= y <= y0):
        # Loop through x values on hazard Curve
        for i in range(len(xCoords)):
            interpVal = (probCoords0[i] * abs(x1 - x) + probCoords1[i] * abs(x - x0)) * (1 / abs(x1 - x0))
            interpolatedProbs.append(interpVal)
        plotInterpolated(xCoords, sI, interpolatedProbs)
    else:
        print('Interpsite not in interpolation bounds')
        exit()

def bilinearinterpolation(s0, s1, s2, s3, sI):
    # Get sites in correct order for interpolation
    p0x, p0y = getUTM(s0)
    p1x, p1y = getUTM(s1)
    p2x, p2y = getUTM(s2)
    p3x, p3y = getUTM(s3)
    x, y = getUTM(sI)
    interpolatedProbs = []
    xCoords = downloadHazardCurve(s0)[0]
    listPXY = [(p0x, p0y, downloadHazardCurve(s0)[1]), (p1x, p1y, downloadHazardCurve(s1)[1]), (p2x, p2y, downloadHazardCurve(s2)[1]), (p3x, p3y, downloadHazardCurve(s3)[1])]
    sortedL = sorted(listPXY, key=lambda x:x[0])
    # Determining S0, S3
    if sortedL[0][1] < sortedL[1][1]:
        # Download hazard curve of site at L[0]
        (x0, y0, probCoords0) = sortedL[0]
        (x3, y3, probCoords3) = sortedL[1]
    else:
        (x0, y0, probCoords0) = sortedL[1]
        (x3, y3, probCoords3) = sortedL[0]
    # Determing S1, S2
    if sortedL[2][1] < sortedL[3][1]:
        (x1, y1, probCoords1) = sortedL[2]
        (x2, y2, probCoords2) = sortedL[3]
    else:
        (x1, y1, probCoords1) = sortedL[3]
        (x2, y2, probCoords2) = sortedL[2]
    # Check if sites form square before interpolating: sides and diagonals
    if (not(9500 <= disFormula(x0,y0,x1,y1) <= 10500) or not(9500 <= disFormula(x1,y1,x2,y2) <= 10500) or 
        not(9500 <= disFormula(x2,y2,x3,y3) <= 10500) or not(9500 <= disFormula(x3,y3,x0,y0) <= 10500) or
        not((9500*math.sqrt(2)) <= disFormula(x0,y0,x2,y2) <= (math.sqrt(2)*10500)) or not((9500*math.sqrt(2)) <= disFormula(x1,y1,x3,y3) <= (math.sqrt(2)*10500))):
        print('Entered sites do not form a square')
        exit()
    # Calculate distances with slanted axis
    yPrime = getDistance(x3, y3, x2, y2, x, y) / 10000
    xPrime =  getDistance(x3, y3, x0, y0, x, y) / 10000
    for i in range(len(xCoords)):
        R1 = (probCoords0[i] * (1-xPrime) + probCoords1[i] * xPrime)
        R2 = (probCoords2[i] * xPrime + probCoords3[i] * (1-xPrime))
        interpVal = (R1 * yPrime + R2 * (1-yPrime))
        interpolatedProbs.append(interpVal)
    # TEMPORARY -> print out interpolated values
    print('\nInterp values')
    for val in interpolatedProbs:
        print(val)
    plotInterpolated(xCoords, sI, interpolatedProbs)

def main():
    # Create comma-separated list of sites from arg
    sites = (args.sitenames[0]).split(',')
    numSites = len(sites)
    for i in range(numSites):
        if i == 0:
            site0 = sites[0]
        elif i == 1:
            site1 = sites[1]
        elif i == 2:
            site2 = sites[2]
        elif i == 3:
            site3 = sites[3]
    #linear interpolation between 2 sites
    if numSites == 2:
        linearinterpolation(site0, site1, args.interpsitename)
    #bilinear interpolation between 4 sites
    elif numSites == 4:
        bilinearinterpolation(site0, site1, site2, site3, args.interpsitename)
    connection.close()

main()