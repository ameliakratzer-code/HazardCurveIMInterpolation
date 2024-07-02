import sqlite3
import argparse
import matplotlib.pyplot as plt
import os
import csv
from utils import Site, interpolate

parser = argparse.ArgumentParser('Allow user to input site name, period')
# User enter sitenames with spaces
parser.add_argument('--sitenames', nargs='+')
parser.add_argument('--interpsitename')
parser.add_argument('--period', default=2)
parser.add_argument('--output', default='Sites',help='Enter name of folder you want to store photos in')
args = parser.parse_args()
# Connect to the database
connection = sqlite3.connect('/scratch1/00349/scottcal/CS_interpolation/study_22_12_lf_indexed.sqlite')

def downloadHazardCurve(nameSite):
    cursor = connection.cursor()
    # Queries to get hazard curve information
    query1 = '''SELECT CyberShake_Runs.Run_ID FROM CyberShake_Sites
                INNER JOIN CyberShake_Runs
                ON CyberShake_Sites.CS_Site_ID = CyberShake_Runs.Site_ID
                INNER JOIN Studies
                ON CyberShake_Runs.Study_ID = Studies.Study_ID
                WHERE CyberShake_Sites.CS_Short_Name = ? AND Studies.Study_Name = 'Study 22.12 LF';
                '''
    cursor.execute(query1, (nameSite,))
    runID = cursor.fetchall()
    # Use query1 value - the run_Id as WHERE Hazard_Curves.Run_ID = query1
    period = float(args.period)
    query2 = '''SELECT * FROM Hazard_Curve_Points
        INNER JOIN Hazard_Curves
        ON Hazard_Curve_Points.Hazard_Curve_ID = Hazard_Curves.Hazard_Curve_ID
        INNER JOIN IM_Types
        ON Hazard_Curves.IM_Type_ID = IM_Types.IM_Type_ID
        WHERE Hazard_Curves.Run_ID = ? AND IM_Types.IM_Type_Value = ? AND IM_Types.IM_Type_Component='RotD50';
        '''
    print(runID, period)
    print(type(runID), type(period))
    cursor.execute(query2, (runID, period))
    result = cursor.fetchall()
    cursor.close()
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
        directory = args.output
        if not os.path.exists(directory):
            os.makedirs(directory)
        fileName = f'{nameSite}' + 'per' + str(args.period) + '.png'
        plt.title(f'{nameSite}, 2 sec RotD50')
        path = os.path.join(directory, fileName)
        #directory + '/' + fileName
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

# Plot with the interpolated curve and actual curve them overlayed
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
    # ({args.output} + '/' + 'Overlayed' + '.png')
    path = os.path.join(args.output, 'Overlayed' + '.png')
    plt.savefig(path)

def linearinterpolation(s0, s1, sI):
    # Prob values for two known sites
    s0 = Site(s0, downloadHazardCurve(s0)[1])
    s1 = Site(s1, downloadHazardCurve(s1)[1])
    pI = Site(sI, downloadHazardCurve(sI)[1])
    xCoords = downloadHazardCurve(s0.name)[0]
    interpolatedProbs = []
    # Check if sI is in between input sites
    if pI.within_x_range(s0, s1) and pI.within_y_range(s0, s1):
        # Loop through x values on hazard Curve
        for i in range(len(xCoords)):
            interpVal = (s0.valsToInterp[i] * abs(s1.x - pI.x) + s1.valsToInterp[i] * abs(pI.x - s0.x)) * (1 / abs(s1.x - s0.x))
            interpolatedProbs.append(interpVal)
        plotInterpolated(xCoords, sI, interpolatedProbs)
    else:
        print('Interpsite not in interpolation bounds')
        exit()
    return interpolatedProbs

def bilinearinterpolation(s0, s1, s2, s3, sI):
    # Use site classes
    p0 = Site(s0, downloadHazardCurve(s0)[1])
    p1 = Site(s1, downloadHazardCurve(s1)[1])
    p2 = Site(s2, downloadHazardCurve(s2)[1])
    p3 = Site(s3, downloadHazardCurve(s3)[1])
    p4 = Site(sI, downloadHazardCurve(sI)[1])
    xCoords = downloadHazardCurve(p4.name)[0]
    listPXY = [p0, p1, p2, p3]
    sortedL = sorted(listPXY, key=lambda site: site.x)
    sortedL.append(p4)
    interpolatedProbs = interpolate(sortedL, xCoords)
    print('\nInterp values')
    for val in interpolatedProbs:
        print(val)
    # Write interpolated vals to file for testing
    fileName = f'Actual{args.interpsitename}.csv'
    # '/Users/ameliakratzer/codescripts/sources/Pasadena/tests' for computer
    filePath = os.path.join(os.getcwd(), fileName)
    with open(filePath, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(['XVals', 'InterpVals'])
        for xVal, interpVal in zip(xCoords, interpolatedProbs):
            write.writerow([xVal, interpVal])
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