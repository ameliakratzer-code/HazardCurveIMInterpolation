#pymysql for database, sqlite for sqlite version
import sqlite3
import argparse
import matplotlib.pyplot as plt
import os
import csv
from utils import Site, interpolate
import numpy as np
from enum import Enum

parser = argparse.ArgumentParser('Allow user to input site names, ')
parser.add_argument('--sitenames', nargs='+')
# User can specify a particular event ID: source, rup, rupVar to do the interpolation for
# One event = specific source, rup, rupVar, one rupture = specific source, rup, all events = nothing specified
parser.add_argument('--source')
parser.add_argument('--rup')
parser.add_argument('--rupVar')
parser.add_argument('--interpsitename')
parser.add_argument('--output')
parser.add_argument('--period', default=2)
args = parser.parse_args()

# Connect to the database
# (host = 'moment.usc.edu', user = 'cybershk_ro', password = 'CyberShake2007', database = 'CyberShake')
connection = sqlite3.connect('/scratch1/00349/scottcal/CS_interpolation/study_22_12_lf_indexed.sqlite')

# Create Mode enumeration and set user_mode variable depending on user arguments
class Mode(Enum):
    ONE_EVENT = 1
    ONE_RUPTURE = 2
    ALL_EVENTS = 3
if args.source != None and args.rup != None and args.rupVar != None:
    userMode = Mode.ONE_EVENT
elif args.source == None and args.rup == None and args.rupVar == None:
    userMode = Mode.ALL_EVENTS
elif args.source != None and args.rup != None and args.rupVar == None:
    userMode = Mode.ONE_RUPTURE

def getIMValues(site0, site1, site2, site3):
    cursor = connection.cursor()
    eventsList = []
    IMVals = []
    baseQuery = '''
            SELECT P.IM_Value 
            FROM CyberShake_Sites S, CyberShake_Runs R, PeakAmplitudes P, Studies T, IM_Types I
            WHERE S.CS_Short_Name = ?
            AND S.CS_Site_ID = R.Site_ID
            AND R.Study_ID = T.Study_ID
            AND T.Study_Name = 'Study 22.12 LF'
            AND R.Run_ID = P.Run_ID
            AND I.IM_Type_Component = 'RotD50'
            AND I.IM_Type_Value = ?
            AND I.IM_Type_ID = P.IM_Type_ID
            '''
    if userMode == Mode.ONE_EVENT:
        for site in [site0, site1, site2, site3, args.interpsitename]:
            qOneEvent = baseQuery + 'AND P.Source_ID = ? AND P.Rupture_ID = ? And P.Rup_Var_ID = ?'
            cursor.execute(qOneEvent, (site, float(args.period), args.source, args.rup, args.rupVar))
            result = cursor.fetchone()
            IMVals.append(result[0])
        eventsList.append((int(args.source),int(args.rup),int(args.rupVar)))
        return eventsList, IMVals
    elif userMode == Mode.ALL_EVENTS:
        # First get list of sharedRups
        sharedRups = []
        for site in [site0, site1, site2, site3]:
            q0 = '''
                    SELECT C.Source_ID, C.Rupture_ID
                    FROM CyberShake_Site_Ruptures C, CyberShake_Sites S
                    WHERE C.CS_Site_ID = S.CS_Site_ID
                    AND C.ERF_ID = 36
                    AND S.CS_Short_Name = ?
                    '''
            cursor.execute(q0, (site,))
            result = cursor.fetchall()
            if sharedRups == []:
                sharedRups = result
            else:
                sharedRups = list(set(sharedRups) & set(result))
    elif userMode == Mode.ONE_RUPTURE:
        sharedRups = [(args.source, args.rup)]
    else:
        print('Please enter one event = specific source, rup, rupVar, one rupture = specific source, rup, or all events = nothing specified')
        exit()
    # Loop through sharedRups and get IM values
    # Need IM value for simulated interpsite
    for site in [site0, site1, site2, site3, args.interpsitename]:
        for (source, rup) in sharedRups:
            q1 = baseQuery + 'AND P.Source_ID = ? AND P.Rupture_ID = ?'
            cursor.execute(q1, (site, float(args.period), source, rup))
            result = cursor.fetchall()
            IMVals.extend(result)
            # Length of result = how many rupture variations
            # Only get event IDs once
            if site == site0:
                for i in range(len(result)):
                    eventsList.append((source, rup, i))
    r = []
    for val in IMVals:
        r.append(val[0])
    cursor.close()
    return eventsList, r

def bilinearinterpolation(s0, s1, s2, s3, sI):
    events, IMTogether = getIMValues(s0, s1, s2, s3)
    # IMTogether has all IMs together IM0, IM1
    IMs = []
    for i in range(0, len(IMTogether), len(events)):
        IMs.append(IMTogether[i:i+len(events)])
    # Use site classes
    p0 = Site(s0, IMs[0])
    p1 = Site(s1, IMs[1])
    p2 = Site(s2, IMs[2])
    p3 = Site(s3, IMs[3])
    p4 = Site(sI, IMs[4])
    xVals = events
    listPXY = [p0, p1, p2, p3]
    sortedL = sorted(listPXY, key=lambda site: site.x)
    sortedL.append(p4)
    interpIMVals = interpolate(sortedL, xVals)
    # Write (event, IM) values to file, user types entire file path
    filePath = args.output + '.csv'
    # Open file in write mode
    with open(filePath, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(['Event', 'IMVal'])
        for event, IMVal in zip(events, interpIMVals):
            write.writerow([event, IMVal])
    print('Data downloaded')
    # Scatterplot of data
    interpScatterplot(p4.valsToInterp, interpIMVals)
    print('Scatterplot plotted')
    # Print out percent error and create a histogram for each event
    listDifferences = []
    for i in range(len(interpIMVals)):
        # Percent difference = (interp-simulated) / simulated not absolute
        percentDifference = ((interpIMVals[i] - p4.interpIMVal[i]) / p4.interpIMVal[i]) * 100
        print(percentDifference)
        listDifferences.append(percentDifference)
    bin_width = 25
    numBins = (max(listDifferences)-min(listDifferences)) // bin_width + 1
    plt.hist(listDifferences, bins = numBins, edgecolor='black')
    plt.title(f'Histogram of Percent Error {sI}')
    plt.xlabel('Percent difference')
    plt.ylabel('Frequency')
    # Save histogram to file
    filePath = args.output + 'histogram.csv'
    plt.savefig(filePath)
    print('Histogram plotted')

# Scatterplot x-axis = simulated IM, y-axis = interp IM
def interpScatterplot(sim, interp):
    numDots = len(sim)
    if numDots <= 20:
        # Size for one event
        size = 100
    elif numDots < 100:
        # Size for all rup vars
        size = 2000 / numDots
    else:
        size = 3
    plt.scatter(sim, interp, color='blue', s = size)
    # Set 1 to 1 ration
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Simulated IMs')
    plt.ylabel('Interpolated IMs')
    if userMode == Mode.ONE_EVENT:
        plt.title(f'{args.interpsitename} IMs for ({args.source}, {args.rup}, {args.rupVar}), {args.period} per RotD50')
    elif userMode == Mode.ONE_RUPTURE:
        plt.title(f'{args.interpsitename} IMs for ({args.source}, {args.rup}, all), {args.period} per RotD50')
    elif userMode == Mode.ALL_EVENTS:
        plt.title(f'{args.interpsitename} IMs for all events, {args.period} per RotD50')
    # Line of best fit to scatterplot
    # Convert x and y to numpy arrays
    xNumpy, yNumpy = np.array(sim), np.array(interp)
    slope, intercept = np.polyfit(xNumpy, yNumpy, 1)
    lineOfBestFit = slope * xNumpy + intercept
    plt.plot(sim, lineOfBestFit, color='green', linestyle = 'dashed')
    # Line y = x
    minVal = min(min(sim), min(interp))
    maxVal = max(max(sim), max(interp))
    plt.plot([minVal, maxVal], [minVal, maxVal], linestyle = 'dashed', color='black')
    # User types in full name of output
    path = args.output + '.png'
    plt.savefig(path)
    
def main():
    sites = (args.sitenames[0]).split(',')
    site0, site1, site2, site3 = sites[0], sites[1], sites[2], sites[3]
    bilinearinterpolation(site0, site1, site2, site3, args.interpsitename)
    connection.close()

main()