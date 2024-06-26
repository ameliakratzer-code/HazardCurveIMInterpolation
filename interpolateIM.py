import pymysql
import argparse
import matplotlib.pyplot as plt
import os
import csv
from utils import Site, linearCheck

# Use when plotting curve so know event ID to include
oneEvent, oneRupture, allEvents = False, False, False

parser = argparse.ArgumentParser('Allow user to input site names, ')
parser.add_argument('--sitenames', nargs='+')
# User can specify a particular event ID: source, rup, rupVar to do the interpolation for
# One event = specific source, rup, rupVar, one rupture = specific source, rup, all events = nothing specified
parser.add_argument('--source')
parser.add_argument('--rup')
parser.add_argument('--rupVar')
parser.add_argument('--interpsitename')
parser.add_argument('--output')
args = parser.parse_args()

# Connect to the database
connection = pymysql.connect(host = 'moment.usc.edu',
                            user = 'cybershk_ro',
                            password = 'CyberShake2007',
                            database = 'CyberShake')

def getIMValues(nameSite):
    with connection.cursor() as cursor:
        # Query changes depending on what part of event ID is entered
        baseQuery = '''
        SELECT P.Source_ID, P.Rupture_ID, P.Rup_Var_ID, P.IM_Value 
        FROM CyberShake_Sites S, CyberShake_Runs R, PeakAmplitudes P, Studies T, IM_Types I
        WHERE S.CS_Short_Name = %s
        AND S.CS_Site_ID = R.Site_ID
        AND R.Study_ID = T.Study_ID
        AND T.Study_Name = 'Study 22.12 LF'
        AND R.Run_ID = P.Run_ID
        AND I.IM_Type_Component = 'RotD50'
        AND I.IM_Type_Value = 2.0
        AND I.IM_Type_ID = P.IM_Type_ID
        '''
        # Want all events for that site
        if args.source == None and args.rup == None and args.rupVar == None:
            cursor.execute(baseQuery, (nameSite))
            # Set global allEvents variable to True
            global allEvents 
            allEvents = True
        # Want all rupture variations for a specific source and rup for that site
        elif args.source != None and args.rup != None and args.rupVar == None:
            query1 = baseQuery + 'AND P.Source_ID = %s AND P.Rupture_ID = %s'
            cursor.execute(query1, (nameSite, args.source, args.rup))
            global oneRupture
            oneRupture = True
        # Want one event only
        elif args.source != None and args.rup != None and args.rupVar != None:
            query2 = baseQuery + 'AND P.Source_ID = %s AND P.Rupture_ID = %s And P.Rup_Var_ID = %s'
            cursor.execute(query2, (nameSite, args.source, args.rup, args.rupVar))
            global oneEvent 
            oneEvent = True
        else:
            print('Please enter one event = specific source, rup, rupVar, one rupture = specific source, rup, or all events = nothing specified')
            exit()
        result = cursor.fetchall()
        # Check to make sure event ID is valid
        if len(result) == 0:
            print('Please enter a valid event for that site')
            exit()
        # Result row = (sourceID, ruptureID, RupVarID, IMVal)
        # EventID = tuple of first three values from result row
        eventID = []
        IMVals = []
        for row in result:
            eventID.append((row[0], row[1], row[2]))
            IMVals.append(row[3])
        return eventID, IMVals

def bilinearinterpolation(s0, s1, s2, s3, sI):
    # Use site classes
    p0 = Site(s0, getIMValues(s0)[1], getIMValues(s0)[0])
    p1 = Site(s1, getIMValues(s1)[1], getIMValues(s1)[0])
    p2 = Site(s2, getIMValues(s2)[1], getIMValues(s2)[0])
    p3 = Site(s3, getIMValues(s3)[1], getIMValues(s3)[0])
    p4 = Site(sI, getIMValues(sI)[1], getIMValues(sI)[0])
    interpolatedIMVals = []
    interpEvents = []
    # xCoords = event IDs, yCoords = IM Vals
    listPXY = [p0, p1, p2, p3]
    sortedL = sorted(listPXY, key=lambda site: site.x)
    sortedL.append(p4)
    s0, s1, s2, s3, yPrime, xPrime = linearCheck(sortedL)
    eventIDs = s0.events, s1.events, s2.events, s3.events
    for i in range(len(s0.events)):
        indexOfEvents = []
        continueOuterLoop = False
        eventID = s0.events[i]
        for list in eventIDs:
            # Event IDs might have different indexes in list of IDs for site
            if eventID in list:
               indexOfEvents.append(list.index(eventID))
            else:
                continueOuterLoop = True
        if continueOuterLoop:
            continue
        R1 = (s0.valsToInterp[indexOfEvents[0]] * (1-xPrime) + s1.valsToInterp[indexOfEvents[1]] * xPrime)
        R2 = (s2.valsToInterp[indexOfEvents[2]] * xPrime + s3.valsToInterp[indexOfEvents[3]] * (1-xPrime))
        interpVal = (R1 * yPrime + R2 * (1-yPrime))
        interpolatedIMVals.append(interpVal)
        interpEvents.append(eventID)
    # Write (event, IM) values to file
    # Specify filename and directory
    filename = args.output + '.csv' if args.output != None else 'unnamed.csv'
    directory = f"/Users/ameliakratzer/Desktop/LinInterpolation/{args.output}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filePath = os.path.join(directory, filename)
    # Open file in write mode
    with open(filePath, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(['Event', 'IMVal'])
        for event, IMVal in zip(interpEvents, interpolatedIMVals):
            write.writerow([event, IMVal])
    print('Data downloaded')
    # Scatterplot of data
    interpScatterplot(getIMValues(sI)[1],interpolatedIMVals, args.interpsitename)
    print('Scatterplot plotted')

# Scatterplot x-axis = simulated IM, y-axis = interp IM
def interpScatterplot(sim, interp, sitename):
    plt.scatter(sim, interp, color='blue')
    plt.xlabel('Simulated IMs')
    plt.ylabel('Interpolated IMs')
    if oneEvent:
        plt.title(f'{args.interpsitename} IMs for ({args.source}, {args.rup}, {args.rupVar}), 2 sec RotD50')
    elif oneRupture:
        plt.title(f'{args.interpsitename} IMs for ({args.source}, {args.rup}, all), 2 sec RotD50')
    elif allEvents:
        plt.title(f'{args.interpsitename} IMs for all events, 2 sec RotD50')
    # Line y = x
    minVal = min(min(sim), min(interp))
    maxVal = max(max(sim), max(interp))
    plt.plot([minVal, maxVal], [minVal, maxVal], linestyle = 'dashed', color='black')
    # Want to save plot to same folder as data
    directory = f"/Users/ameliakratzer/Desktop/LinInterpolation/{args.output}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fileName = f'{sitename}' + '.png'
    path = os.path.join(directory, fileName)
    plt.savefig(path)
    
def main():
    sites = (args.sitenames[0]).split(',')
    site0, site1, site2, site3 = sites[0], sites[1], sites[2], sites[3]
    bilinearinterpolation(site0, site1, site2, site3, args.interpsitename)
    connection.close()

main()