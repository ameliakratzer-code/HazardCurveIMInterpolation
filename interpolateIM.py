import pymysql
import argparse
import matplotlib.pyplot as plt
import pyproj

parser = argparse.ArgumentParser('Allow user to input site names, ')
parser.add_argument('--sitenames', nargs='+')
# User can specify a particular event ID: source, rup, rupVar to do the interpolation for
# One event = specific source, rup, rupVar, one rupture = specific source, rup, all events = nothing specified
parser.add_argument('--source')
parser.add_argument('--rup')
parser.add_argument('--rupVar')
parser.add_argument('--interpsitename')
args = parser.parse_args()

# Connect to the database
connection = pymysql.connect(host = 'moment.usc.edu',
                            user = 'cybershk_ro',
                            password = 'CyberShake2007',
                            database = 'CyberShake')

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
        # Want all rupture variations for a specific source and rup for that site
        elif args.source != None and args.rup != None and args.rupVar == None:
            query1 = baseQuery + 'AND P.Source_ID = %s AND P.Rupture_ID = %s'
            cursor.execute(query1, (nameSite, args.source, args.rup))
        # Want one event only
        elif args.source != None and args.rup != None and args.rupVar != None:
            query2 = query1 + 'And P.Rup_Var_ID = %s'
            cursor.execute(query1, (nameSite, args.source, args.rup, args.rupVar))
        else:
            print('Please enter one event = specific source, rup, rupVar, one rupture = specific source, rup, or all events = nothing specified')
            exit()
        result = cursor.fetchall()
        # Result row = (sourceID, ruptureID, RupVarID, IMVal)
        # EventID = tuple of first three values from result row
        eventID = []
        IMVals = []
        for row in result:
            eventID.append((result[0], result[1], result[2]))
            IMVals.append(result[3])
        return eventID, IMVals

def bilinearinterpolation(s0, s1, s2, s3, sI):
    pass

def interpScatterplot():
    # Make scatterplot of actual IM values versus interpolated
    # Have y = x as reference
    pass

def main():
    sites = (args.sitenames[0]).split(',')
    site0, site1, site2, site3 = sites[0], sites[1], sites[2], sites[3]
    # Just bilin interpolation for now, can add 1d linear later
    bilinearinterpolation(site0, site1, site2, site3, args.interpsitename)
    connection.close()

main()