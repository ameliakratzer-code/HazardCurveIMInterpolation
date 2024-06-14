import pymysql
import argparse
import matplotlib.pyplot as plt
import pyproj

parser = argparse.ArgumentParser('Allow user to input site name, period')
# User enter sitenames with spaces
parser.add_argument('--sitenames', nargs='+')
parser.add_argument('--interpsitename')
parser.add_argument('--period', default=2)
parser.add_argument('--output', default='y')
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
    plt.figure()
    # plot of hazard curve using matplotlib
    plt.xscale('linear')
    plt.xlim(0, 2)
    plt.yscale('log')
    plt.ylim(1e-6,1)
    plt.xlabel('Accel (g)')
    plt.ylabel('Prob')
    plt.title(f'Hazard Curve {nameSite}')
    plt.plot(xVals, yVals, marker='^')
    plt.grid(axis = 'y')
    # If any output argument is provided, store the image under the site name, period
    if args.output == 'y':
        plt.savefig(f'{nameSite}' + 'per' + str(args.period) + '.png')
    else:
        plt.show(block=False)
        plt.pause(5)
        plt.close()

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

def linearinterpolation(s0, s1, sI):
    #prob values for two known sites
    xCoords, probCoords0 = (downloadHazardCurve(s0))
    probCoords1 = (downloadHazardCurve(s1))[1]
    interpolatedProbs = []
    # Convert from lat/lon to UTM
    x0, y0 = getUTM(s0)
    x1, y1 = getUTM(s1)
    x, y = getUTM(sI)
    # Loop through x values on hazard Curve
    # Apply formula
    for i in range(51):
        interpVal = (probCoords0[i] * abs(x1 - x) + probCoords1[i] * abs(x - x1)) * (1 / abs(x1 - x0))
        interpolatedProbs.append(interpVal)
    print(interpolatedProbs)
    plotHazardCurve(xCoords,interpolatedProbs, sI+' Interpolated')
    downloadHazardCurve(sI)

def main():
    #break apart sites from list provided
    numSites = len(args.sitenames)
    for i in range(numSites):
        if i == 0:
            site0 = args.sitenames[0]
        elif i == 1:
            site1 = args.sitenames[1]
        elif i == 2:
            site2 = args.sitenames[2]
        elif i == 3:
            site3 = args.sitenames[3]
    #linear interpolation between 2 sites
    if numSites == 2:
        linearinterpolation(site0, site1, args.interpsitename)
    #bilinear interpolation between 4 sites
    elif numSites == 4:
        pass
    connection.close()

main()