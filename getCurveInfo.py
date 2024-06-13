import pymysql
import argparse
import matplotlib.pyplot as plt
import pyproj

def downloadHazardCurve():
    # Handle command line arguments
    parser = argparse.ArgumentParser('Allow user to input site name, period')
    parser.add_argument('--sitename')
    parser.add_argument('--period')
    parser.add_argument('--outputName')
    args = parser.parse_args()

    # Check arguments
    if args.sitename == None and args.period == None:
        print('A sitename and period are required.')
        exit()
    elif args.sitename == None:
        print('A sitename is required in addition to the period.')
        exit()
    elif args.period == None:
        print('A period is required in addition to the sitename.')
        exit()

    # Connect to the database 
    connection = pymysql.connect(host = 'moment.usc.edu',
                                user = 'cybershk_ro',
                                password = 'CyberShake2007',
                                database = 'CyberShake')

    with connection.cursor() as cursor:
        # Queries to get hazard curve information
        query1 = '''SELECT CyberShake_Runs.Run_ID FROM CyberShake_Sites
                    INNER JOIN CyberShake_Runs
                    ON CyberShake_Sites.CS_Site_ID = CyberShake_Runs.Site_ID
                    INNER JOIN Studies
                    ON CyberShake_Runs.Study_ID = Studies.Study_ID
                    WHERE CyberShake_Sites.CS_Short_Name = %s AND Studies.Study_Name = 'Study 22.12 LF';
                    '''
        cursor.execute(query1, (args.sitename))
        runID = cursor.fetchall()
        # Use query1 value - the run_Id as WHERE Hazard_Curves.Run_ID = query1
        period = float(args.period)
        query2 = '''SELECT * FROM Hazard_Curve_Points
            INNER JOIN Hazard_Curves
            ON Hazard_Curve_Points.Hazard_Curve_ID = Hazard_Curves.Hazard_Curve_ID
            INNER JOIN IM_Types
            ON Hazard_Curves.IM_Type_ID = IM_Types.IM_Type_ID
            WHERE Hazard_Curves.Run_ID = %s AND IM_Types.IM_Type_Value = %s AND IM_Types.IM_Type_Component='RotD50';'''
        cursor.execute(query2, (runID, period))
        result = cursor.fetchall()
    connection.close()

    # plot of hazard curve using matplotlib
    plt.xscale('linear')
    plt.xlim(0, 2)
    plt.yscale('log')
    plt.ylim(1e-6,1)
    plt.xlabel('Accel (g)')
    plt.ylabel('Prob')
    plt.title('Hazard Curve')
    #get list of x and y coordinates from result tuple
    xCoords = []
    yCoords = []
    for row in result:
        xCoords.append(row[2])
        yCoords.append(row[3])
    plt.plot(xCoords, yCoords, marker='^')
    plt.grid(axis = 'y')

    # If outputName provided, save the plot to the given name
    if args.outputName != None:
        plt.savefig(args.outputName + '.png')
    else:
        plt.show()

# Convert from lat/lon to UTM
def getUTM(site):
    lon =
    lat = 
    myProj = pyproj.Proj(proj ='utm', zone = 11, ellps = 'WGS84', preserve_units=True)
    return myProj(lon, lat)

def main():
    downloadHazardCurve()

main()