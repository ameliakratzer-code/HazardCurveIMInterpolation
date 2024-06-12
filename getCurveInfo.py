import pymysql
import argparse

# Handle command line arguments
# By default, arguments are strings
parser = argparse.ArgumentParser('Allow user to input site name, period')
parser.add_argument('--sitename')
args = parser.parse_args()

# Connect to the database 
connection = pymysql.connect(host = 'moment.usc.edu',
                             user = 'cybershk_ro',
                             password = 'CyberShake2007',
                             database = 'CyberShake')

with connection.cursor() as cursor:
    # Queries to get hazard curve information
    query1 = f'''SELECT CyberShake_Runs.Run_ID FROM CyberShake_Sites
                INNER JOIN CyberShake_Runs
                ON CyberShake_Sites.CS_Site_ID = CyberShake_Runs.Site_ID
                INNER JOIN Studies
                ON CyberShake_Runs.Study_ID = Studies.Study_ID
                WHERE CyberShake_Sites.CS_Short_Name = '{args.sitename}' AND Studies.Study_Name = 'Study 22.12 LF';
    '''
    cursor.execute(query1)
    runID = cursor.fetchall()
    # Use query1 value - the run_Id as WHERE Hazard_Curves.Run_ID = query1
    query2 = '''SELECT * FROM Hazard_Curve_Points
           INNER JOIN Hazard_Curves
           ON Hazard_Curve_Points.Hazard_Curve_ID = Hazard_Curves.Hazard_Curve_ID
           INNER JOIN IM_Types
           ON Hazard_Curves.IM_Type_ID = IM_Types.IM_Type_ID
           WHERE Hazard_Curves.Run_ID = %s AND IM_Types.IM_Type_Value = 2 AND IM_Types.IM_Type_Component='RotD50'
           '''
    cursor.execute(query2, (runID))
    result = cursor.fetchall()

# Print out result
for row in result:
    print(row)
print(len(result))

connection.close()

