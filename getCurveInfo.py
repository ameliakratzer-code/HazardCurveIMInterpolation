import pymysql

# Connect to the database 
connection = pymysql.connect(host = 'moment.usc.edu',
                             user = 'cybershake_ro',
                             password = 'CyberShake2007',
                             database = 'myDB')

with connection.cursor() as cursor:
    # Queries to get hazard curve information
    #change WHERE Site_Name to command line argument
    query1 = '''SELECT CyberShake_Runs.Run_ID FROM CyberShake_Sites
                INNER JOIN CyberShake_Runs
                ON CyberShake_Sites.CS_Site_ID = CyberShake_Runs.Site_ID
                INNER JOIN Studies
                ON CyberShake_Runs.Study_ID = Studies.Study_ID
                WHERE CyberShake_Sites.CS_Site_Name = 'Pasadena' AND Studies.Study_Name = 'Study 22.12 LF';
    '''
    cursor.execute(query1)
    runID = cursor.fetchone()
    #figure out how to use query1 value - the run_Id as WHERE Hazard_Curves.Run_ID = query1
    query2 = '''SELECT * FROM Hazard_Curve_Points
           INNER JOIN Hazard_Curves
           ON Hazard_Curve_Points.Hazard_Curve_ID = Hazard_Curves.Hazard_Curve_ID
           INNER JOIN IM_Types
           ON Hazard_Curves.IM_Type_ID = IM_Types.IM_Type_ID
           WHERE Hazard_Curves.Run_ID = %s AND IM_Types.IM_Type_Value = 2 AND IM_Types.IM_Type_Component=’RotD50’'''
    cursor.execute(query2, (runID))
    result = cursor.fetchall()
    for row in result:
        print(row)

connection.close()

