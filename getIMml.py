import csv
import sys
import sqlite3
from utils import getUTM, disFormula

# Only argument is filePath (must run on Frontera to connect to DB)
outputPath = sys.argv[1]
connection = sqlite3.connect('/scratch1/00349/scottcal/CS_interpolation/study_22_12_lf_indexed.sqlite')
cursor = connection.cursor()

# Gets all events for this siteGroup
with open(outputPath, 'w', newline='') as file:
    siteGroup = ('s385','s429','s431','s387','COO')
    # First event in sharedRups
    eventNum = 0
    sharedRups = []
    IMVals = []
    eventsList = []
    for site in siteGroup[:-1]:
        # Get shared rups
        q0 = '''
                SELECT C.Source_ID, C.Rupture_ID
                FROM CyberShake_Site_Ruptures C, CyberShake_Sites S
                WHERE C.CS_Site_ID = S.CS_Site_ID
                AND C.ERF_ID = 36
                AND S.CS_Short_Name = ?;
        '''
        cursor.execute(q0, (site,))
        result = cursor.fetchall()
        if sharedRups == []:
            sharedRups = result
        else:
            sharedRups = list(set(sharedRups) & set(result))
    for site in siteGroup:
        for (source, rup) in sharedRups:
            q1 = '''
                    SELECT P.IM_Value 
                    FROM CyberShake_Sites S, CyberShake_Runs R, PeakAmplitudes P, Studies T, IM_Types I
                    WHERE S.CS_Short_Name = ?
                    AND S.CS_Site_ID = R.Site_ID
                    AND R.Study_ID = T.Study_ID
                    AND T.Study_Name = 'Study 22.12 LF'
                    AND R.Run_ID = P.Run_ID
                    AND I.IM_Type_Component = 'RotD50'
                    AND I.IM_Type_Value = 2
                    AND I.IM_Type_ID = P.IM_Type_ID
                    AND P.Source_ID = ?
                    AND P.Rupture_ID = ?
                    '''
            cursor.execute(q1, (site, source, rup))
            result = cursor.fetchall()
            IMVals.extend(result)
            if site == siteGroup[0]:
                for i in range(len(result)):
                    eventsList.append((source, rup, i))
            r = []
            for val in IMVals:
                r.append(val[0])
    IMs = []
    for i in range(0, len(r), len(eventsList)):
        IMs.append(r[i:i+len(eventsList)])
    # Write vals to csv file
    writer = csv.writer(file)
    # Headers = event IDs
    writer.writerows(['siteName'] + eventsList)
    # One row = site s385 and all IM vals for the sharedRups
    for i in range(5):
        writer.writerows(siteGroup[i] + IMs[i])
        # 5th column down = y labels for interpsite
cursor.close()
connection.close()