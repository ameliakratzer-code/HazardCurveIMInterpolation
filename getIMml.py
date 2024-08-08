import csv
import sys
import sqlite3
from utils import getUTM, disFormula

# 1 argument = name of which site list
listNum = sys.argv[1]
if listNum == 1: 
    sites = [('s123','s165','s167','s125','s145'),('s165','s207','s209','s167','s187'),('s159','s201','s203','s161','PDE'),
('s195','s234','s236','s197','WSS'),('s197','s236','s238','s199','P20'),('s199','s238','s240','s201','FFI'),('s207','s246','s248','s209','s228'),('s209','s248','s250','s211','ALP'),
('s234','s271','s273','s236','P23'), ('s724','s768','s770','s726','PERR')]
elif listNum == 2:
    sites = [('s238','s275','s277','s240','P2'),('s238','s275','s277','s240','P1'),('s246','s283','s285','s248','s266'),('s271','s307','s309','s273','P25'),
('s273','s309','s311','s275','s292'),('s275','s311','s313','s277','P4'),('s281','s317','s319','s283','ACTN'),('s283','s319','s321','s285','s302'),
('s307','s345','s347','s309','P16')]
elif listNum == 3:
    sites = [('s307','s345','s347','s309','P17'),('s307','s345','s347','s309','CCP'),('s307','s345','s347','s309','P18'),('s307','s345','s347','s309','s328'),('s307','s345','s347','s309','s346'),('s307','s345','s347','s309','P19'),
('s311','s349','s351','s313','P6'),('s311','s349','s351','s313','P5'),('s317','s355','s357','s319','ALIS'), ('s429', 's470', 's472', 's431', 'DLA')]
elif listNum == 4:
    sites = [('s319','s357','s359','s321','s339'),
('s345','s387','s389','s347','s365'),('s345','s387','s389','s347','s388'),('s345','s387','s389','s347','s366'),('s345','s387','s389','s347','USC'),
('s347','s389','s391','s349','P14'),('s347','s389','s391','s349','LADT'),('s347','s389','s391','s349','P10'),('s347','s389','s391','s349','P9')]
elif listNum == 5:
    sites = [('s347','s389','s391','s349','P8'),
('s349','s391','s393','s351','PAS'),('s355','s397','s399','s357','PACI2'),('s357','s399','s401','s359','s378'),('s383','s427','s429','s385','RHCH'),('s383','s427','s429','s385','TRA'),('s383','s427','s429','s385','HUMC'),
('s385','s429','s431','s387','CSDH'),('s385','s429','s431','s387','COO')]
elif listNum == 6:
    sites = [('s387','s431','s433','s389','s410'),('s387','s431','s433','s389','STNI'),
('s389','s433','s435','s391','P12'),('s397','s441','s443','s399','PACI'),('s429','s470','s472','s431','s451'),('s431','s472','s474','s433','s453'),('s435','s476','s478','s437','EMCH'),('s437','s478','s480','s439','SGRTT'),
('s439','s480','s482','s441','SGCD'), ('s688','s730','s732','s690','s689')]
elif listNum == 7:
    sites = [
('s439','s480','s482','s441','MRSD'),('s443','s484','s486','s445','s465'),('s470','s510','s512','s472','s491'),('s472','s512','s514','s474','s493'),('s474','s514','s516','s476','OLI'),('s484','s524','s526','s486','s505'),
('s510','s550','s552','s512','s531'),('s520','s560','s562','s522','s541'),('s524','s564','s566','s526','s545')]
elif listNum == 8:
    sites = [('s552','s593','s595','s554','SABD'),('s558','s599','s601','s560','CHN'),('s560','s601','s603','s562','PDU'),('s564','s605','s607','s566','s586'),
('s591','s632','s634','s593','STG'),('s603','s644','s646','s605','s624'),('s636','s678','s680','s638','PLS'),('s638','s680','s682','s640','s660'),('s642','s684','s686','s644','PEDL'),('s688','s730','s732','s690','s731')]
elif listNum == 9:
    sites = [('s644','s686','s688','s646','s666'),('s646','s688','s690','s648','s668'),
('s680','s722','s724','s682','LMAT'),('s680','s722','s724','s682','GAVI'),('s682','s724','s726','s684','MKBD'),('s682','s724','s726','s684','GOPH'),
('s684','s726','s728','s686','UCR'),('s686','s728','s730','s688','SBSM'),('s688','s730','s732','s690','s710'), ('s724','s768','s770','s726','PERRM')]
elif listNum == 10:
    sites = [('s726','s770','s772','s728','MRVY'),('s730','s774','s776','s732','SVD'),('s720','s764','s766','s722','s765'),('s036', 's080', 's082', 's038', 's081'),('s068', 's111', 's113', 's070', 'MOP'),('s197', 's236', 's238', 's199', 'P21'),('s238', 's275', 's277', 's240', 'P3'),
('s273', 's309', 's311', 's275', 'HLL'),('s472', 's512', 's514', 's474', 'BRE'),('s435', 's476', 's478', 's437', 'RIO')]

connection = sqlite3.connect('/scratch1/00349/scottcal/CS_interpolation/study_22_12_lf_indexed.sqlite')
cursor = connection.cursor()

for group in sites:
    outputPath = f'/scratch1/10000/ameliakratzer14/IMMLInputs/{group[4]}.csv'
    with open(outputPath, 'w', newline='') as file:
        writer = csv.writer(file)
        # Headers
        writer.writerow(['d1', 'd2', 'd3', 'd4', 's1v', 's1z1', 's1z2', 's2v', 's2z1', 's2z2', 's3v', 's3z1', 's3z2', 's4v', 's4z1', 's4z2', 'sIv', 'sIz1', 'sIz2', 'IMLB', 'IMRB', 'IMRT', 'IMLT', 'IMInterp'])
        xInterpSite, yInterpSite = getUTM(group[4])
        sharedRups = []
        IMVals = []
        eventsList = []
        distance = []
        velocityVals = []
        for site in group[:-1]:
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
        for site in group:
            # Add distances to list
            if site != group[4]:
                x, y = getUTM(site)
                d = disFormula(x,y,xInterpSite,yInterpSite)
                distance.append(d)
            # Add velocities to list
            q2 = '''
                SELECT R.Model_Vs30, R.Z1_0, R.Z2_5 FROM CyberShake_Runs R, Studies T, CyberShake_Sites S
                WHERE T.Study_Name = 'Study 22.12 LF'
                AND T.Study_ID = R.Study_ID
                AND S.CS_Site_ID = R.Site_ID
                AND S.CS_Short_Name = ?
            '''
            cursor.execute(q2, (site,))
            result = cursor.fetchone()
            velocityVals.extend(result)
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
                if site == group[0]:
                    for i in range(len(result)):
                        eventsList.append((source, rup, i))
        r = []
        for val in IMVals:
            r.append(val[0])
        IMs = []
        for i in range(0, len(r), len(eventsList)):
            IMs.append(r[i:i+len(eventsList)])
        for i in range(len(eventsList)):
            writer.writerow(distance + velocityVals + [IMs[0][i], IMs[1][i], IMs[2][i], IMs[3][i], IMs[4][i]])
cursor.close()
connection.close()

