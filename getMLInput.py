import csv
import sys
import sqlite3
from utils import getUTM, disFormula

# Only argument is filePath (on computer: Desktop/nameOfFile)
outputPath = sys.argv[1]
connection = sqlite3.connect('/scratch1/00349/scottcal/CS_interpolation/study_22_12_lf_indexed.sqlite')
# Create list of all site combinations
sites = [('s034','s078','s080','s036','s035'),('s076','s119','s121','s078','OSI'),('s070','s113','s115','s072','FIL'),('s123','s165','s167','s125','s145'),('s165','s207','s209','s167','s187'),('s159','s201','s203','s161','PDE'),
('s195','s234','s236','s197','WSS'),('s197','s236','s238','s199','P20'),('s199','s238','s240','s201','FFI'),('s207','s246','s248','s209','s228'),('s209','s248','s250','s211','ALP'),
('s234','s271','s273','s236','P23'),('s238','s275','s277','s240','P2'),('s238','s275','s277','s240','P1'),('s246','s283','s285','s248','s266'),('s271','s307','s309','s273','P25'),
('s273','s309','s311','s275','s292'),('s275','s311','s313','s277','P4'),('s281','s317','s319','s283','ACTN'),('s283','s319','s321','s285','s302'),
('s307','s345','s347','s309','P16'),('s307','s345','s347','s309','P17'),('s307','s345','s347','s309','CCP'),('s307','s345','s347','s309','P18'),('s307','s345','s347','s309','s328'),('s307','s345','s347','s309','s346'),('s307','s345','s347','s309','P19'),
('s311','s349','s351','s313','P6'),('s311','s349','s351','s313','P5'),('s317','s355','s357','s319','ALIS'),('s319','s357','s359','s321','s339'),
('s345','s387','s389','s347','s365'),('s345','s387','s389','s347','s346'),('s345','s387','s389','s347','s388'),('s345','s387','s389','s347','s366'),('s345','s387','s389','s347','USC'),
('s347','s389','s391','s349','P14'),('s347','s389','s391','s349','LADT'),('s347','s389','s391','s349','P10'),('s347','s389','s391','s349','P9'),('s347','s389','s391','s349','P8'),
('s349','s391','s393','s351','PAS'),('s355','s397','s399','s357','PACI2'),('s357','s399','s401','s359','s378'),('s383','s427','s429','s385','RHCH'),('s383','s427','s429','s385','TRA'),('s383','s427','s429','s385','HUMC'),
('s385','s429','s431','s387','CSDH'),('s385','s429','s431','s387','COO'),('s387','s431','s433','s389','s410'),('s387','s431','s433','s389','STNI'),
('s389','s433','s435','s391','P12'),('s397','s441','s443','s399','PACI'),('s429','s470','s472','s431','s451'),('s431','s472','s474','s433','s453'),('s435','s476','s478','s437','EMCH'),('s437','s478','s480','s439','SGRTT'),
('s439','s480','s482','s441','SGCD'),('s439','s480','s482','s441','MRSD'),('s443','s484','s486','s445','s465'),('s470','s510','s512','s472','s491'),('s472','s512','s514','s474','s493'),('s474','s514','s516','s476','OLI'),('s484','s524','s526','s486','s505'),
('s510','s550','s552','s512','s531'),('s520','s560','s562','s522','s541'),('s524','s564','s566','s526','s545'),('s552','s593','s595','s554','SABD'),('s558','s599','s601','s560','CHN'),('s560','s601','s603','s562','PDU'),('s564','s605','s607','s566','s586'),
('s591','s632','s634','s593','STG'),('s603','s644','s646','s605','s624'),('s636','s678','s680','s638','PLS'),('s638','s680','s682','s640','s660'),('s642','s684','s686','s644','PEDL'),('s644','s686','s688','s646','s666'),('s646','s688','s690','s648','s668'),
('s680','s722','s724','s682','LMAT'),('s680','s722','s724','s682','GAVI'),('s682','s724','s726','s684','MKBD'),('s682','s724','s726','s684','GOPH'),
('s684','s726','s728','s686','UCR'),('s686','s728','s730','s688','SBSM'),('s588','s730','s732','s690','s710'),('s724','s768','s770','s726','PERR'),('s724','s768','s770','s726','PERRM'),
('s726','s770','s772','s728','MRVY'),('s730','s774','s776','s732','SVD'),('s720','s764','s766','s722','s765'),('s036', 's080', 's082', 's038', 's081'),('s068', 's111', 's113', 's070', 'MOP'),('s197', 's236', 's238', 's199', 'P21'),('s238', 's275', 's277', 's240', 'P3'),
('s273', 's309', 's311', 's275', 'HLL'),('s720', 's764', 's766', 's722', 's765'),('s472', 's512', 's514', 's474', 'BRE'),('s246', 's283', 's285', 's248', 's266'),('s435', 's476', 's478', 's437', 'RIO'),('s429', 's470', 's472', 's431', 'DLA')]

with open(outputPath, 'w', newline='') as file:
    write = csv.writer(file)
    # Headers
    write.writerow(['LBProb, RBProb, RTProb, LTProb, simVal, d1, d2, d3, d4, interpSiteName'])
    # Have list of groups of sites we are using [(s0, s1, s2, s3, interpsite), (s0, s1, s2, s3, interpsite)]
    for group in sites:
        xInterpSite, yInterpSite = getUTM(group[4])
        probVals = []
        distanceVals = []
        cursor = connection.cursor()
        print(group)
        for i in range(5):
            q1 = '''SELECT CyberShake_Runs.Run_ID FROM CyberShake_Sites
                    INNER JOIN CyberShake_Runs
                    ON CyberShake_Sites.CS_Site_ID = CyberShake_Runs.Site_ID
                    INNER JOIN Studies
                    ON CyberShake_Runs.Study_ID = Studies.Study_ID
                    WHERE CyberShake_Sites.CS_Short_Name = ? AND Studies.Study_Name = 'Study 22.12 LF';
                '''
            cursor.execute(q1, (group[i],))
            runID = cursor.fetchone()[0]
            desiredXVal = 0.50119
            period = 2
            q2 = '''SELECT Hazard_Curve_Points.Y_Value FROM Hazard_Curve_Points
                        INNER JOIN Hazard_Curves
                        ON Hazard_Curve_Points.Hazard_Curve_ID = Hazard_Curves.Hazard_Curve_ID
                        INNER JOIN IM_Types
                        ON Hazard_Curves.IM_Type_ID = IM_Types.IM_Type_ID
                        WHERE Hazard_Curves.Run_ID = ? AND IM_Types.IM_Type_Value = ? AND IM_Types.IM_Type_Component='RotD50' AND Hazard_Curve_Points.X_Value = ?;
                    '''
            cursor.execute(q2, (runID, period, desiredXVal))
            result = cursor.fetchone()
            # Appending site prob to list, including interpsite
            probVals.append(result[0])
            # Calculate distance from interpsite to input site
            x, y = getUTM(group[i])
            d = disFormula(x,y,xInterpSite,yInterpSite)
            distanceVals.append(d)
        # Write vals for this group to file
        write.writerow(probVals + distanceVals + [group[4]])
    
            

