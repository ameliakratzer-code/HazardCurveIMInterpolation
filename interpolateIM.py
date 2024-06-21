import pymysql
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Allow user to input site names, ')
parser.add_argument('--sitenames', nargs='+')
# User can specify a particular event ID (source, rup, rupVar) to do the interpolation for
parser.add_argument('--eventID')
parser.add_argument('--interpsitename')
args = parser.parse_args()

# Connect to the database
connection = pymysql.connect(host = 'moment.usc.edu',
                            user = 'cybershk_ro',
                            password = 'CyberShake2007',
                            database = 'CyberShake')

def getIMValues(nameSite):
    with connection.cursor() as cursor:
        IMVals = '''
        SELECT P.Source_ID, P.Rupture_ID, P.Rup_Var_ID, P.IM_Value 
        FROM CyberShake_Sites S, CyberShake_Runs R, PeakAmplitudes P, Studies T, IM_Types I
        WHERE S.CS_Short_Name = %s
        AND S.CS_Site_ID = R.Site_ID
        AND R.Study_ID = T.Study_ID
        AND T.Study_Name = 'Study 22.12 LF'
        AND R.Run_ID = P.Run_ID
        AND I.IM_Type_Component = 'RotD50'
        AND I.IM_Type_Value = 2.0
        AND I.IM_Type_ID = P.IM_Type_ID;
        '''
        cursor.execute(IMVals, (nameSite))
        result = cursor.fetchall()

def main():
    # Create comma-separated list of sites from arg