import sqlite3
import pyproj
import math


def getUTM(siteName):
    connection = sqlite3.connect('/scratch1/00349/scottcal/CS_interpolation/study_22_12_lf_indexed.sqlite')
    cursor = connection.cursor()
    query3 = '''SELECT CS_Site_Lat, CS_Site_Lon FROM CyberShake_Sites
                    WHERE CS_Short_Name = ?
        '''
    cursor.execute(query3, (siteName,))
    location = cursor.fetchall()
    lat, lon = location[0][0], location[0][1]
    cursor.close()
    connection.close()
    myProj = pyproj.Proj(proj ='utm', zone = 11, ellps = 'WGS84')
    return myProj(lon, lat)

# Create instance of Site class by doing site0 = Site('USC', getIM or downloadHazardCurves)
class Site:
    def __init__(self, name, valsToInterp):
        self.name = name
        self.valsToInterp = valsToInterp
        self.x, self.y = getUTM(name)

    # Define comparison operators
    def within_x_range(self, s0, s1):
        return (s0.x <= self.x <= s1.x) or (s1.x <= self.x <= s0.x)

    def within_y_range(self, s0, s1):
        return (s0.y <= self.y <= s1.y) or (s1.y <= self.y <= s0.y)

    def y_less_than(self, other):
        return self.y < other.y

    def __repr__(self):
        return f'Sitename {self.name}'

# Shared functions between getCurveInfo and interpolateIM.py
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

# Used when checking if input sites form a square
def disFormula(x0, y0, x1, y1):
    dsquared = (x0-x1)**2 + (y0-y1)**2
    d = dsquared**0.5
    return d

def interpolate(sortedL, xVals, velYesorNo):
    # Determining S0, S3
    if sortedL[0].y_less_than(sortedL[1]):
        # Download hazard curve of site at L[0]
        s0 = sortedL[0]
        s3 = sortedL[1]
    else:
        s0 = sortedL[1]
        s3 = sortedL[0]
    # Determing S1, S2
    if sortedL[2].y_less_than(sortedL[3]):
        s1 = sortedL[2]
        s2 = sortedL[3]
    else:
        s1 = sortedL[3]
        s2 = sortedL[2]
    # Check if sites form square before interpolating: sides and diagonals
    if (not(9900 <= disFormula(s0.x,s0.y,s1.x,s1.y) <= 10100) or not(9900 <= disFormula(s1.x,s1.y,s2.x,s2.y) <= 10100) or 
        not(9900 <= disFormula(s2.x,s2.y,s3.x,s3.y) <= 10100) or not(9900 <= disFormula(s3.x,s3.y,s0.x,s0.y) <= 10100) or
        not((9900*math.sqrt(2)) <= disFormula(s0.x,s0.y,s2.x,s2.y) <= (math.sqrt(2)*10100)) or not((9900*math.sqrt(2)) <= disFormula(s1.x,s1.y,s3.x,s3.y) <= (math.sqrt(2)*10100))):
        print('Entered sites do not form a square')
        exit()
    # Calculate distances with slanted axis
    yPrime = getDistance(s3.x, s3.y, s2.x, s2.y, sortedL[-1].x, sortedL[-1].y) / 10000
    xPrime =  getDistance(s3.x, s3.y, s0.x, s0.y, sortedL[-1].x, sortedL[-1].y) / 10000
    interpolatedProbs = []
    if velYesorNo == True:
        # Taking into account velocity model
        connection = sqlite3.connect('/scratch1/00349/scottcal/CS_interpolation/study_22_12_lf_indexed.sqlite')
        cursor = connection.cursor()
        scales = []
        # Get velocity model vals for interp site
        q = '''
                SELECT R.Model_Vs30, R.Z1_0, R.Z2_5 FROM CyberShake_Runs R, Studies T, CyberShake_Sites S
                WHERE T.Study_Name = 'Study 22.12 LF'
                AND T.Study_ID = R.Study_ID
                AND S.CS_Site_ID = R.Site_ID
                AND S.CS_Short_Name = ?
            '''
        cursor.execute(q, (sortedL[4].name,))
        result = cursor.fetchone()
        sIVs30, sIZ1, sIZ2 = result[0], result[1], result[2]
    # Compare ratios and calculate scale factor for each site
    for site in [s0.name, s1.name, s2.name, s3.name]:
        totalDifference = 0
        if velYesorNo == True:
            cursor.execute(q, (site,))
            res = cursor.fetchone()
            Vs30Ratio = sIVs30 / res[0]
            Z1Ratio = sIZ1 / res[1]
            Z2Ratio = sIZ2 / res[2]
            # Decide if increasing or decreasing ground motions
            if Vs30Ratio > 1:
                # Negative contribution to total distance since want to decrease ground motions
                totalDifference += 1-Vs30Ratio
            else:
                totalDifference += Vs30Ratio-1
            if  Z1Ratio > 1:
                # Positive contribution to total distance since want to increase ground motions
                totalDifference += Z1Ratio-1
            else:
                totalDifference += 1-Z1Ratio
            if  Z2Ratio > 1:
                # Positive contribution to total distance since want to increase ground motions
                totalDifference += Z2Ratio-1
            else:
                totalDifference += 1-Z2Ratio
            # Modeling after 0.25 scale = 1.5 difference for site s505
            scaleFactor = 6.5
            scale = totalDifference / scaleFactor
            scales.append(1+scale)
    for i in range(len(xVals)):
        if velYesorNo == True:
            R1 = (s0.valsToInterp[i] * scales[0] * (1-xPrime) + s1.valsToInterp[i] * scales[1] * xPrime)
            R2 = (s2.valsToInterp[i] * scales[2] * xPrime + s3.valsToInterp[i] * scales[3] * (1-xPrime))
            interpVal = (R1 * yPrime + R2 * (1-yPrime))
            interpolatedProbs.append(interpVal)
        else:
            R1 = (s0.valsToInterp[i] * (1-xPrime) + s1.valsToInterp[i] * xPrime)
            R2 = (s2.valsToInterp[i] * xPrime + s3.valsToInterp[i] * (1-xPrime))
            interpVal = (R1 * yPrime + R2 * (1-yPrime))
            interpolatedProbs.append(interpVal)
    cursor.close()
    connection.close()
    return interpolatedProbs