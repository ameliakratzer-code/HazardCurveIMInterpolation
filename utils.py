import pymysql
import pyproj

# Connect to the database
connection = pymysql.connect(host = 'moment.usc.edu',
                            user = 'cybershk_ro',
                            password = 'CyberShake2007',
                            database = 'CyberShake')

# Create instance of Site class by doing site0 = Site('USC', getIM or downloadHazardCurves)
class Site:
    def __init__(self, name, valsToInterp):
        self.name = name
        self.valsToInterp = valsToInterp
        self.x = None
        self.y = None

    def getUTM(self, name):
        with connection.cursor() as cursor:
            query3 = '''SELECT CS_Site_Lat, CS_Site_Lon FROM CyberShake_Sites
                        WHERE CS_Short_Name = %s
            '''
            cursor.execute(query3, (name))
            location = cursor.fetchall()
            lat, lon = location[0][0], location[0][1]
        myProj = pyproj.Proj(proj ='utm', zone = 11, ellps = 'WGS84')
        # Update self x and y
        self.x, self.y = myProj(lon, lat)
        return self.x, self.y
