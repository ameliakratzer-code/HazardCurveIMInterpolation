import unittest
import csv
import sys
sys.path.append('/home1/10000/ameliakratzer14/Pasadena')
from getCurveInfo import bilinearinterpolation

class testHazardCurveInterpolater(unittest.TestCase):
    # Testing accuracy of interpolation function
    def testInterpolationCalcs(self):
        # Get ref results list
        referenceFile = 'ReferenceCOO.csv'
        refResultsL = []
        with open(referenceFile, 'r') as file:
            read = csv.reader(file)
            # Skip headers, vals read as strings
            next(read)
            for row in read:
                refResultsL.append(float(row[1]))
        # Compare ref results list to current results
        args = ['--sitename', 'S385,S429,S431,S387', '--interpsitename', 'COO', '--output', '$SCRATCH']
        currentResultsL = bilinearinterpolation(args)
        errorTolerance = 0.001 / 100
        for i in range(len(refResultsL)):
            difference = abs(refResultsL[i]-currentResultsL[i])
            # Unittest method that checks whether difference is <= errorTolerance
            self.assertLessEqual(difference, errorTolerance, 'Difference exceeds error tolerence')

if __name__ == '__main__':
    unittest.main()