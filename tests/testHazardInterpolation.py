import unittest
import csv
import sys
sys.path.append('/home1/10000/ameliakratzer14/Pasadena')
from getCurveInfo import main
from unittest.mock import patch
import argparse

class testHazardCurveInterpolater(unittest.TestCase):
    # Testing accuracy of interpolation function
    def testInterpolationCalcs(self):
        with patch('argparse.ArgumentParser.parse_args',
                   return_value=argparse.Namespace(
                       sitenames=['S385', 'S429', 'S431', 'S387'],
                       interpsitename='COO',
                       output='$SCRATCH')):
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
            currentResultsL = main()
            errorTolerance = 0.001 / 100
            for i in range(len(refResultsL)):
                difference = abs(refResultsL[i]-currentResultsL[i])
                # Unittest method that checks whether difference is <= errorTolerance
                self.assertLessEqual(difference, errorTolerance, 'Difference exceeds error tolerence')

if __name__ == '__main__':
    unittest.main()