import unittest
import csv
import sys
import os
# Relative path: get path of user's current directory and add on Pasadena
current_dir = os.path.dirname(os.path.abspath(__file__))
pasadena_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(pasadena_dir)
import getCurveInfo

class TestHazardInterp(unittest.TestCase):
    def test_calculations(self):
        errorTolerance = 0.001 / 100
        # Add script name since getCurveInfo parser parses from args[1:]
        args = ['getCurveInfo.py', '--sitenames', 's345,s387,s389,s347', '--interpsitename', 'USC', '--output', '$SCRATCH']
        getCurveInfo.main(argv=args)
        # Reference file stored in tests
        referenceFile = 'ReferenceUSC.csv'
        currentFile = 'ActualUSC.csv'
        refResultsL = []
        with open(referenceFile, 'r') as file:
            read = csv.reader(file)
            next(read)
            for row in read:
                refResultsL.append(float(row[1]))
        curResultsL = []
        with open(currentFile, 'r') as file:
            read = csv.reader(file)
            next(read)
            for row in read:
                curResultsL.append(float(row[1]))
        for i in range(len(refResultsL)):
            difference = abs(refResultsL[i]-curResultsL[i])
            self.assertLessEqual(difference, errorTolerance, 'Difference exceeds error tolerence')

if __name__ == "__main__":
    unittest.main()