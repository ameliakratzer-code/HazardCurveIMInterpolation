import unittest
import csv
import sys
sys.path.append('/home1/10000/ameliakratzer14/Pasadena')
import getCurveInfo

class TestHazardInterp(unittest.TestCase):
    def test_calculations(self):
        errorTolerance = 0.001 / 100
        # '/home1/10000/ameliakratzer14/Pasadena/getCurveInfo.py'
        args = [
        '--sitenames', 's345,s387,s389,s347',
        '--interpsitename', 'USC',
        '--output', '$SCRATCH'
        ]
        print(args)
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