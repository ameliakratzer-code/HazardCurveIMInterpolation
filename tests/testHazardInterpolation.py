import subprocess
import unittest
import csv
import os

def call_script():
    script_name = '/Users/ameliakratzer/codescripts/sources/Pasadena/getCurveInfo.py'
    sitenames = 'S345,S387,S389,S347'
    interpsitename = 'USC'
    # Desktop on laptop, SCRATCH on Frontera
    output = '$SCRATCH'

    # Construct the command to run the second script with arguments
    command = [
        'python3', script_name,
        '--sitename', sitenames,
        '--interpsitename', interpsitename,
        '--output', output
    ]
    # Call the second script using subprocess
    print(os.getcwd())
    result = subprocess.run(command, cwd = os.getcwd(), capture_output=True, text=True)

class TestHazardInterp(unittest.TestCase):
    def test_calculations(self):
        errorTolerance = 0.001 / 100
        call_script()
        currentFile = 'ActualUSC.csv'
        # Reference file stored in tests
        referenceFile = 'ReferenceUSC.csv'
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