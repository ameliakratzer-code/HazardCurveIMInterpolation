These codes were designed to interpolate two CyberShake data products, hazard curves and intensity measures, using both bilinear interpolation and machine learning methods. With these interpolations, the goal is to avoid having to run computationally intensive CyberShake simulations for all sites of interest.

File Descriptions

1. Bilinear interpolation hazard curve
- Utils.py: bilinear interpolation library with Site class definition and functions to convert lat and lon to UTM and execute the bilinear interpolation formula
- getCurveInfo.py: uses Utils.py to bilinearly interpolate the hazard curve for a site of interest using 4 input sites. This code takes the input sites, interpolation site, period, and output directory as command line arguments. The user can also choose to include the velocity flag if they want scalar factors, derived from the velocity structures of the input sites, to be used in the bilinear interpolation formula. This script outputs a CSV file with the interpolated values and an overlaid hazard curve plot including the simulated and interpolated hazard curves.
- testHazardInterpolation.py: this script tests whether the difference between the simulated and interpolated hazard curve probabilities are less than or equal to 0.00001.
  
2. Bilinear interpolation IMs
- interpolateIM.py: uses Utils.py to bilinearly interpolate the intensity measure values for a site of interest using 4 input sites. This script can be used to interpolate a single event, all events for a single rupture, or all events for the site of interest. The input sites, interpolation site, period, output directory, velocity flag, source, rupture, and rupture variation are command line arguments. When interpolating all events, the code only runs the bilinear interpolation on events which all 4 input sites share. This script outputs a histogram of percent error between simulated and interpolated IM values, a scatterplot of simulated versus interpolated IMs, and a scatterplot by magnitude.

3. Machine learning hazard curve
- getMLInput.py: since our first ML model just interpolated the hazard cuve probabilities at x = 0.50119 g only, this script creates a CSV file with the hazard curve probabilities at x = 0.50119 g, distances, and velocity structures for all non-10km site groups. This CSV file will serve as input to the hazard curve ML interpolation network.
- getHazardData.py: this script does the same thing as getMLInput.py, but includes the hazard curve probabilities for all 51 x values for the input sites.
- I created three separate scripts, each implementing a different machine learning network structure, to interpolate the hazard curves. MLHazardCurve.py takes as input the CSV file produced from getMLInput.py since it is designed to interpolate only the hazard curve probabilities at x = 0.50119 g. I then created two different scripts ml50Networks.py and MLHazardAllPoints.py which take as input the CSV file produced from getHazardData.py since they are designed to interpolate the hazard curve probabilities at all 51 x values. MLHazardAllPoints.py is one large network with 208 inputs and 51 outputs since it takes as input the input sites hazard curve probabilities at the 51 x values and their distances to the interpolated site and outputs the interpolated hazard curve probabilties at the 51 x values. While ml50Networks.py includes code to train 51 smaller networks which each handle the interpolation of hazard curve probabilities at a different x value.

4. Machine learning IMs
- getIMml.py: for a given list of site groups, this script generates the input CSV file for the IM ML model which includes distances, velocity structures, and IM values for the 4 input sites and interpolated site in the group for each event.
- processData.py: this script takes as a command line argument the IM ML model input file, and preprocesses the data. This includes breaking the data into X input data and y label data, splitting the data into testing and training data, and normalizing the data. It also prepares the USC and s505 data for inference by our ML model. It then uses joblib to store the scalers and data sets.
- IMML1.py: this script takes as input a CSV file containing the distances and IM values for 4 input sites and interpolated site for a single group. It preprocesses the data and trains the model which has 8 inputs, 1 output, and 5 hidden layers with the softplus activation function. The code plots training versus testing loss and performs inference on the test data to create a simulated versus interpolated IM value scatterplot.
- IMML2.py: uses joblib to load the results of processData.py and then trains the model on these results. The model has 23 inputs (as it includes the velocity structures of the input sites since many site groups are included as input now), 1 output, and 3 hidden layers with the softplus activation function. The code plots training versus testing loss and performs inference on USC and s505, creating simulated versus interpolated scatterplots. 


Workflow
- To bilinearly interpolate hazard curves, run getCurveInfo.py and test your results with testHazardInterpolation.py
- To bilinearly interpolate IMs, run interpolateIM.py
- To ML interpolate hazard curves for all 51 x values, run getHazardData.py to get an input CSV file which is used when you run ml50Networks.py.
- To ML interpolate IMs, run getIMml.py with your desired site groups. Then, use the CSV file output as input to processData.py to preprocess your data. Run IMML2.py to train the neural network.
