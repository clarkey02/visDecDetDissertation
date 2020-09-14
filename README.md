# visDecDetDissertation
A public repository showing the code for visualisation software produced in relation to the following dissertation title: "An Explorative Study into the Effects of Visualisation on Deception Detection Accuracy".

To run in terminal:

1 - Download all files/directories from the Master Branch you are currently on

2 - Navigate to the directory where these are stored locally

3 - Enter the following command:

    python facialVisualisationSoftware.py

4 - After software finishes, processed frames should appear in 'images' folder

5 - Enter the following command to produce video output:

    python videoConvert.py

6 - File titled 'output.avi' should be created.

TROUBLESHOOTING:

- An FLV supported media player may be required to run the video files
- Various pip install packages including: numpy, scipy, dlib, openCV, imutils...
- Two test flv files have been provided
- By default the software applies a point visualisation to test2.flv
- To change visualisation type and file, enter facialVisualisationSoftware.py. Sensitivities may need to be altered.
