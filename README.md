# Drowsy Driver Detection System

This project detects driver drowsiness in real-time using eye aspect ratio (EAR) analysis from facial landmarks.

## Features
- Real-time webcam-based monitoring
- Alerts when eyes stay closed beyond a safe threshold
- Uses EAR (Eye Aspect Ratio) with facial landmarks

## Technologies
- Python
- OpenCV
- Dlib
- Scipy
- Imutils

## How to Run
1. **Install Dependencies**
   (in bash)
   pip install -r requirements.txt

2. Download the Dlib Model
Download from: shape_predictor_68_face_landmarks.dat.bz2

&nbsp;&nbsp;**Extract it:**
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

&nbsp;&nbsp;Place the .dat file in the project root folder

3. Run the Script
python drowsy_driver_detector.py


Note
- This program uses your system webcam
- Press q to stop the program
- If running this in GitHub Codespaces, note that webcam access may be limited. For full functionality, run the script locally.
