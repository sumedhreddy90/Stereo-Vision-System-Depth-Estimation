# StereoVisionSystem

# Instructions to Run the Program

```
git clone https://github.com/sumedhreddy90/StereoVisionSystem.git
cd StereoVisionSystem
cd code 
python3 depth_estimation.py

```
# Inputs 

 1. curule dataset 
 2. octagon dataset 
 3. pendulum dataset 
 Please select the dataset to determine the depth estimate: 2

 Select your dataset for depth estimation

 # Outputs

## Calibration Image Processing:

 Fundamental Matrix :\  [[-4.01998335e-10 -2.06586677e-08  3.47888159e-05]
 [ 4.92569519e-08 -1.00397767e-07 -1.66207555e-03]
 [-4.74547906e-05  1.71957189e-03 -2.40244547e-03]]\
 
Essential Matrix:\  [[-4.72041496e-04 -2.28067026e-02  1.49934322e-02]
 [ 5.10735428e-02 -5.27644655e-02 -9.97204888e-01]
 [-9.83596848e-03  9.98268008e-01 -5.29988959e-02]]\
 
Estimated Rotation:\  [[ 0.99961331 -0.00248165  0.02769617]
 [ 0.00393755  0.99860595 -0.05263693]
 [-0.02752694  0.05272563  0.99822957]]\
 
Estimated translation:\  [0.99962734 0.01377738 0.02356598]


## Rectification Image Processing:
<img width="461" alt="epipolar_2" src="https://user-images.githubusercontent.com/24978535/163831761-7378f604-ea17-4f02-88dd-3275c6d78856.png">
<img width="461" alt="rectified_epipolar_2" src="https://user-images.githubusercontent.com/24978535/163831792-a9a8a2c7-33eb-4f7d-aef1-235c5281671b.png">

## Correspondence Image Processing:
<img width="441" alt="disparity_2" src="https://user-images.githubusercontent.com/24978535/163831810-2d971074-47ef-4e98-ab6b-12ce6b75a544.png">

## Depth Image Processing:
<img width="385" alt="depth_2" src="https://user-images.githubusercontent.com/24978535/163831821-7cdc0201-79c3-4a99-b60c-c4e3bb579434.png">

