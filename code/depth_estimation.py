from helper import *

data_selection = input('\n 1. curule dataset \n 2. octagon dataset \n 3. pendulum dataset \n Please select the dataset to determine the depth estimate: ')
#Defining Calibration data and baselines from given dataset
if data_selection == 1: #curule dataset
    print("Using 'curule' dataset...")
    cam0 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    cam1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    baseline = 88.39  
    f = cam0[0,0]
    path = '/home/starfleeet-robotics/Desktop/StereoVisionSystem/data/curule'

elif data_selection == 2:   #octagon dataset
    print("Using 'octagon' dataset...")
    cam0 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    cam1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    baseline = 221.76 
    f = cam0[0,0]
    path = '/home/starfleeet-robotics/Desktop/StereoVisionSystem/data/octagon'

elif data_selection == 3:   #pendulum dataset
    print("Using 'pendulum' dataset...")
    cam0 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    cam1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    baseline = 537.75
    print(baseline)
    f = cam0[0,0]
    path = '/home/starfleeet-robotics/Desktop/StereoVisionSystem/data/pendulum'


# Calibrating the selected dataset
print("Reading images from the selected dataset...")
images = dataLoader(path)
print(images)
# sift_image for feature matching
sift_image = cv2.xfeatures2d.SIFT_create()
image0 = images[0].copy()
image1 = images[1].copy()

# Convert Images to Grayscale
image0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

print("Finding matches between images using sift_image...")
kp1, des1 = sift_image.detectAndCompute(image0_gray, None)
kp2, des2 = sift_image.detectAndCompute(image1_gray, None)
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x :x.distance)
chosen_matches = matches[0:100]