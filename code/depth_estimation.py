from helper import *

data_selection = input('\n 1. curule dataset \n 2. octagon dataset \n 3. pendulum dataset \n Please select the dataset to determine the depth estimate: ')

#selecting curule dataset
if data_selection =='1': 
    print("Using 'curule' dataset...")
    cam0 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    cam1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    baseline = 88.39  
    f = cam0[0,0]
    images_path = '/Users/sumedhreddy/Desktop/StereoVisionSystem/data/curule'

#selecting octagon dataset
elif data_selection =='2':   
    print("Using 'octagon' dataset...")
    cam0 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    cam1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    baseline = 221.76 
    f = cam0[0,0]
    images_path = '/Users/sumedhreddy/Desktop/StereoVisionSystem/data/octagon'

#selecting pendulum dataset
elif data_selection =='3':
    print("Using 'pendulum' dataset...")
    cam0 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    cam1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    baseline = 537.75
    f = cam0[0,0]
    images_path = '/Users/sumedhreddy/Desktop/StereoVisionSystem/data/pendulum'


# Calibrating the selected dataset
print("Reading images from the selected dataset...")
images = dataLoader(images_path)
# Feature Matching, Fundamental Matrix and RANSAC
# sift_image for feature matching
sift_image = cv2.xfeatures2d.SIFT_create()
image0 = images[0].copy()
image1 = images[1].copy()

# Convert the input images to Grayscale
image0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

print("Finding matches between two images using sift_image")
# Finding the Matching points between two images
kp1, des_1 = sift_image.detectAndCompute(image0_gray, None)
kp2, des_2 = sift_image.detectAndCompute(image1_gray, None)
matcher = cv2.BFMatcher()
matched = matcher.match(des_1,des_2)
# sorting matched points
matched = sorted(matched, key = lambda x :x.distance)
# choosing 0:100
selected_matches = matched[0:100]
features = []
for i, mth in enumerate(selected_matches):
    point1 = kp1[mth.queryIdx].pt
    point2 = kp2[mth.trainIdx].pt
    features.append([point1[0], point1[1], point2[0], point2[1]])
    
features = np.array(features).reshape(-1, 4)
# Match Outlier Rejection via RANSAC
# Get Inliner RANSAC 
# Calculating F matrix and obtaining maximized number of inliers

required_idx = []
Fundamental_matrix = 0
iterations = 1000
inliers_threshold = 0
error_threshold = 0.03

for i in range(0, iterations):
    idxs = []
    # choosing 8 correspondences randomly
    n_rows = features.shape[0]
    random_idx = np.random.choice(n_rows, size=8)
    point_8_features = features[random_idx, :] 
    x_1 = point_8_features[:,0:2]
    x_2 = point_8_features[:,2:4]
    # Calculating Fundamental Matrix
    F = calcFundamentalMatrix(x_1, x_2)
    for j in range(n_rows):
        feature = features[j]
        x1,x2 = feature[0:2], feature[2:4]
        x1_tmp=np.array([x1[0], x1[1], 1]).T
        x2_tmp=np.array([x2[0], x2[1], 1])
        err = np.dot(x1_tmp, np.dot(F, x2_tmp))
        err = np.abs(err)
        if err < error_threshold:
            idxs.append(j)

    if len(idxs) > inliers_threshold:
        inliers_threshold = len(idxs)
        required_idx = idxs
        Fundamental_matrix = F

# Matched Inliners
matched_inliers = features[required_idx, :]
print("Fundamental Matrix : ", Fundamental_matrix)
