from helper import *

# Pipeline for creating a Stereo Vision System

data_selection = input('\n 1. curule dataset \n 2. octagon dataset \n 3. pendulum dataset \n Please select the dataset to determine the depth estimate: ')

#selecting curule dataset
if data_selection =='1': 
    print("Using 'curule' dataset...")
    cam0 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    cam1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    baseline = 88.39  
    f = cam0[0,0]
    images_path = '../data/curule'

#selecting octagon dataset
elif data_selection =='2':   
    print("Using 'octagon' dataset...")
    cam0 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    cam1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    baseline = 221.76 
    f = cam0[0,0]
    images_path = '../data/octagon'

#selecting pendulum dataset
elif data_selection =='3':
    print("Using 'pendulum' dataset...")
    cam0 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    cam1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    baseline = 537.75
    f = cam0[0,0]
    images_path = '../data/pendulum'

# Step: 1 :: Calibration: 
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
fundamental_matrix = 0
iterations = 1000
inliers_threshold = 0
error_threshold = 0.02

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
        fundamental_matrix = F

# Matched Inliners
matched_inliers = features[required_idx, :]
print("Fundamental Matrix : ", fundamental_matrix)

# Calculating Essential Matrix
E = cam1.T.dot(fundamental_matrix).dot(cam0)
U,s,V = np.linalg.svd(E)
s = [1,1,0]
essential_matrix = np.dot(U,np.dot(np.diag(s),V))
print("Essential Matrix: ", essential_matrix)

# Estimating Camera Pose from Essential Matrix
# Calculating rotation and translation from E
rotation, translation = estimateCameraPose(essential_matrix)

# Triangulation Check for Cheirality Condition
# Calculating Triangulate Points
points_3D = []
Rot_1 = np.identity(3)
Trans_1 = np.zeros((3,1))
I = np.identity(3)
P1 = np.dot(cam0, np.dot(Rot_1, np.hstack((I, -Trans_1.reshape(3,1)))))

for i in range(len(translation)):
    x1 = matched_inliers[:,0:2].T
    x2 = matched_inliers[:,2:4].T

    P2 = np.dot(cam1, np.dot(rotation[i], np.hstack((I, -translation[i].reshape(3,1)))))

    X = cv2.triangulatePoints(P1, P2, x1, x2)  
    points_3D.append(X)

# Estimating best Rotation and Translation
est_rotation, est_translation = calcBestRT(points_3D, rotation, translation)

print("Estimated Rotation: ", est_rotation)
print("Estimated translation: ",est_translation)
print("Calibration Complete")

# Step: 2 :: Rectification
print(" Rectification Image Processing")
# Apply perspective transformation to make sure that the epipolar lines are
# horizontal for both the images.
pair_1, pair_2 = matched_inliers[:,0:2], matched_inliers[:,2:4]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pair_2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
Epipolar_img0,_ = drawlines(image0,image1,lines1,pair_1,pair_2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pair_1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
Epipolar_img1,_ = drawlines(image1,image0,lines2,pair_2,pair_1)
# Plotting Epipolar lines
Epipolar_img0 = cv2.cvtColor(Epipolar_img0, cv2.COLOR_BGR2RGB)
plt.subplot(211),plt.imshow(Epipolar_img0),plt.title('Epipolar lines Image 0')
plt.axis('off')
Epipolar_img1 = cv2.cvtColor(Epipolar_img1, cv2.COLOR_BGR2RGB)
plt.subplot(212),plt.imshow(Epipolar_img1), plt.title('Epipolar lines Image 1')
plt.axis('off')
plt.show()

h1, w1 = image0.shape[:2]
h2, w2 = image1.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pair_1), np.float32(pair_2), fundamental_matrix, imgSize=(w1, h1))

# Warp the images and Transforming Images
img0 = images[0].copy()
img1 = images[1].copy()
img1_rect = cv2.warpPerspective(img0, H1, (w1, h1))
img2_rect = cv2.warpPerspective(img1, H2, (w2, h2))

img1_rect_gray = cv2.warpPerspective(image0_gray, H1, (w1, h1))
img2_rect_gray = cv2.warpPerspective(image1_gray, H2, (w2, h2))

pair_1_rect = cv2.perspectiveTransform(pair_1.reshape(-1, 1, 2), H1).reshape(-1,2)
pair_2_rect = cv2.perspectiveTransform(pair_2.reshape(-1, 1, 2), H2).reshape(-1,2)

# Solving Rectified Fundamental Matrix
H2_T_inv =  np.linalg.inv(H2.T)
H1_inv = np.linalg.inv(H1)
F_rectified = np.dot(H2_T_inv, np.dot(fundamental_matrix, H1_inv))

# Drawing Rectified Epipolar lines 
# Find Rectified epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pair_2_rect.reshape(-1,1,2), 2,F_rectified)
lines1 = lines1.reshape(-1,3)
rect_epipolar_img0,_ = drawlines(img1_rect,img2_rect,lines1,pair_1_rect,pair_2_rect)

# Find Rectified epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pair_1_rect.reshape(-1,1,2), 1,F_rectified)
lines2 = lines2.reshape(-1,3)
rect_epipolar_img1,_ = drawlines(img2_rect,img1_rect,lines2,pair_2_rect,pair_1_rect)
# Plotting Rectified Epipolar lines
rect_epipolar_img0 = cv2.cvtColor(rect_epipolar_img0, cv2.COLOR_BGR2RGB)
plt.subplot(211),plt.imshow(rect_epipolar_img0), plt.title('Rectified Epipolar lines Image 0')
plt.axis('off')
rect_epipolar_img1 = cv2.cvtColor(rect_epipolar_img1, cv2.COLOR_BGR2RGB)
plt.subplot(212),plt.imshow(rect_epipolar_img1),plt.title('Rectified Epipolar lines Image 1')
plt.axis('off')
plt.show() 

# Step: 2 :: Correspondence
# For each epipolar line, applying the matching windows concept (discussed in
# class such as SSD).
print("Correspondence Image Processing")

no_disparities = 15 
pix_block_size = 5      
h, w = img1_rect_gray.shape
disparity_mask = np.zeros(shape = (h,w))

# Apply Matching Windows concept with SSD
# Using SSD, compare matches along epipolar lines of the two pictures to compute the Disparity Map.
for i in range(pix_block_size, img1_rect_gray.shape[0] - pix_block_size - 1):
    for j in range(pix_block_size + no_disparities, img1_rect_gray.shape[1] - pix_block_size - 1):
        SSD = np.empty([no_disparities, 1])
        l = img1_rect_gray[(i - pix_block_size):(i + pix_block_size), (j - pix_block_size):(j + pix_block_size)]
        h, w = l.shape
        for d in range(0, no_disparities):
            r = img2_rect_gray[(i - pix_block_size):(i + pix_block_size), (j - d - pix_block_size):(j - d + pix_block_size)]
            SSD[d] = np.sum((l[:,:]-r[:,:])**2)
        disparity_mask[i, j] = np.argmin(SSD)


print("Disparity: ", SSD)

# Rescaling the disparity to be from 0-255 and saving the resulting image

disparity_rescale = ((disparity_mask/disparity_mask.max())*255).astype(np.uint8)

# saving the disparity as a gray scale and color image using heat map conversion
print("Creating heatmap ")
cmap = plt.get_cmap('inferno')
heat_map = (cmap(disparity_rescale) * 2**16).astype(np.uint16)[:,:,:3]
heat_map = cv2.cvtColor(heat_map, cv2.COLOR_RGB2BGR)

# Plotting Disparity gray and heat map
plt.subplot(211),plt.imshow(disparity_rescale),plt.title(' Disparity grayscale vs heat map')
plt.axis('off')
plt.subplot(212),plt.imshow(heat_map),plt.title(' Disparity grayscale vs heat map')
plt.axis('off')
plt.show() 


# computing the depth information for each image pixel.
# The resulting depth image has the same dimensions of the disparity image but it has depth information instead.

depth_image = np.zeros(shape=img1_rect_gray.shape).astype(float)
depth_image[disparity_rescale > 0] = (f * baseline) / (disparity_rescale[disparity_rescale > 0])
depth_rescaled = ((depth_image/depth_image.max())*255).astype(np.uint8)

# saving the depth image as a gray scale and color using heat map conversion
color_map = plt.get_cmap('inferno')
depth_heatmap = (color_map(depth_rescaled) * 2**16).astype(np.uint16)[:,:,:3]
depth_heatmap  = cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR)

# Plotting Depth gray and heat map
plt.subplot(211),plt.imshow(depth_rescaled),plt.title(' Depth grayscale')
plt.axis('off')
plt.subplot(212),plt.imshow(depth_heatmap),plt.title(' Depth heat map')
plt.axis('off')
plt.show()

# Displaying Output according to the choosen dataset
if data_selection== '1':

    # epipolarLine 
    cv2.imwrite("epipolarLines_crule_cam0.jpg", Epipolar_img0)
    cv2.imwrite("epipolarLines_crule_cam1.jpg", Epipolar_img1)
    # epipolarLine Rectified
    cv2.imwrite("epipolarLines_crule_rectified_cam0.jpg", rect_epipolar_img0)
    cv2.imwrite("epipolarLines_crule_rectified_cam1.jpg", rect_epipolar_img1)
    # Disparity and its Heatmap 
    disparity_rescale = cv2.cvtColor(disparity_rescale, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Disparity_gray_crule.jpg", disparity_rescale)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Disparity_heat_map_crule.jpg", heat_map)
    # Depth and its Heatmap
    depth_rescaled = cv2.cvtColor(depth_rescaled, cv2.COLOR_BGR2RGB)
    depth_heatmap = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Depth_gray_crule.jpg", depth_rescaled)
    cv2.imwrite("Depth_heat_map_crule.jpg", depth_heatmap)

elif data_selection== '2':

    # epipolarLine 
    cv2.imwrite("epipolarLines_octagon_cam0.jpg", Epipolar_img0)
    cv2.imwrite("epipolarLines_octagon_cam1.jpg", Epipolar_img1)
    # epipolarLine Rectified
    cv2.imwrite("epipolarLines_octagon_rectified_cam0.jpg", rect_epipolar_img0)
    cv2.imwrite("epipolarLines_octagon_rectified_cam1.jpg", rect_epipolar_img1)
    # Disparity and its Heatmap 
    disparity_rescale = cv2.cvtColor(disparity_rescale, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Disparity_gray_octagon.jpg", disparity_rescale)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Disparity_heat_map_octagon.jpg", heat_map)
    # Depth and its Heatmap
    depth_rescaled = cv2.cvtColor(depth_rescaled, cv2.COLOR_BGR2RGB)
    depth_heatmap = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Depth_gray_octagon.jpg", depth_rescaled)
    cv2.imwrite("Depth_heat_map_octagon.jpg", depth_heatmap) 

elif data_selection== '3':
 
    # epipolarLine 
    cv2.imwrite("epipolarLines_pendulum_cam0.jpg", Epipolar_img0)
    cv2.imwrite("epipolarLines_pendulum_cam1.jpg", Epipolar_img1)
    # epipolarLine Rectified
    cv2.imwrite("epipolarLines_pendulum_rectified_cam0.jpg", rect_epipolar_img0)
    cv2.imwrite("epipolarLines_pendulum_rectified_cam1.jpg", rect_epipolar_img1)
    # Disparity and its Heatmap 
    disparity_rescale = cv2.cvtColor(disparity_rescale, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Disparity_gray_pendulum.jpg", disparity_rescale)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Disparity_heat_map_pendulum.jpg", heat_map)
    # Depth and its Heatmap
    depth_rescaled = cv2.cvtColor(depth_rescaled, cv2.COLOR_BGR2RGB)
    depth_heatmap = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Depth_gray_pendulum.jpg", depth_rescaled)
    cv2.imwrite("Depth_heat_map_pendulum.jpg", depth_heatmap) 