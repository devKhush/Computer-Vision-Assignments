# %% [markdown]
# # **Khushdev Pandit**
# # **Roll no: 2020211**
# # *Assignment Question-1*

# %% [markdown]
# #

# %% [markdown]
# # **Q1 Part-1**
# ##### (5 points) Report the estimated intrinsic camera parameters, i.e., focal length(s), skew parameter and principal point along with error estimates if available.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.facecolor'] = 'white'

# load all the images
images = glob.glob('Chess_Images/*.jpg')
print("Number of Chessboard Images clicked : ", len(images))

# %%
# define the size of the checkerboard
checkerboard_size = (5, 5)

# define the world coordinates of the checkerboard
objects_points = []
image_points = []

# termination criteria for the sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# loop over all the images to find the chessboard corners
for image_name in tqdm(images):
    img = cv2.imread(image_name)
    inverted_img = np.array(255 - img, dtype=np.uint8)
    gray_img = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)

    # 3D points in real world space
    objp = np.zeros(
        (1, checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard_size[0],
                              0:checkerboard_size[1]].T.reshape(-1, 2)

    # Find the chessboard corners
    cornersFound, corners = cv2.findChessboardCorners(
        image=gray_img, patternSize=checkerboard_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

    if cornersFound == True:
        # Refine the corners
        corners_refined = cv2.cornerSubPix(
            gray_img, corners, (11, 11), (-1, -1), criteria)
        objects_points.append(objp)
        image_points.append(corners_refined)

        # Draw the corners on the image
        corners_img = cv2.drawChessboardCorners(
            img, checkerboard_size, corners_refined, cornersFound)
        for corner in corners_refined.squeeze():
            coord = (int(corner[0]), int(corner[1]))
            cv2.circle(img=corners_img, center=coord, radius=33,
                       color=(255, 0, 0), thickness=15)
        # plt.imsave("Chess_Corners/" + str(image_name.split('\\')[-1].split('.')[0] + '.png'), corners_img)
    else:
        print("Corners not found for image: ", image_name)

# %%
# Calibrate the camera for all the given images
img = cv2.imread(images[0])
retVal, cameraInternalMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objects_points, image_points, img.shape[:2], None, None)

print("Error estimate: \n", retVal, "\n")
print("Internal Camera matrix : \n", cameraInternalMatrix, "\n")
print("Radial Distortion \n", distCoeffs[0], "\n")
print("Focal length :")
print("fx = ", cameraInternalMatrix[0, 0])
print("fy = ", cameraInternalMatrix[1, 1])
print("\nPrincipal point :")
print("cx = ", cameraInternalMatrix[0, 2])
print("cy = ", cameraInternalMatrix[1, 2])
print("\nSkew paramerer:")
print("s = ", cameraInternalMatrix[0, 1])

# %%
np.savez('calibration.npz', mtx=cameraInternalMatrix,
         dist=distCoeffs, rvecs=rvecs, tvecs=tvecs)

# %%
data = np.load('calibration.npz')
cameraInternalMatrix = data['mtx']
distCoeffs = data['dist']
rvecs = data['rvecs']
tvecs = data['tvecs']

# %% [markdown]
# # **Q1 Part-2**
# ##### (5 points) Report the estimated extrinsic camera parameters, i.e., rotation matrix and translation vector for each of the selected images.

# %%
print("Length of Translation Vector: ", len(tvecs))
print("Length of Rotation Vector: ", len(rvecs))
print("Translation Vector Shape: ", np.array(tvecs).shape)
print("Rotation Vector Shape: ", np.array(rvecs).shape)

# %%
for i in range(len(tvecs)):
    tranlation_vector = tvecs[i].squeeze()
    Rotation_matrix, Jacobian_matrix = cv2.Rodrigues(rvecs[i])
    print("Translation Vector for Image-" + str(i+1) + ": ", tranlation_vector)
    print("Rotation Matrix for Image-" + str(i+1) + ": \n", Rotation_matrix)
    print()

# %% [markdown]
# # **Q1 Part-3**
# ##### (5 points) Report the estimated radial distortion coefficients. Use the radial distortion coefficients to undistort 5 of the raw images and include them in your report. Observe how straight lines at the corner of the images change upon application of the distortion coefficients. Comment briefly on this observation.

# %%
print("Lens distortion coefficients :")
print("k1 = ", distCoeffs[0, 0])
print("k2 = ", distCoeffs[0, 1])
print("p1 = ", distCoeffs[0, 2])
print("p2 = ", distCoeffs[0, 3])
print("k3 = ", distCoeffs[0, 4])

# %%
images_5 = ['Chess_Images/IMG_20230402_131656.jpg', 'Chess_Images/IMG_20230402_131411.jpg',
            'Chess_Images/IMG_20230402_131614.jpg', 'Chess_Images/IMG_20230402_131648.jpg',
            'Chess_Images/IMG_20230402_131426.jpg']

for image in images_5:
    img = cv2.imread(image)
    # perform the undistortion operation on the image
    undistorted = cv2.undistort(img, cameraInternalMatrix, distCoeffs)

    # Save the original and undistorted images
    # plt.imsave("Undistorted_Image/Raw_image__" + str(image.split('\\')[-1].split('.')[0] + '.png'), img)
    # plt.imsave("Undistorted_Image/Undistorted_image__" + str(image.split('\\')[-1].split('.')[0] + '.png'), undistorted)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img)
    ax[0].set_title("Raw Image")
    ax[0].set_aspect('equal')
    ax[0].set_facecolor('white')
    ax[0].grid(False)
    ax[0].set_frame_on(False)
    ax[1].imshow(undistorted)
    ax[1].set_title("Undistorted Image")
    ax[1].set_aspect('equal')
    ax[1].set_facecolor('white')
    ax[1].grid(False)
    ax[1].set_frame_on(False)
    plt.show()

# %% [markdown]
# #### Observation for the straight lines at the corner of the images change upon application of the distortion coefficients:
#
# Straight lines at the corners of the image become somewhat more straight and less distorted on applying the radial distortion coefficients. Radial distortion is caused by the the curvature of the lens. When we perform undistortion of an image (such as for pincushion distortion), we distortion is corrected and a more accurate image scene is obtained.

# %% [markdown]
# # **Q1 Part-4**
# ##### (5 points) Compute and report the re-projection error using the intrinsic and ex-trinsic camera parameters for each of the 25 selected images. Plot the error using a bar chart. Also report the mean and standard deviation of the re-projection error.

# %%
print("Images Processed:")
errors = []
i = 0
for image_name in tqdm(images):
    # project 3D points to image plane
    reproj_image_points, _ = cv2.projectPoints(
        objects_points[i], rvecs[i], tvecs[i], cameraInternalMatrix, distCoeffs)

    # calculate error
    error = cv2.norm(image_points[i], reproj_image_points,
                     cv2.NORM_L2)/len(reproj_image_points)
    errors.append(error)
    i += 1

# %%
plt.bar(np.arange(1, len(errors) + 1), errors, 0.8, color='r')
plt.title('Re-projection errors for each image')
plt.xlabel('Image number ')
plt.ylabel('Re-projection error for each image')
plt.xlim([0, len(images)+1])
plt.show()

# %%
# calculate mean and standard deviation of re-projection error
mean_error = np.mean(errors)
std_dev_error = np.std(errors)

print('Mean Re-projection error for all Images:', mean_error)
print('Standard deviation of Re-projection error for all Images:', std_dev_error)

# %% [markdown]
# # **Q1 Part-5**
# ##### 5. (10 points) Plot figures showing corners detected in the image along with the corners after the re-projection onto the image for all the 25 images. Comment on how is the reprojection error computed.

# %%
i = 0
# Find the Re-projection corners coordinates for all the images
for image_name in images:
    img = cv2.imread(image_name)
    reproj_image_points, _ = cv2.projectPoints(objects_points[i], rvec=rvecs[i], tvec=tvecs[i],
                                               cameraMatrix=cameraInternalMatrix, distCoeffs=distCoeffs)

    # Draw the corners on the image
    corners_img = cv2.drawChessboardCorners(
        img.copy(), checkerboard_size, image_points[i], True)
    for corner in image_points[i].squeeze():
        coord = (int(corner[0]), int(corner[1]))
        cv2.circle(img=corners_img, center=coord, radius=38,
                   color=(255, 0, 0), thickness=20)

    # Draw the re-projection corners on the image
    reprojected_img = cv2.drawChessboardCorners(
        img.copy(), checkerboard_size, reproj_image_points, True)
    for corner in reproj_image_points.squeeze():
        coord = (int(corner[0]), int(corner[1]))
        cv2.circle(img=reprojected_img, center=coord,
                   radius=38, color=(0, 255, 0), thickness=20)
    i += 1

    # plt.imsave("Chess_Corners_With_ReProjection/Without_Reproj_" + str(image_name.split('\\')[-1].split('.')[0] + '.png'), corners_img)
    # plt.imsave("Chess_Corners_With_ReProjection/With_Reproj_" + str(image_name.split('\\')[-1].split('.')[0] + '.png'), reprojected_img)
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(img)
    ax[0].set_title("Raw Image")
    ax[1].imshow(corners_img)
    ax[1].set_title("Image with Corners (Detected by OpenCV)")
    ax[2].imshow(reprojected_img)
    ax[2].set_title("Image with Corners (Reprojected by OpenCV)")
    plt.show()

# %% [markdown]
# #### Comment on how is the reprojection error computed.
#
# The Reprojection error between the detected corners and the Re-projected corners is computed by the average of "L2 Norm" error or average "Euclidean distance" error for every image.

# %% [markdown]
# # **Q1 Part-6**
# ##### (10 points) Compute the checkerboard plane normals nCi, i ∈ {1, ..25} for each of the 25 selected images in the camera coordinate frame of reference (Oc)

# %%
# Find the normal vector to the image plane
normal_to_image_planes = []
i = 0

for image_name in images:
    img = cv2.imread(image_name)

    # Find the normal vector to the image plane
    _, rvec, tvec = cv2.solvePnP(
        objects_points[i], image_points[i], cameraInternalMatrix, distCoeffs)

    # Normal vector will be the last column of the rotation matrix
    Rotation_Matrix, _ = cv2.Rodrigues(rvec)
    normal_to_plane = Rotation_Matrix.dot(np.array([0, 0, 1]))
    normal_to_image_planes.append(normal_to_plane)
    i += 1

# %%
# Plotting the normal vectors on the image plane
for i in range(len(images)):
    img = cv2.imread(images[i])
    mean_u = np.mean(image_points[i][:, :, 0])
    mean_v = np.mean(image_points[i][:, :, 1])

    # calculate z coordinate of the image plane
    z = -(normal_to_image_planes[i][0] * mean_u +
          normal_to_image_planes[i][1] * mean_v) / normal_to_image_planes[i][2]

    # calculate end point of line of the normal vector on the image plane
    end_point = 200*np.array([mean_u, mean_v, z])

    # Drawing the normal vector on the image
    image_with_normal = cv2.line(img, (int(mean_u), int(mean_v)), (int(
        end_point[0]), int(end_point[1])), (255, 0, 0), thickness=25, lineType=8, shift=0)

    if i == 0 or i == 7 or i == 8 or i == 11 or i == 14 or i == 26:
        plt.figure(figsize=(8, 8))
        plt.imshow(image_with_normal)
        plt.title(f"Image-{i} with Normal Vector")
        plt.show()
        # name = images[i].split('\\')[-1].split('.')[0]
        # plt.imsave(f"Image_with_Normal_Vectors/Image-{name}_with_Normal_Vector.png", image_with_normal)

# %%
for i in range(len(normal_to_image_planes)):
    print("Image Number: ", i+1)
    print("Normal Vector to the Image Plane: ",
          normal_to_image_planes[i], "\n")
