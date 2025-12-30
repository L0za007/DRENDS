
import numpy as np
import cv2, tqdm
from scipy.spatial.transform import Rotation
from scipy.linalg import rq  # RQ decomposition
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt

########################################################################
### Helpers for stereo calibration
########################################################################
  
def calibrate(L_centres, L_mask, R_centres, R_mask, pattern, pattern_size, img_size, verbose=False):
    """ Function to calibrate a stereo camera.
        Steps:
        1. Create the object points for the pattern.
        2. Extract the valid image points for the left and right images.
        3. Calibrate the left and right cameras separatelly.
        4. Calibrate the stereo camera.
    Args:
        - L_centres, R_centres: List of 2D points detected in the left and right images.
        - mask: List of booleans to select the valid points.
        - pattern: Pattern used for calibration.
        - pattern_size: Size of the pattern.
        - img_size: Size of the images.
        - verbose: Print the calibration results.
    Returns:
        - calibration_data: Dictionary with the calibration data
    """
    # 1. Params for calibration
    obj_pattern = create_pattern(pattern, pattern_size)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 500, 1e-8)
    flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS   # Use provided initial intrinsics M
    # + cv2.CALIB_FIX_PRINCIPAL_POINT   # Keep cx, cy at image center
    # + cv2.CALIB_ZERO_TANGENT_DIST     # Disable p1, p2
    # + cv2.CALIB_FIX_K3                # Disable k3
    # + cv2.CALIB_FIX_K4                # Disable k4 (if enabled by default)
    # + cv2.CALIB_FIX_K5                # Disable k5
    # + cv2.CALIB_FIX_K6                # Disable k6 (for 8-coeff model)
    )

    # 2. Calibrate individual cameras
    print("Mono calibration")
    ## Left camera
    objpoints = [obj_pattern.copy() for _ in range(sum(L_mask))]
    Limg_pts = np.array(L_centres[L_mask].tolist(), dtype=np.float32)
    deformation = np.zeros(5, dtype=np.float32)
    M = np.array([[img_size[0], 0, img_size[0]/2], [0, img_size[0], img_size[1]/2], [0, 0, 1]], dtype=np.float32)
    retL, camera_matrix_L, dist_coeffs_L, _, _ = cv2.calibrateCamera(objpoints, Limg_pts, img_size, M, deformation, criteria=criteria,flags=flags)
    ## Right camera
    objpoints = [obj_pattern.copy() for _ in range(sum(R_mask))]
    Rimg_pts = np.array(R_centres[R_mask].tolist(), dtype=np.float32)
    deformation = np.zeros(5, dtype=np.float32)
    M = np.array([[img_size[0], 0, img_size[0]/2], [0, img_size[0], img_size[1]/2], [0, 0, 1]], dtype=np.float32)
    retR, camera_matrix_R, dist_coeffs_R, _, _ = cv2.calibrateCamera(objpoints, Rimg_pts,img_size, M, deformation, criteria=criteria,flags=flags)

    print("\tLeft camera calibration error: ", retL)
    if verbose: print(camera_matrix_L,'\n', dist_coeffs_L.ravel())
    print("\tRight camera calibration error: ", retR)
    if verbose: print(camera_matrix_R,'\n', dist_coeffs_R.ravel())

    # 3. Stereo calibration
    print("Stereo calibration")
    mask = L_mask & R_mask
    objpoints = [obj_pattern.copy() for _ in range(sum(mask))]
    Limg_pts = np.array(L_centres[mask].tolist(), dtype=np.float32)
    Rimg_pts = np.array(R_centres[mask].tolist(), dtype=np.float32)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS #+ cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_INTRINSIC
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 500, 1e-9)
    ret_stereo, left_cameraMatrix, left_distCoeffs, right_cameraMatrix, right_distCoeffs, R, T, E, F = cv2.stereoCalibrate(objpoints, Limg_pts, Rimg_pts,
                                                    camera_matrix_L, dist_coeffs_L,
                                                    camera_matrix_R, dist_coeffs_R, img_size, 
                                                    flags=flags, 
                                                    criteria=criteria
                                                    )
    print("\tStereo calibration error ", ret_stereo)
    
    calibration_data = {
        "left_K": left_cameraMatrix,
        "left_D": left_distCoeffs.ravel(),
        "right_K": right_cameraMatrix,
        "right_D": right_distCoeffs.ravel(),
        "stereo_T": T.ravel(),
        "stereo_R": Rotation.from_matrix(R).as_euler('xyz', degrees=True),
        "stereo_M": R,
    }
    if verbose:
        for key, value in calibration_data.items():
            print(f"{key}:\n {value}")
    return calibration_data

def create_pattern(pattern, distance_dots):
    """ Function to create the pattern for the calibration
    
    Args:
        - pattern: size of the pattern
        - distance_dots: distance between dots in mm
    Returns:
        - objp: object points
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= distance_dots /1000 # Convert to meters
    return objp

def rectify_objects(imgs:np.ndarray, pts:np.ndarray, calib:Dict[str,np.ndarray], out_dir=None, verbose=False):
    """ Function to rectify images and 2D points."""
    LK, LD, LR, LP = calib["left_K"], calib["left_D"], calib["left_R"], calib["left_P"]
    RK, RD, RR, RP = calib["right_K"], calib["right_D"], calib["right_R"], calib["right_P"]
    # Rectify images if needed
    if imgs is not None:
        Limgs, Rimgs = imgs
        path_format = isinstance(Limgs[0], str) or isinstance(Limgs[0], Path)
        img_size = cv2.imread(str(Limgs[0])).shape[:2][::-1] if path_format else Limgs[0].shape[:2][::-1]
        map1x, map1y = cv2.initUndistortRectifyMap(LK, LD, LR, LP, img_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(RK, RD, RR, RP, img_size, cv2.CV_32FC1)
        rect_Limgs, rect_Rimgs = [], []
        pbar = tqdm.tqdm(range(len(Limgs)), desc="Rectifying images")
        for idx in range(len(Limgs)):
            if path_format:
                img1 = cv2.imread(str(Limgs[idx]))
                img2 = cv2.imread(str(Rimgs[idx]))
            else:
                img1,img2 = Limgs[idx].astype(np.float32), Rimgs[idx].astype(np.float32)
            img1_rectified = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
            img2_rectified = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
            rect_Limgs.append(img1_rectified.astype(np.uint8)), rect_Rimgs.append(img2_rectified.astype(np.uint8))
            # Save images if needed
            if out_dir is not None:
                out_dir.mkdir(exist_ok=True)
                row = np.concatenate((img1_rectified, img2_rectified), axis=1)
                for i in range(0, row.shape[0], 50):
                    cv2.line(row, (0, i), (row.shape[1], i), (0, 255, 0), 1)
                if isinstance(Limgs[idx], Path):
                    file_name = out_dir / Limgs[idx].name
                else:
                    file_name = out_dir / f"{idx:05d}.png"
                cv2.imwrite(str(file_name), row.astype(np.uint8))
            pbar.update(1)
        pbar.close()
        # Return images
        rect_Limgs, rect_Rimgs = np.array(rect_Limgs, dtype=object), np.array(rect_Rimgs, dtype=object)
        if pts is None:
            return {'imgs': (rect_Limgs, rect_Rimgs), 'pts': None}
    # Rectify 2D points
    if pts is not None:
        Lpts, Rpts = pts
        Lpts_rect, Rpts_rect = [], []
        for idx in range(len(Lpts)):
            if Lpts[idx] is not None:
                Lpts_rect.append(cv2.undistortPoints(Lpts[idx].astype(np.float32), LK, LD, R=LR, P=LP))
            else:
                Lpts_rect.append(None)
            if Rpts[idx] is not None:
                Rpts_rect.append(cv2.undistortPoints(Rpts[idx].astype(np.float32), RK, RD, R=RR, P=RP))
            else:
                Rpts_rect.append(None)
        # Return points
        Lpts_rect, Rpts_rect = np.array(Lpts_rect, dtype=object), np.array(Rpts_rect, dtype=object)
        if imgs is None:
            return {'imgs': None, 'pts': (Lpts_rect, Rpts_rect)}
    return {'imgs': (rect_Limgs, rect_Rimgs), 'pts': (Lpts_rect, Rpts_rect)}

def stereo_triangulation(pts, calib):
    """ Function to triangulate the 3D points from the 2D points detected in the left and right images.
    """
    centersL, centersR = pts
    LP = calib["left_P"].astype(np.float32)
    RP = calib["right_P"].astype(np.float32)
    points_3d = []
    flag = False
    for cL, cR in zip(centersL, centersR):
        if cL is not None and cR is not None:
            pointL = np.squeeze(cL).astype(np.float32) # shape (12, 2)
            pointR = np.squeeze(cR).astype(np.float32)
            points_4d = cv2.triangulatePoints(LP, RP, pointL.T, pointR.T)
            points_3d.append((points_4d[:3] / points_4d[3]).T)
        else:
            points_3d.append(None)
            flag = True
    if flag:
        return np.array(points_3d, dtype=object)
    return np.array(points_3d)

def plot_triangulation(pts_3d, images, paths, pattern_shape, out_folder, valid=None):
    """ Function to plot the triangulation of the 3D points
    Args:
        - pts_3d: List of 3D points.
        - images: List of left and right images.
        - paths: List of paths to the images.
        - pattern_size: Size of the pattern.
        - out_folder: Path to the output folder.
        - valid: List of booleans to select the valid points.
    """
    # Unpack t
    imagesL, imagesR = images
    pathsL, _ = paths
    if valid is None:
        valid = [True] * len(pts_3d)

    # Create folder
    if not out_folder.exists(): out_folder.mkdir()

    pbar = tqdm.tqdm(range(len(pts_3d)), desc="Plotting 3D points")
    colors = ['r', 'orange', 'y', 'g', 'c', 'b', 'm']
    c = []
    for i in range(pattern_shape[1]):
        c+= [colors[i%len(colors)]]*pattern_shape[0]
    for idx, points_3d in enumerate(pts_3d): #shape (N, pts, 1, 2)
        if points_3d is None or not valid[idx]:
            pbar.update(1)
            continue
        # Plot results 
        fig = plt.figure(figsize=(18, 16))
        ##  3D plot
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c=c, marker='o')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend(['3D points'])
        ax.set_title('3D View')
        ## X-Y view
        ax = fig.add_subplot(222)
        ax.scatter(points_3d[:,0], -points_3d[:,1], c=c, marker='o')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('X-Y View')
        ## X-Z view
        ax = fig.add_subplot(223)
        ax.scatter(points_3d[:,0], points_3d[:,2], c=c, marker='o')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title('X-Z View')
        ax.set_ylim(0.01, 0.750)
        ax = fig.add_subplot(224)
        if imagesR is not None:
            row = np.concatenate((imagesL[idx], imagesR[idx]), axis=0).astype(np.uint8)
        else:
            row = imagesL[idx].astype(np.uint8)
        ax.imshow(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title('Rectified Image')
        file_name = out_folder / f"{pathsL[idx].stem}.png"
        plt.savefig(file_name)
        plt.close()
        pbar.update(1)
    pbar.close()

def reproject3D(pts3D, images, paths, calib, out_dir, verbose=False):
    
    LP =  calib["left_P"]
    RP =  calib["right_P"]
    LK,LR,LT = decompose_projection_matrix(LP)
    RK,RR,RT = decompose_projection_matrix(RP)
    SR, ST = calib["stereo_R"], calib["stereo_T"]
    if images is not None: 
        if not out_dir.exists(): out_dir.mkdir()
        Limgs, Rimgs = images
    Lreprj, Rreprj = [], []
    if verbose:  pbar = tqdm.tqdm(range(len(pts3D)), desc="Reprojecting points on rectified images")
    for idx in range(len(pts3D)):
        if images is not None:
            left_img = Limgs[idx].astype(np.uint8)
            right_img =  Rimgs[idx].astype(np.uint8)
        if pts3D[idx] is not None:
            points3D_left = np.dot(LR, pts3D[idx].T).T
            rvec_left = np.zeros((3, 1))  # No rotation: points3D_left are in the left camera's coordinate system.
            tvec_left = np.zeros((3, 1))
            proj_points_left, _ = cv2.projectPoints(points3D_left, rvec_left, tvec_left, LK, np.zeros(5))
            proj_points_left = proj_points_left.reshape(-1, 2)
            Lreprj.append(proj_points_left)

            points3D_right = np.dot(RR, pts3D[idx].T).T + RT.reshape(1, 3)
            rvec_right = np.zeros((3, 1))
            tvec_right = np.zeros((3, 1))
            proj_points_right, _ = cv2.projectPoints(points3D_right, rvec_right, tvec_right, RK, np.zeros(5))
            proj_points_right = proj_points_right.reshape(-1, 2)
            Rreprj.append(proj_points_right)

            if images is not None:
                for pt in proj_points_left:
                    cv2.circle(left_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                for pt in proj_points_right:
                    cv2.circle(right_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        if out_dir is not None and images is not None:
            row = np.concatenate((left_img, right_img), axis=1)
            file_name = out_dir / paths[idx].name
            cv2.imwrite(str(file_name), row)
        if verbose: pbar.update(1)
    if verbose: pbar.close()
    
    return Lreprj, Rreprj

def decompose_projection_matrix(P):
    # Extract K' (first 3x3 part) and T (last column)
    K_prime = P[:, :3]  # First three columns
    T = P[:, 3]  # Last column (Translation vector)
    # RQ decomposition to separate K and R
    K, R_rect = rq(K_prime)
    # Normalize K to ensure K[2,2] == 1
    K /= K[2, 2]
    T = T/K[0,0]
    return K, R_rect, T
    
