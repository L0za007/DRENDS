import numpy as np
import cv2, tqdm
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from scipy.linalg import rq  # RQ decomposition
from scipy.spatial.transform import Rotation as R

########################################################################
### Helpers for stereo-helios calibration
########################################################################

def get_ptCloud_from_paths(ptC_paths, img_size=(480,640), supress_outliers=False, verbose=False):
    pt_clouds = []
    if verbose: pbar = tqdm.tqdm(range(len(ptC_paths)), desc="- Processing point clouds")
    for i,path in enumerate(ptC_paths):
        pt_cloud = np.load(path)
        pt_cloud = pt_cloud.reshape((*img_size,3))
        if supress_outliers: 
            pt_cloud = supress_outliers_in_ptCloud(pt_cloud)
        pt_clouds.append(pt_cloud)
        if verbose: pbar.update(1)
    if verbose: pbar.close()
    pt_clouds = np.array(pt_clouds, dtype=np.float32)
    return pt_clouds

def supress_outliers_in_ptCloud(point_cloud):
    """ Detect missing values in the pointcloud (NaN values and distorded reflections, 
    i.e. points far away from camera) (H, W, 3) using KDTree nearest neighbors.
    
    Parameters:
    - point_cloud (np.array): Point cloud (H, W, 3), 
    Returns:
    - Corrected point cloud matrix.
    """
    H, W, _ = point_cloud.shape

    # Create a grid of indices for point positions
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    grid_points = np.column_stack((x.ravel(), y.ravel()))
    # Find valid points
    valid_mask = ~np.isnan(point_cloud) & ~(point_cloud>1)
    valid_mask = valid_mask[..., 0] & valid_mask[..., 1] & valid_mask[..., 2]
    valid_mask = valid_mask.ravel()
    missing_mask = ~valid_mask
    # Process each channel independently
    for c in range(3):
        values = point_cloud[..., c].ravel()
        # Find valid points
        valid_points = grid_points[valid_mask]
        valid_values = values[valid_mask]

        # Build KDTree and query missing points
        kdtree = KDTree(valid_points)
        missing_points = grid_points[missing_mask]
        
        if missing_points.size > 0:
            _, indices = kdtree.query(missing_points)
            filled_values = valid_values[indices]
            # Update the point cloud
            values[missing_mask] = filled_values

        point_cloud[..., c] = values.reshape(H, W)

    return point_cloud

def get_pattern_calibration_in_ptCloud(ptC_paths, img_centres2D, mask, img_size, approx_excess, verbose=False):
    """ Load point clouds and extract the 3D points corresponding to the calibration pattern centres.
    Args:
        - ptC_paths: List of paths to the point clouds
        - img_centres2D: List of 2D image coordinates of the calibration pattern centres
        - mask: List of boolean values indicating which images to process
        - img_size: Tuple (H, W) of the image size
        - approx_excess: Approximate excess
    Returns:
        - ptc_centres: List of 3D points corresponding to the calibration pattern centres
        - pt_clouds: List of point clouds
    """
    ptc_centres, pt_clouds = [], []
    missing_values=False
    if verbose: pbar = tqdm.tqdm(range(len(ptC_paths)), desc="- Processing point clouds")
    for i,path in enumerate(ptC_paths):
        if mask[i]:
            if isinstance(path, str) or isinstance(path, Path):
                pt_cloud = np.load(path)
                pt_cloud = pt_cloud.reshape((*img_size,3))
                # pt_cloud = supress_outliers_in_ptCloud(pt_cloud)
            elif isinstance(path, np.ndarray):
                pt_cloud = path
            if approx_excess:
                img_centres = np.rint(img_centres2D[i][:,0]).astype(int) ## INT conversion to the nearest pixel ##
                current_centre = np.array([pt_cloud[pt[1], pt[0]] for pt in img_centres])
            else:   
                current_centre = sample_from_grid(pt_cloud, img_centres2D[i])
        else:
            current_centre, pt_cloud = None, None
            missing_values = True
        pt_clouds.append(pt_cloud)
        ptc_centres.append(current_centre)
        if verbose: pbar.update(1)
    if verbose: pbar.close()
    pt_clouds = np.array(pt_clouds, dtype=object if missing_values else np.float32)
    ptc_centres = np.array(ptc_centres, dtype=object if missing_values else np.float32)
    return ptc_centres, pt_clouds

def sample_from_grid(grid, pts):
    """ Sample the grid at the given points.
    Args:
        - grid: 3D grid
        - pts: List of 2D points
    Returns:
        - samples: List of samples
    """
    grid_x = np.arange(grid.shape[1])
    grid_y = np.arange(grid.shape[0])
    interpolator = RegularGridInterpolator((grid_y, grid_x), grid, method='linear', bounds_error=False, fill_value=None)
    samples = []
    for pt in pts:
        samples.append(interpolator(pt[...,::-1]))
    samples =np.concatenate(samples, axis=0)
    return samples

def HelRGB_calibrate(PC_pts, RGB_3Dpts, RGB_pts, calib, valid, verbose=False):
    """ Helios-RGB calibration:
    Find the rotation (R) and translation (t) between a depth camera and an RGB camera
    using nonlinear least squares. The error in this function is calculated on the 
    image plane of the RGB image.

    Parameters:
    - PC_pts: (N, 3) ndarray of 3D points in the depth camera space
    - RGB_pts: (N, 2) ndarray of corresponding 2D points in the RGB camera image space
    - K_rgb: (3, 3) intrinsic matrix of the RGB camera

    Returns:
    - R_opt: Optimal rotation matrix (3x3)
    - t_opt: Optimal translation vector (3,)
    """
    PC_pts = [PC_pts[i] for i,v in enumerate(valid) if v]
    RGB_pts = [RGB_pts[i] for i,v in enumerate(valid) if v]
    RGB_3Dpts = [RGB_3Dpts[i] for i,v in enumerate(valid) if v]
    PC_pts  = np.concatenate(PC_pts , axis=0).astype(np.float32)
    RGB_pts = np.squeeze(np.concatenate(RGB_pts, axis=0)).astype(np.float32)
    RGB_3Dpts = np.concatenate(RGB_3Dpts, axis=0).astype(np.float32)

    # Initialize parameters: rotation (Rodrigues vector) and translation
    rvec_init = np.array([0.0, 0.0, 0.0])  # No initial rotation
    t_init = np.array([0.0, 0.0, 0.0])     # No initial translation

    params_init = np.hstack((rvec_init, t_init))
    K_rgb,_,_ = decompose_projection_matrix(calib["left_P"])
    dist_rgb = np.zeros(5)

    # Define the residual function
    def reprojection_error(params):
        # Extract rvec and t from params
        rvec = params[:3]
        t = params[3:]
        reproj_c, reproj_c3D = cv2.projectPoints(PC_pts, rvec, t, K_rgb, dist_rgb)
        reproj_c = np.squeeze(reproj_c)
        # print("reproj_c: ", reproj_c.shape, "RGB_pts: ", RGB_pts.shape)
        residuals = np.hstack([(RGB_pts[:, 0] - reproj_c[:, 0]), (RGB_pts[:, 1] - reproj_c[:, 1])])
        return residuals

    # Solve using least squares
    result = least_squares(reprojection_error, params_init)
    print(f"Helios->RGB Calibration result, success: {result.success}")

    # Extract optimized rvec and t
    rvec_opt = result.x[:3]
    t_opt = result.x[3:]

    pts2D = np.squeeze(cv2.projectPoints(PC_pts, rvec_opt, t_opt, K_rgb, dist_rgb)[0])
    error2D = np.linalg.norm(pts2D - RGB_pts, axis=1)
    print(f"\t2D reprojection error: mean={np.mean(error2D):.4f} px, std={np.std(error2D):.4f} px, max={np.max(error2D):.4f} px")
    pts3D = (cv2.Rodrigues(rvec_opt)[0] @ PC_pts.T).T + t_opt
    error3D = np.linalg.norm(pts3D - RGB_3Dpts, axis=1)
    print(f"\t3D alignment error: mean={np.mean(error3D)*1000:.2f} mm, std={np.std(error3D)*1000:.2f} mm, max={np.max(error3D)*1000:.2f} mm")

    rot_matrix, _ = cv2.Rodrigues(rvec_opt)
    calib['he_lap_T'] = t_opt
    calib['he_lap_rvec'] = rvec_opt
    calib['he_lap_M'] = rot_matrix
    calib['he_lap_R'] = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
    if verbose:
        print(f"Helios->RGB transform: rvec={calib['he_lap_R']}, tvec={t_opt}")

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

