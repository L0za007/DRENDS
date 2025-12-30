import cv2, numpy as np
import json
from pathlib import Path
from matplotlib import pyplot as plt
import tqdm
from PIL import Image, ImageDraw

### Data reader
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def check_folder(data_folder, subfolders, take_n_samples=None, verbose=True):
    """ Function to check all the folders have the same img ids
    Args:
        - data_folder: folder with the images
        - subfolders: list of subfolders
        - verbose: print the number of images in each subfolder
    Returns:
        - img_paths: dictionary with the image paths for each subfolder
    """
    # Read the imag paths for each subfolder
    sub_f = [data_folder / subfolder for subfolder in subfolders]
    img_paths = {sub.name:list(sub.glob('*')) for sub in sub_f}
    # Find common ids
    img_steam = {sub.name:[img.stem for img in img_paths[sub.name]] for sub in sub_f}
    common_ids = set(img_steam[sub_f[0].name])
    for sub in sub_f[1:]:
        common_ids = common_ids & set(img_steam[sub.name])
    # Filter the images
    img_paths = {sub.name:sorted([img for img in img_paths[sub.name] if img.stem in common_ids]) for sub in sub_f}
    n_samples = len(img_paths[sub_f[0].name])
    if verbose: print(f"Found {n_samples} common images in folders: {', '.join([sub.name for sub in sub_f])}")
    # Take n samples
    if take_n_samples is not None:
        if verbose: print(f"Taking {take_n_samples} samples")
        sample_indices = np.linspace(0, n_samples - 1, min(n_samples, take_n_samples), dtype=int)
        img_paths = {sub.name: [img_paths[sub.name][i] for i in sample_indices] for sub in sub_f}
    # check that there is only png, jpg, bmp
    img_paths = {sub.name: [img for img in img_paths[sub.name] if img.suffix in ['.png', '.jpg', '.bmp']] for sub in sub_f}
    return img_paths

### Pattern detection
def detect_square_pattern(images, pattern_size, tag='', return_imgpoints=False, verbose=False):
    """ Function to process the images and find the squares in images. The order
    of the detected points is reversed if the first point is not the top left corner.

    Args:
        images (list[Path,str,narray]): List of images to process.
        pattern_size (tuple[int]): Size of the pattern (rows, columns).
        params (cv2.SimpleBlobDetector_Params, optional): Parameters for the blob detector. Defaults to STEREO_PARAM.
        tag (str, optional): Tag to show in the progress bar. Defaults to ''.
        return_imgpoints (bool, optional): Return the images with the detected points. Defaults to False.
    Returns:
    tuple: 
        - centres (list[narray]): List of detected points for each image.
        - valid_imgs (list[boolean]): List of booleans indicating if the pattern was found in each image.
        - imgs (list[narray], optional): List of images with detected points drawn, if return_imgpoints is True.
    """
    n_patterns = np.prod(pattern_size)
    imgs, valid_imgs, centres = [], [], []
    if verbose: pbar = tqdm.tqdm(range(len(images)), desc="- {}. Detecting pattern".format(tag))
    for idx,image in enumerate(images):
        if isinstance(image, Path) or isinstance(image, str):
            image = cv2.imread(str(image))
        found, corners = cv2.findChessboardCorners(image, pattern_size, 
                            flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
        found = found and (corners.shape[0] == n_patterns)
        if found:
            # Refine the corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)
            # Order the points
            corners = order_points([corners], [found], [images[idx]], pattern_shape=pattern_size)[0]
            if return_imgpoints:
                cv2.drawChessboardCorners(image, pattern_size, corners, True)
        elif verbose:
            if isinstance(image, Path):
                print(f"Pattern not found in {images[idx].parent.name} {images[idx].name}")
            else:
                print(f"Pattern not found in img idx {idx}")
        # Append the results
        imgs.append(image), valid_imgs.append(found), centres.append(corners)
        if verbose: pbar.update(1)
    if verbose: pbar.close()
    # Convert to numpy arrays
    valid_imgs = np.array(valid_imgs)
    imgs = np.array(imgs)
    
    centres = np.array(centres, dtype=object)
    if return_imgpoints:
        return centres, valid_imgs, imgs
    return centres, valid_imgs

def order_points(pts, mask, paths, pattern_shape, decreasing=False):
    """ Function to order the points in the pattern
    
    Parameters:
        - pts: points to order
        - mask: mask to filter the points
        - decreasing: order the points in decreasing order
        - tag: tag to show
    Returns:
        - ordered points
    """
    cols, rows = pattern_shape
    total_points = cols * rows
    for i in range(len(mask)):
        if mask[i]:
            assert len(pts[i].shape) == 3, f"idx {i} has shape {pts[i].shape}. Expected [n, 1, 2]"
            _pts = pts[i].reshape((rows, cols, 2))  # Assuming OpenCV returns in row-major order

            # Find the point with smallest y, then x (true top-left in image space)
            flat_pts = _pts.reshape((-1, 2))
            top_left_idx = np.lexsort((flat_pts[:, 0], flat_pts[:, 1]))[0]
            top_left_pos = np.unravel_index(top_left_idx, (rows, cols))

            # Generate all 4 possible orientations
            orientations = [
                _pts,                                 # original
                np.flip(_pts, axis=1),                # flip horizontally
                np.flip(_pts, axis=0),                # flip vertically
                np.flip(np.flip(_pts, axis=0), axis=1)  # flip both
            ]

            # Choose orientation with top-left-most corner in image space
            best = min(orientations, key=lambda p: (p[0,0,0]**2 + p[0,0,1]**2)**0.5)  # compare y, then x
            if best[0,1,0]< best[1,0,0]:
                _pts =[best[:,i,:] for i in range(rows)]
            best = np.stack(_pts, axis=0)
            pts[i] = best.reshape((-1, 1, 2)).astype(np.float32)
    return pts

### Loggers and visualizers
def save_images_with_pattern(images, img_paths, output_folder):
    """ Function to save the images with the detected pattern
    Args:
        - images: list of list images
        - img_paths: list of list image paths
        - output_folder: folder to save the images
    """
    if not output_folder.exists():
        output_folder.mkdir()
    n_imgs = len(images[0])

    pbar = tqdm.tqdm(range(n_imgs), desc="Saving images with patterns")

    for i in range(n_imgs):
        # check file nalmes
        img_ids = [path[i].stem for path in img_paths]
        if not all(current_id == img_ids[0] for current_id in img_ids):
            raise ValueError(f"Image paths at index {i} do not have the same stem: {img_ids}")
        # Get all images to the same size
        imgs = [img[i] for img in images]
        imgs = resize_to_smallest(imgs)
        image_name = str(output_folder / f"{img_ids[0]}.png")
        row = np.concatenate(imgs, axis=1).astype(np.uint8)
        cv2.imwrite(image_name, row)
        pbar.update(1)

    pbar.close()

def resize_to_smallest(images):
    """Resize all images in a list to the smallest image size (height and width)."""
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    resized = [cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA) for img in images]
    return resized

def pattern_3D_dimensions_error(pts3D, paths, valid, pattern_shape, pattern_size, file_name, verbose=False):
    """ Function to quantify the error in the dimensions of 3D pattern points
        compared to a ground truth distance in a batch."""
    # Get the stats per image
    x_means, x_stds, y_means, y_stds = [],[],[],[]
    for i in range(len(valid)):
        if valid[i]:    
            _xmean,_xstd,_ymean,_ystd = dimension_error(pts3D[i],pattern_size/1000, pattern_shape)
            x_means.append(_xmean),x_stds.append(_xstd)
            y_means.append(_ymean),y_stds.append(_ystd)

    if verbose:
        _, ax = plt.subplots(figsize=(15, 7))
        x = np.arange(len(x_means))
        width = 0.35
        x_label = [f"{i}:{paths[i].stem}" for i in range(len(paths)) if valid[i]]
        ax.bar(x - width/2, x_means, width, yerr=x_stds, label='X means', capsize=5)
        ax.bar(x + width/2, y_means, width, yerr=y_stds, label='Y means', capsize=5)
        ax.axhline(y=pattern_size*0.05, color='r', linestyle='--', linewidth=1)
        ax.axhline(y=-pattern_size*0.05, color='r', linestyle='--', linewidth=1)
        ax.set_xticklabels(x_label, rotation=90)
        ax.set_ylabel('Mean Error (mm)')
        ax.set_title(f"Mean Errors for X and Y with Standard Deviation. GT:{pattern_size}mm +/-5%")
        ax.set_xticks(x)
        ax.legend(['+5% error threshold', '-5% error threshold', 'X means', 'Y means'])
        plt.savefig(file_name)
    print(f"Pattern Dimensions Error (GT:{pattern_size}mm):")
    print("\tMean x error: {:.2f} +/- {:.2f} mm".format(np.mean(x_means), np.mean(x_stds)))
    print("\tMean y error: {:.2f} +/- {:.2f} mm".format(np.mean(y_means), np.mean(y_stds)))

    return x_means, x_stds, y_means, y_stds

def dimension_error(pts3D, gt_distance, pattern_shape):
    """ Quantifies the error on the dimensions of a 3D pattern compared to a ground truth distance between points.
    Parameters:
        pts3D (numpy.ndarray): The 3D points of the pattern, expected to be reshaped according to pattern_shape.
        gt_distance (float): The ground truth distance between points in the pattern.
        pattern_shape (tuple): The shape of the pattern (rows, columns).
    """
    pts3D = pts3D.reshape(pattern_shape[1], pattern_shape[0], -1)
    x_dist = pts3D[:, 1:] - pts3D[:, :-1] # x size in m
    y_dist = pts3D[1:, :] - pts3D[:-1, :] # y size in m
    norm_x = np.linalg.norm(x_dist, axis=-1)
    norm_y = np.linalg.norm(y_dist, axis=-1)

    scale = 1000 # to mm
    x_mean = np.mean(norm_x- gt_distance)*scale 
    y_mean = np.mean(norm_y- gt_distance)*scale
    x_std = np.std(norm_x- gt_distance)*scale
    y_std = np.std(norm_y- gt_distance)*scale
    
    return x_mean, x_std, y_mean, y_std

def reprojection_error(vertices_A, vertices_B, tags, paths, valid, verbose=False):
    """ Function to quantify the reprojection error between two sets of 2D points in a batch.
        The error is calculated as the Euclidean distance between corresponding points."""
    assert len(vertices_A) == len(vertices_B) == len(tags) == 2, "All inputs must have length 2."
    if verbose:
        _, ax = plt.subplots(figsize=(15, 7))
        ax.set_title("Reprojection Error")
        x = np.arange(sum(valid))
        x_label = [f"{i}:{paths[i].stem}" for i in range(len(paths)) if valid[i]]
        ax.set_xticklabels(x_label, rotation=90)
        ax.set_xticks(x+0.4)
        ax.set_ylabel('Mean Error (px)')
        width = (1-0.2)/len(tags)

    print("Reprojection Error:")
    for idx, current_tag in enumerate(tags):
        _mean, _std = [], []
        _batchA, _batchB = vertices_A[idx], vertices_B[idx]
        for _ptsA,ptsB in zip(_batchA, _batchB):
            errors = []
            for pointA, pointB in zip(_ptsA, ptsB):
                error = np.linalg.norm(pointA - pointB)
                errors.append(error)
            _mean.append(np.mean(errors))
            _std.append(np.std(errors))
        print(f"\t{current_tag} Reprojection Error: Mean: {np.mean(_mean):.2f} +/- {np.mean(_std):.2f} px")
        if verbose: ax.bar(x + width/2 + idx*width, 
            _mean, width, yerr=_std, label='X means', capsize=5)
            
    if verbose:
        ax.legend(tags)
        plt.savefig( paths[0].parent.parent / "Log_ReprojectionError.png")
        plt.close()

def plot_3Dpts_error(pts3D_A, pts3D_B, paths, calib, valid):
    """ Function to transform ptsA to ptsB frame using the calibration parameters and plot the 3D point cloud error."""
    assert len(pts3D_A) == len(pts3D_B) == len(paths) == len(valid), "All inputs must have the same length."
    Rot = calib['he_lap_M'] # 3X3 rotation matrix
    Trans = calib['he_lap_T'] # 3X1 translation vector
    _,ax = plt.subplots(figsize=(15, 7))
    ax.set_title("3D Point Cloud Error")
    x = np.arange(sum(valid))
    x_label = [f"{i}:{paths[i].stem}" for i in range(len(paths)) if valid[i]]
    ax.set_xticklabels(x_label, rotation=90)
    ax.set_xticks(x)
    ax.set_ylabel('Mean Error (mm)')
    width = 0.4
    mean_errors, std_errors = [], []
    for idx,v in enumerate(valid):
        if not v:
            continue
        assert pts3D_A[idx].shape == pts3D_B[idx].shape, "Inconsistent 3D point cloud shapes."
        ptsA = pts3D_A[idx].reshape(-1,3)@Rot.T + Trans.T
        ptsB = pts3D_B[idx].reshape(-1,3)
        
        # Compute and log the 3D point cloud error
        error = np.linalg.norm(ptsA - ptsB, axis=-1)
        mean_errors.append(np.mean(error)*1000)  # Convert to mm
        std_errors.append(np.std(error)*1000)    # Convert to mm
    ax.bar(x, mean_errors, width, yerr=std_errors, label='3D Point Cloud Error', capsize=5)
    plt.savefig( paths[0].parent.parent / "Log_3DPointCloudError.png")
    plt.close()


def plot_depths(pt_clouds, paths, valid, pts2D=None, verbose=False):
    """ Function to plot the depth maps from point clouds (z values) and optionally overlay 2D points."""
    assert len(pt_clouds) == len(paths) == len(valid), "All inputs must have the same length."
    if pts2D is not None:
        assert len(pts2D) == len(pt_clouds), "pts2D must have the same length as pt_clouds."
    depth_folder = paths[0].parent.parent / "Log_HeliosDepth"
    depth_folder.mkdir(exist_ok=True)
    if verbose: pbar = tqdm.tqdm(range(len(pt_clouds)), desc="Plotting Depth Maps")
    for idx, pt_cloud in enumerate(pt_clouds):
        if not valid[idx]:
            continue
        depth = pt_cloud[:, :, 2].copy()  # Assuming depth is the third channel
        mask = ~np.isnan(depth)

        d_min, d_max = np.min(depth[mask]), np.max(depth[mask])
        # print(f"Min depth: {d_min}, Max depth: {d_max}")
        depth[~mask] = d_max
        depth = (depth - d_min) / (d_max - d_min)  # Normalize depth to [0, 1]

        cmap = plt.get_cmap('viridis')
        depth = cmap(depth)  # Apply colormap
        depth = (depth[:, :, :3]*255).astype(np.uint8)  # Convert to RGB format
        depth[~mask] = 0
        
        depth = Image.fromarray(depth)
        if pts2D is not None:
            draw = ImageDraw.Draw(depth)
            for pt in pts2D[idx].reshape(-1, 2):
                if pt is not None:
                    x, y = int(pt[0]), int(pt[1])
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0))  # Draw red circle at the point
        depth.save(depth_folder / f"{paths[idx].name}.png")
        if verbose: pbar.update(1)
    if verbose: pbar.close()
