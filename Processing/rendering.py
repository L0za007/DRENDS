import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import open3d as o3d, matplotlib.pyplot as plt, numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.spatial, cv2, copy
from scipy.linalg import rq  # RQ decomposition
from scipy.spatial.transform import Rotation as Rscipy

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)  # To suppress info messages

# region Point cloud
def get_ptCloud_from_paths(ptC_path, img_size=(640,480), supress_outliers=False, crop=False, crop_coords=None):
    """ Load point cloud from .npy file and reshape it to (H, W, 3).
        Optionally supress outliers and crop the point cloud."""
    pt_cloud = np.load(ptC_path)
    pt_cloud = pt_cloud.reshape((*img_size[::-1],3))
    if supress_outliers: 
        pt_cloud, valid_mask = supress_outliers_in_ptCloud(pt_cloud)
    if crop and crop_coords is not None:
        x_min, x_max, y_min, y_max = crop_coords
        pt_cloud = pt_cloud[y_min:y_max, x_min:x_max]
        valid_mask = valid_mask[y_min:y_max, x_min:x_max]
    return pt_cloud, valid_mask

def get_depth_from_hdf5(h5_obj, idx, supress_outliers=False, crop=False, crop_coords=None):
    point_cloud =  np.array(h5_obj["xyz"][idx], dtype=np.float32)/1000.0  # (H, W, 3)
    if supress_outliers:
        point_cloud, valid_mask = supress_outliers_in_ptCloud(point_cloud)
    if crop and crop_coords is not None:
        x_min, x_max, y_min, y_max = crop_coords
        point_cloud = point_cloud[y_min:y_max, x_min:x_max]
        valid_mask = valid_mask[y_min:y_max, x_min:x_max]
    return point_cloud, valid_mask


def supress_outliers_in_ptCloud(point_cloud):
    """ Detect missing values in the pointcloud (NaN values and distorded reflections, 
        i.e. points far away from camera) (H, W, 3) using KDTree nearest neighbors.
    Args:
        ptC_path: path to the .npy file containing the point cloud.
        img_size: (width, height) of the point cloud image.
        supress_outliers: if True, supress outliers in the point cloud.
        crop: if True, crop the point cloud to the region defined by crop_coor.
        crop_coor: (x_min, x_max, y_min, y_max) coordinates
    Returns:
        point_cloud: (H, W, 3) point cloud with outliers supressed.
        valid_mask: (H, W) boolean mask indicating valid points.
    """
    H, W, _ = point_cloud.shape

    # Create a grid of indices for point positions
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    grid_points = np.column_stack((x.ravel(), y.ravel()))
    # Find valid points
    valid_mask = ~np.isnan(point_cloud) & (np.abs(point_cloud)<2)
    valid_mask = np.all(valid_mask, axis=-1)
    valid_mask = valid_mask.ravel()
    missing_mask = ~valid_mask
    # Process each channel independently
    for c in range(3):
        values = point_cloud[..., c].ravel()
        # Find valid points
        valid_points = grid_points[valid_mask]
        valid_values = values[valid_mask]

        # Build KDTree and query missing points
        kdtree = scipy.spatial.KDTree(valid_points)
        missing_points = grid_points[missing_mask]
        
        if missing_points.size > 0:
            _, indices = kdtree.query(missing_points)
            filled_values = valid_values[indices]
            # Update the point cloud
            values[missing_mask] = filled_values

        point_cloud[..., c] = values.reshape(H, W)

    return point_cloud, valid_mask.reshape(H, W)

def temporal_filter(depth_seq, win_size=4, sigma_t=2.0, sigma_r=5.0, tau=None, **kwargs):
    """ Causal temporal bilateral filter over a per-pixel depth sequence.
    Args:
        depth_seq : (M, H, W) array (e.g., mm). Assumed clean (no NaN/Inf).
        win_size  : number of past frames to consider (history depth). Uses up to win_size past + current.
        sigma_t   : temporal sigma (frames). Larger -> longer temporal support. (Peso relacionado al tiempo)
        sigma_r   : range sigma (same units as depth). Larger -> more blending across depth changes. (Peso relacionado a los cambios de profundidad)
        tau       : optional hard gate (same units as depth). If set, samples with |d - ref| > tau are ignored.
    Returns: (M, H, W) filtered sequence.
    """
    if depth_seq.ndim != 3:
        raise ValueError("depth_seq must be (M, H, W)")
    M, H, W = depth_seq.shape
    out = np.empty_like(depth_seq)

    # t=0: seed
    out[0] = depth_seq[0]

    # Precompute temporal weights for offsets 0..K
    # w_t[o] corresponds to frame t-o (o=0 is current, o=K is oldest in window)
    offsets = np.arange(0, win_size+1, dtype=np.float32)
    w_t = np.exp(-0.5 * (offsets / float(sigma_t))**2)  # shape (K+1,)

    for t in range(1, M):
        start = max(0, t - win_size)
        # Window frames (causal): [start..t], length L
        win = depth_seq[start:t+1]                  # (L, H, W)
        L = win.shape[0]

        # Reference for range weights: previous filtered frame (stabilized)
        ref = win[-1]                              # (H, W)

        # Temporal weights for the actual L used (align from the end)
        # offsets_used: 0 for current, 1 for t-1, ..., L-1 for oldest in this small window
        w_t_used = w_t[:L]                          # (L,)
        w_t_used = w_t_used.reshape(L, 1, 1)        # (L,1,1)

        # Range weights relative to ref
        w_r = np.exp(-0.5 * ((win - ref) / float(sigma_r))**2)  # (L,H,W)

        # Optional hard gate
        if tau is not None:
            gate = (np.abs(win - ref) <= tau).astype(win.dtype) # (L,H,W)
        else:
            gate = 1.0

        weights = w_t_used * w_r * gate                         # (L,H,W)

        # Normalize safely (ensure at least current frame has some weight)
        num = np.sum(weights * win, axis=0)                     # (H,W)
        den = np.sum(weights, axis=0)                           # (H,W)
        y = num / den                                           # (H,W)

        out[t] = y

    return out

def spatial_filtering(point_cloud,d=5, sigma_s=3.0, sigma_r_mm=5.0, verbose=False, **kwargs):
    """ Apply bilateral filtering to the depth channel of the point cloud.
    Args:
        point_cloud : (H, W, 3) array (e.g., mm). Assumed clean (no NaN/Inf).
        d           : Diameter of each pixel neighborhood that is used during filtering.
        sigma_s     : Filter sigma in the spatial domain (pixels).
        sigma_r_mm  : Filter sigma in the range domain (same units as depth).
    Returns: (H, W, 3) filtered point cloud.
    """
    point_cloud[..., -1] = cv2.bilateralFilter(point_cloud[..., -1], d=d, sigmaColor=sigma_r_mm, sigmaSpace=sigma_s)
    return point_cloud
# endregion

# region Mesh
def create_mesh_from_grid(grid_points):
    """Create a triangle mesh from a structured (H, W, 3) 3D grid of points."""
    H, W, _ = grid_points.shape
    vertices = grid_points.reshape(-1, 3)

    faces = []  # List to store triangle indices
    for i in range(H - 1):
        for j in range(W - 1):
            # Convert 2D grid indices to 1D point indices
            idx0 = i * W + j
            idx1 = i * W + (j + 1)
            idx2 = (i + 1) * W + j
            idx3 = (i + 1) * W + (j + 1)
            # Create two triangles for each quad
            faces.append([idx0, idx1, idx2])  # First triangle
            faces.append([idx1, idx3, idx2])  # Second triangle
    # Convert faces to a NumPy array
    faces = np.array(faces, dtype=np.int32)

    # Create Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # Compute normals for better visualization
    mesh.compute_vertex_normals()

    mesh_material = o3d.visualization.rendering.MaterialRecord()
    mesh_material.base_color = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)  # Gray color
    mesh_material.shader = "defaultLit"  # Use lighting

    return mesh, mesh_material

def filter_normals(mesh, theta=60, bidirectional=False, enabled=True):
    """
    Keep only triangles whose face normals are within theta of the Z-axis.
    Also compacts vertices so that unused points are removed.
    Args:
        mesh: The input mesh to filter.
        theta: angle threshold in degrees relative to the Z-axis.
        bidirectional: 
            - True  => accept normals near +Z **or** -Z (|angle| ≤ theta)
            - False => accept normals near +Z only (angle to +Z ≤ theta)
    Returns:
        filtered_mesh
    """
    normals = np.asarray(mesh.vertex_normals)
    if not enabled:
        return np.ones(len(normals), dtype=bool)

    # Angle-to-Z test via cosine threshold (avoid arccos for speed/robustness)
    # cos(angle) = n · z_hat = n_z  (since z_hat = [0,0,1])
    cos_thresh = np.cos(np.deg2rad(theta))
    nz = normals[:, 2]
    if bidirectional:
        keep_mask = np.abs(nz) >= cos_thresh
    else:
        keep_mask = nz >= cos_thresh

    # Early exit if nothing survives
    if not np.any(keep_mask):
        raise ValueError("No triangles survived the filtering.")
    
    return keep_mask

def transform_mesh(mesh, R,T, compensation=None):
    """ Apply rigid transformation to the mesh vertices.
    Args:
        mesh: Open3D TriangleMesh
        R: Rotation (3,) Euler angles in degrees or (3,3) rotation matrix
        T: Translation (3,) vector
        compensation: Optional dict with 'R' (3,) Euler angles in degrees and 'T' (3,) translation vector
                        to be applied after the main transformation (inverse).
    Returns:
        new_mesh: Open3D TriangleMesh
    """
    # Normals are NOT transformed here. 
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles).copy()
    normals = np.asarray(mesh.vertex_normals).copy()
    if R.shape == (3,):
        R = Rscipy.from_euler('xyz', R, degrees=True).as_matrix()
    else: assert R.shape == (3,3), "R must be (3,) or (3,3)"
    pointcloud = (R @ vertices.T).T + T
    if compensation is not None:
        R_comp = Rscipy.from_euler('xyz', compensation['R'], degrees=True).as_matrix().T
        T_comp = -np.array(compensation['T'])
        pointcloud = (R_comp @ pointcloud.T).T + T_comp
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(pointcloud)
    new_mesh.triangles = o3d.utility.Vector3iVector(faces)
    new_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    return new_mesh

def remove_points_from_mesh(mesh, mask):
    """ Remove vertices from the mesh where mask is False.
    Args:
        mesh: Open3D TriangleMesh
        mask: Boolean array of shape (num_vertices,) indicating which vertices to keep.
    Returns:
        new_mesh: Open3D TriangleMesh with filtered vertices and updated faces.
    """
    mesh = copy.deepcopy(mesh)
    # Remove vertices where mask is False
    vertices = np.asarray(mesh.vertices)
    nv = len(vertices)
    vertices = vertices[mask]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # Remove vert normals
    normals = np.asarray(mesh.vertex_normals)
    normals = normals[mask]
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    # Update faces
    idx_map = -np.ones(nv, dtype=np.int64)
    idx_map[mask] = np.arange(len(vertices))
    faces = np.asarray(mesh.triangles)
    faces = idx_map[faces]
    faces = faces[~np.any(faces == -1, axis=1)]
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def filter_camera_view(mesh, params, K, padding=20):
    """ Create a mask for the mesh vertices that are within the camera view.
    Args:
        mesh: Open3D TriangleMesh
        params: dict with 'img_size': (width, height)
        K: Camera intrinsic matrix (3,3)
        padding: Number of pixels to pad the image boundaries.
    Returns:
        mask: Boolean array of shape (num_vertices,) indicating which vertices are within the camera view.
    """
    pts3D = np.asarray(mesh.vertices).copy()
    rvec_left = np.zeros((3, 1))  
    tvec_left = np.zeros((3, 1))
    proj_points, _ = cv2.projectPoints(pts3D, rvec_left, tvec_left, K, np.zeros(5))
    proj_points = proj_points.reshape(-1, 2)

    # mask positions that are within camera view
    width, height = params["img_size"]
    mask = (proj_points[..., 0] >= 0 - padding) & (proj_points[..., 0] < width + padding) & \
           (proj_points[..., 1] >= 0 - padding) & (proj_points[..., 1] < height + padding)
    
    return mask
# endregion

# region Rendering
def decompose_projection_matrix(P):
    """ Decompose the projection matrix P into intrinsic K, rotation R, and translation T."""
    # Extract K' (first 3x3 part) and T (last column)
    K_prime = P[:, :3]  # First three columns
    T = P[:, 3]  # Last column (Translation vector)
    # RQ decomposition to separate K and R
    K, R_rect = rq(K_prime)
    # Normalize K to ensure K[2,2] == 1
    K /= K[2, 2]
    T = T/K[0,0]
    return K, R_rect, T

def project_to_img_plane(pts3D, K):
    """ Project 3D points to 2D image plane using camera intrinsic matrix K."""
    shape = pts3D.shape[:-1]
    pts3D = pts3D.reshape(-1, 3)
    rvec_left = np.zeros((3, 1))
    tvec_left = np.zeros((3, 1))
    proj_points, _ = cv2.projectPoints(pts3D, rvec_left, tvec_left, K, np.zeros(5))
    proj_points = proj_points.reshape(-1, 2).reshape((*shape, 2)) # x,y
    return proj_points

def render_depth_map(mesh, M,K, width, height):
    """ Render a depth map from the mesh using the camera extrinsic M and intrinsic K."""
    import platform
    if platform.system() == "Darwin":
        mesh, _ = mesh
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        # Set up raycasting scene
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        # Intrinsic parameters
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create a grid of pixel coordinates
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)

        # Convert pixel coords to normalized camera directions
        x = (xs - cx) / fx
        y = (ys - cy) / fy
        z = np.ones_like(x)
        directions = np.stack((x, y, z), axis=-1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

        # Flatten and transform to world space
        directions = directions.reshape(-1, 3)
        origins = np.zeros_like(directions)

        R = M[:3, :3]
        t = M[:3, 3]
        directions = (R @ directions.T).T
        origins = np.tile(t, (directions.shape[0], 1))

        # Stack origins and directions into ray tensor
        rays = o3d.core.Tensor(np.hstack([origins, directions]), dtype=o3d.core.Dtype.Float32)

        # Cast rays
        ans = scene.cast_rays(rays)

        # Get hit distances (depth values)
        depth = ans['t_hit'].numpy().reshape((height, width))
        depth[np.isinf(depth)] = 0  # Optional: set misses to 0

        return o3d.geometry.Image(depth.astype(np.float32))
    
    else:
        print("Using offscreen visualiser")
        # Initialize the OffscreenRenderer to render the scene from the virtual camera
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        # Add geometries
        mesh, mesh_material = mesh
        renderer.scene.add_geometry("mesh", mesh, mesh_material)
        # Set the camera
        K = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        renderer.setup_camera(K.intrinsic_matrix, M, width, height)
        renderer.scene.camera.set_projection(K.intrinsic_matrix, 0.05, 0.60, width, height)
        # Render the depth map
        depth_image = renderer.render_to_depth_image(z_in_view_space=True)
        return np.asarray(depth_image)

def get_occlusion_mask(pts2D, pts3D, depth_map, ref_depth_map, depth_range, img_size):
    """ Create occlusion masks for the depth map and the reprojected points.
    Args:
        pts2D: (N, 2) array of 2D projected points.
        pts3D: (N, 3) array of corresponding 3D points.
        depth_map: (H, W) rendered depth map.
        ref_depth_map: (H, W) reference depth map (from non-filtered depth).
        depth_range: (min_d, max_d) valid depth range.
        img_size: (width, height) of the images.
    Returns:
        map_mask: (H, W) boolean mask for valid depth map pixels.
        pts_mask: (N,) boolean mask for valid reprojected points.
    """
    # mask for the depth map
    min_d, max_d = depth_range

    ref_depth_map = np.asarray(ref_depth_map)
    depth_map = np.asarray(depth_map)

    map_mask = (depth_map > min_d) & (depth_map < max_d)
    # ref_mask = (ref_depth_map > min_d) & (ref_depth_map < max_d)
    occ_mask = (depth_map < (ref_depth_map+0.005))
    # print("num of valid pixels:", np.sum(occ_mask))
    map_mask = map_mask & occ_mask
    # print("num of valid pixels:", np.sum(map_mask))
    # mask for the reprojected points
    width, height = img_size
    pts2D = pts2D.reshape(-1, 2)
    ptsZ = pts3D.reshape(-1, 3)[...,-1]
    _pts_mask = (pts2D[..., 0] >= 0) & (pts2D[..., 0] < width) & (pts2D[..., 1] >= 0) & (pts2D[..., 1] < height)
    _pts_mask = _pts_mask & (ptsZ > min_d) & (ptsZ < max_d)
    _pts2D, ptsZ = pts2D[_pts_mask], ptsZ[_pts_mask]
    pts_depth = ref_depth_map[_pts2D[...,1].astype(int), _pts2D[...,0].astype(int)]
    pts_mask = _pts_mask.copy() 
    pts_mask[_pts_mask] = pts_mask[_pts_mask] & (ptsZ < pts_depth + 0.005)  # Allow a small tolerance
    return map_mask, pts_mask

def create_heat_map(depth_map, map_mask=None, pts2d=None, bw_mask=False, bkground_img=None):
    """ Create a heat map from the depth map."""
    # Convert the depth image to a numpy array
    depth_np = np.asarray(depth_map)
    # Remove invalid values
    depth_np[np.isinf(depth_np)] = np.nan
    if map_mask is not None and (np.sum(map_mask) > 0):
        max_depth = np.nanmax(depth_np[map_mask])
        min_depth = np.nanmin(depth_np[map_mask])
    else:
        max_depth = np.nanmax(depth_np) 
        min_depth = np.nanmin(depth_np)
    mask = np.isnan(depth_np)
    depth_np[mask] = max_depth
    # Normalize the depth values to the range [0, 1]
    normiliser = plt.Normalize(vmin=min_depth, vmax=max_depth)
    # normiliser = plt.Normalize(vmin=0.01, vmax=1)
    norm_depth_np = normiliser(depth_np)
    colormap = plt.get_cmap('autumn')
    heatmap = colormap(norm_depth_np)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    if map_mask is not None:
        if bkground_img is not None:
            heatmap[~map_mask] = bkground_img[~map_mask]
        else:
            norm_depth_np = plt.Normalize()(depth_np)
            heatmap[~map_mask] = norm_depth_np[~map_mask,None] * np.array([1,1,1])*255 if bw_mask else np.array([0,0,0])
    heatmap = Image.fromarray(heatmap)
    # Write depth values range on the bottom right side of the image
    draw = ImageDraw.Draw(heatmap)
    font = ImageFont.load_default()
    x, y = heatmap.size
    # Define text
    min_text = f"Min: {min_depth:.3f} m"
    max_text = f"Max: {max_depth:.3f} m"
    # Calculate text size
    min_text_bbox = draw.textbbox((0, 0), min_text, font=font)
    max_text_bbox = draw.textbbox((0, 0), max_text, font=font)
    min_text_size = (min_text_bbox[2] - min_text_bbox[0], min_text_bbox[3] - min_text_bbox[1])
    max_text_size = (max_text_bbox[2] - max_text_bbox[0], max_text_bbox[3] - max_text_bbox[1])
    # Define rectangle size
    rect_width = max(min_text_size[0], max_text_size[0]) + 10
    rect_height = min_text_size[1] + max_text_size[1] + 10
    rect_x0 = x - rect_width - 10
    rect_y0 = y - rect_height - 10
    rect_x1 = x - 10
    rect_y1 = y - 10
    # Draw semi-transparent rectangle
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(255, 255, 255, 128))
    # Draw text
    draw.text((rect_x0 + 5, rect_y1 - max_text_size[1] - 5), min_text, fill="black", font=font)
    draw.text((rect_x0 + 5, rect_y0 + 5), max_text, fill="black", font=font)
    # Draw the vertices of the mesh if they are provided
    if pts2d is not None:
        shift = 1
        for pt in pts2d:
            draw.ellipse([pt[0] - shift, pt[1] - shift, pt[0] + shift, pt[1] + shift], fill="red")

    return heatmap