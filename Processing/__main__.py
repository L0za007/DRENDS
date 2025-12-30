
import argparse, shutil, tifffile
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from utils import *
from rendering import *


def get_args():
    parser = argparse.ArgumentParser(description="Postprocessing for dVRK Data Acquisition, Leeds University. Use a configuration file to change parameters.")
    parser.add_argument('--config_file', type=str, default='Postprocessing/config.json', help="Path to the config file")
    parser.add_argument('--seq_dir'    , type=str, default=None, nargs='+', help="Path to the sequence directory. If not provided, it will be read from the config file")
    parser.add_argument('--single_img' , action='store_true', help="Process a single image instead of a folder of images")
    parser.add_argument('--verbose'    , action='store_true', help="Enable verbose output and store intermediate results")
    parser.add_argument('--remove_all' , action='store_true', help="Remove all the generated files by this script (tool masks, rectified images, depth maps, raw data)")
    parser.add_argument('--naive_extraction', action='store_true', help="Extract data from videos and hdf5 files without any postprocessing")
    parser.add_argument('--denoise_videos', action='store_true', help="Apply denoising to RGB videos")
    args = parser.parse_args()

    return args

# Rotation and translation to change coordinate system. Open3D -> Blender
R_O2B = np.array([180.0, 0.0, 0.0])
T_O2B = np.array([0, 0, 0])


def process_folder(seq_dir:Path, config, cam_view='left', crop_coords=None, single_img=False, verbose=False):
    """ Function to process a folder of images. 
    Args:
        seq_dir (Path): Path to the data directory.
        config (dict): Configuration dictionary.
        cam_view (str): Camera view to process ('left' or 'right').
        crop_coords (tuple): Coordinates to crop the point cloud (x_min, x_max, y_min, y_max).
        single_img (bool): If True, process a single image instead of a folder of images.
        verbose (bool): If True, print verbose output and save intermediate results. 
    """
    sub_folders = ['left', 'right', 'intensity']
    for sf in sub_folders:
        assert (seq_dir / f'{sf}.mp4').exists(), f"File {f'{sf}.mp4'} does not exist for sequence {seq_dir.name}"
    
    ##### Read camera params #####
    params = load_calibration(config['calibration_file'])
    
    ##### Rectify images #####
    if not (seq_dir / 'rect_left').exists() or not (seq_dir / 'rect_right').exists():
        for sf in sub_folders:
            if not (seq_dir / sf).exists():
                denoise = config.get('denoise_rgb_videos', False) if sf in ['left', 'right'] else False
                video_to_frames(seq_dir / f'{sf}.mp4', seq_dir / sf, denoise=denoise, verbose=verbose)
        files_dirs =  check_folder(seq_dir, sub_folders, verbose=verbose)
        if verbose: print("Rectifying images...")
        rectify_images((files_dirs['left'], files_dirs['right']), params, seq_dir, verbose=verbose)
    else:
        if verbose: print("Rectified images already exist. Skipping rectification step.")

    ##### Generate tool masks #####
    if config['mask_tools']:
        left_mask_dir = seq_dir / 'tool_masks_left'
        right_mask_dir = seq_dir / 'tool_masks_right'
        helios_mask_dir = seq_dir / 'tool_masks_helios'
        if not left_mask_dir.exists() or not right_mask_dir.exists() or not helios_mask_dir.exists():
            for sf in sub_folders:
                if not (seq_dir / sf).exists():
                    video_to_frames(seq_dir / f'{sf}.mp4', seq_dir / sf, verbose=verbose)
            generate_tools_masks(seq_dir, params)
        else:
            if verbose:  print("Tool masks already exist. Skipping tool mask generation step.")
        if verbose: 
            log_dir = seq_dir / 'Log_masks'
            log_dir.mkdir(exist_ok=True)
            files_dirs =  check_folder(seq_dir, ['rect_left', 'rect_right', 'tool_masks_left', 'tool_masks_right', 'tool_masks_helios'], verbose=verbose)
            generate_tool_mask_video(files_dirs['rect_left'], files_dirs['tool_masks_left'], log_dir / 'left_tools.mp4')
            generate_tool_mask_video(files_dirs['rect_right'], files_dirs['tool_masks_right'], log_dir / 'right_tools.mp4')
            generate_tool_mask_video(seq_dir/'intensity.mp4', files_dirs['tool_masks_helios'], log_dir / 'helios_tools.mp4')
            print(f"Saved tool segmentation videos to {log_dir / 'helios_tools.mp4'}, {log_dir / 'left_tools.mp4'}, {log_dir / 'right_tools.mp4'}")

    ##### Remove redundant folders #####
    for folder in sub_folders:
        folder_path = seq_dir / folder
        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)
    ##### Get point clouds and meshes #####
    point_clouds, meshes, helios_masks, stereo_masks = {}, {}, {}, {}
    if True:
        if  verbose: # save meshes and depth values
            raw_data = seq_dir / f'Log_{cam_view}'
            raw_data.mkdir(exist_ok=True)
    
        # Read point clouds
        assert (seq_dir / 'point_clouds.hdf5').exists(), f"File point_clouds.hdf5 does not exist for sequence {seq_dir.name}"
        hdf5_pc = h5py.File(seq_dir / 'point_clouds.hdf5', 'r')
        id_items = list(range(hdf5_pc['xyz'].shape[0])) if not single_img else [0]
        if  verbose: pbar = tqdm(total=len(id_items), desc="Reading Point Clouds")
        for idx in id_items:
            _pt_cloud, mask = get_depth_from_hdf5(hdf5_pc, idx, supress_outliers=True,
                                 crop=config['mask_workspace'], crop_coords=crop_coords)
            point_clouds[f"{idx:05d}"] = _pt_cloud
            helios_masks[f"{idx:05d}"] = mask
            if  verbose: 
                Image.fromarray((mask).astype(np.uint8)*255).save(raw_data / f'mask_helios_{idx:05d}.png')
                pbar.update(1)
        if  verbose: pbar.close()

        # Filtering pointclouds
        if config['temporal_filtering']['enabled']:
            if verbose: print("Applying temporal filtering to point clouds...")
            _point_clouds = [point_cloud for point_cloud in point_clouds.values()]
            _point_clouds = np.stack(_point_clouds, axis=0)  # (M,H,W,3)
            _point_clouds[..., -1] = temporal_filter(_point_clouds[..., -1], **config['temporal_filtering'])
            for i, file in enumerate(point_clouds.keys()):
                point_clouds[file][..., -1] = _point_clouds[i, ..., -1]

        # Spatial filtering of point clouds
        if config['spatial_filtering']['enabled']:
            if verbose: pbar = tqdm(total=len(point_clouds), desc="Spatial Filtering of Point Clouds")
            for file, _pt_cloud in point_clouds.items():
                _pt_cloud = spatial_filtering(_pt_cloud, **config['spatial_filtering'])
                point_clouds[file] = _pt_cloud
                if verbose: pbar.update(1)
            if verbose: pbar.close()

        # Create meshes from point clouds
        if verbose: pbar = tqdm(total=len(point_clouds), desc="Creating Meshes from Point Clouds")
        for file, _pt_cloud in point_clouds.items():
            map_size = _pt_cloud.shape[:-1]
            mesh, mesh_material = create_mesh_from_grid(_pt_cloud)
            meshes[file] = mesh
            if verbose: 
                o3d.io.write_triangle_mesh(raw_data / f'mesh_{file}.ply', transform_mesh(mesh, R=R_O2B, T=T_O2B))
                pbar.update(1)
        if verbose: pbar.close()

        # Detect degenerated normals
        if verbose: pbar = tqdm(total=len(meshes), desc="Filter Meshes by Normal Direction")
        for file, mesh in meshes.items():
            norm_maks = filter_normals(mesh, **config['normal_filtering']).reshape(map_size)
            stereo_masks[file] = norm_maks
            if verbose: 
                Image.fromarray((norm_maks).astype(np.uint8)*255).save(raw_data / f'mask_norm_{file}.png')
                pbar.update(1)
        if verbose: pbar.close()

    ##### Transform mesh to left camera view ##### 
    if  verbose: pbar = tqdm(total=len(meshes), desc="Transform Meshes to Left Camera View")
    if config['compensation_transform']:
        compensation = load_config(seq_dir.parent / 'fine_transform.json')
        compensation = compensation[seq_dir.name]
        # print(f"Applying compensation transform: R={compensation['R']}, T={compensation['T']}")
    else: compensation=None
    
    local_K, local_R, local_T = decompose_projection_matrix(params[f'{cam_view}_P'])
    for file, mesh in meshes.items():
        # Helios to left camera transformation
        mesh = transform_mesh(mesh, R = params['he_lap_R'],T = params['he_lap_T'], compensation=compensation)
        # camera view parameters
        mesh = transform_mesh(mesh, R=local_R, T=local_T)
        view_mask = filter_camera_view(mesh, params, local_K).reshape(map_size)
        meshes[file] = mesh
        stereo_masks[file] = stereo_masks[file] & view_mask
        if verbose: 
            Image.fromarray((view_mask).astype(np.uint8)*255).save(raw_data / f'mask_view_{file}.png')
            pbar.update(1)
    if verbose: pbar.close()

    # Meshes ready. Rendering depth maps...
    print("Meshes ready. Rendering depth maps...")

    # Render depth from RGB camera view
    M = np.eye(4)  # Overwrite M for the camera position because this transformation was already considered in the mesh transformation
    w,h = params['img_size']
    (seq_dir / f'GT_{cam_view}').mkdir(exist_ok=True)
    depth_dir = seq_dir / f'GT_{cam_view}' / 'depth_maps'
    depth_dir.mkdir(exist_ok=True)
    mask_dir = seq_dir / f'GT_{cam_view}' / 'masks'
    mask_dir.mkdir(exist_ok=True)
    frames = []
    for file, mesh in meshes.items():
        # Use the full mesh to get the reference depth map (detect occlusions)
        ref_mesh = remove_points_from_mesh(mesh, helios_masks[file].reshape(-1))
        ref_depth_map = render_depth_map((ref_mesh, mesh_material), M, local_K, w,h)
        # Use the filtered mesh to get the valid depth map
        filt_mask = stereo_masks[file] & helios_masks[file]
        if config['mask_tools']: # Remove tool points from the mesh
            tool_mask = read_tool_mask(helios_mask_dir / f'{file}.png', crop=config['mask_workspace'], crop_coor=crop_coords)
            filt_mask = filt_mask & (~tool_mask)
        filt_mesh = remove_points_from_mesh(mesh, filt_mask.reshape(-1))
        depth_map = render_depth_map((filt_mesh, mesh_material), M, local_K, w,h)
        pts3D = np.asarray(filt_mesh.vertices)
        pts2D = project_to_img_plane(pts3D, local_K)
        if verbose: o3d.io.write_triangle_mesh(raw_data / f'mesh_filt_{file}.ply', transform_mesh(filt_mesh, R=R_O2B, T=T_O2B))

        if config['remove_occlusions']:
            map_mask,pts_mask = get_occlusion_mask(pts2D, pts3D, depth_map, ref_depth_map,config['depth_range'], params['img_size'])
            pts2D = pts2D.reshape(-1, 2)
            pts2D = pts2D[pts_mask]
            if  verbose: filt_mask[filt_mask] = pts_mask
        else:
            map_mask = None

        if config['mask_tools']: # Remove tool points from the depth map
            tool_mask_dir = left_mask_dir if cam_view=='left' else right_mask_dir
            tool_mask = read_tool_mask(tool_mask_dir / f'{file}.png')
            map_mask = map_mask & (~tool_mask) if map_mask is not None else ~tool_mask
            pts_mask = map_mask[pts2D[:,1].astype(int), pts2D[:,0].astype(int)]
            pts2D = pts2D[pts_mask]
            if  verbose: filt_mask[filt_mask] = pts_mask
        # Save output depth maps and masks
        depth_map = np.asarray(depth_map)
        tifffile.imwrite(depth_dir / f'{file}.tiff', (depth_map*1000).astype(np.uint16)) # in mm
        Image.fromarray((map_mask).astype(np.uint8)*255).save(mask_dir / f'{file}.png')
        # Create masked depth map visualization
        bkground_img = seq_dir / f'rect_{cam_view}' / f'{file}.png'
        bkground_img = np.array(Image.open(bkground_img))
        img_depth_map = create_heat_map(depth_map, map_mask, pts2D, bkground_img=bkground_img)
        frames.append(np.array(img_depth_map))
        if  verbose: # log
            img_depth_map = create_heat_map(depth_map, map_mask, pts2D, bkground_img=bkground_img)
            img_depth_map.save(raw_data / f'depth_stereo_masked_{file}.png')
            create_heat_map(ref_depth_map).save(raw_data / f'depth_stereo_{file}.png')
            _pts_cloud = point_clouds[file]
            create_heat_map(_pts_cloud[...,-1], filt_mask, bw_mask=True).save(raw_data / f'depth_helios_{file}.png')

    #imageio.mimwrite(seq_dir / f'depth_{cam_view}.mp4', frames)#, fps=10)
    save_frames_in_video(frames, seq_dir / f"depth_{cam_view}.mp4", fps=20, colorize=True)
    print(f"Saved depth video to {seq_dir / f'depth_{cam_view}.mp4'}")

def remove_generated_files(seq_dir:Path):
    """ Function to remove all generated files in the data directory."""
    objects_to_retain = ['left.mp4', 'right.mp4', 'intensity.mp4', 'timestamps.csv', 'point_clouds.hdf5']
    for item in objects_to_retain:
        assert (seq_dir / item).exists(), f"File {item} does not exist for sequence {seq_dir.name}"
    for folder in seq_dir.iterdir():
        if folder.name in objects_to_retain:
            continue
        if folder.is_dir():
            shutil.rmtree(folder)
        else:
            folder.unlink()

if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_file)
    if args.seq_dir is not None:
        seq_dirs = sorted([Path(dd) for dd in args.seq_dir if 'Vid' in dd or 'Seq' in dd])
    else:
        seq_dirs = [Path(config['seq_directory'])]

    config['calibration_file'] = seq_dirs[0].parent / config['calibration_file']
    load_calibration(config['calibration_file'])
    crop_coor = None 
    for seq_dir in seq_dirs:
        assert seq_dir.exists(), f"The provided data directory {seq_dir} does not exist"
        if args.remove_all:
            print(f"Removing all generated files in {seq_dir}")
            remove_generated_files(seq_dir)
            continue
        if args.naive_extraction:
            print("Naive frame extraction")
            for video in seq_dir.glob("*.mp4"):
                denoise = config.get('denoise_rgb_videos', False) if video.stem in ['left', 'right'] else False
                video_to_frames(video, seq_dir / video.stem, denoise=denoise, single_img=args.single_img, verbose=args.verbose)
            h5_file = seq_dir / 'point_clouds.hdf5'
            hdf5_to_numpy(h5_file, out_dir=seq_dir / h5_file.stem, single_img=args.single_img, verbose=args.verbose)
            continue
        if args.denoise_videos:
            rgb_videos = [seq_dir / 'left.mp4', seq_dir / 'right.mp4']
            denoise_videos_opencv(rgb_videos)
            continue
        print(f"Processing folder: {seq_dir}")
        # try:
        if config['mask_workspace'] and crop_coor is None:
            crop_coor = define_workspace(seq_dir.parent / config['image_to_define_workspace'], 
                                     save_path=seq_dir.parent)
        process_folder(seq_dir, config, crop_coords=crop_coor, single_img=args.single_img, verbose=args.verbose)
        process_folder(seq_dir, config, cam_view='right', crop_coords=crop_coor, single_img=args.single_img, verbose=args.verbose)
        print(f"Folder {seq_dir} processed.")
        # except Exception as e:
        #     print(f"Error processing folder {seq_dir}: {e}")
        print("--------------------------------------------------")