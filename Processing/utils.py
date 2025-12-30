import json, cv2, torch, imageio, h5py
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.linalg import rq  # RQ decomposition
from pathlib import Path

from typing import List

def save_frames_in_video(frames, out_path, fps=20, colorize=True, vmin=None, vmax=None):
    """ Save a list of frames (HxW numpy arrays) as a video file.
    """
    out_path = str(out_path)

    # ---- Normalize / colorize to 8-bit RGB for compatibility ----
    fr_list = list(frames)
    if not fr_list:
        raise ValueError("No frames to write")

    # Compute global min/max once (for float/uint16 depth)
    if vmin is None or vmax is None:
        vals_min, vals_max = np.inf, -np.inf
        for f in fr_list:
            f = np.asarray(f)
            if f.ndim == 3 and f.shape[2] == 3:
                continue
            vals_min = min(vals_min, np.nanmin(f))
            vals_max = max(vals_max, np.nanmax(f))
        if not np.isfinite(vals_min): vals_min = 0.0
        if not np.isfinite(vals_max): vals_max = 1.0
        vmin = vals_min if vmin is None else vmin
        vmax = vals_max if vmax is None else vmax
        if vmax == vmin:
            vmax = vmin + 1e-6

    # Convert all frames to uint8 RGB (HxWx3)
    try:
        import matplotlib.cm as cm
        lut = (cm.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:
        lut = None  # fallback to gray

    rgb_frames = []
    H, W = None, None
    for f in fr_list:
        a = np.asarray(f)
        if a.ndim == 3 and a.shape[2] == 3:
            if a.dtype.kind == "f":
                a = np.clip(a, 0, 1) * 255.0
            a = np.clip(a, 0, 255).astype(np.uint8)
            rgb = a
        else:
            a = a.squeeze()
            a_norm = (a - vmin) / (vmax - vmin)
            a_u8 = np.clip(a_norm, 0, 1)
            a_u8 = (a_u8 * 255.0).astype(np.uint8)
            if colorize and lut is not None:
                rgb = lut[a_u8]
            else:
                rgb = np.repeat(a_u8[..., None], 3, axis=2)
        if H is None:
            H, W = rgb.shape[:2]
        elif rgb.shape[:2] != (H, W):
            raise ValueError(f"Frame size mismatch: got {rgb.shape[:2]}, expected {(H, W)}")
        rgb_frames.append(rgb)

    # ---- Write via ffmpeg backend (most compatible) ----
    # Keep params minimal to avoid version-specific issues.
    writer = imageio.get_writer(
        out_path,
        format="ffmpeg",   # force ffmpeg
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p"  # plays in QuickTime/browsers
    )
    try:
        for rgb in rgb_frames:
            writer.append_data(rgb)
    finally:
        writer.close()

def load_calibration(calib_file):
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    calib = {k:np.array(v) for k,v in calib.items()}
    return calib

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def define_workspace(image_file, save_path):
    """ Read an image and display to the user to select 4 points on the image. 
    Args:
        image: image file directory
    Return:
        workspace: 2 points defined by the min and max corners of the selected region.
    """
    work_space_mask = save_path / 'workspace_mask.png'
    if not work_space_mask.exists():
        print(f"Work space mask has been required but {work_space_mask} not found. Select 4 point on the image to define the workspace.")
        points = []
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select 4 Points", param)

        img = cv2.imread(image_file)
        cv2.imshow("Select 4 Points", img)
        cv2.setMouseCallback("Select 4 Points", click_event, img)

        while True:
            cv2.imshow("Select 4 Points", img)
            if len(points) >= 4:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        pts = np.array(points[:4])
        min_pt = np.min(pts, axis=0)
        max_pt = np.max(pts, axis=0)
        img_mask = np.zeros_like(img)
        img_mask[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = cv2.imread(image_file)[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] + 10
        img_mask = np.clip(img_mask, 0, 255).astype(np.uint8)
        cv2.imwrite(str(work_space_mask), img_mask)
    # Return the crop coordinates
    _ws_mask = np.array(Image.open(work_space_mask)) > 0 
    coordinates = np.where(_ws_mask)
    crop_coor = (coordinates[1].min(), coordinates[1].max(), coordinates[0].min(), coordinates[0].max())
    return crop_coor

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

def extract_camera_parameters(calib, camera):
    if camera == 'left':
        K,R,T = decompose_projection_matrix(calib['left_P'])
        
    elif camera == 'right':
        K,R,T = decompose_projection_matrix(calib['right_P'])
        
    else:
        raise ValueError("Invalid camera")
    
    M = build_extrinsic(R, T)
    return K,M

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
    if verbose: print(f"Found {n_samples} common images")
    # Take n samples
    if take_n_samples is not None:
        if verbose: print(f"Taking {take_n_samples} samples")
        sample_indices = np.linspace(0, n_samples - 1, min(n_samples, take_n_samples), dtype=int)
        img_paths = {sub.name: [img_paths[sub.name][i] for i in sample_indices] for sub in sub_f}
    return img_paths

def rectify_images(imgs:np.ndarray, params, out_dir:Path=None, tag='rect', verbose=False):
    """ Function to rectify images and 2D points."""
    LK, LD, LR, LP = params["left_K"], params["left_D"], params["left_R"], params["left_P"]
    RK, RD, RR, RP = params["right_K"], params["right_D"], params["right_R"], params["right_P"]
    # Rectify images 
    Limgs, Rimgs = imgs
    path_format = isinstance(Limgs[0], str) or isinstance(Limgs[0], Path)
    img_size = cv2.imread(str(Limgs[0])).shape[:2][::-1] if path_format else Limgs[0].shape[:2][::-1]
    map1x, map1y = cv2.initUndistortRectifyMap(LK, LD, LR, LP, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(RK, RD, RR, RP, img_size, cv2.CV_32FC1)
    rect_Limgs, rect_Rimgs = [], []
    if  verbose: pbar = tqdm(total=len(Limgs), desc="Rectifying Images")
    for idx in range(len(Limgs)):
        if path_format:
            img1 = cv2.imread(str(Limgs[idx]))
            img2 = cv2.imread(str(Rimgs[idx]))
        else:
            img1,img2 = Limgs[idx].astype(np.float32), Rimgs[idx].astype(np.float32)
        img1_rectified = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        rect_Limgs.append(img1_rectified.astype(np.uint8)), rect_Rimgs.append(img2_rectified.astype(np.uint8) )
        # Save images if needed
        if out_dir is not None:
            out_dir_left, out_dir_right = out_dir / f'{tag}_left', out_dir / f'{tag}_right'
            out_dir_left.mkdir(exist_ok=True)
            out_dir_right.mkdir(exist_ok=True)
            if isinstance(Limgs[idx], Path):
                file_name = Limgs[idx].name
            else:
                file_name = f"{idx:05d}.png"
            cv2.imwrite(str(out_dir_left / file_name), img1_rectified)
            cv2.imwrite(str(out_dir_right / file_name), img2_rectified)
        if  verbose: pbar.update(1)
    if  verbose: pbar.close()

    return rect_Limgs, rect_Rimgs

def generate_tools_masks(data_dir:Path, params, verbose=False):
    # read propts
    save_path = data_dir.parent / "tool_prompts.json"
    assert save_path.exists(), f"Prompt file {save_path} does not exist"
    with open(save_path, "r") as f:
        prompts = json.load(f)

    # Read left & right images
    frames_dir =  check_folder(data_dir, ['left', 'right', 'intensity'])

    # Run SAM2
    left_seg_frames, left_masks = run_sam(frames_dir['left'], prompts)
    right_seg_frames, right_masks = run_sam(frames_dir['right'], prompts)
    intensity_seg_frames, intensity_masks = run_sam(frames_dir['intensity'], prompts)

    # Save the mask
    rect_left_masks, rect_right_masks = rectify_images((left_masks, right_masks), params, out_dir=data_dir, tag='tool_masks')
    left_seg_frames, right_seg_frames = rectify_images((left_seg_frames, right_seg_frames), params)
    out_int = data_dir / 'tool_masks_helios'
    out_int.mkdir(exist_ok=True)
    for idx, file_dir in enumerate(frames_dir['intensity']):
        cv2.imwrite(str(out_int / file_dir.name), intensity_masks[idx])
        
def run_sam(frame_files:Path, prompts:dict):
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    """Generate tool masks for the video frames"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[SAM2]: using device: {device}")
    frames_dir = frame_files[0].parent

    # prompts
    prompt = {}
    _name = frames_dir.name if 'intensity' not in frames_dir.name else "intensity"
    prompt_key = str(Path(frames_dir.parent.name) / _name)
    if prompt_key not in prompts:
        raise ValueError(f"No prompts found for the folder {frames_dir}")
    for idx, n_pts in enumerate(prompts[prompt_key]):
        n_pts = np.array(n_pts, dtype=np.float32)
        prompt[idx] = n_pts, np.ones(len(n_pts), np.int32)
    print(f"[SAM2]: Generating tool masks for the folder {frames_dir} with the prompt{prompt}")
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to(device)
    

    #with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(str(frames_dir))
    # Make inference in initial mask
    for key,val in prompt.items():
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,frame_idx=0,
                                                                        obj_id=key,points=val[0],labels=val[1])
    # Propagate mask
    seg_frames, masks = [], []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        tool_mask = [(out_mask_logits[i] > 0.0).cpu().numpy() 
                        for i, out_obj_id in enumerate(out_obj_ids)]
        tool_mask = np.concatenate(tool_mask, axis=0)
        tool_mask = np.any(tool_mask,axis=0)
        try:
            _image = imageio.imread(frame_files[out_frame_idx])
        except Exception as e:
            print(f"[WARN] Skipping unreadable frame: {frame_files[out_frame_idx]} ({e})")
            continue
        if _image.ndim == 2:
            _image = np.stack([_image]*3, axis=-1)
        _image[...,1] =  np.clip((_image[...,1].astype(np.int32)+tool_mask*40),a_min=0,a_max=255).astype(np.uint8)
        seg_frames.append(_image)
        masks.append(tool_mask.astype(np.uint8) * 255)
    return seg_frames, masks, 

def generate_tool_mask_video(images:List[Path | np.ndarray] | Path, masks: List[Path | np.ndarray], out_file:Path):
    """ Generate a video with the tool masks overlayed on the images"""
    if isinstance(images, Path):
        assert 'mp4' in images.suffix, "If images is a Path, it must be a video file"
        # read video frames
        reader = imageio.get_reader(images, format="ffmpeg")
        images = [frame for frame in reader]
        images = images[:-1] # remove duplicated last frame
        reader.close()
    assert len(images) == len(masks), "Number of images and masks must be the same"
    overlay_frames = []
    for img_file, mask_file in zip(images, masks):
        if isinstance(img_file, (str, Path)):
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_file
        if isinstance(mask_file, (str, Path)):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        else:
            mask = mask_file
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        overlay = img.copy()
        overlay[mask > 0, 1] = np.clip(overlay[mask > 0, 1].astype(np.int32) + 40, a_min=0, a_max=255).astype(np.uint8)
        overlay_frames.append(overlay)
    # save video
    save_frames_in_video(overlay_frames, out_file, fps=20)

def read_tool_mask(file_dir:Path, crop=False, crop_coor=None):
    """ Read the tool masks and resize them to the img_size"""
    assert file_dir.exists(), f"Mask file {file_dir} does not exist"
    mask = cv2.imread(str(file_dir), cv2.IMREAD_GRAYSCALE)
    if crop and crop_coor is not None:
        x_min, x_max, y_min, y_max = crop_coor
        mask = mask[y_min:y_max, x_min:x_max]
    return mask > 0

def video_to_frames(video_file: Path, out_dir: Path = None, ext = "png", denoise = False, single_img = False, verbose = True):
    """Extract frames from a video file and save them as individual image files. """
    if out_dir is None:
        out_dir = video_file.parent / video_file.stem
    if out_dir.exists():
        print(f"Folder {out_dir} already exists. Skipping video extraction.")
        return
    out_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        print(f"Extracting frames from {video_file.name} → {out_dir}")
        pbar = tqdm(unit="frames")

    # --- Extraction loop ---
    reader = imageio.get_reader(video_file, format="ffmpeg")
    img_paths = []
    for i, frame in enumerate(reader):
        if denoise:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.fastNlMeansDenoisingColored(
                frame_bgr, None,
                h=8,          # Luminance filter strength (increase for stronger)
                hColor=6,      # Chrominance filter strength
                templateWindowSize=7,
                searchWindowSize=21
            )
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_file = out_dir / f"{i:05d}.{ext}"
        imageio.imwrite(out_file, frame)
        img_paths.append(out_file)
        if verbose:
            pbar.update(1)
        if single_img:
            break

    reader.close()
    if verbose:
        pbar.close()

    # --- Remove duplicate last frame ---
    if len(img_paths) > 1:
        last_frame = img_paths[-1]
        prev_frame = img_paths[-2]
        try:
            import numpy as np
            img_last = imageio.imread(last_frame)
            img_prev = imageio.imread(prev_frame)
            if np.array_equal(img_last, img_prev):
                last_frame.unlink()  # delete duplicated last frame
                img_paths.pop()
                if verbose:
                    print(f"Removed duplicated last frame: {last_frame.name}")
        except Exception as e:
            if verbose:
                print(f"Warning: could not verify duplicate last frame ({e})")

    if verbose:
        print(f"Saved {len(img_paths)} frames to {out_dir}")

def hdf5_to_numpy(hdf5_file: Path, out_dir: Path = None, single_img = False, verbose = True):
    h5_file = Path(hdf5_file)
    if out_dir is None:
        out_dir = h5_file.parent / h5_file.stem
    if out_dir.exists():
        print(f"Folder {out_dir} already exists. Skipping hdf5 extraction.")
        return
    out_dir.mkdir(exist_ok=True, parents=True)

    with h5py.File(h5_file, "r") as h5:
        if "xyz" not in h5:
            raise KeyError(f"'xyz' dataset not found in {h5_file}")
        dset = h5["xyz"]
        T, H, W, C = dset.shape
        if verbose:
            print(f"Extracting {T} point clouds from {h5_file.name} → {out_dir}")
            pbar = tqdm(total=T, unit="frames")

        pc_paths = []
        for t in range(T):
            pc = np.array(dset[t], dtype=np.float32)/1000.0  # load one frame
            out_file = out_dir / f"{t:05d}.npy"
            np.save(out_file, pc)
            pc_paths.append(out_file)
            if verbose:
                pbar.update(1)
            if single_img:
                break

        if verbose:
            pbar.close()

    if verbose:
        print(f"Saved {len(pc_paths)} point clouds to {out_dir}")

    return pc_paths

def denoise_videos_opencv(video_paths):
    """
    Apply strong OpenCV Non-Local Means denoising to a list of .mp4 videos.
    Saves a new version in the same folder with '_denoised' added to the filename.
    """
    for video_path in video_paths:
        video_path = Path(video_path)
        if not video_path.exists() or video_path.suffix.lower() != ".mp4":
            print(f"Skipping invalid file: {video_path}")
            continue

        print(f"Processing: {video_path.name}")

        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

        output_path = video_path.with_name(video_path.stem + "_denoised.mp4")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=frame_count, unit="frames", desc="Denoising video")
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply strong denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                frame, None,
                h=8,          
                hColor=6,      
                templateWindowSize=7,
                searchWindowSize=21
            )

            out.write(denoised)
            pbar.update(1)
        pbar.close()

        cap.release()
        out.release()
        print(f"→ Denoised saved as: {output_path.name}")