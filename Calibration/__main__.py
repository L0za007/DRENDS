import argparse, shutil
import numpy as np
from pathlib import Path

from utils import *
import stereo_calibration as Stereo
import helios_calibration as Helios

np.set_printoptions(precision=4, suppress=True)

def get_args():
    parser = argparse.ArgumentParser(description="Calibration for dVRK Data Acquisition, Leeds University. Use a configuration file to change parameters.")
    parser.add_argument('--config_file', type=str, default='Calibration/config.json', help="Path to the config file")
    parser.add_argument('--calib_dir', type=str, default=None, nargs='+', help="Path to the calibration folder. If not provided, it will be read from the config file")
    parser.add_argument('--calibration', type=str, default='Full', help="Calibration type: Stereo, HtoS (Helios to Stereo) or Full (Stereo + HtoS)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output and store intermediate results")
    parser.add_argument('--remove_all', action='store_true', help="Remove all the generated files by this script")
    args = parser.parse_args()
    config = load_config(args.config_file)
    if args.verbose:
        print("Configuration loaded from:", args.config_file)

    return args, config

def remove_all(calib_dir:Path):
    for folder in calib_dir.glob('Log_*'):
        folder_path = calib_dir / folder
        if folder_path.exists(): 
            if folder_path.is_dir(): shutil.rmtree(folder_path)
            else: folder_path.unlink()

def stereo_calibration(args, config):
    paths = check_folder(calib_dir, ['left','right'], 
                         take_n_samples=config['calibration_samples'], 
                         verbose=args.verbose)
    
    L_vertices,L_valid,L_imgs = detect_square_pattern(paths['left'], 
                                    config['pattern_shape'], tag='Left',  
                                    return_imgpoints=True, verbose=args.verbose)
    R_vertices,R_valid,R_imgs = detect_square_pattern(paths['right'], 
                                    config['pattern_shape'], tag='Right', 
                                    return_imgpoints=True, verbose=args.verbose)
    if args.verbose:
        print(f"Left:Found {len(L_imgs)} images with {sum(L_valid)} valid vertices")
        print(f"Right:Found {len(R_imgs)} images with {sum(R_valid)} valid vertices")
        save_images_with_pattern(images=(L_imgs ,R_imgs), img_paths=(paths['left'], paths['right']), 
                                 output_folder=calib_dir / 'Log_PatternDetection_LR')
    
    img_size = [L_imgs[0].shape[1], L_imgs[0].shape[0]] #(width, height)
    print(f"Image size detected: {img_size[0]}x{img_size[1]} (width x height)")
    stereo_calib = Stereo.calibrate(L_vertices, L_valid, 
                         R_vertices, R_valid, 
                         config['pattern_shape'], config['square_size_mm'],
                         img_size, verbose=args.verbose)
    
    LR, RR, LP, RP, Q, roi1, roi2 = cv2.stereoRectify(stereo_calib['left_K'], stereo_calib['left_D'], 
                                            stereo_calib['right_K'], stereo_calib['right_D'], 
                                            img_size, 
                                            stereo_calib['stereo_M'], stereo_calib['stereo_T'], 
                                            flags=cv2.CALIB_ZERO_DISPARITY, 
                                            alpha=config['alpha'])
    _rect_data = {'left_R':LR, 'right_R':RR, 'left_P':LP, 'right_P':RP, 'Q':Q}
    if args.verbose:
        for key, value in _rect_data.items():
            print(f"{key}:\n {value}")
    stereo_calib.update(_rect_data)
    out = Stereo.rectify_objects(imgs=(L_imgs, R_imgs) if args.verbose else None, 
            pts=(L_vertices, R_vertices), 
            calib=stereo_calib,  verbose=args.verbose,
            out_dir=calib_dir / "Log_StereoRectifiedImgs" if args.verbose else None)
    rect_imgs, rect_vertices = out['imgs'], out['pts']
    L_rect_pts = np.squeeze(np.array(rect_vertices[0][L_valid & R_valid].tolist()))
    R_rect_pts = np.squeeze(np.array(rect_vertices[1][L_valid & R_valid].tolist()))

    # 3D error calculation
    stereo_centres3D = Stereo.stereo_triangulation(rect_vertices, stereo_calib)
    if args.verbose:
        folder = calib_dir / "Log_StereoTriangulation"
        folder.mkdir(exist_ok=True)
        Stereo.plot_triangulation(stereo_centres3D, rect_imgs, (paths['left'], paths['right']), 
                                  config['pattern_shape'], folder, L_valid & R_valid)
    pattern_3D_dimensions_error(stereo_centres3D, paths['left'], L_valid & R_valid, 
                                config['pattern_shape'], config['square_size_mm'], 
                                calib_dir/'Log_StereoTriangulatedPatternError.png',
                                verbose=args.verbose)
    
    # 2D Reprojection error calculation
    L_reprojected_pts, R_reprojected_pts = Stereo.reproject3D(pts3D=stereo_centres3D, 
                images=rect_imgs, paths=paths['left'], calib=stereo_calib, 
                out_dir=calib_dir / "Log_ReprojectedPatterns" if args.verbose else None,
                verbose=args.verbose)
    L_reprojected_pts, R_reprojected_pts = np.array(L_reprojected_pts), np.array(R_reprojected_pts)
    reprojection_error((L_rect_pts, R_rect_pts), (L_reprojected_pts, R_reprojected_pts), tags=('Left','Right'), 
                      paths=paths['left'], valid=L_valid & R_valid, verbose=args.verbose)
    stereo_calib['img_size'] = np.array(img_size) #(width, height)
    # Save calibration data
    _calibration_data = {k:v.tolist() for k,v in stereo_calib.items()}
    with open(str(calib_dir / "Log_Stereo_Calibration.json"), 'w') as f:
        json.dump(_calibration_data, f, indent=4)

def helios_to_stereo_calibration(args, config):
    calib_stereo =json.load(open(str(calib_dir /'Log_Stereo_Calibration.json'), 'r'))
    param_helios =json.load(open(str(calib_dir /'ToF_params.json'), 'r'))
    calib = {**calib_stereo, **param_helios}
    calib = {k:np.array(v) for k,v in calib.items()}
    # Read images and detect patterns
    paths = check_folder(calib_dir, ['intensity','left','right'], take_n_samples=None, verbose=args.verbose)
    L_vertices,L_valid,L_imgs = detect_square_pattern(paths['left'], 
                                    config['pattern_shape'], tag='Left',  
                                    return_imgpoints=True, verbose=args.verbose)
    R_vertices,R_valid,R_imgs = detect_square_pattern(paths['right'], 
                                    config['pattern_shape'], tag='Right',  
                                    return_imgpoints=True, verbose=args.verbose)
    D_vertices,D_valid,D_imgs = detect_square_pattern(paths['intensity'], 
                                    config['pattern_shape'], tag='Helios',  
                                    return_imgpoints=True, verbose=args.verbose)
    valid = L_valid & R_valid & D_valid
    if args.verbose:
        print(f"Left:Found {len(L_imgs)} images with {sum(L_valid)} valid vertices")
        print(f"Right:Found {len(R_imgs)} images with {sum(R_valid)} valid vertices")
        print(f"Helios:Found {len(D_imgs)} images with {sum(D_valid)} valid vertices")
        save_images_with_pattern(images=(L_imgs ,R_imgs, D_imgs), 
                                 img_paths=(paths['left'], paths['right'], paths['intensity']), 
                                 output_folder=calib_dir / 'Log_PatternDetection_LRH')
    # Load point clouds
    PC_paths = [Path(str(path).replace("intensity", "point_clouds").replace("bmp", "npy").replace("png", "npy")) 
            for path in paths['intensity']]
    pointclouds = Helios.get_ptCloud_from_paths(PC_paths, supress_outliers=True)
    if args.verbose:
        plot_depths(pt_clouds=pointclouds, paths=paths['intensity'], 
                    valid=valid, pts2D=D_vertices, verbose=True)
    
    ptc_centres, pointclouds = Helios.get_pattern_calibration_in_ptCloud(
                                PC_paths, D_vertices, valid, calib['ToF_img_size'][::-1], approx_excess=False)
    print("Helios pattern")
    pattern_3D_dimensions_error(ptc_centres, paths['intensity'], valid,
                                config['pattern_shape'], config['square_size_mm'], 
                                calib_dir/'Log_HeliosPatternError.png',
                                verbose=args.verbose)
    if args.verbose:
        folder = calib_dir / "Log_Helios3DPattern"
        folder.mkdir(exist_ok=True)
        Stereo.plot_triangulation(ptc_centres, (D_imgs,None), (paths['intensity'],None),  
                                  config['pattern_shape'], folder, valid)

    # Stereo rectification and triangulation
    out = Stereo.rectify_objects(imgs=(L_imgs, R_imgs) if args.verbose else None, 
            pts=(L_vertices, R_vertices), 
            calib=calib,  verbose=args.verbose,
            out_dir=calib_dir / "Log_StereoRectifiedImgs" if args.verbose else None)
    rect_imgs, rect_vertices = out['imgs'], out['pts']
    stereo_centres3D = Stereo.stereo_triangulation(rect_vertices, calib)

    # Transform Helios points to Stereo frame
    Helios.HelRGB_calibrate(ptc_centres, stereo_centres3D, rect_vertices[0], 
                                         calib, valid, verbose=args.verbose)
    
    # Plot 3D error after calibration
    if args.verbose:
        plot_3Dpts_error(ptc_centres, stereo_centres3D, paths['intensity'],calib, valid)
        new_centres = []
        Rot = calib['he_lap_M'] 
        Trans = calib['he_lap_T']
        for idx in range(len(valid)):
            if valid[idx]:
                centres = (Rot @ ptc_centres[idx].T).T + Trans.T
                new_centres.append(centres)
            else:
                new_centres.append(None)
        new_centres = np.array(new_centres, dtype=np.float32)
        folder = calib_dir / "Log_ProjectionHtoS"
        folder.mkdir(exist_ok=True)
        Stereo.reproject3D(pts3D=new_centres, 
                images=rect_imgs, paths=paths['left'], calib=calib, 
                out_dir=folder if args.verbose else None,
                verbose=args.verbose)


    # Save calibration parameters
    output_file = calib_dir / "calibration.json"
    _calibration_data = {k:v.tolist() for k,v in calib.items()}
    with open(str(output_file), 'w') as f:
        json.dump(_calibration_data, f, indent=4)
    print("Calibration parameters stored in {}".format(str(output_file)))

if __name__ == "__main__":
    args, config = get_args()
    if args.calib_dir is None:
        calib_directories = [config['calib_directory']]
    else:
        calib_directories = args.calib_dir
    for calib_dir in calib_directories:
        calib_dir = Path(calib_dir)
        print(f"Using calibration directory: {calib_dir}")
        if not calib_dir.is_dir():
            print(f"Sikpping")
            continue
        if args.remove_all:
            remove_all(calib_dir)
            print("All generated files have been removed.")
            continue
        if args.calibration in ['Stereo', 'Full']:
            print("Starting Stereo Calibration...")
            stereo_calibration(args, config)
            print("Stereo Calibration completed.")
        if args.calibration in ['HtoS', 'Full']:
            if (calib_dir /'Log_Stereo_Calibration.json').exists() and (calib_dir /'ToF_params.json').exists():
                print("Starting Helios to Stereo Calibration...")
                helios_to_stereo_calibration(args, config)
                print("Helios to Stereo Calibration completed.")
            else:
                raise FileNotFoundError("Required calibration files not found. Please run Stereo calibration first or check Helios parameters.")
    


