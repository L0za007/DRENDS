#  DRENDS: A Dataset for Depth in Robotic Endoscopy with Dynamic Scenarios
<center> <h3>
 🌎 <a href="https://l0za007.github.io/DRENDS">Project page</a>.
 📝 <a href="">Paper</a>. 
 💾 <a href="https://zenodo.org/records/17598453">Dataset</a>. 
 📋 <a href="https://github.com/MattiPoli97/DRENDS_Eval">Evaluation code</a>. 
</h3></center>
This repository contains all the code to run the calibration of the cameras and the processing pipeline. The last one rectifies every frame in the videos and generates the ground-truth depth maps for each rectified frame, along with a mask for the pixels with valid ground-truth.

## 🔧 Installation
We recommend using conda for the creation of an environment and pip to install all dependencies. We tested our code in Ubuntu 22.04.5 with PyTorch 2.5.0 and Cuda 12.2. The DRENDS environment can be created with the instructions below.
```
conda env create -f env.yml
```
Install PyTorch following the instructions in the official [documentation](https://pytorch.org/). After that, run the command below to install the rest of the dependencies.
```
pip install -r requirements.txt
```

## 🧾 Download and extract data
The full dataset can be downloaded from [DRENDS-Zenodo](https://zenodo.org/records/17598453). Once the data is downloaded, follow the commands below to extract the data from the zip files and obtain the folder structure shown below.
```
unzip DRENDS.zip
for f in DRENDS_ExVivo_*.zip; do 
 echo "Extracting $f..."
 unzip -o "$f" -d "DRENDS/ExVivo/"
done
for f in DRENDS_Phantom_*.zip; do 
 echo "Extracting $f..."
 unzip -o "$f" -d "DRENDS/Phantom/"
done
```
Additionally, if you wish to run the calibration pipeline. You can download the calibration data using the following comands or by accessing the following link [DRENDS-CalibrationData](https://www.dropbox.com/scl/fo/4q3vwun91ztr51i1s8iet/ABsH8qfCVCtimHB3UUnTe4o?rlkey=7jy4r3ai656o7rs8f3zfm1n7s&st=si9bym00&dl=0).
```
wget -q --content-disposition https://www.dropbox.com/scl/fi/j1yutthone5ws87qfey89/DRENDS_ExVivo_Calibration.zip?rlkey=i0659nlubmuq00w86mroctcvj&st=k7sz54c8&dl=1
unzip -o "DRENDS_ExVivo_Calibration.zip" -d "DRENDS/ExVivo/"

wget -q --content-disposition https://www.dropbox.com/scl/fi/5uq5j84g03u62oekwgh4b/DRENDS_Phantom_Calibration.zip?rlkey=eq9anmj6l85btokuzfrlcd85w&st=jqu9hjgp&dl=1
unzip -o "DRENDS_Phantom_Calibration.zip" -d "DRENDS/Phantom/"
```


## 📁 Data structure overview
The dataset can be found at: ....
The dataset is structured by batch, and each batch contains the following data:

```
DRENDS
├── ExVivo
│   ├── Calibration
│   ├── Seq00_Colon_Ext
│       ├── left.mp4
│       ├── right.mp4
│       ├── intensity.mp4
│       ├── point_clouds.h5
│       ├── timestamps.csv
│   ├── ...
│   ├── calibration.json
│   ├── fine_transform.json
│   ├── tool_prompts.json
│   ├── workspace_mask.png
├── Phantom
│   ...
```

## 💻 Code usage
The calibration results can be recreated using the command below:
```
conda activate DRENDS
python Calibration --calib_dir path/to/the/Dataset/DRENDS/ExVivo --verbose
python Calibration --calib_dir path/to/the/Dataset/DRENDS/Phantom --verbose
```
This command creates a `calibration.json` file containing all output parameters. The `--verbose` flag creates a log of the calibration, including rectified images of the reprojected patterns. More about the available functionalities can be seen by adding the flag `-h`. Some of the calibration configurations are in the file `./Calibration/config.json`.

The command below can be used to generate the depth ground truth in a folder within the dataset for each modality.
```
conda activate DRENDS
python Processing --config_file .Postprocessing/exvivo-config.json --seq_dir path/to/the/Dataset/DRENDS/ExVivo/Seq00_Colon_Ext
python Processing --config_file .Postprocessing/phantom-config.json --seq_dir path/to/the/Dataset/DRENDS/Phantom/Seq00_Colon_Ext
```
The command below can also be used to run the processing pipeline on the whole dataset:
```
conda activate DRENDS
python Processing --config_file .Postprocessing/exvivo-config.json --seq_dir path/to/the/Dataset/DRENDS/ExVivo/*
python Processing --config_file .Postprocessing/phantom-config.json --seq_dir path/to/the/Dataset/DRENDS/Phantom/*
```
More details about the processing pipeline configuration can be found in `Processing/*config.json`. Additional functionalities can be found by adding the `-h` flag to any of the previous commands.
## 📋 Notes
* The data is publicly available. However, you must give appropriate credit by citing the paper.
* MP4 videos use visually lossless encoding (CRF=0,H.264 encoder, 4:4:4 chroma, no color subsampling). VSCode can directly open these videos. If you want to use a different media player, make sure you use the needed decoder.

## 🧩 Contact & Support
**Maintainer:** Gerardo Loza and Mattia Magro  
**Contact:** scgelg@leeds.ac.uk, mattia.magro@polimi.it 

## 📰 Cite
```
 @misc{DRENDS2025,
 title={DRENDS: A Dataset for Depth in Robotic Endoscopy with Dynamic Scenarios},
 author={Gerardo Loza and Mattia Magro and Benjamin Calmé and Junlei Hua nd Emanuele Ruffaldi  and Dominic Jones and Elena de Momi and Sharib Ali and Pietro Valdastri},
 archivePrefix={arXiv},
 year={2025}
 }
```