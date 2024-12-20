# Self-Supervised Audiovisual Deepfake Detection on the Presidential Deepfakes Dataset (PDD)


## Overview

This project introduces a state-of-the-art deepfake detection system designed to identify manipulated videos by detecting inconsistencies between audio and visual streams. Utilizing a self-supervised learning approach, it eliminates the need for labeled training data, ensuring efficient and robust anomaly detection.


## Features

- **Self-Supervised Learning:** No labeled data required for training.
- **Audio-Visual Synchronization Analysis:** Exploits temporal inconsistencies in deepfake videos.
- **Flexible Architectures:** Supports multiple state-of-the-art backbones:
  - **Visual Encoders:** ResNet-18 2D+3D, Vision Transformer (ViT)
  - **Audio Encoders:** VGG-M, Wav2Vec 2.0
  - **Fusion Module:** Transformer-based architecture for cross-modal feature integration.
- **Pretrained Model Support:** Includes pretrained weights for faster experimentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Xuening0322/deep-fake-detection.git
   cd deep-fake-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets:
   - **LRS2 Dataset:** Use the provided `download_lrs2.py` script.
   - **Presidential Deepfakes Dataset (PDD):** Use the `download_pdd.py` script (requires `yt_dlp`).

4. Download pretrained model weights: [Google Drive Link](https://drive.google.com/file/d/1-RGElrYZquO6RGE9Xjf-ODkb6UXWBR3g/view?usp=sharing)



## Datasets

### LRS2 Dataset
The **Lip Reading Sentences 2 (LRS2)** dataset contains 97,000 speech videos of cropped face tracks. Download it using the provided script:
```bash
python download_lrs2.py --username your_user_name \
                       --password your_password \
                       --output_dir /path/to/output_directory
```

### Presidential Deepfakes Dataset (PDD)
To download the PDD dataset:
1. Install `yt_dlp`:
   ```bash
   pip install yt_dlp
   ```
2. Run the script:
   ```bash
   python download_pdd.py
   ```

## Usage

### Training

1. Ensure all datasets are downloaded.
2. Run the training script:
   ```bash
   python train.py \
       --train_list /path/to/train_list.txt \
       --device cuda:0 \
       --output_dir ./checkpoints \
       --lr 1e-4 \
       --epochs_0 30 \
       --epochs_1 30
   ```
   #### Explanation of Arguments:
   - `--train_list`: Path to the training list file that contains the list of videos for training.
   - `--device`: Specifies the device for computation (e.g., `cuda:0` for GPU or `cpu`).
   - `--output_dir`: Directory where model checkpoints and logs will be saved.
   - `--lr`: Learning rate for the optimizer.
   - `--epochs_0`: Number of epochs for the first training phase with cross-video negatives.
   - `--epochs_1`: Number of epochs for the second training phase with within-video negatives.


### Inference

#### Single Video
```bash
python detect.py \
    --test_video_path test.mp4 \
    --device cuda:0 \
    --output_dir out/
```

#### Explanation of Arguments:
- `--test_video_path`: Path to the input video file for inference.
- `--device`: Specifies the computation device (e.g., `cuda:0` for GPU or `cpu`).
- `--output_dir`: Directory where the results and visualizations will be saved.


You can view the generated visualizations in `out/`.

![Generated Visualizations](out/analysis_video_0.png)


#### Dataset Inference
```bash
python detect.py \
    --test_video_path /path/to/fake_videos.txt \
    --device cuda:0 \
    --output_dir out/
```

See `config_deepfake.py` for additional details and configurable parameters.


## Project Structure


```
deep-fake-detection/
├── backbone/                # Backbone model architectures
│   ├── __pycache__/         # Python cache files
│   ├── __init__.py          # Module initializer
│   ├── audio_process.py     # Audio processing utilities
│   ├── i3d.py               # I3D architecture implementation
│   ├── resnet_2d3d.py       # 2D+3D ResNet implementation
│   ├── s3dg.py              # S3DG model implementation
│   ├── select_backbone.py   # Select and configure backbone models
│   ├── transformer3d.py     # 3D Transformer architecture
├── checkpoints/             # Model checkpoints
├── out/                     # Output directory for inference results
├── README.md                # Project documentation
├── __init__.py              # Module initializer
├── audio_process.py         # Audio processing utilities
├── config_deepfake.py       # Configuration settings
├── create_train_list.py     # Script for generating training list
├── deep_fake_data.py        # Data processing for deepfake videos
├── detect.py                # Detection script
├── download_lrs.py          # Script to download LRS2 dataset
├── download_pdd.py          # Script to download Presidential Deepfakes Dataset
├── download_txt_files.py    # Script to download LRS2 text files
├── fake_celeb_dataset.py    # Processing Fake Celebrity Dataset
├── fake_celeb_dataset_detect.py # Fake Celebrity Dataset detection script
├── load_audio.py            # Audio loading utilities
├── load_video.py            # Video loading utilities
├── model.py                 # Main model implementation
├── pca.pkl                  # PCA model for dimensionality reduction
├── requirements.txt         # Project dependencies
├── test.mp4                 # Test video for inference
├── test.wav                 # Test audio file
├── train.py                 # Training script
├── train_list.txt           # Training list for LRS2
├── transformer_component.py # Transformer components for the model
├── avfeature_regressive_model.pth # Pretrained model from the paper
├── dist_regressive_model.pth # Pretrained model from the paper
```


## Acknowledgments

This project's implementation is largely adopted from Feng, Chao, et al. "Self-supervised video forensics by audio-visual anomaly detection." CVPR, 2023.

```bibtex
@inproceedings{feng2023self,
  title={Self-supervised video forensics by audio-visual anomaly detection},
  author={Feng, Chao and Chen, Ziyang and Owens, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10491--10503},
  year={2023}
}