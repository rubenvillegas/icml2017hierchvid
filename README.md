# Learning to Generate Long-term Future via Hierarchical Prediction

This is the code for the ICML 2017 paper [Learning to Generate Long-term Future via Hierarchical Prediction](https://arxiv.org/pdf/1704.05831.pdf) by Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryull Sohn, Xunyu Lin, Honglak Lee.

Please follow the instructions to run the code.

## Requirements
This code works with
* Linux
* NVIDIA Titan X GPU
* Tensorflow version 1.3.0

## Installing Dependencies (Anaconda installation is recommended)
* pip install scipy
* pip install imageio
* pip install pyssim
* pip install scikit-image
* pip install opencv-python
* pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl

FFMPEG needs to be installed as well to generate gif videos.
If using anaconda, ffmpeg can be installed as follows:
* conda install -c menpo ffmpeg=3.1.3

## Data download and preprocessing
Penn Action:  
* Download data:  
Download from [Penn Action](https://dreamdragon.github.io/PennAction) and extract into ./datasets/PennAction/
* Download pose estimated using hourglass network:
```
./datasets/PennAction/download_hourglass.sh
```
* Preprocess:
```
python ./datasets/PennAction/preprocess.py
```
Human 3.6M:  
* Download and convert:  
Download from [Human 3.6M](http://vision.imar.ro/human3.6m/description.php) into ./datasets/Human3.6M/ and preprocess by first converting the pose CDF files into .mat using MATLAB and running the matlab script in ./datasets/Human3.6M/:
```
cdf2mat.m
```
* Download pose estimated using hourglass network:
```
./datasets/Human3.6M/download_hourglass.sh
```
* Preprocess:
```
python ./datasets/Human3.6M/preprocess.py
```

## Download pre-trained perceptual models for feature loss
```
./perceptual_models/download.sh
```

## Penn Action training/testing
Training LSTM (can run in parallel with image generator training):
```
CUDA_VISIBLE_DEVICES=GPU_ID python lstm_src/train_det_rnn_penn.py --gpu=GPU_ID
```
Training Image Generator (can run in parallel with LSTM training):
```
CUDA_VISIBLE_DEVICES=GPU_ID python imggen_src/train_penn.py --gpu=GPU_ID
```
Predict future pose from models trained with the above commands:
```
CUDA_VISIBLE_DEVICES=GPU_ID python lstm_src/test_det_rnn_penn.py --gpu=GPU_ID --prefix=PENNACTION_DET_LSTM_num_class=8_learning_rate=0.001_image_size=128_batch_size=256_lm_size=13_fut_step=32_num_layer=1_lstm_units=1024_seen_step=10_input_size=26_keep_prob=1.0 --steps=64
```
Predict video from networks trained with the above commands:
```
CUDA_VISIBLE_DEVICES=GPU_ID python imggen_src/test_penn.py --gpu=GPU_ID --imggen_prefix=PENNACTION_ANALOGY_imgsize=128_layer=3_alpha=1.0_beta=1.0_gamma=1.0_lr=0.0001 --lstm_prefix=PENNACTION_DET_LSTM_num_class=8_learning_rate=0.001_image_size=128_batch_size=256_lm_size=13_fut_step=32_num_layer=1_lstm_units=1024_seen_step=10_input_size=26_keep_prob=1.0
```
Resulting images and videos will be located at:
```
./results/images/PENNACTION_ANALOGY_imgsize=128_layer=3_alpha=1.0_beta=1.0_gamma=1.0_lr=0.0001/
```

## Human 3.6M training/testing
Training LSTM (can run in parallel with image generator training):
```
CUDA_VISIBLE_DEVICES=GPU_ID python lstm_src/train_det_rnn_h36m.py --gpu=GPU_ID
```
Training Image Generator (can run in parallel with LSTM training):
```
CUDA_VISIBLE_DEVICES=GPU_ID python imggen_src/train_h36m.py --gpu=GPU_ID
```
Predict future pose from models trained with the above commands:
```
CUDA_VISIBLE_DEVICES=GPU_ID python lstm_src/test_det_rnn_h36m.py --gpu=GPU_ID --prefix=HUMAN3.6M_DET_LSTM_fskip=4_keep_prob=1.0_image_size=128_batch_size=256_lm_size=32_fut_step=32_num_layer=1_lstm_units=1024_seen_step=10_input_size=64_learning_rate=0.001 --steps=128
```
Predict video from networks trained with the above commands:
```
CUDA_VISIBLE_DEVICES=GPU_ID python imggen_src/test_h36m.py --gpu=GPU_ID --imggen_prefix=HUMAN3.6M_ANALOGY_imgsize=128_layer=3_alpha=1.0_beta=1.0_gamma=1.0_lr=0.0001 --lstm_prefix=HUMAN3.6M_DET_LSTM_fskip=4_keep_prob=1.0_image_size=128_batch_size=256_lm_size=32_fut_step=32_num_layer=1_lstm_units=1024_seen_step=10_input_size=64_learning_rate=0.001
```
Resulting images and videos will be located at:
```
./results/images/HUMAN3.6M_ANALOGY_imgsize=128_layer=3_alpha=1.0_beta=1.0_gamma=1.0_lr=0.0001/
```

## Citation

If you find this useful, please cite our work as follows:
```
@inproceedings{villegas17hierchvid,
  title={{Learning to Generate Long-term Future via Hierarchical Prediction}},
  author={Villegas, Ruben and Yang, Jimei and Zou, Yuliang and Sohn, Sungryull and Lin, Xunyu and Lee, Honglak},
  booktitle=ICML,
  year={2017}
}
```

Please contact "ruben.e.villegas@gmail.com" if you have any questions.

