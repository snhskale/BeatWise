# BeatWise
BeatWise: A 3-part Python pipeline (Preprocessing, 1D-ResNet Classification, Communication) for end-to-end ECG arrhythmia classification from raw MIT-BIH data.

## The Dataset: From Raw Signals to a Balanced Training Set

This project is built on two "gold standard" benchmarks, providing a diverse and comprehensive source of data:
* **MIT-BIH Arrhythmia Database** (48 patient records)
* **MIT-BIH Supraventricular Arrhythmia Database** (78 patient records)

### 1. Source Data
Combined, these datasets provide **126 unique patient records** of 30-minute, two-channel ECG recordings, sampled at 360 Hz. Each record consists of:
* **`.dat`**: The raw, binary ECG signal.
* **`.hea`**: The header file (containing metadata like sampling rate).
* **`.atr`**: The expert cardiologist's "ground truth" beat annotations.

### 2. Dataset Generation
A custom preprocessing pipeline was executed on all 126 records to create the final, clean dataset used for this project. This automated process involved:
1.  **Filtering:** A 0.5-40 Hz Butterworth bandpass filter (`scipy`) was applied to remove baseline wander and muscle noise.
2.  **R-Peak Detection:** The Pan-Tompkins algorithm (`biosppy`) was used to find the location of all heartbeats.
3.  **Segmentation:** Each beat was extracted as a **280-sample** window (100 samples pre-R, 180 samples post-R) from the filtered signal.
4.  **Normalization:** Each individual segment was Min-Max scaled to a [0, 1] range to ensure the model learns *shape*, not *amplitude*.
5.  **Alignment:** Each of our detected segments was programmatically matched to the nearest expert label from the `.atr` files to create our "ground truth".

This pipeline resulted in a final, aligned dataset of **271,885** clean, labeled heartbeats (of the six classes we targeted), which were then prepared for model training.

### 3. Training & Test Set Creation
To ensure a fair and realistic evaluation, the data was prepared as follows:
1.  **Stratified Split:** The full 271,885-beat dataset was split 80% for training and 2
