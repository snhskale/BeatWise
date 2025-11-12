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

This pipeline resulted in a final, aligned dataset of about **260,000** clean, labeled heartbeats (of the six classes we targeted), which were then prepared for model training.

### 3. Training & Test Set Creation
To ensure a fair and realistic evaluation, the data was prepared as follows:
1.  **Stratified Split:** The full beat dataset was split 80% for training and 20% for testing. The split was *stratified* to ensure the test set retained the original, real-world class imbalance.
2.  **Hybrid Resampling (Training Set Only):** The 80% training set was then balanced to prevent model bias:
    * **SMOTE (Oversampling):** All five minority arrhythmia classes were synthetically oversampled to **20,000** samples each.
    * **Random Undersampling:** The majority 'N' (Normal) class was undersampled to **40,000** samples.

This resulted in a final, balanced **training set of 140,000 heartbeats**, while the test set remained in its natural, imbalanced state to provide a true test of real-world performance.

## Project Pipeline Architecture

The system is a 3-part Python pipeline that processes data sequentially. The `app.py` (Flask server) coordinates the execution of these modules.

1.  **Preprocessing Module (`preprocessing_agent.py`)**
    * **Input:** A raw file path.
    * **Process:** Loads the raw signal using `wfdb`, applies the 0.5-40 Hz Butterworth filter, detects all R-peaks using `biosppy` (Pan-Tompkins), and segments the filtered signal into 280-sample heartbeats.
    * **Output:** A dictionary containing the NumPy array of normalized segments, R-peak locations, and the sampling frequency.

2.  **Classification Module (`classification_agent.py`)**
    * **Input:** The dictionary from the preprocessing module.
    * **Process:** Loads the pre-trained `final_model.pth` (our 1D-ResNet). It then performs beat-by-beat inference on the NumPy array of segments.
    * **Output:** An array of predicted string labels (e.g., `['N', 'N', 'V', 'N', ...]`).

3.  **Communication Module (`communication_agent.py`)**
    * **Input:** The prediction array and the R-peak locations.
    * **Process:** This is a *scripted* (hard-coded) report generator. It first calculates clinical statistics (Avg. BPM, HRV, PVC count, V-Tach runs, etc.). It then formats these numbers into two distinct, structured reports.
    * **Output:** A technical **Doctor's Report** and a simple, jargon-free **Patient's Report**.

## Model Architecture: 1D Residual Network (ResNet)

The core of the Classification Module is a 1D-ResNet, an architecture chosen for its ability to handle the "vanishing gradient" problem and learn deep, complex morphological features from the 1D heartbeat signals.

* **Initial Block:** A large `Conv1d` layer (kernel=7, stride=2) followed by `MaxPool1d` to quickly downsample the signal and extract broad features.
* **Residual Blocks:** A stack of `ResidualBlock`s (inspired by ResNet-18). Each block contains two `Conv1d` layers (kernel=3) and a "skip connection," which adds the block's input to its output. This allows the model to learn much deeper patterns without losing information.
* **Classifier Head:** An `AdaptiveAvgPool1d` layer summarizes the features into a fixed-size vector, which is then passed to a final `Linear` (fully connected) layer for classification.
* **Training:** This model was trained on the **140,000-sample balanced training set** described above.

## Key Finding: 99.4% (Model) vs. 73.50% (System)

This project's main finding is the critical difference between "model accuracy" and "system accuracy."

* **Model Accuracy (99.4%):** Our 1D-ResNet model is exceptionally accurate when tested on the *aligned* test set (from `model_building.ipynb`, Cell 7). This data was perfectly segmented, providing a "clean" academic benchmark.

* **System Accuracy (73.50%):** When running the full, end-to-end pipeline on a raw file, the accuracy drops.

* **Reason (Pipeline Misalignment):** This is not a model failure. The `biosppy` R-peak detector in the preprocessor, while robust, does not segment beats at the *exact* same sample as the original expert annotations. This "off-center" segmentation (especially on 'V' beats) confuses the model, which was trained on "perfectly" centered beats. This 73.50% represents a realistic, real-world baseline for the *current* system.

## Technologies Used

* **Python 3.10+**
* **Deep Learning:** PyTorch
* **Data Science:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (for SMOTE)
* **Signal Processing:** SciPy, WFDB, Biosppy
* **Web Framework:** Flask

## How to Run the Web App

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/BeatWise-Project.git](https://github.com/YOUR_USERNAME/BeatWise-Project.git)
    cd BeatWise-Project
    ```

2.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` in the root folder and paste the following:
    ```
    pandas
    numpy
    torch
    scikit-learn
    imbalanced-learn
    matplotlib
    seaborn
    wfdb
    biosppy
    Flask
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place Data:**
    * Ensure your `final_model.pth` is in the root folder.
    * Ensure your raw data folder(after converted and made using preprocessing agent) is in the root folder.

5.  **Run the Flask Web App:**
    ```bash
    python app.py
    ```

6.  **Use the App:**
    * Open your web browser to `http://127.0.0.1:5000`.
    * Upload a `.dat` file from your dataset folder(MIT Arrhythmia Database or Supraventricular Arrhythmia Database) and click "Analyze".
    * **Note:** You must also have the corresponding `.hea` file (e.g., `208.hea`) in the *same folder* for the `wfdb` library to read the `.dat` file.
