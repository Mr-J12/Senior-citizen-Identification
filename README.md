# Senior Citizen Identification System

**This project is an extension of my training project [Your Training Project Name Here], submitted for the [Internship Name] internship.**

## Problem Statement

This project implements a real-time system to detect persons in a video feed, identify their age and gender, and flag individuals over 60 as "Senior Citizens." All detections of senior citizens are logged with a timestamp, age, and gender into a `.csv` file.

This system builds on my training project by [Explain how it connects, e.g., "by taking the CNN classification skills I learned and applying them to a more complex, multi-output, real-time detection pipeline."].

## Methodology

The system works as a two-stage pipeline:

1.  **Face Detection:** I used a pre-trained SSD-based (Single Shot Detector) face detector from OpenCV's DNN module to find all faces in a frame.
2.  **Age & Gender Classification:** I "created my own model" by fine-tuning a pre-trained **MobileNetV2** (or ResNet, etc.) on the **UTKFace dataset**.
    * The original classification head was removed and replaced with two separate output heads: one for gender (binary classification) and one for age (regression).
    * The notebook used for this training can be found here: [Link to your public Model_Training.ipynb notebook].
3.  **Logging:** A Python script using OpenCV captures video, runs the pipeline on each frame, and logs all senior citizen data. The results are saved to `senior_citizen_log.csv` using Pandas.

## Results

* **Age Model (MAE):** The fine-tuned model achieved a Mean Absolute Error of **[Your MAE, e.g., 3.8] years** on the validation set.
* **Gender Model (Accuracy):** The model achieved **[Your Accuracy, e.g., 96%]%** accuracy on the validation set.
* **Real-Time Performance:** The system runs at approximately [Your FPS, e.g., 10-15] FPS on a [Your CPU/GPU] CPU.

*(Insert a screenshot or GIF of your project working here)*

## How to Run

1.  Clone the repository:
    ```bash
    git clone [Your Repo URL]
    cd senior-citizen-detection-project
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application (on webcam):
    ```bash
    python main.py
    ```
4.  Press 'q' to quit. The `senior_citizen_log.csv` file will be generated in the root directory.