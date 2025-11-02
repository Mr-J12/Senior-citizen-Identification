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

## Model metrics & reports

When you run the training script (`model_training.py`) it computes and saves useful evaluation artifacts for both the age regression and gender classification tasks. The training script produces the following files inside a `reports/` directory (created automatically):

- `reports/training_history.csv` — per-epoch loss and metric values (CSV table).
- `reports/training_history.png` — plotted training & validation curves (loss, accuracy, MAE) as an image.
- `reports/final_metrics_summary.csv` — one-line summary of final metrics (train/val loss, final gender acc, final age MAE).
- `reports/gender_classification_report.txt` — human-readable classification report for gender (precision/recall/f1-support).
- `reports/gender_classification_report.csv` — same as above in CSV (structured) form.
- `reports/gender_confusion_matrix.png` — confusion matrix image for gender predictions.
- `reports/gender_confusion_matrix.csv` — confusion matrix as CSV.

How to generate these files

1. Ensure the UTKFace dataset is extracted to the `UTKFace/` folder and dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Run the training script (this will train the model and write the reports):

```bash
python model_training.py
```

3. After training completes you can open the `reports/` folder to view the plots and CSVs. Example quick checks in Python:

```python
import pandas as pd
pd.read_csv('reports/training_history.csv').tail()

# Print classification report
print(open('reports/gender_classification_report.txt').read())
```

Notes on interpretation

- Accuracy: check `final_gender_val_acc` in `reports/final_metrics_summary.csv` and the plotted validation accuracy curve in `training_history.png`.
- Loss: review `val_loss` (and per-head losses) in `training_history.csv` and the PNG plot.
- Classification report: contains precision, recall and f1-score per class (Male/Female). Use this to evaluate class-wise performance.
- Confusion matrix: shows the count of true vs predicted classes — useful to see systematic misclassification.

If you'd like, I can add example screenshots or embed the latest classification report and confusion matrix directly into this README once you've run training and share the resulting files here.

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