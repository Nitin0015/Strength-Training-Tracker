# **Strength Training Tracker**

This repository contains a project that explores automated tracking of free weight exercises using wristband accelerometer and gyroscope data. The goal is to develop machine learning models that mimic human personal trainers by identifying exercises, counting repetitions, and detecting improper form.

---

## **Overview**

Free weight strength training lacks comprehensive tracking mechanisms in wearable devices. This project addresses this gap by leveraging data from wristband sensors to analyze barbell exercises. Using supervised learning techniques, models are developed and evaluated to classify exercises, count repetitions, and detect improper form.

The dataset includes data from 5 participants performing various barbell exercises with medium to heavy weights. Models are trained on features extracted from accelerometer and gyroscope readings.

---

## **Dataset**

- **Source**: Data collected using the MbientLabs wristband sensor research kit.
- **Participants**: 5 individuals (4 males, 1 female) performing barbell exercises.
- **Exercises**:
  - Bench Press
  - Deadlift
  - Barbell Row
  - Squat
  - Overhead Press
- **Features**:
  - Accelerometer readings ($$x, y, z$$)
  - Gyroscope readings ($$x, y, z$$)
  - Derived features: PCA components, temporal abstractions, frequency-domain features.
- **Target Variables**:
  - Exercise type
  - Repetition count
  - Form classification (correct/improper).

---

## **Project Workflow**

1. **Data Collection**:
   - Wristband sensors recorded accelerometer (12.5 Hz) and gyroscope (25 Hz) data during exercise sessions.
   - Participants performed sets with varying repetitions (5 or 10) to explore model generalization across weights.

2. **Data Preprocessing**:
   - Outlier detection using Chauvenet's criterion.
   - Missing value imputation using interpolation.
   - Noise reduction with Butterworth low-pass filtering.

3. **Feature Engineering**:
   - Raw sensor data processed into enriched features (e.g., statistical moments, PCA).
   - Temporal and frequency-domain features added for better representation.

4. **Modeling**:
   - Supervised learning algorithms evaluated: Random Forests, Neural Networks, KNN, Naive Bayes, etc.
   - Feature selection optimized using forward selection techniques.

5. **Evaluation**:
   - Models assessed for accuracy in exercise classification, repetition counting, and form detection.
   - Metrics include accuracy, precision, recall, F1-score.

---

## **Results**

- **Exercise Classification**: Random Forest achieved an accuracy of 99.58%.
- **Repetition Counting**: Low-pass filtered accelerometer data yielded a miscount rate of ~5%.
- **Form Detection**: Random Forest identified improper bench press form with an accuracy of 98.53%.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/StrengthTrainingTracker.git
   cd StrengthTrainingTracker
   ```

2. Ensure that the dataset files (`sensor_data.csv`) are in the same directory as the code.

3. Open the Jupyter Notebook or script file and run all cells/steps to execute the code.

---

## **Acknowledgments**

- Data collected using [MbientLab Wristband Sensors](https://mbientlab.com/).
- Special thanks to participants for contributing to the dataset.
- References include prior research on wearable sensors and machine learning for activity recognition.

---
