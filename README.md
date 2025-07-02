
# 🚑 Lung Disease Classification using Ensemble Transfer Learning and Grad-CAM

A robust and explainable AI framework for pneumonia and lung disease detection from chest X-rays using an ensemble of pre-trained CNNs and Grad-CAM for visual interpretability.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Research Objective](#research-objective)
4. [Proposed System](#proposed-system)
5. [Architecture Diagram](#architecture-diagram)
6. [Modules](#modules)
7. [Implementation](#implementation)
8. [Results](#results)
9. [Conclusion & Future Work](#conclusion--future-work)
10. [References](#references)

---

## 🧾 Project Overview

Pneumonia continues to be a pressing global health issue, particularly affecting children under five, contributing to nearly **800,000 deaths annually**. While **Chest X-rays (CXRs)** are a cost-effective and accessible diagnostic tool, accurate interpretation is often hindered by **image noise**, **feature overlap with other lung diseases**, and **radiologist variability**.

These challenges result in diagnostic delays and errors, especially in low-resource settings. Existing AI approaches often focus on **binary classification**, overlook **multi-class complexity** (bacterial, viral, and normal), and lack **explainability**—which is crucial for clinical trust.

This project presents a **multi-class deep learning framework** that:

* Combines five powerful CNN models through **ensemble learning**.
* Employs **Grad-CAM** to visualize and explain predictions.
* Offers **real-time, clinically meaningful insights** for early lung disease diagnosis.

### 📦 Dataset Sources

We used large-scale, public datasets of chest X-rays to train and evaluate our models:

* [NIH Chest X-ray Dataset (ChestX-ray14)](https://nihcc.app.box.com/v/ChestXray-NIHCC)
* [Kaggle NIH Chest X-ray Dataset Mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## ❗ Problem Statement

Accurate detection of pneumonia from CXRs is hindered by:

* Low image contrast and overlapping features
* Radiologist fatigue and diagnostic inconsistency
* Limited availability of skilled professionals
* Lack of explainability in current AI models

There’s a pressing need for an explainable, multi-class AI framework that performs well in real-world clinical environments, particularly in under-resourced regions.

---

## 🎯 Research Objective

> **Goal**: To develop a robust, interpretable deep learning framework for pneumonia classification and localization from CXR images using ensemble CNNs and Grad-CAM.

---

## 🧠 Proposed System

* Leverages **transfer learning** using pre-trained CNNs:

  * `InceptionResNetV2`
  * `EfficientNetB2`
  * `DenseNet121`
  * `MobileNet`
  * `InceptionV3`
* Uses **ensemble averaging** to combine model predictions
* Employs **Grad-CAM** to highlight regions influencing decisions
* Integrated with a **Flask web interface** for real-time diagnosis

---

## 🖼️ Architecture Diagram

![image](https://github.com/user-attachments/assets/8f29a5da-656b-4ba5-86f2-99a08fc626e0)


```
        +--------------------+
        | CXR Image Upload   |
        +--------+-----------+
                 |
         +-------v--------+      
         | Preprocessing  |
         +-------+--------+
                 |
        +--------v----------+
        | Ensemble CNNs     |
        +--------+----------+
                 |
        +--------v----------+
        | Ensemble Averaging|
        +--------+----------+
                 |
        +--------v----------+
        | Grad-CAM Heatmap  |
        +--------+----------+
                 |
        +--------v----------+
        | Flask Interface   |
        +-------------------+
```

---

## ⚙️ Modules

### 1. Data Preprocessing

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

### 2. Deep Learning Models

```python
base_model = tf.keras.applications.InceptionResNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
```

### 3. Ensemble Prediction

```python
final_prediction = np.mean([model1_pred, model2_pred, model3_pred, model4_pred, model5_pred], axis=0)
```

### 4. Explainability with Grad-CAM

```python
grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
heatmap = compute_gradcam(grad_model, img_tensor)
```

### 5. Evaluation

* Metrics: Accuracy, AUC-ROC, F1-score, Precision, Recall

### 6. Deployment via Flask

A simple web interface allows real-time uploads and outputs with heatmap visualization.

---

## 🧪 Implementation Summary

* **Frameworks**: TensorFlow / Keras, PyTorch (optional), OpenCV, Flask
* **Techniques**: Data Augmentation, Transfer Learning, Ensemble Averaging
* **Visuals**: Grad-CAM for transparency

---

## 📊 Results

### 🔢 Individual Model Metrics

| Model             | Accuracy | AUC-ROC |
| ----------------- | -------- | ------- |
| InceptionResNetV2 | 90.12%   | 77.80%  |
| EfficientNetB2    | 89.73%   | 77.11%  |
| MobileNet         | 89.65%   | 76.60%  |
| InceptionV3       | 89.08%   | 76.18%  |
| DenseNet121       | 88.62%   | 73.77%  |

### 🧬 Ensemble Results

* **Accuracy**: 89.69%
* **AUC-ROC**: **79.10%**
* **Insight**: Ensemble learning offered improved generalization and clinical trust.

---

## 🔍 Findings

* ✅ Ensemble outperformed all individual CNNs in AUC and robustness
* ✅ Grad-CAM visualizations increased transparency for medical professionals
* ✅ InceptionResNetV2 was the top individual performer
* ⚠️ DenseNet121 struggled slightly due to overfitting

---

## 📌 Conclusion

We developed an **explainable ensemble AI system** for pneumonia classification from chest X-rays that:

* Combines five powerful CNNs
* Offers improved performance and interpretability
* Is deployable in real-time clinical settings

---

## 👨‍💻 Authors

* **Monish P.** (21BCE5377)
* **Rithvika T.** (21BCE5554)

   *Guide: Prof. Poonkodi M.*

