# Career-Path-Recommendation-System
An intelligent machine learning pipeline and recommendation system that predicts optimal career paths based on candidate skills, education, and background.
# 🎯 AI-Based Career Pathfinder & Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
The **AI Career Pathfinder** is an end-to-end Machine Learning pipeline designed to solve a core problem in human resources and career counseling: *Predicting the optimal career path for an individual based on their unique skill set, education level, and background.*

Rather than relying on basic rule-based filtering, this system utilizes advanced **Natural Language Processing (NLP)**, **Dimensionality Reduction**, and **Ensemble Learning** to provide highly accurate, multi-angle career recommendations.

---

## 🚀 Key Features & Pipeline

This project was built using a robust 5-stage Data Mining & Machine Learning pipeline:

1. **Intelligent Data Preprocessing:**
   - Applied `TF-IDF (Term Frequency-Inverse Document Frequency)` to extract mathematical weights from text-based skills, giving higher importance to rare/niche skills.
   - Handled categorical data using `LabelEncoder` (preserving ordinality in education levels) and `StandardScaler` for numerical consistency.

2. **Dimensionality Reduction (PCA):**
   - Compressed a highly sparse feature matrix (102 features) down to the most critical components explaining **90% of the variance** using Principal Component Analysis (PCA), significantly reducing computational noise.

3. **Pattern Discovery (Unsupervised Learning):**
   - Implemented **K-Means Clustering** (optimized via Silhouette Score & Elbow Method) to group candidates with similar latent profiles.
   - Utilized **Apriori Algorithm** to mine frequent skill itemsets and discover hidden co-occurrence rules among top technical skills.

4. **Content-Based Filtering (Recommender System):**
   - Engineered a custom recommendation engine using **Cosine Similarity**.
   - Calculates the exact distance between a candidate's profile and established career centroids to recommend the **Top 5 alternative careers** with match percentages.

5. **Predictive Modeling (Classification):**
   - Evaluated multiple algorithms (Random Forest, Gradient Boosting, KNN).
   - Deployed a highly tuned **Random Forest Classifier** with SMOTE for handling class imbalances.
   - **Performance:** Achieved an impressive **Top-3 Accuracy of ~86%**, ensuring realistic and flexible career predictions.

---

## 📊 Visual Insights & Architecture
*(Screenshots of System Architecture, Cluster Visualizations, and PCA Plots will be displayed here)*

> **Note:** This repository serves as a professional showcase of the system's architecture and engineering logic. The full proprietary dataset and source code are kept private.
