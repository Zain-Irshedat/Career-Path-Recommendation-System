# Career-Path-Recommendation-System
An intelligent machine learning pipeline and recommendation system that predicts optimal career paths based on candidate skills, education, and background.
<div align="center">

# 🚀 AI-Powered Career Pathfinder & Recommendation Engine

*An Intelligent, Multi-Tiered Career Guidance System using NLP, Vector Similarity & Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Production--Ready-success?style=for-the-badge)]()

</div>

---

## 📌 Executive Summary

**Career Pathfinder AI** is a Machine Learning system designed to accurately map candidate skill sets, educational background, and experience to optimal career trajectories.

The platform utilizes a dual-engine architecture:
- **Primary Classification Engine:** Predicts the most dominant career fit using tuned Ensemble Learning.
- **Similarity Recommendation Engine:** Computes vector distances (Cosine Similarity) to provide top alternative career pivots with exact percentage matching.

---

## 🛠️ System Architecture

```text
[ Candidate Inputs: Skills, Education, Age ] 
                     │
                     ▼
  [ Feature Pipeline: TF-IDF + Ordinal Encoder + Scaler ]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 [ Primary Classifier ]  [ Recommendation Engine ]
 (Random Forest + SMOTE)   (Cosine Similarity Vector Space)
         │                       │
         ▼                       ▼
 🎯 Primary Career       💡 Top 5 Alternative Careers
   Prediction               & Match Percentage
