"""
=============================================================================
Project: AI-Based Career Pathfinder & Recommendation Engine
Module: Environment Setup & Core Dependencies
=============================================================================
"""

# 1. Data Manipulation & Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 2. Native Python Utils (Used for Custom Apriori Implementation)
from collections import Counter
from itertools import combinations

# 3. Data Preprocessing, NLP & Balancing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# 4. Unsupervised Learning & Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 5. Recommendation Engine (Vector Distance)
from sklearn.metrics.pairwise import cosine_similarity

# 6. Supervised Learning (Classification Models)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# 7. Validation & Performance Metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix

print("✅ All libraries loaded successfully. Pipeline is ready.")
