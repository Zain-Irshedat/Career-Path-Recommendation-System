"""
=============================================================================
Project: AI-Based Career Pathfinder & Recommendation Engine
Module: Data Preprocessing & Feature Engineering
=============================================================================
Description:
This script handles the transformation of raw candidate data into a 
mathematical format suitable for machine learning models. It combines 
Categorical Encoding, Feature Scaling, and Natural Language Processing (NLP).
=============================================================================
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------
# 1. Categorical Encoding (Education)
# ---------------------------------------------------------
# Preserving the ordinal nature of education levels (e.g., Bachelor < Master)
le_edu = LabelEncoder()
df['Education_Enc'] = le_edu.fit_transform(df['Education'])

print("✅ Education Encoding Mapping:")
for label, enc in zip(le_edu.classes_, le_edu.transform(le_edu.classes_)):
    print(f"   {label:<12} -> {enc}")

# ---------------------------------------------------------
# 2. Numerical Scaling (Age)
# ---------------------------------------------------------
# Standardizing age to zero mean and unit variance to prevent it from 
# dominating the distance metrics (like Cosine Similarity or K-Means).
scaler = StandardScaler()
df['Age_Scaled'] = scaler.fit_transform(df[['Age']])

# ---------------------------------------------------------
# 3. NLP Feature Extraction (Skills & Interests)
# ---------------------------------------------------------
# Combining text columns and applying TF-IDF to assign mathematical weights.
# sublinear_tf=True is used to scale down extremely frequent generic skills.
df['text_combined'] = (df['Skills'].str.replace(';', ' ') + ' ' + 
                       df['Interests'].str.replace(';', ' '))

tfidf = TfidfVectorizer(max_features=100, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(df['text_combined']).toarray()

print(f"\n✅ TF-IDF Matrix Extracted: Shape {X_tfidf.shape}")

# ---------------------------------------------------------
# 4. Final Feature Matrix Assembly
# ---------------------------------------------------------
# Horizontally stacking numerical and NLP features into a single matrix.
X_numeric = np.hstack([
    df['Age_Scaled'].values.reshape(-1, 1),
    df['Education_Enc'].values.reshape(-1, 1)
])
X_full = np.hstack([X_tfidf, X_numeric])

# Encode the Target Variable (Careers)
le_target = LabelEncoder()
y = le_target.fit_transform(df['Recommended_Career'])

print(f"\n✅ Final Feature Matrix (X_full) : {X_full.shape}")
print(f"✅ Target Vector (y)             : {y.shape} ({len(le_target.classes_)} unique career classes)")
