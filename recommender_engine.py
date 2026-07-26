"""
=============================================================================
Project: AI-Based Career Pathfinder & Recommendation Engine
Module: Content-Based Recommender System (Cosine Similarity)
=============================================================================
Description:
This script computes the mathematical distance between a candidate's 
profile and established career paths. It uses Cosine Similarity on the 
vectorized NLP features and scaled numeric features to recommend the top 
5 alternative careers with exact match percentages.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_career_recommendations(user_profile_vector, career_centroids, target_encoder, top_n=5):
    """
    Calculates cosine similarity between a candidate's processed feature 
    vector (102 dimensions) and the aggregated career centroids (32 classes).
    
    Parameters:
    - user_profile_vector: Array containing TF-IDF weights + scaled numeric data.
    - career_centroids: The average feature vector for each target career.
    - target_encoder: The fitted LabelEncoder to decode career IDs back to text.
    - top_n: Number of top recommendations to return.
    """
    
    print("🔍 Calculating vector distances (Cosine Similarity)...")
    
    # 1. Compute Cosine Similarity
    # We use Cosine Similarity instead of Euclidean distance because it evaluates 
    # the angle/orientation of the skills, effectively prioritizing skill exact-matches
    # over raw magnitude in the sparse TF-IDF matrix.
    similarities = cosine_similarity(user_profile_vector, career_centroids)[0]
    
    # 2. Sort and extract Top N matches
    # np.argsort sorts ascending, so we use [::-1] to reverse it to descending order
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # 3. Decode and Format the Output
    recommendations = []
    for rank, idx in enumerate(top_indices, start=1):
        # Decode the numeric ID back to the readable Career Name
        career_name = target_encoder.inverse_transform([idx])[0]
        
        # Calculate matching percentage
        match_percentage = round(similarities[idx] * 100, 2)
        
        recommendations.append({
            "Rank": rank,
            "Recommended_Career": career_name, 
            "Match_Score": f"{match_percentage}%"
        })
        
    return pd.DataFrame(recommendations)

# =============================================================================
# Example of Expected Output:
#    Rank    Recommended_Career    Match_Score
# 0     1          Data Analyst         85.40%
# 1     2          BI Developer         78.20%
# 2     3        Data Scientist         74.10%
# 3     4     Machine Learning          65.50%
# 4     5      Database Admin           61.30%
# =============================================================================
