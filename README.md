# Goodreads Prediction Systems

## Overview
This project implements multiple predictive models on Goodreads review data,
covering implicit feedback, explicit ratings, and text-based classification.

## Tasks
1. **Read Prediction**
   - Predict whether a user will read a book (binary)
2. **Rating Prediction**
   - Predict star ratings using a latent factor model with bias terms
3. **Category Prediction**
   - Predict book genre from review text using TF-IDF features

## Methods
- Rating prediction: bias-based latent factor model with regularization
- Read prediction: popularity baseline + collaborative filtering heuristics
- Category prediction: TF-IDF (unigrams + bigrams) with linear classification

## Results
All models outperform provided baselines on held-out evaluation sets.

## Files
- `goodreads_models.py` — model implementations
- `method_overview.txt` — concise explanation of modeling decisions

## Notes
Dataset and evaluation scripts are not included due to size and licensing.
