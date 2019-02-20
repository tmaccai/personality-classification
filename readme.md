# Put all the files in the same directory

## requirements
sklearn, pickle, numpy, keras, sys, tensorflow 1.12


# Make sure run the following steps sequentially
# To save time, you can run steps[j:] where j in [1,2,3,4]

## steps

STEP 1. Data Preprocessing and Word Embedding

Command: python process_data.py
Input: GoogleNews-vectors-negative300.txt, essays.csv, mairesse.csv 
Output: essays_mairesse.p
Time Consumption: 5 hours or more

STEP 2. Feature Extraction with CNN

Command: python cnn_feature.py
Input: essays_mairesse.p
Output: EXT.p, NEU.p, AGR.p, OPN.p, CON.p (corresponding to five personality traits)
Time Consumption: 10 hours or more

STEP 3. TF-IDF Feature Extraction
Extraction of TF-IDF features consists of the following three steps, reading&preprocessing, vectorization, and saving.

Command: python tfidf_feature.py
Input essays_mairesse.p
Output: tfidf.p
Time Consumption: 1 hour


STEP 4. Classification

The following command outputs train and test accuracies for class openness using different models different features. If want to classify other classes, just replace OPN.p by other class data files (AGR.p, CON.p, EXT.p, NEU.p)

Command: python classification.py OPN.p tfidf.p
Input: tfidf.p, EXT.p, NEU.p, AGR.p, OPN.p, CON.p
Output: printed
Time Consumption: 20 min for each personality

