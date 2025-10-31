import numpy as np
import pandas as pd

# === Step 0: Load dataset ===
df = pd.read_csv(r"C:\Users\pc\Desktop\prostratecancerdetection\dataset\prostate_cancer_prediction.csv")

print("✅ Dataset loaded")
print("Columns available:", df.columns.tolist())

# === Step 1: Define expected columns ===
symptom_cols = [
    'Difficulty_Urinating', 'Weak_Urine_Flow', 'Blood_in_Urine',
    'Pelvic_Pain', 'Back_Pain', 'Erectile_Dysfunction',
    'Exercise_Regularly', 'Healthy_Diet', 'Smoking_History',
    'Hypertension', 'Diabetes', 'Follow_Up_Required',
    'Genetic_Risk_Factors', 'Previous_Cancer_History',
    'Early_Detection', 'Survival_5_Years'
]

binary_map = {'Yes': 1, 'No': 0}
abnormal_map = {'Abnormal': 1, 'Normal': 0}

# === Step 2: Fix data types (only if columns exist) ===
if 'PSA_Level' in df.columns:
    df['PSA_Level'] = pd.to_numeric(df['PSA_Level'], errors='coerce')
    df = df.dropna(subset=['PSA_Level'])

if 'Biopsy_Result' in df.columns:
    df['Biopsy_Result'] = df['Biopsy_Result'].replace({'Yes': 'Benign'})  # Fix if mislabeled

# Binary mappings
for col in ['Family_History', 'Race_African_Ancestry']:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

if 'DRE_Result' in df.columns:
    df['DRE_Result'] = df['DRE_Result'].map(abnormal_map)

# Apply mapping to symptom columns if present
for col in symptom_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

if 'Cholesterol_Level' in df.columns:
    df['Cholesterol_Level'] = df['Cholesterol_Level'].map({'Normal': 0, 'High': 1})

# === Step 3: Reassign Biopsy_Result probabilistically ===
if all(c in df.columns for c in ['Age', 'PSA_Level', 'Family_History', 'DRE_Result', 'Race_African_Ancestry']):
    np.random.seed(42)
    age_norm = df['Age'] - 50
    logit = (
        -6.0
        + 0.05 * age_norm
        + 0.7 * df['Family_History'].fillna(0)
        + 1.2 * df['DRE_Result'].fillna(0)
        + 0.2 * df['PSA_Level']
        + 0.5 * df['Race_African_Ancestry'].fillna(0)
    )
    prob = 1 / (1 + np.exp(-logit))
    df['Biopsy_Result'] = np.random.binomial(1, prob).astype(int)
    df['Biopsy_Result'] = df['Biopsy_Result'].map({0: 'Benign', 1: 'Malignant'})

    # Step 4: Adjust PSA based on new label
    for idx, row in df.iterrows():
        if row['Biopsy_Result'] == 'Malignant':
            df.at[idx, 'PSA_Level'] = np.clip(
                np.random.lognormal(np.log(6.5), 0.8), 4, 30
            )
        else:
            df.at[idx, 'PSA_Level'] = np.clip(
                np.random.lognormal(np.log(2.5), 0.8), 0.5, 10
            )

# === Step 5: Assign stages, treatments, survival ===
df['Cancer_Stage'] = 'None'
df['Treatment_Recommended'] = 'None'
df['Survival_5_Years'] = 'Yes'

if 'Biopsy_Result' in df.columns:
    malignant_idx = df[df['Biopsy_Result'] == 'Malignant'].index
    stages = np.random.choice(
        ['Localized', 'Advanced', 'Metastatic'],
        size=len(malignant_idx),
        p=[0.7, 0.2, 0.1]
    )
    df.loc[malignant_idx, 'Cancer_Stage'] = stages

    stage_treat_map = {
        'Localized': ['Active Surveillance', 'Radiation', 'Surgery', 'Hormone Therapy'],
        'Advanced': ['Hormone Therapy', 'Chemotherapy', 'Radiation'],
        'Metastatic': ['Chemotherapy', 'Immunotherapy', 'Hormone Therapy']
    }

    for idx in malignant_idx:
        stage = df.at[idx, 'Cancer_Stage']
        df.at[idx, 'Treatment_Recommended'] = np.random.choice(stage_treat_map[stage])
        if stage == 'Localized':
            df.at[idx, 'Survival_5_Years'] = np.random.choice(['Yes', 'No'], p=[0.95, 0.05])
        elif stage == 'Advanced':
            df.at[idx, 'Survival_5_Years'] = np.random.choice(['Yes', 'No'], p=[0.8, 0.2])
        else:
            df.at[idx, 'Survival_5_Years'] = np.random.choice(['Yes', 'No'], p=[0.5, 0.5])

# === Step 6: Adjust symptoms based on stage ===
stage_prob_map = {'None': 0.1, 'Localized': 0.3, 'Advanced': 0.6, 'Metastatic': 0.8}
for col in symptom_cols[:6]:  # Only real symptoms
    if col in df.columns:
        probs = df['Cancer_Stage'].map(stage_prob_map)
        df[col] = np.random.binomial(1, probs)

# === Step 7: Early detection ===
if 'Early_Detection' in df.columns:
    df['Early_Detection'] = np.where(
        (df['Cancer_Stage'] == 'Localized') |
        ((df['Biopsy_Result'] == 'Benign') & (df['PSA_Level'] < 4)),
        'Yes', 'No'
    )

# === Step 8: Convert back to Yes/No ===
reverse_map = {1: 'Yes', 0: 'No'}
for col in symptom_cols + ['Family_History', 'Race_African_Ancestry']:
    if col in df.columns:
        df[col] = df[col].map(reverse_map)

if 'DRE_Result' in df.columns:
    df['DRE_Result'] = df['DRE_Result'].map({1: 'Abnormal', 0: 'Normal'})

if 'Cholesterol_Level' in df.columns:
    df['Cholesterol_Level'] = df['Cholesterol_Level'].map({0: 'Normal', 1: 'High'})

# === Step 9: Save final dataset ===
df.to_csv("tuned_prostate_cancer_prediction.csv", index=False)
print("💾 Tuned dataset saved as 'tuned_prostate_cancer_prediction.csv'")
print(df.head())
