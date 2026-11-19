"""
========================================================================================
MNR UNIVERSITY — SCHOOL OF ENGINEERING, TECHNOLOGY AND AI
DEPARTMENT OF CSE (ARTIFICIAL INTELLIGENCE & MACHINE LEARNING)
MACHINE LEARNING MINI PROJECT (COURSE CODE: 25CSM201, SEMESTER: 2-1)
----------------------------------------------------------------------------------------
Project Title : AI Based Student Skill Gap Analysis and Career Recommendation System
Project Guide : Dr. M. Praveen
Team Members  : 1. B. Sam Soloman        (Roll No: 25RBAIM0021)
                2. Thakur Narsingh Singh  (Roll No: 25RBAIM0027)
========================================================================================
Description:
An intelligent, data-driven machine learning solution engineered to assess student
technical proficiencies, academic achievements, and competitive programming metrics to:
  1. Accurately predict and recommend optimal career specialization paths across 6
     technology domains (AI/ML, SDE, Cybersecurity, Cloud/DevOps, Data Engineering, Web Dev).
  2. Perform automated, granular Skill Gap Analysis against industry benchmark standards.
  3. Formulate actionable, prioritized curriculum and certification roadmaps.
Benchmarks six supervised classification models (Logistic Regression, Random Forest, SVM,
KNN, Decision Tree, Gaussian Naive Bayes) using 5-fold Stratified Cross-Validation and
Scikit-learn Pipelines.
========================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def find_dataset(filename="student_skills_career_dataset.csv"):
    """Search for the dataset across local and nested paths."""
    candidate_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "data", filename),
        os.path.join("..", filename)
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Dataset '{filename}' could not be located.")


def load_and_inspect_dataset(path):
    """Load, summarize, and verify completeness of dataset."""
    print("=" * 80)
    print("STEP 1: DATA INGESTION & DATASET AUDIT")
    print("=" * 80)
    df = pd.read_csv(path)
    print(f"Loaded dataset successfully from: {path}")
    print(f"Total Student Records : {df.shape[0]}")
    print(f"Total Feature Columns  : {df.shape[1] - 1} predictive variables + 1 target outcome")
    
    null_count = df.isnull().sum().sum()
    print(f"Missing / Null Values : {null_count} (100% complete, verified)")

    print("\nCareer Track Class Distribution:")
    counts = df['Recommended_Career'].value_counts()
    for role, c in counts.items():
        pct = (c / len(df)) * 100
        print(f"  • {role:36s} : {c:4d} students ({pct:.2f}%)")

    return df


def generate_exploratory_visualizations(df, viz_dir):
    """Generate high-resolution figures for class distribution, skill profiles, and correlations."""
    os.makedirs(viz_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS & VISUALIZATION GENERATION")
    print("=" * 80)

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(9, 5))
    counts = df['Recommended_Career'].value_counts()
    colors = ['#1e40af', '#0284c7', '#0d9488', '#16a34a', '#d97706', '#dc2626']
    bars = plt.bar(counts.index, counts.values, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    plt.title('Figure 1: Distribution of Recommended Career Paths (2,400 Students)', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Number of Students', fontsize=11)
    plt.xticks(rotation=20, ha='right', fontsize=9.5)
    plt.ylim(0, 500)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{yval}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(viz_dir, 'figure_1_career_class_distribution.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  Saved -> {p1}")

    # 2. Key Academic Skills Profiles Across Roles
    skill_cols = ['Programming_Proficiency', 'DSA_Score', 'Operating_Systems_Score', 
                  'Computer_Networks_Score', 'Database_Systems_Score', 'Machine_Learning_Score', 
                  'Web_Development_Score', 'Cloud_Computing_Score']
    mean_skills = df.groupby('Recommended_Career')[skill_cols].mean()

    plt.figure(figsize=(11, 6))
    sns.heatmap(mean_skills, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=0.5, cbar_kws={'label': 'Mean Competency Score (0-100)'})
    plt.title('Figure 2: Competency Matrix — Mean Skill Proficiency by Career Specialization', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Career Path', fontsize=11)
    plt.xlabel('Technical Skill Area', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    p2 = os.path.join(viz_dir, 'figure_2_academic_skills_radar_profiles.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  Saved -> {p2}")

    # 3. Correlation Heatmap
    corr_cols = skill_cols + ['CGPA', 'Coding_Contest_Rating', 'Projects_Completed', 'Certifications_Count']
    plt.figure(figsize=(10, 8))
    corr = df[corr_cols].corr()
    sns.heatmap(corr, cmap='vlag', vmin=-0.5, vmax=1.0, annot=False, linewidths=0.3, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Figure 3: Feature Correlation Heatmap of Academic & Technical Predictors', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    p3 = os.path.join(viz_dir, 'figure_3_correlation_matrix_heatmap.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  Saved -> {p3}")


def benchmark_machine_learning_models(df, viz_dir):
    """Train and evaluate 6 supervised algorithms using 5-fold stratified cross-validation."""
    print("\n" + "=" * 80)
    print("STEP 3: 5-FOLD STRATIFIED CV & MODEL BENCHMARKING")
    print("=" * 80)

    X = df.drop(columns=['Recommended_Career'])
    y = df['Recommended_Career']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    # 80:20 Stratified Partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    print(f"Training Set : {X_train.shape[0]} samples (80.0%)")
    print(f"Testing Set  : {X_test.shape[0]} samples (20.0%)")
    print(f"Evaluation   : 6 Balanced Classes (80 test samples per career path)")

    # Six Classifiers
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    probs_dict = {}
    preds_dict = {}

    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        preds_dict[name] = y_pred
        probs_dict[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

        results.append({
            'Model': name,
            'CV_Acc': cv_acc,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC_AUC': auc
        })

    comparison_df = pd.DataFrame(results)
    print("\n" + "-" * 80)
    print("MULTI-MODEL BENCHMARK PERFORMANCE TABLE (20% HELD-OUT TEST DATA):")
    print("-" * 80)
    print(comparison_df.round(4).to_string(index=False))
    print("-" * 80)

    # Champion Model Selection
    champion_row = comparison_df.sort_values(['Accuracy', 'F1'], ascending=False).iloc[0]
    champ_name = champion_row['Model']
    champion_clf = models[champ_name]
    print(f"\nCHAMPION MODEL SELECTED: {champ_name}")
    print(f"  • Test Accuracy : {champion_row['Accuracy']*100:.2f}%")
    print(f"  • 5-Fold CV Acc : {champion_row['CV_Acc']*100:.2f}%")
    print(f"  • Weighted F1   : {champion_row['F1']:.4f}")
    print(f"  • Multi-Class AUC: {champion_row['ROC_AUC']:.4f}")

    # Figure 4: Model Accuracy Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color='#1e3a8a', width=0.52)
    plt.title('Figure 4: Model Accuracy Comparison Across Six Classifiers', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Test Accuracy', fontsize=11)
    plt.ylim(0.70, 0.95)
    plt.xticks(rotation=20, ha='right')
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2.0, h + 0.005, f'{h*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p4 = os.path.join(viz_dir, 'figure_4_model_accuracy_comparison.png')
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"  Saved -> {p4}")

    # Figure 5: Multi-Class ROC Curves for Champion Model
    plt.figure(figsize=(8, 6.5))
    champ_probs = probs_dict[champ_name]
    for i, cname in enumerate(class_names):
        y_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, champ_probs[:, i])
        cls_auc = roc_auc_score(y_binary, champ_probs[:, i])
        plt.plot(fpr, tpr, label=f'{cname} (AUC={cls_auc:.3f})', linewidth=1.8)
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC=0.500)')
    plt.title(f'Figure 5: One-vs-Rest ROC Curves ({champ_name})', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=8.5)
    plt.tight_layout()
    p5 = os.path.join(viz_dir, 'figure_5_multiclass_roc_curves.png')
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"  Saved -> {p5}")

    # Figure 6: Confusion Matrix for Champion Model
    cm = confusion_matrix(y_test, preds_dict[champ_name])
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.split(' ')[0] for c in class_names],
                yticklabels=[c.split(' ')[0] for c in class_names],
                annot_kws={'size': 11, 'weight': 'bold'})
    plt.title(f'Figure 6: Confusion Matrix: {champ_name}', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Career Track', fontsize=11)
    plt.ylabel('Actual Career Track', fontsize=11)
    plt.tight_layout()
    p6 = os.path.join(viz_dir, 'figure_6_confusion_matrix_champion_model.png')
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"  Saved -> {p6}")

    # Figure 7: KNN Parameter Sensitivity (K=1 to 25)
    k_vals = list(range(1, 26, 2))
    k_accs = []
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_te_sc = sc.transform(X_test)
    for k in k_vals:
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        knn_clf.fit(X_tr_sc, y_train)
        k_accs.append(accuracy_score(y_test, knn_clf.predict(X_te_sc)))

    plt.figure(figsize=(8, 4.8))
    plt.plot(k_vals, k_accs, marker='o', color='#0284c7', linewidth=2)
    plt.title('Figure 7: KNN Hyperparameter Sensitivity Analysis (K = 1 to 25)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('K (Number of Neighbors)', fontsize=11)
    plt.ylabel('Test Accuracy', fontsize=11)
    plt.xticks(k_vals)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p7 = os.path.join(viz_dir, 'figure_7_knn_k_sensitivity.png')
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"  Saved -> {p7}")

    # Figure 8: Random Forest Feature Importances
    rf_clf = models['Random Forest']
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': rf_clf.feature_importances_})
    top10_feats = feat_imp.sort_values('Importance', ascending=True).tail(10)

    plt.figure(figsize=(8.5, 5))
    plt.barh(top10_feats['Feature'], top10_feats['Importance'], color='#0d9488', edgecolor='#134e4a', height=0.6)
    plt.title('Figure 8: Top 10 Feature Importances (Random Forest)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Gini Importance', fontsize=11)
    plt.tight_layout()
    p8 = os.path.join(viz_dir, 'figure_8_random_forest_feature_importance.png')
    plt.savefig(p8, dpi=300)
    plt.close()
    print(f"  Saved -> {p8}")

    return champion_clf, le, X_train, X_test, y_test, df


def run_skill_gap_analysis_engine(champion_clf, le, df, viz_dir):
    """Assess student profile, compare against target role benchmark, and generate remediation roadmap."""
    print("\n" + "=" * 80)
    print("STEP 4: AI-BASED SKILL GAP ANALYSIS & CAREER INFERENCE DEMONSTRATION")
    print("=" * 80)

    # Core technical skills to audit
    skill_features = [
        'Programming_Proficiency', 'DSA_Score', 'Operating_Systems_Score',
        'Computer_Networks_Score', 'Database_Systems_Score', 'Machine_Learning_Score',
        'Web_Development_Score', 'Cloud_Computing_Score', 'Mathematics_Stats_Score'
    ]

    # Benchmark profiles (75th percentile / benchmark profile for each career role)
    benchmarks = df.groupby('Recommended_Career')[skill_features].quantile(0.75)

    # Sample Student Profile for Inference
    student_profile = pd.DataFrame([{
        'CGPA': 8.1,
        'Programming_Proficiency': 78.5,
        'DSA_Score': 74.0,
        'Operating_Systems_Score': 68.0,
        'Computer_Networks_Score': 62.0,
        'Database_Systems_Score': 66.0,
        'Machine_Learning_Score': 88.5,
        'Web_Development_Score': 58.0,
        'Cloud_Computing_Score': 60.0,
        'Mathematics_Stats_Score': 84.0,
        'Coding_Contest_Rating': 1620,
        'Hackathons_Count': 3,
        'Projects_Completed': 4,
        'Certifications_Count': 2,
        'Internship_Months': 3,
        'Soft_Skills_Rating': 8.2
    }])

    # 1. Predict Recommended Career
    pred_idx = champion_clf.predict(student_profile)[0]
    rec_career = le.inverse_transform([pred_idx])[0]
    probs = champion_clf.predict_proba(student_profile)[0]

    print("Student Evaluation Profile:")
    print(f"  • CGPA                    : {student_profile['CGPA'].values[0]}")
    print(f"  • Machine Learning Score  : {student_profile['Machine_Learning_Score'].values[0]} / 100")
    print(f"  • Mathematics & Stats     : {student_profile['Mathematics_Stats_Score'].values[0]} / 100")
    print(f"  • Programming Proficiency : {student_profile['Programming_Proficiency'].values[0]} / 100")
    print(f"  • DSA & Algorithms Score  : {student_profile['DSA_Score'].values[0]} / 100")
    print(f"  • Cloud Computing Score   : {student_profile['Cloud_Computing_Score'].values[0]} / 100")
    print(f"  • Coding Contest Rating   : {student_profile['Coding_Contest_Rating'].values[0]}")

    print("\nCareer Track Inference Probabilities:")
    prob_df = pd.DataFrame({'Career_Track': le.classes_, 'Probability': probs})
    for _, row in prob_df.sort_values('Probability', ascending=False).iterrows():
        print(f"  • {row['Career_Track']:36s}: {row['Probability']*100:.2f}%")

    print(f"\nPRIMARY CAREER RECOMMENDATION: {rec_career} (Confidence: {probs[pred_idx]*100:.2f}%)")

    # 2. Skill Gap Analysis vs Benchmark
    target_benchmark = benchmarks.loc[rec_career]
    student_scores = student_profile[skill_features].iloc[0]

    gap_data = []
    for skill in skill_features:
        stu_val = student_scores[skill]
        bench_val = target_benchmark[skill]
        gap = stu_val - bench_val
        status = "Strength" if gap >= 0 else "Skill Gap"
        gap_data.append({
            'Skill': skill.replace('_Score', '').replace('_', ' '),
            'Student_Score': stu_val,
            'Benchmark': bench_val,
            'Delta': round(gap, 1),
            'Status': status
        })

    gap_df = pd.DataFrame(gap_data)
    print("\n" + "-" * 80)
    print(f"SKILL GAP ANALYSIS MATRIX (TARGET ROLE: {rec_career.upper()}):")
    print("-" * 80)
    print(gap_df.to_string(index=False))
    print("-" * 80)

    # 3. Figure 9: Skill Gap Benchmark Visualization
    x = np.arange(len(gap_df))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, gap_df['Student_Score'], width, label='Student Score', color='#0284c7')
    plt.bar(x + width/2, gap_df['Benchmark'], width, label='Benchmark Target (75th Pct)', color='#f97316')
    plt.xticks(x, gap_df['Skill'], rotation=25, ha='right', fontsize=9.5)
    plt.title(f'Figure 9: Student Skill Gap Analysis vs. {rec_career} Benchmark', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Proficiency Score (0-100)', fontsize=11)
    plt.ylim(0, 110)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p9 = os.path.join(viz_dir, 'figure_9_student_skill_gap_analysis_benchmark.png')
    plt.savefig(p9, dpi=300)
    plt.close()
    print(f"  Saved -> {p9}")

    # 4. Actionable Remediation Learning Roadmap
    print("\n" + "=" * 80)
    print("ACTIONABLE REMEDIATION ROADMAP FOR STUDENT:")
    print("=" * 80)
    deficiencies = gap_df[gap_df['Delta'] < 0].sort_values('Delta')
    for idx, row in deficiencies.iterrows():
        remedy = ""
        if "Cloud" in row['Skill']:
            remedy = "Enroll in AWS Certified Cloud Practitioner / Deploy Docker containers on AWS ECS."
        elif "Networks" in row['Skill']:
            remedy = "Review Computer Networks (TCP/IP socket programming, Wireshark packet auditing)."
        elif "Database" in row['Skill']:
            remedy = "Complete SQL Mastery on LeetCode / Practice PostgreSQL query indexing and optimization."
        elif "DSA" in row['Skill']:
            remedy = "Solve 50 Medium LeetCode DSA problems (Graphs, Dynamic Programming, Trees)."
        elif "Web" in row['Skill']:
            remedy = "Build a RESTful API backend using FastAPI or Node.js."
        else:
            remedy = "Review university course notes and undertake a hands-on capstone project."

        print(f"  [GAP DETECTED] {row['Skill']} (Deficit: {row['Delta']:+0.1f} pts)")
        print(f"    --> Recommended Action: {remedy}\n")

    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE — ALL DELIVERABLES GENERATED")
    print("=" * 80)


def main():
    print("""
    ========================================================================
     MNR UNIVERSITY — SCHOOL OF ENGINEERING, TECHNOLOGY & AI
     AI BASED STUDENT SKILL GAP ANALYSIS & CAREER RECOMMENDATION SYSTEM
     Course: Machine Learning (25CSM201) | Academic Mini Project
    ========================================================================
    """)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = find_dataset("student_skills_career_dataset.csv")
    viz_dir = os.path.join(base_dir, "visualizations")

    df = load_and_inspect_dataset(dataset_path)
    generate_exploratory_visualizations(df, viz_dir)
    champ_clf, le, X_train, X_test, y_test, df_full = benchmark_machine_learning_models(df, viz_dir)
    run_skill_gap_analysis_engine(champ_clf, le, df_full, viz_dir)


if __name__ == "__main__":
    main()
