import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the base directory path
BASE_DIR = r"C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek"

# Create models directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

def create_advanced_features(df):
    """Create more sophisticated features from sequences"""
    df = df.copy()
    
    # Basic features
    df['sequence_length'] = df['event_sequence'].apply(len)
    
    # Event type counts
    for event in [
        'Expert Consultation', 'Witness Statement', 'Police Report Filed', 
        'Damage Documentation', 'Assessment'
    ]:
        df[f'count_{event.lower().replace(" ", "_")}'] = df['event_sequence'].apply(
            lambda x: sum(1 for e in x if e == event)
        )
    
    # Position features
    df['first_event'] = df['event_sequence'].apply(lambda x: x[0] if len(x) > 0 else '')
    df['last_event'] = df['event_sequence'].apply(lambda x: x[-1] if len(x) > 0 else '')
    
    # Pattern features
    df['has_assessment_before_documentation'] = df['event_sequence'].apply(
        lambda x: any('Assessment' in x[i] and 'Documentation' in x[i+1] 
                     for i in range(len(x)-1)) if len(x) > 1 else False
    )
    
    df['payment_before_assessment'] = df['event_sequence'].apply(
        lambda x: any('Payment' in x[i] and any('Assessment' in x[j] for j in range(i+1, len(x)))
                     for i in range(len(x)-1)) if len(x) > 1 else False
    )
    
    # Convert sequence to text for TF-IDF
    df['sequence_text'] = df['event_sequence'].apply(lambda x: ' '.join(x))
    
    return df

def create_sequence_features(df, vectorizer, use_tfidf=True):
    """Create features from sequences"""
    if use_tfidf:
        # TF-IDF features
        features_matrix = vectorizer.fit_transform(df['sequence_text'])
        feature_names = vectorizer.get_feature_names_out()
        features_df = pd.DataFrame(features_matrix.toarray(), columns=feature_names)
    else:
        # Binary presence features
        features_matrix = vectorizer.fit_transform(df['sequence_text'])
        feature_names = vectorizer.get_feature_names_out()
        features_df = pd.DataFrame(features_matrix.toarray() > 0, columns=feature_names).astype(int)
    
    # Add engineered features
    engineered_features = df[[
        'sequence_length', 
        'count_expert_consultation', 'count_witness_statement',
        'count_police_report_filed', 'count_damage_documentation',
        'count_assessment', 'has_assessment_before_documentation',
        'payment_before_assessment'
    ]]
    
    # Combine all features
    feature_df = pd.concat([engineered_features.reset_index(drop=True), features_df], axis=1)
    
    return feature_df, vectorizer

def train_enhanced_models():
    # Load data
    data_path = os.path.join(BASE_DIR, "data", "synthetic_claims.csv")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run data/generate_synthetic_data.py first to create the synthetic data.")
        return None, None, None, None
    
    df = pd.read_csv(data_path)
    df['event_sequence'] = df['event_sequence'].apply(eval)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraudulent claims: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Create sequence features
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9
    )
    
    X, vectorizer = create_sequence_features(df, vectorizer, use_tfidf=True)
    y = df['is_fraud'].values
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', {0: 1, 1: 3}]  # Give more weight to fraud class
    }
    
    # Train Random Forest with GridSearch
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Use best estimator
    best_rf = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test_scaled)
    y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
    
    print("\n=== ENHANCED RANDOM FOREST RESULTS ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    
    # Plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(BASE_DIR, "models", "precision_recall_curve.png"))
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Enhanced Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(BASE_DIR, "models", "confusion_matrix_enhanced.png"))
    plt.close()
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 15 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(15))
    
    # Create a separate directory for models (not the same as CSV file)
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save models to the models directory
    joblib.dump(vectorizer, os.path.join(model_dir, "enhanced_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "enhanced_scaler.pkl"))
    joblib.dump(best_rf, os.path.join(model_dir, "enhanced_fraud_classifier.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "enhanced_feature_names.pkl"))
    
    print(f"\nEnhanced models saved successfully to {model_dir}!")
    
    return best_rf, scaler, vectorizer, feature_names

if __name__ == "__main__":
    train_enhanced_models()