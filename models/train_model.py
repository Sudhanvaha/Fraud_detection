import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_sequences(df):
    """Convert sequences to text for processing"""
    df['sequence_text'] = df['event_sequence'].apply(lambda x: ' '.join(x))
    return df

def create_features(df, vectorizer_type='tfidf', max_features=100):
    """Create features from sequences"""
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
    else:  # count vectorizer
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
    
    # Fit and transform the sequences
    features_matrix = vectorizer.fit_transform(df['sequence_text'])
    
    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    features_df = pd.DataFrame(features_matrix.toarray(), columns=feature_names)
    
    # Combine with original features
    feature_df = pd.concat([
        df[['sequence_length', 'total_duration']].reset_index(drop=True),
        features_df
    ], axis=1)
    
    return feature_df, vectorizer

def analyze_patterns(df):
    """Analyze common patterns in fraudulent vs legitimate claims"""
    print("=== PATTERN ANALYSIS ===")
    
    # Analyze sequence length
    fraud_lengths = df[df['is_fraud'] == 1]['sequence_length']
    legit_lengths = df[df['is_fraud'] == 0]['sequence_length']
    print(f"Fraudulent claims: {fraud_lengths.mean():.2f} ± {fraud_lengths.std():.2f} events")
    print(f"Legitimate claims: {legit_lengths.mean():.2f} ± {legit_lengths.std():.2f} events")
    
    # Analyze duration
    fraud_durations = df[df['is_fraud'] == 1]['total_duration']
    legit_durations = df[df['is_fraud'] == 0]['total_duration']
    print(f"Fraudulent claims: {fraud_durations.mean():.2f} ± {fraud_durations.std():.2f} days")
    print(f"Legitimate claims: {legit_durations.mean():.2f} ± {legit_durations.std():.2f} days")
    
    # Analyze common event patterns
    print("\nMost common sequences in fraudulent claims:")
    fraud_sequences = df[df['is_fraud'] == 1]['sequence_text'].value_counts().head(5)
    for seq, count in fraud_sequences.items():
        print(f"  {count} claims: {seq}")
    
    print("\nMost common sequences in legitimate claims:")
    legit_sequences = df[df['is_fraud'] == 0]['sequence_text'].value_counts().head(5)
    for seq, count in legit_sequences.items():
        print(f"  {count} claims: {seq}")

def train_models():
    # Load data
    df = pd.read_csv("data/synthetic_claims.csv")
    df['event_sequence'] = df['event_sequence'].apply(eval)  # Convert string to list
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraudulent claims: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Legitimate claims: {len(df) - df['is_fraud'].sum()} ({(1 - df['is_fraud'].mean())*100:.2f}%)")
    
    # Analyze patterns
    analyze_patterns(df)
    
    # Preprocess sequences
    df = preprocess_sequences(df)
    
    # Create features
    feature_df, vectorizer = create_features(df, vectorizer_type='tfidf', max_features=50)
    X = feature_df.values
    y = df['is_fraud'].values
    
    # Get feature names for later use
    feature_names = feature_df.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Isolation Forest for anomaly detection
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.3,  # Matches our fraud ratio
        random_state=42
    )
    iso_forest.fit(X_train_scaled)
    
    # Convert Isolation Forest predictions to binary labels (0=normal, 1=fraud)
    y_pred_iso = iso_forest.predict(X_test_scaled)
    y_pred_iso_binary = [1 if x == -1 else 0 for x in y_pred_iso]
    
    print("\n=== ISOLATION FOREST RESULTS ===")
    print(classification_report(y_test, y_pred_iso_binary))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_iso_binary):.4f}")
    
    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42, 
        class_weight='balanced',
        max_depth=10
    )
    rf_classifier.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf_classifier.predict(X_test_scaled)
    y_pred_proba_rf = rf_classifier.predict_proba(X_test_scaled)[:, 1]
    
    print("\n=== RANDOM FOREST RESULTS ===")
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    
    # Plot confusion matrix for Random Forest
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10))
    
    # Save models and feature names
    joblib.dump(vectorizer, "models/sequence_vectorizer.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(rf_classifier, "models/fraud_classifier.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    
    print("\nModels saved successfully!")
    
    return rf_classifier, scaler, vectorizer, feature_names

if __name__ == "__main__":
    train_models()