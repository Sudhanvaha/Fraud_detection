import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_sequences(df):
    """Prepare sequences for Doc2Vec training"""
    tagged_docs = []
    for i, row in df.iterrows():
        tagged_docs.append(TaggedDocument(words=row['event_sequence'], tags=[str(i)]))
    return tagged_docs

def train_doc2vec(tagged_docs, vector_size=100, window=5, epochs=40):
    """Train a Doc2Vec model on the sequences"""
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        epochs=epochs
    )
    
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model

def create_embedding_features(df, doc2vec_model):
    """Create embedding features for each claim"""
    embeddings = []
    for i, row in df.iterrows():
        embedding = doc2vec_model.infer_vector(row['event_sequence'])
        embeddings.append(embedding)
    
    # Convert to DataFrame
    embedding_cols = [f'embedding_{i}' for i in range(len(embeddings[0]))]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
    # Combine with original features
    feature_df = pd.concat([
        df[['sequence_length', 'total_duration']].reset_index(drop=True),
        embedding_df
    ], axis=1)
    
    return feature_df, embedding_cols

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
    fraud_sequences = df[df['is_fraud'] == 1]['event_sequence'].apply(lambda x: ' '.join(x)).value_counts().head(5)
    for seq, count in fraud_sequences.items():
        print(f"  {count} claims: {seq}")
    
    print("\nMost common sequences in legitimate claims:")
    legit_sequences = df[df['is_fraud'] == 0]['event_sequence'].apply(lambda x: ' '.join(x)).value_counts().head(5)
    for seq, count in legit_sequences.items():
        print(f"  {count} claims: {seq}")

def train_models():
    # Load data
    df = pd.read_csv(r"C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek\data\synthetic_claims.csv")
    df['event_sequence'] = df['event_sequence'].apply(eval)  # Convert string to list
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraudulent claims: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Legitimate claims: {len(df) - df['is_fraud'].sum()} ({(1 - df['is_fraud'].mean())*100:.2f}%)")
    
    # Analyze patterns
    analyze_patterns(df)
    
    # Prepare sequences and train Doc2Vec
    tagged_docs = prepare_sequences(df)
    doc2vec_model = train_doc2vec(tagged_docs, vector_size=100)
    
    # Create features
    feature_df, embedding_cols = create_embedding_features(df, doc2vec_model)
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
    plt.savefig(r'C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek\models\confusion_matrix.png')
    plt.close()
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10))
    
    # Save models
    doc2vec_model.save(r"C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek\models\claim_embedding_model.doc2vec")
    joblib.dump(scaler, r"C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek\models\scaler.pkl")
    joblib.dump(rf_classifier, r"C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek\models\fraud_classifier.pkl")
    joblib.dump(feature_names, r"C:\Users\sudha\OneDrive\Desktop\My_workspace\deepseek\models\feature_names.pkl")
    
    print("\nModels saved successfully!")
    
    return rf_classifier, scaler, doc2vec_model, feature_names

if __name__ == "__main__":
    train_models()