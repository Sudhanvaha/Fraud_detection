import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import StandardScaler
import joblib

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
    
    return feature_df

def save_models(doc2vec_model, scaler, file_prefix):
    """Save trained models"""
    doc2vec_model.save(f"{file_prefix}_doc2vec.model")
    joblib.dump(scaler, f"{file_prefix}_scaler.pkl")

def load_models(file_prefix):
    """Load trained models"""
    doc2vec_model = Doc2Vec.load(f"{file_prefix}_doc2vec.model")
    scaler = joblib.load(f"{file_prefix}_scaler.pkl")
    return doc2vec_model, scaler