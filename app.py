import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from preprocessing import create_embedding_features

# Set page config
st.set_page_config(
    page_title="Fraud Detection via Sequence Embeddings",
    page_icon="🔍",
    layout="wide"
)

# Load models (with caching)
@st.cache_resource
def load_models():
    try:
        doc2vec_model = Doc2Vec.load("models/claim_embedding_model.doc2vec")
        scaler = joblib.load("models/scaler.pkl")
        classifier = joblib.load("models/fraud_classifier.pkl")
        return doc2vec_model, scaler, classifier
    except FileNotFoundError:
        st.error("Models not found. Please run the training script first.")
        return None, None, None

# Load sample data for similarity analysis
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("data/synthetic_claims.csv")
        df['event_sequence'] = df['event_sequence'].apply(eval)
        return df
    except FileNotFoundError:
        return None

# Define event types for the dropdown
EVENT_TYPES = [
    "First Notice of Loss",
    "Initial Assessment",
    "Damage Documentation",
    "Witness Statement",
    "Police Report Filed",
    "Expert Consultation",
    "Repair Authorization",
    "Parts Ordered",
    "Repair In Progress",
    "Quality Check",
    "Final Assessment",
    "Payment Processing",
    "Claim Closed"
]

def find_similar_sequences(input_embedding, sample_df, doc2vec_model, top_n=5):
    """Find the most similar sequences in the training data"""
    # Get embeddings for all samples
    sample_embeddings = []
    for _, row in sample_df.iterrows():
        sample_embedding = doc2vec_model.infer_vector(row['event_sequence'])
        sample_embeddings.append(sample_embedding)
    
    # Calculate cosine similarities
    similarities = cosine_similarity([input_embedding], sample_embeddings)[0]
    
    # Get top N most similar sequences
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        results.append({
            'sequence': sample_df.iloc[idx]['event_sequence'],
            'similarity': similarities[idx],
            'is_fraud': sample_df.iloc[idx]['is_fraud'],
            'length': sample_df.iloc[idx]['sequence_length']
        })
    
    return results

def main():
    st.title("🔍 Fraud Detection via Sequence Embeddings")
    st.markdown("""
    This application detects potentially fraudulent insurance claims by analyzing 
    the sequence of events in a claim's history. The model uses Doc2Vec embeddings 
    to represent claim sequences and a classifier to identify anomalous patterns.
    """)
    
    # Load models and sample data
    doc2vec_model, scaler, classifier = load_models()
    sample_df = load_sample_data()
    
    if doc2vec_model is None:
        st.warning("Please train the models first by running `python models/train_model.py`")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Claim Sequence")
        
        # Initialize session state for events
        if 'events' not in st.session_state:
            st.session_state.events = []
        
        # Add event interface
        event_type = st.selectbox("Select event type:", EVENT_TYPES)
        add_event = st.button("Add Event")
        
        if add_event:
            st.session_state.events.append(event_type)
            st.success(f"Added: {event_type}")
        
        # Display current sequence
        if st.session_state.events:
            st.subheader("Current Sequence:")
            for i, event in enumerate(st.session_state.events, 1):
                st.write(f"{i}. {event}")
            
            # Clear sequence button
            if st.button("Clear Sequence"):
                st.session_state.events = []
                st.rerun()
        
        # Example sequences
        st.subheader("Example Sequences:")
        example_sequences = {
            "Normal Claim": [
                "First Notice of Loss",
                "Initial Assessment",
                "Damage Documentation",
                "Repair Authorization",
                "Repair In Progress",
                "Quality Check",
                "Final Assessment",
                "Payment Processing",
                "Claim Closed"
            ],
            "Suspicious Claim (Too Fast)": [
                "First Notice of Loss",
                "Payment Processing",
                "Claim Closed"
            ],
            "Suspicious Claim (Multiple Consultations)": [
                "First Notice of Loss",
                "Expert Consultation",
                "Expert Consultation",
                "Expert Consultation",
                "Payment Processing"
            ],
            "Custom Sequence": st.session_state.events.copy() if st.session_state.events else []
        }
        
        example = st.selectbox("Load example:", list(example_sequences.keys()))
        if st.button("Load Example") and example_sequences[example]:
            st.session_state.events = example_sequences[example]
            st.rerun()
    
    with col2:
        st.header("Analysis Results")
        
        if st.session_state.events:
            # Prepare the sequence for prediction
            sequence_df = pd.DataFrame({
                'event_sequence': [st.session_state.events],
                'sequence_length': [len(st.session_state.events)],
                'total_duration': [len(st.session_state.events) * 2]  # Estimate duration
            })
            
            # Create embeddings
            embedding_features = create_embedding_features(sequence_df, doc2vec_model)
            input_embedding = doc2vec_model.infer_vector(st.session_state.events)
            
            # Scale features
            scaled_features = scaler.transform(embedding_features)
            
            # Make prediction
            prediction = classifier.predict(scaled_features)[0]
            probability = classifier.predict_proba(scaled_features)[0][1]
            
            # Display results
            st.subheader("Prediction:")
            if prediction == 1:
                st.error(f"🚨 Fraudulent Claim (Probability: {probability:.2%})")
                
                # Provide explanation
                st.info("""
                **Potential reasons for fraud detection:**
                - Unusual sequence pattern
                - Missing expected events
                - Suspicious event repetitions
                - Illogical event ordering
                """)
            else:
                st.success(f"✅ Legitimate Claim (Probability: {1-probability:.2%})")
            
            # Show probability gauge
            st.subheader("Fraud Probability:")
            st.progress(float(probability))
            st.write(f"{probability:.2%}")
            
            # Show similar sequences from training data
            if sample_df is not None:
                st.subheader("Similar Sequences in Training Data:")
                similar_sequences = find_similar_sequences(input_embedding, sample_df, doc2vec_model)
                
                for i, similar in enumerate(similar_sequences, 1):
                    status = "🚨 Fraud" if similar['is_fraud'] else "✅ Normal"
                    st.write(f"{i}. Similarity: {similar['similarity']:.3f} - {status}")
                    st.write(f"   Length: {similar['length']} events")
                    with st.expander("View sequence"):
                        for j, event in enumerate(similar['sequence'], 1):
                            st.write(f"   {j}. {event}")
            
            # Show feature importance (if available)
            if hasattr(classifier, 'feature_importances_'):
                st.subheader("Top Influential Features:")
                feature_names = embedding_features.columns.tolist()
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': classifier.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                st.bar_chart(importance_df.set_index('feature'))
        
        else:
            st.info("Please add events to the sequence to analyze.")
    
    # Add some explanation
    st.markdown("---")
    st.subheader("How It Works:")
    st.markdown("""
    1. **Sequence Embedding**: Each claim sequence is converted to a fixed-length vector using Doc2Vec
    2. **Feature Engineering**: Additional features like sequence length and duration are added
    3. **Classification**: A trained model detects patterns indicative of fraud
    4. **Similarity Analysis**: Compares input sequence to known examples in training data
    
    Common fraud indicators:
    - Abnormally short sequences (claims processed too quickly)
    - Repeated events (multiple assessments, consultations)
    - Missing key events (no police report for large claims)
    - Unusual event ordering
    
    The model learns from diverse examples and can generalize to new sequences beyond the predefined patterns.
    """)

if __name__ == "__main__":
    main()