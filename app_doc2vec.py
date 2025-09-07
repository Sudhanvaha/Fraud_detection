import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import joblib
from sklearn.metrics.pairwise import cosine_similarity

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
        feature_names = joblib.load("models/feature_names.pkl")
        return doc2vec_model, scaler, classifier, feature_names
    except FileNotFoundError:
        st.error("Models not found. Please run the training script first.")
        return None, None, None, None

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

# Define comprehensive examples
FRAUD_EXAMPLES = [
    ["First Notice of Loss", "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Expert Consultation", "Expert Consultation", "Expert Consultation", "Payment Processing"],
    ["First Notice of Loss", "Initial Assessment", "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Payment Processing", "Damage Documentation", "Claim Closed"],
    ["First Notice of Loss", "Police Report Filed", "Payment Processing"],
]

LEGITIMATE_EXAMPLES = [
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Repair Authorization", 
     "Repair In Progress", "Quality Check", "Final Assessment", "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Police Report Filed", 
     "Expert Consultation", "Repair Authorization", "Repair In Progress", "Final Assessment", 
     "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Final Assessment", 
     "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Witness Statement", 
     "Parts Ordered", "Repair In Progress", "Quality Check", "Final Assessment", "Payment Processing", "Claim Closed"],
]

def create_features_from_sequence(sequence, doc2vec_model, sequence_length, feature_names):
    """Create features from a sequence for prediction using Doc2Vec"""
    # Get the embedding for the sequence
    embedding = doc2vec_model.infer_vector(sequence)
    
    # Create feature DataFrame
    embedding_df = pd.DataFrame([embedding], columns=[f'embedding_{i}' for i in range(len(embedding))])
    
    # Add sequence length and duration features
    embedding_df['sequence_length'] = sequence_length
    embedding_df['total_duration'] = sequence_length * 2  # Estimate duration
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in embedding_df.columns:
            embedding_df[feature] = 0
    
    # Reorder columns to match training data
    embedding_df = embedding_df[feature_names]
    
    return embedding_df, embedding

def find_similar_sequences(input_embedding, sample_df, doc2vec_model, top_n=5):
    """Find the most similar sequences in the training data using Doc2Vec embeddings"""
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
    st.title("🔍 Fraud Detection via Sequence Embeddings (Doc2Vec)")
    st.markdown("""
    This application detects potentially fraudulent insurance claims by analyzing 
    the sequence of events in a claim's history. The model uses Doc2Vec embeddings
    to represent claim sequences and a classifier to identify anomalous patterns.
    """)
    
    # Load models and sample data
    doc2vec_model, scaler, classifier, feature_names = load_models()
    sample_df = load_sample_data()
    
    if doc2vec_model is None:
        st.warning("Please train the models first by running `python models/train_model_doc2vec.py`")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Sequence Analysis", "Examples", "How It Works"])
    
    with tab1:
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
            
            # Quick load buttons
            st.subheader("Quick Load Examples")
            col1a, col2a = st.columns(2)
            
            with col1a:
                st.write("Fraudulent Examples:")
                for i, example in enumerate(FRAUD_EXAMPLES[:3], 1):
                    if st.button(f"Fraud Example {i}", key=f"fraud_{i}"):
                        st.session_state.events = example
                        st.rerun()
            
            with col2a:
                st.write("Legitimate Examples:")
                for i, example in enumerate(LEGITIMATE_EXAMPLES[:3], 1):
                    if st.button(f"Legit Example {i}", key=f"legit_{i}"):
                        st.session_state.events = example
                        st.rerun()
        
        with col2:
            st.header("Analysis Results")
            
            if st.session_state.events:
                # Create features from the sequence
                sequence_length = len(st.session_state.events)
                features, input_embedding = create_features_from_sequence(
                    st.session_state.events, doc2vec_model, sequence_length, feature_names
                )
                
                # Scale features
                scaled_features = scaler.transform(features)
                
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
                    - Unusually short sequence
                    - Missing critical events (assessment, documentation)
                    - Suspicious event repetitions
                    - Illogical event ordering
                    - Rapid claim processing
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
            
            else:
                st.info("Please add events to the sequence to analyze.")
    
    with tab2:
        st.header("Fraudulent Claim Examples")
        st.write("These are patterns that typically indicate fraudulent claims:")
        
        for i, example in enumerate(FRAUD_EXAMPLES, 1):
            st.write(f"**Example {i}:**")
            for j, event in enumerate(example, 1):
                st.write(f"  {j}. {event}")
            st.write("**Why it's suspicious:**")
            if i == 1:
                st.write("  - Too few events, skipping critical steps like assessment and documentation")
            elif i == 2:
                st.write("  - Excessive expert consultations without other necessary events")
            elif i == 3:
                st.write("  - Missing damage documentation step")
            elif i == 4:
                st.write("  - Illogical ordering: payment before damage documentation")
            elif i == 5:
                st.write("  - Only police report, missing assessment and documentation")
            st.write("---")
        
        st.header("Legitimate Claim Examples")
        st.write("These are patterns that typically indicate legitimate claims:")
        
        for i, example in enumerate(LEGITIMATE_EXAMPLES, 1):
            st.write(f"**Example {i}:**")
            for j, event in enumerate(example, 1):
                st.write(f"  {j}. {event}")
            st.write("**Why it's legitimate:**")
            if i == 1:
                st.write("  - Complete process with all necessary steps")
            elif i == 2:
                st.write("  - Includes investigation steps (police report, expert consultation)")
            elif i == 3:
                st.write("  - Standard process for minor claims")
            elif i == 4:
                st.write("  - Includes witness statement and parts ordering")
            st.write("---")
    
    with tab3:
        st.header("How It Works:")
        st.markdown("""
        ## Doc2Vec Approach
        
        1. **Sequence Embedding**: Each claim sequence is converted to a fixed-length vector using Doc2Vec
        2. **Feature Engineering**: Additional features like sequence length and duration are added
        3. **Classification**: A trained model detects patterns indicative of fraud
        4. **Similarity Analysis**: Compares input sequence to known examples in training data
        
        **How Doc2Vec works:**
        - Treats each event as a "word" and each sequence as a "document"
        - Lears distributed representations of events in a continuous vector space
        - Captures semantic relationships between events
        - Similar sequences have similar vector representations
        
        **Common fraud indicators:**
        - Abnormally short sequences (claims processed too quickly)
        - Repeated events (multiple assessments, consultations)
        - Missing key events (no damage documentation, no assessment)
        - Unusual event ordering (payment before assessment)
        - Suspicious patterns (only investigation events without repair/assessment)
        
        **Legitimate claim characteristics:**
        - Logical progression of events
        - Complete documentation process
        - Appropriate investigation steps when needed
        - Reasonable processing time
        
        The model learns from diverse examples and can generalize to new sequences.
        """)

if __name__ == "__main__":
    main()