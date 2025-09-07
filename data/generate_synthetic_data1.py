import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define event types
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

# Define some anomalous event patterns that might indicate fraud
FRAUD_PATTERNS = [
    ["First Notice of Loss", "Payment Processing", "Claim Closed"],  # Too fast
    ["First Notice of Loss", "Expert Consultation", "Expert Consultation", "Expert Consultation", "Payment Processing"],  # Multiple expert consultations
    ["First Notice of Loss", "Damage Documentation", "Repair Authorization", "Final Assessment", "Final Assessment", "Payment Processing"],  # Duplicate assessments
    ["First Notice of Loss", "Police Report Filed", "Witness Statement", "Witness Statement", "Witness Statement", "Payment Processing"],  # Multiple witness statements
]

NORMAL_PATTERNS = [
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Repair Authorization", "Repair In Progress", "Quality Check", "Final Assessment", "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Police Report Filed", "Expert Consultation", "Repair Authorization", "Repair In Progress", "Final Assessment", "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Witness Statement", "Parts Ordered", "Repair In Progress", "Quality Check", "Final Assessment", "Payment Processing", "Claim Closed"],
    ["First Notice of Loss", "Initial Assessment", "Damage Documentation", "Police Report Filed", "Expert Consultation", "Parts Ordered", "Repair In Progress", "Quality Check", "Final Assessment", "Payment Processing", "Claim Closed"],
]

def generate_claim_sequence(is_fraud):
    """Generate a sequence of events for a claim"""
    if is_fraud:
        pattern = random.choice(FRAUD_PATTERNS)
        # Add some random noise to fraud patterns
        if random.random() > 0.7:
            pattern.insert(random.randint(1, len(pattern)-1), random.choice(EVENT_TYPES))
    else:
        pattern = random.choice(NORMAL_PATTERNS)
    
    # Generate timestamps for each event
    start_date = datetime.now() - timedelta(days=random.randint(1, 365))
    sequence = []
    
    for i, event in enumerate(pattern):
        days_after_start = random.randint(0, 10) if i == 0 else random.randint(1, 5)
        start_date += timedelta(days=days_after_start)
        sequence.append({
            "event_type": event,
            "timestamp": start_date,
            "days_since_previous": days_after_start
        })
    
    return sequence

def generate_synthetic_dataset(num_claims=1000, fraud_ratio=0.2):
    """Generate a synthetic dataset of claims"""
    claims = []
    sequences = []
    labels = []
    
    num_fraud = int(num_claims * fraud_ratio)
    num_normal = num_claims - num_fraud
    
    # Generate fraudulent claims
    for i in range(num_fraud):
        claim_id = f"F_{i+1}"
        sequence = generate_claim_sequence(is_fraud=True)
        
        claims.append({
            "claim_id": claim_id,
            "is_fraud": 1,
            "sequence_length": len(sequence),
            "total_duration": sum(event["days_since_previous"] for event in sequence)
        })
        sequences.append([event["event_type"] for event in sequence])
        labels.append(1)
    
    # Generate normal claims
    for i in range(num_normal):
        claim_id = f"N_{i+1}"
        sequence = generate_claim_sequence(is_fraud=False)
        
        claims.append({
            "claim_id": claim_id,
            "is_fraud": 0,
            "sequence_length": len(sequence),
            "total_duration": sum(event["days_since_previous"] for event in sequence)
        })
        sequences.append([event["event_type"] for event in sequence])
        labels.append(0)
    
    # Create DataFrame
    df = pd.DataFrame(claims)
    df["event_sequence"] = sequences
    
    return df

if __name__ == "__main__":
    print("Generating synthetic claim data...")
    df = generate_synthetic_dataset(num_claims=2000)
    df.to_csv("synthetic_claims.csv", index=False)
    print(f"Generated {len(df)} claims ({df['is_fraud'].sum()} fraudulent)")
    print("Data saved to synthetic_claims.csv")