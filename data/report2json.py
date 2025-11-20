import pandas as pd
import json

# Load the medical report dataset from CSV file
# This dataset contains radiology reports with associated image references and clinical information
df = pd.read_csv('./report.csv')

# Shuffle the dataset randomly with fixed seed for reproducible results
# Using random_state=42 ensures consistent shuffling across runs
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset into training (80%) and testing (20%) subsets
# This follows standard machine learning practice for model evaluation
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

def to_json(dataframe):
    """
    Transforms medical report DataFrame into structured JSON format for vision-language model training.
    
    Converts tabular medical data into conversation-style format with image references,
    creating appropriate prompt-response pairs for training LLaVA-style models.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame containing medical report data with columns:
            - image_files: semicolon-separated image filenames
            - Problems: patient's chief complaints
            - indication: clinical indications for the study
            - findings: radiological observations from the images
            - impression: diagnostic conclusions
    
    Returns:
        list: List of dictionaries formatted for LLaVA training, each containing:
            - id: unique zero-padded identifier
            - image: filename of associated chest X-ray
            - conversations: list of human-AI dialogue turns
            - domain: modality classification flags
            - type: data type identifier
    """
    data = []
    
    # Process each medical report case in the dataset
    for _, row in train_df.iterrows():
        # Extract and split image filenames (some cases have multiple views)
        image_path = row['image_files'].split(';')
        
        # Construct clinical prompt combining patient problems and study indications
        # This creates a realistic radiology consultation scenario for the model
        prompt = "You are a radiologist. Analyze the given chest X-ray based on the provided Problems and Indication. Describe the key Findings observed in the image that correspond to these inputs. Finally, provide an Impression summarizing the overall diagnostic conclusion. Chief Complaint: " + row['Problems'] + str(row['indication'])
        
        # Create data entry for primary/first image in the study
        row_dict1 = {
            "prompt": prompt,
            "input_image": './images/' + image_path[0],  # Full path to first image
            "response": f"Findings: {row['findings']} Impression: {row['impression']}"  # Ground truth radiology report
        }
        
        # Handle cases with multiple images (e.g., PA and lateral views)
        if len(image_path) > 1:
            row_dict2 = {
                "prompt": prompt,
                "input_image": './images/' + image_path[1],  # Path to secondary image
                "response": f"Findings: {row['findings']} Impression: {row['impression']}"  # Same report for additional view
            }
            data.append(row_dict2)
        
        # Add primary image entry to dataset
        data.append(row_dict1)
    
    # Convert intermediate format to final LLaVA-compatible JSON structure
    output = []
    for i, d in enumerate(data):
        output.append({
            "id": str(i).zfill(5),  # Zero-padded ID for consistent sorting and referencing
            "image": d["input_image"].split("/")[-1],  # Extract filename only from path
            "conversations": [
                {"from": "human", "value": d["prompt"]},  # Clinician's query/instruction
                {"from": "gpt", "value": d["response"]}   # AI's expected response (ground truth)
            ],
            # Domain classification for data filtering and model specialization
            "domain": {
                "chest_xray": True,   # Primary modality: chest radiography
                "mri": False,         # Not magnetic resonance imaging
                "ct_scan": False,     # Not computed tomography  
                "histology": False,   # Not histopathology
                "gross": False        # Not gross pathology
            },
            "type": "conversation"  # Data type for conversation-based training
        })
    
    return output

# Generate formatted training and test datasets
train = to_json(train_df)
test = to_json(test_df)

# Save training dataset to JSON file with proper Unicode handling
with open("./train_report.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

# Save test dataset to JSON file for model evaluation
with open("./test_report.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

# Output dataset statistics for verification
print(f"✅ Generated {len(train)} training entries and saved to train_report.json")
print(f"✅ Generated {len(test)} test entries and saved to test_report.json")
