import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='transfer report into json')
parser.add_argument('--report_file', type=str, required=False, help='Report File', default='./report.csv')
parser.add_argument('--seed', type=str, required=False, help='Seed', default=42)
args = parser.parse_args()
report_file = args.report_file
seed = args.seed

# Load the medical report dataset from CSV file
# This dataset contains radiology reports with associated image references and clinical information
df = pd.read_csv(report_file)

# Shuffle the dataset randomly with fixed seed for reproducible results
# Using random_state ensures consistent shuffling across runs
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

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
    for _, row in dataframe.iterrows():
        # Extract and split image filenames (some cases have multiple views)
        image_path = row['image_files'].split(';')
        
        # Construct clinical prompt combining patient problems and study indications
        # This creates a realistic radiology consultation scenario for the model
        prompt = "You are a radiologist. Analyze the given chest X-ray based on the provided Problems and Indication. Describe the key Findings observed in the image that correspond to these inputs. Finally, provide an Impression summarizing the overall diagnostic conclusion. Chief Complaint: " + row['Problems'] + str(row['indication'])
        
        # Create data entry for primary/first image in the study
        row_dict1 = {
            "prompt": prompt,
            "input_image": './images/images_normalized/' + image_path[0],  # Full path to first image
            "response": f"Findings: {row['findings']} Impression: {row['impression']}"  # Ground truth radiology report
        }
        
        # Handle cases with multiple images (e.g., PA and lateral views)
        if len(image_path) > 1:
            row_dict2 = {
                "prompt": prompt,
                "input_image": './images/images_normalized/' + image_path[1],  # Path to secondary image
                "response": f"Findings: {row['findings']} Impression: {row['impression']}"  # Same report for additional view
            }
            data.append(row_dict2)
        
        # Add primary image entry to dataset
        data.append(row_dict1)
    return data

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

