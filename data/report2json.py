import pandas as pd
import json

# Load the CSV file containing medical report data
df = pd.read_csv('./report.csv')

# Shuffle the dataset with a fixed random state for reproducibility, then reset index
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into training (80%) and testing (20%) sets
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

def to_json(dataframe):
    """
    Convert DataFrame rows into a structured JSON format suitable for training vision-language models.
    Args: dataframe: Pandas DataFrame containing medical report data.
    Returns: List of dictionaries in the required JSON format for model training.
    """
    data = []
    
    # Iterate through each row in the DataFrame
    for _, row in train_df.iterrows():
        # Split image paths (some cases may have multiple images)
        image_path = row['image_files'].split(';')
        
        # Construct the prompt for the AI model using patient problems and indications
        prompt = "You are a radiologist. Analyze the given chest X-ray based on the provided Problems and Indication. Describe the key Findings observed in the image that correspond to these inputs. Finally, provide an Impression summarizing the overall diagnostic conclusion. Chief Complaint: " + row['Problems'] + str(row['indication'])
        
        # Create the first dictionary entry for the primary image
        row_dict1 = {
            "prompt": prompt,
            "input_image": './images/' + image_path[0],
            "response": f"Findings: {row['findings']} Impression: {row['impression']}"
        }
        
        # If there's a second image, create an additional entry for it
        if len(image_path) > 1:
            row_dict2 = {
                "prompt": prompt,
                "input_image": './images/' + image_path[1],
                "response": f"Findings: {row['findings']} Impression: {row['impression']}"
            }
            data.append(row_dict2)
        
        # Add the primary image entry to the data list
        data.append(row_dict1)
    
    # Transform the data into the final JSON structure
    output = []
    for i, d in enumerate(data):
        output.append({
            "id": str(i).zfill(5),  # Zero-padded ID for consistent formatting
            "image": d["input_image"].split("/")[-1],  # Extract just the filename
            "conversations": [
                {"from": "human", "value": d["prompt"]},  # User/doctor query
                {"from": "gpt", "value": d["response"]}   # AI assistant response
            ],
            # Domain classification for the image type
            "domain": {
                "chest_xray": True, 
                "mri": False, 
                "ct_scan": False, 
                "histology": False, 
                "gross": False
            },
            "type": "conversation"  # Type of data entry
        })
    
    return output

# Generate training and test datasets in JSON format
train = to_json(train_df)
test = to_json(test_df)

# Save training data to JSON file
with open("./train_report.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

# Save test data to JSON file
with open("./test_report.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

# Print confirmation message with dataset sizes
print(f"✅ Generated {len(train)} training entries and saved to train_report.json")
print(f"✅ Generated {len(test)} test entries and saved to test_report.json")