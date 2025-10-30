import re
import json
import pandas as pd
import zipfile

def trim_string(string, length):
    """
    Trims a string to a given length.
    """
    if len(string) > length:
        return string[:length]
    return string

def extract_log_data(filename):
    """
    Extracts specific fields from a log file containing JSON objects.

    Args:
        filename (str): The path to the log file.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data for all log entries.
    """
    
    # This regex matches the log prefix and *captures* the timestamp (the part in parentheses).
    # This allows re.split() to return the delimiters (timestamps) in the results.
    log_entry_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - my_app_logger - INFO - "
    )
    #if the file is a zip file, read from the zip file
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            content = zip_ref.read("debug.log").decode('utf-8')
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

    # Split the entire file content by the log prefix.
    # Because the timestamp is a capturing group, the result will be like:
    # ['', 'timestamp_1', 'json_body_1', 'timestamp_2', 'json_body_2', ...]
    parts = log_entry_pattern.split(content)

    extracted_data = []

    # The first part (parts[0]) is anything before the first log entry (usually empty).
    # We iterate over the list in pairs: (timestamp, json_string)
    # We start at index 1 and step by 2.
    count = 0
    failed_to_parse = 0
    for i in range(1, len(parts), 2):
        timestamp = parts[i]
        
        # Check if there's a corresponding JSON body (for files ending unexpectedly)
        if i + 1 >= len(parts):
            print(f"Warning: Found timestamp {timestamp} without a following JSON body.")
            continue
            
        json_string = parts[i+1].strip()
        
        # Skip empty strings or non-JSON content
        if not json_string or not json_string.startswith('{'):
            print(f"Warning: Skipping non-JSON content for timestamp {timestamp}: {json_string[:100]}...")
            continue
        
        try:
            # Parse the string as JSON
            data = json.loads(json_string)
            extracted_data.append({
                "timestamp": timestamp,
                "batch_number": data.get("batch_number"),
                "prompt": data.get("prompt"),
                "solution": data.get("completion"),
                "test_reward": data.get("test_reward"),
                "error_reward": data.get("error_reward")
            })
            count += 1
        except json.JSONDecodeError as e:
            failed_to_parse += 1
            print(f"Warning: Failed to parse: {trim_string(json_string, 100)} Error: {e}")
    print(f"Total count: {count}, failed to parse: {failed_to_parse}")
    return pd.DataFrame(extracted_data)
