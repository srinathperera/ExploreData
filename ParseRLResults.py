import re
import json
import pandas as pd
import matplotlib.pyplot as plt

def group_and_aggregate(df, field_name, aggregate_column, aggregate_function):
    """
    Groups the dataframe by the given field and aggregates the values using the given function.
    """
    results_df = df.groupby(field_name)[aggregate_column].aggregate(aggregate_function).to_frame()
    results_df = results_df.reset_index()
    #sort by the aggregate column descending
    results_df = results_df.sort_values(by=aggregate_column, ascending=False)
    return results_df
    #results_df = df.groupby(field_name, as_index=False)[aggregate_column].aggregate(aggregate_function)
    #return results_df
    #results_df = df.groupby(field_name)[aggregate_column].mean().reset_index()
    #return results_df

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

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

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
            print(f"Warning: Failed to parse: {json_string} Error: {e}")
    print(f"Total count: {count}, failed to parse: {failed_to_parse}")
    return pd.DataFrame(extracted_data)

# --- Main execution ---
if __name__ == "__main__":
    
    # Create a dummy log file named 'app.log' for this example
    log_file_name = "data/rl-debug-oct23.txt"
    
    

    try:
        # Run the extraction function
        df = extract_log_data(log_file_name)
        #print df dimensions 
        print("df.shape", df.shape)

        # Add reward column
        df["reward"] = df["test_reward"] + df["error_reward"]

        #create a dictionary entry for each unique entry in the prompt column
        
        uniqure_prompt_values = df["prompt"].unique()
        prompt_dict = {}
        for i, prompt in enumerate(uniqure_prompt_values):
            prompt_dict[prompt] = i
        #save the dict as a dataframe
        prompt_df = pd.DataFrame(prompt_dict.items(), columns=["Prompt", "PromptIndex"])
        #make PromptIndex the first feild of the dataframe
        prompt_df = prompt_df.reindex(columns=["PromptIndex", "Prompt"])
        prompt_df.to_csv("temp/prompt_dict.csv", index=False)

        #print("prompt_dict", len(prompt_dict))
        
        df["PromptIndex"] = df["prompt"].map(prompt_dict)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        min_timestamp = df["timestamp"].min()
        max_timestamp = df["timestamp"].max()
        df["timestamp_quater"] = df["timestamp"].apply(lambda x: round(4 * (x - min_timestamp) / (max_timestamp - min_timestamp)))
        print(df.head())
        #break timestammp value into four 


        print("== Top 20 prompts by reward: ===")
        grouped_df = group_and_aggregate(df, "PromptIndex", "reward", "mean")
        #format the dataframe to 2 decimal places
        grouped_df = grouped_df.round(3)
        print(grouped_df.head(20).to_string(index=False))

        print("== Top 20 prompts by max reward: ===")
        grouped_df = group_and_aggregate(df, "PromptIndex", "reward", "max")
        grouped_df = grouped_df.round(3)
        print(grouped_df.head(20).to_string(index=False))

        print("Rewards by timestamp quater: ===")
        grouped_df = group_and_aggregate(df, "timestamp_quater", "reward", "mean")
        grouped_df = grouped_df.round(3)
        print(grouped_df.head(20).to_string(index=False))

        #for each prompt, in the reward values for that prompt, find the difference between the max and the min
        print("Prompt Reward Improvement (Max - Min) highest to lowest: ===")
        data = []
        for i, prompt in enumerate(uniqure_prompt_values):
            prompt_reward_values = df[df["PromptIndex"] == i]["reward"]
            max_reward = prompt_reward_values.max()
            min_reward = prompt_reward_values.min()
            data.append({
                "Prompt": i,
                "Max Reward - Min Reward": max_reward - min_reward
            })
        #save the data as a dataframe
        data_df = pd.DataFrame(data)
        #sort by the Max Reward - Min Reward descending
        data_df = data_df.sort_values(by="Max Reward - Min Reward", ascending=False)
        print(data_df.head(20).to_string(index=False))
        data_df.to_csv("temp/prompt_reward_improvement.csv", index=False)


        #scatter plot of the reward vs the timestamp
        plt.scatter(df["timestamp"], df["reward"])
        plt.show()

        print("=== Reward over time ===")
        grouped_df = group_and_aggregate(df, "timestamp", "reward", "mean")
        #sort by the timestamp ascending
        grouped_df = grouped_df.sort_values(by="timestamp", ascending=True)
        #plot the reward over time
        plt.plot(grouped_df["timestamp"], grouped_df["reward"])
        plt.show()
        

        

    except IOError as e:
        print(f"Error writing dummy file: {e}")