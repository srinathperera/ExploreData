import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from core_utils import extract_log_data
from dp_utils import enrich_dataframe_with_mismatched_keys
from dp_utils import group_and_aggregate
import re
from plot_utils import plot_multiple_y_axes
from plot_utils import show_confidence_intervals_with_many_y_axes
from plot_utils import generate_weighted_density_plot
import seaborn as sns
from dp_utils import group_by_many_fields_and_aggregate
from dp_utils import assert_no_null_nans_or_infinity_or_empty
import os

def load_pompt_data():
    prompt_metadata_dir = "data/prompt-metadata/"
    prompt_metadata_list = []
    for filename in os.listdir(prompt_metadata_dir):
        if filename.endswith(".json"):
            prompt_data = pd.read_json(os.path.join(prompt_metadata_dir, filename), orient='records', lines=True)
            if "prompt" in prompt_data.columns:
                prompt_data["Prompt"] = prompt_data["prompt"]
                #remove the Prompt column
                prompt_data = prompt_data.drop(columns=["prompt"])
            prompt_data = cleanup_prompt(prompt_data, "Prompt")
            prompt_metadata_list.append(prompt_data)
    prompt_data = pd.concat(prompt_metadata_list)
    print("prompt_data.shape", prompt_data.shape, "prompt_data.columns", prompt_data.columns)
    return prompt_data
            

def cleanup_prompt(df, column_name):
    delimiter_literal = "<|im_start|>user\n"
    escaped_delimiter = re.escape(delimiter_literal) 
    df[column_name] = df[column_name].str.split(escaped_delimiter, n=1, regex=True).str[1]
    return df

def viulaize_outcomes_over_time(enriched_df):
    # timestap vs average reward, average advantage, average difficulty
    grouped_df1 = group_and_aggregate(enriched_df, "timestamp", "reward", "max")
    grouped_df2 = group_and_aggregate(enriched_df, "timestamp", "advantage", "max")
    grouped_df3 = group_and_aggregate(enriched_df, "timestamp", "difficulty", "mean")

    grouped_df = pd.merge(grouped_df1, grouped_df2, on="timestamp", how="left")
    grouped_df = pd.merge(grouped_df, grouped_df3, on="timestamp", how="left")
    #assert that the shape of the grouped_df is the same as the shape of the enriched_df
    assert grouped_df.shape[0] == grouped_df1.shape[0], "The shape of the grouped_df is not the same as the shape of the grouped_df1" + str(grouped_df.shape) + " != " + str(grouped_df1.shape)
    #sort the grouped_df by the timestamp
    grouped_df = grouped_df.sort_values(by="timestamp")
    grouped_df["difficulty"] = grouped_df["difficulty"]*2/100

    plot_multiple_y_axes(grouped_df, "timestamp", ["reward", "advantage", "difficulty"])

    print(grouped_df.head())
    grouped_df.to_csv("temp/grouped_df.csv", index=False)

def check_and_create_prompt_template_data():
    promot_additional_data_filename = "data/promot_additional_data.csv"
    if os.path.exists(promot_additional_data_filename):
        prompt_data = load_pompt_data(promot_additional_data_filename)
    else:
        prompt_data_df = advantages_df[["prompt"]].drop_duplicates()
        prompt_data_df["difficulty"] = -1
        prompt_data_df["tldr"] = -1

        #prompt_data_df.to_csv(promot_additional_data_filename, index=False)
        #save df as json
        json_filename = promot_additional_data_filename.replace(".csv", ".json")
        prompt_data_df.to_json(json_filename, orient="records", lines=True)

        #verify file is readable
        try:
            _ = pd.read_json(json_filename, orient='records', lines=True)
        except Exception as e:
            print("Error reading json file: ", e)

        print("provide prompt in ", json_filename, " to the user")
        exit(1)



if __name__ == "__main__":
    #log_file_name = "data/rl-debug-oct23.txt"
    #log_file_name = "data/grpo-debug-oct27.zip"

    log_file_name = "data/debug-11-30.log"
    # Run the extraction function
    df = extract_log_data(log_file_name)
    #print df dimensions 
    print("df.shape", df.shape)

    # Add reward column
    df["reward"] = df["test_reward"] + df["error_reward"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.to_csv("temp/df_temp5.csv", index=False)
    
    # Process prompt column to remove the prefix
    df = cleanup_prompt(df, "prompt")
    #df["prompt"] = df["prompt"].str.split("<|im_start|>").str[1]

    #check_and_create_prompt_template_data()
    prompt_data = load_pompt_data()

    #calculate the advantages
    advantages = []
    batches = df["batch_number"].unique()
    for i, batch in enumerate(batches):
        batch_df = df[df["batch_number"] == batch]
        mean = batch_df["reward"].mean()
        std = batch_df["reward"].std()
        #print(f"Batch {batch}: Mean {mean}, Std {std}")
        #for each row in the batch_df, calculate the z-score of the reward
        #batch_df["advantage"] = abs((batch_df["reward"] - mean) / (std + 1e-6))
        batch_df["advantage"] = (batch_df["reward"] - mean) / (std + 1e-6)
        for j, row in batch_df.iterrows():
            advantages.append({
                "prompt": row["prompt"],
                "timestamp": row["timestamp"],
                "batch_number": batch,
                "advantage": row["advantage"],
                "reward": row["reward"]
            })
    
    advantages_df = pd.DataFrame(advantages)
    #advantage_by_Promot = group_and_aggregate(advantages_df, "PromptIndex", "advantage", "mean")
    #print(advantage_by_Promot.head())
    #advantages_df.to_csv("temp/advantages_df.csv", index=False)



    enriched_df = enrich_dataframe_with_mismatched_keys(advantages_df, prompt_data, "prompt", "Prompt")
    print("enriched_df.shape", enriched_df.shape)
    print(enriched_df.shape, enriched_df.columns)
    enriched_df.to_csv("temp/enriched_df.csv", index=False)
    assert_no_null_nans_or_infinity_or_empty(enriched_df)

    #print(enriched_df.head())

    #feilds Index(['prompt', 'timestamp', 'batch_number', 'advantage', 'reward', 'PromptIndex', 'tldr', 'difficulty']
    viulaize_outcomes_over_time(enriched_df)
    print("Done viulaize_outcomes_over_time V1")

    #print("Showing confidence intervals")
    # Convert difficulty to numeric before plotting
    enriched_df["difficulty"] = pd.to_numeric(enriched_df["difficulty"], errors='coerce')
    # Convert timestamp to numeric for regression analysis
    enriched_df["difficulty_scaled"] = enriched_df["difficulty"]*2/100
    enriched_df["timestamp_numeric"] = enriched_df["timestamp"].astype('int64') / 10**9  # Convert to seconds since epoch
    show_confidence_intervals_with_many_y_axes(enriched_df, "timestamp_numeric", ["reward", "advantage", "difficulty_scaled"])
    print("Done viulaize_outcomes_over_time V2")

    # difficulty vs average reward, average advantage
    enriched_df_sorted_by_difficulty = enriched_df.sort_values(by="difficulty")
    plot_multiple_y_axes(enriched_df_sorted_by_difficulty, "difficulty", ["reward", "advantage"])
    print("Done reward, advantage vs difficulty")

    #find the max advantage for each PromptIndex
    grouped_df = group_and_aggregate(enriched_df, "PromptIndex", "advantage", "max")
    #print(grouped_df.head())
    grouped_df = grouped_df.merge(prompt_data, on="PromptIndex", how="left")
    #print(grouped_df.head())
    grouped_df.to_csv("temp/max_advantage_by_PromptIndex.csv", index=False)
    #print("Done")

    #plot difficulty vs average advantage using regression plot
    enriched_df_sorted_by_difficulty = enriched_df.sort_values(by="difficulty")
    show_confidence_intervals_with_many_y_axes(enriched_df_sorted_by_difficulty, "difficulty", ["advantage"])
    show_confidence_intervals_with_many_y_axes(enriched_df_sorted_by_difficulty, "difficulty", ["reward"])

    #generate_weighted_density_plot(enriched_df, "timestamp_numeric", "difficulty", "advantage")


    #for each batch number and prompt index, calculate difference between the max reward and the min reward
    batch_prompt_reward_diff = enriched_df.groupby(["batch_number", "PromptIndex"])["reward"].apply(lambda x: x.max() - x.min())
    batch_prompt_reward_diff = batch_prompt_reward_diff.reset_index()
    print(batch_prompt_reward_diff.head())
    #scatter plot batch_number vs reward_diff
    sns.scatterplot(x="batch_number", y="reward", data=batch_prompt_reward_diff)
    plt.savefig("temp/batch_number_vs_reward_diff.png", dpi=300)
    batch_prompt_reward_diff.to_csv("temp/batch_prompt_reward_diff.csv", index=False)

    print("Done")
        
