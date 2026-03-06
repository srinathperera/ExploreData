import re
import json
import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
from plot_utils import plot_aggregated_data_with_error_bars
from plot_utils import scatter_plot_with_many_y_axes
from dp_utils import enrich_dataframe_with_mismatched_keys
from dp_utils import group_and_aggregate

from core_utils import extract_log_data

# --- Main execution ---
if __name__ == "__main__":
    
    log_file_name = "data/rl-debug-oct23.txt"
    
    try:
        # Run the extraction function
        df = extract_log_data(log_file_name)
        #print df dimensions 
        print("df.shape", df.shape)

        # Add reward column
        df["reward"] = df["test_reward"] + df["error_reward"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])





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

        advantages = []
        batches = df["batch_number"].unique()
        for i, batch in enumerate(batches):
            batch_df = df[df["batch_number"] == batch]
            mean = batch_df["reward"].mean()
            std = batch_df["reward"].std()
            print(f"Batch {batch}: Mean {mean}, Std {std}")
            #for each row in the batch_df, calculate the z-score of the reward
            batch_df["advantage"] = abs((batch_df["reward"] - mean) / (std + 1e-6))
            for j, row in batch_df.iterrows():
                advantages.append({
                    "timestamp": row["timestamp"],
                    "batch_number": batch,
                    "PromptIndex": row["PromptIndex"],
                    "advantage": row["advantage"],
                    "reward": row["reward"]
                })
        
        advantages_df = pd.DataFrame(advantages)
        #advantage_by_Promot = group_and_aggregate(advantages_df, "PromptIndex", "advantage", "mean")
        #print(advantage_by_Promot.head())
        advantages_df.to_csv("temp/advantages_df.csv", index=False)

        scatter_plot_with_many_y_axes(advantages_df, "timestamp", ["reward", "advantage"])

        plot_aggregated_data_with_error_bars(advantages_df, "timestamp", ["reward", "advantage"])

        #scatter plot of the reward vs the timestamp
        #plt.scatter(df["timestamp"], df["reward"])
        #plt.savefig("temp/reward_vs_timestamp.png")
        

        #print("=== Reward over time ===")
        #grouped_df = group_and_aggregate(df, "timestamp", "reward", "mean")
        #sort by the timestamp ascending
        #grouped_df = grouped_df.sort_values(by="timestamp", ascending=True)
        #plot the reward over time
        #plt.plot(grouped_df["timestamp"], grouped_df["reward"])
        #plt.savefig("temp/reward_over_time.png")

        import seaborn as sns

        plt.figure(figsize=(12, 6))

        # The seaborn.lineplot function handles the grouping and coloring in one call
        sns.lineplot(
            data=df,
            x='timestamp',
            y='reward',
            hue='PromptIndex',  # This sets the color/line for each unique PromptIndex
            marker='o'          # Add markers for better visibility
        )

        plt.title('Reward vs. Timestamp for Each Prompt Index (Seaborn)')
        plt.xlabel('Timestamp')
        plt.ylabel('Reward')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Prompt Index')
        plt.tight_layout()

        plt.savefig('temp/reward_vs_timestamp_by_prompt_index_seaborn.png')

        #plot promot Index over time
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=df,
            x='batch_number',
            y='PromptIndex',
            marker='o'
        )
        plt.savefig('temp/prompt_index_over_time.png')

        prompt_data = pd.read_json("data/filtered_prompts_with_tldr.json")
        prompt_data = prompt_data[["Prompt", "tldr", "difficulty"]]
        prompt_data["Prompt"] = prompt_data["Prompt"].str.split("<|im_start|>").str[1]
        df["prompt"] = df["prompt"].str.split("<|im_start|>").str[1]

        enriched_df = enrich_dataframe_with_mismatched_keys(df, prompt_data, "prompt", "Prompt")
        print("enriched_df.shape", enriched_df.shape)

        #plot the tldr vs the reward
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=enriched_df,
            x='difficulty',
            y='reward',
            marker='o'
        )
        plt.savefig('temp/difficulty_vs_reward.png')
        
    except Exception as e:
        print(f"Error: {e}")