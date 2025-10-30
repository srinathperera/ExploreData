import pandas as pd
import numpy as np

def enrich_dataframe_with_mismatched_keys(main_df, supplemental_df, main_key, supplemental_key, how='left'):
    """
    Enriches the main DataFrame with data from the supplemental DataFrame 
    by performing a merge operation using keys with different names.

    Args:
        main_df (pd.DataFrame): The primary DataFrame to be enriched (left side of merge).
        supplemental_df (pd.DataFrame): The secondary DataFrame providing enrichment (right side of merge).
        main_key (str): The column name in main_df to use for joining.
        supplemental_key (str): The column name in supplemental_df to use for joining.
        how (str): Type of merge to be performed. Defaults to 'left'.
                   Common values are 'left', 'right', 'inner', 'outer'.

    Returns:
        pd.DataFrame: The enriched DataFrame.
    """

    #check supplemental_df for duplicates
    supplemental_df.to_csv("temp/supplemental_df_temp.csv", index=False)

    temp_df = supplemental_df[[supplemental_key]]
    temp_df.to_csv("temp/temp_df_temp.csv", index=False)
    key_duplicates = temp_df.duplicated(keep='first')
    if key_duplicates.any():
        print("Duplicates found in supplemental_df with key: " + supplemental_key)
        duplicate_rows = supplemental_df[key_duplicates]
        print("supplemental_df.shape", supplemental_df.shape, "duplicate_rows.shape", duplicate_rows.shape)
        print(duplicate_rows.head())
        duplicate_rows.to_csv("temp/duplicate_rows.csv", index=False)
        raise Exception("Duplicates found in supplemental_df with key: " + supplemental_key)
    
    # Perform the merge using left_on and right_on
    enriched_df = pd.merge(
        main_df,
        supplemental_df,
        left_on=main_key,       # Use this column from the main_df
        right_on=supplemental_key, # Use this column from the supplemental_df
        how=how                 # Type of join (e.g., 'left' to keep all rows from main_df)
    )
    
    # Optional: Drop the redundant key column from the supplemental DataFrame 
    # if it's not needed after the merge, to clean up the result.
    if supplemental_key in enriched_df.columns:
        # Renaming of the key columns can happen automatically if they have the same 
        # name *after* the merge but *before* the join. However, since they have 
        # different names here, we only need to drop the redundant key.
        # Note: Be careful not to drop the primary key if it was needed in the main_df
        # before the join. We are only dropping the key from the supplemental_df.
        enriched_df = enriched_df.drop(columns=[supplemental_key])

    #assert that the main_df and the enriched_df have the same number of rows
    assert main_df.shape[0] == enriched_df.shape[0], "The main_df and the enriched_df have different number of rows" + str(main_df.shape[0]) + " != " + str(enriched_df.shape[0])
        
    return enriched_df






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

def group_by_many_fields_and_aggregate(df, field_names, aggregate_column, aggregate_function):
    """
    Groups the dataframe by the given fields and aggregates the values using the given function.
    """
    results_df = df.groupby(field_names)[aggregate_column].aggregate(aggregate_function).to_frame()
    results_df = results_df.reset_index()
    #sort by the aggregate column descending
    results_df = results_df.sort_values(by=aggregate_column, ascending=False)
    return results_df

def assert_no_null_nans_or_infinity_or_empty(df):
    """
    Asserts that the dataframe has no null, nans, infinity, or empty values.
    """
    assert df.isnull().sum().sum() == 0, "The dataframe has null values"
    assert df.isin([np.nan, np.inf, -np.inf]).sum().sum() == 0, "The dataframe has nan or infinity values"
    assert df.empty == False, "The dataframe is empty"
