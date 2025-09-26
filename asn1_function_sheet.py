
import pandas as pd
import numpy as np
import math

def age_splitter(df, col_name, age_threshold):
    """
    Splits the dataframe into two dataframes based on an age threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    col_name (str): The name of the column containing age values.
    age_threshold (int): The age threshold for splitting.

    Returns:
    tuple: A tuple containing two dataframes:
        - df_below: DataFrame with rows where age is below the threshold.
        - df_above_equal: DataFrame with rows where age is above or equal to the threshold.
    """
    below_threshold = df[df[col_name] < age_threshold]
    above_or_equal_threshold = df[df[col_name] >= age_threshold]
    return below_threshold, above_or_equal_threshold


    
def effectSizer(df, num_col, cat_col):
    """
    Calculates the effect sizes of binary categorical classes on a numerical value.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    num_col (str): The name of the numerical column.
    cat_col (str): The name of the binary categorical column.

    Returns:
    float: Cohen's d effect size between the two groups defined by the categorical column.
    Raises:
    ValueError: If the categorical column does not have exactly two unique values.
    """
    # Drop rows with NA in relevant columns
    sub = df[[num_col, binary_col]].dropna()

    # Make sure the binary column has exactly 2 unique values
    unique_vals = sub[binary_col].unique()
    if len(unique_vals) != 2:
        raise ValueError(f"'{binary_col}' must have exactly 2 unique values. Found: {unique_vals}")

        # Ensure group with value 1 is used as x1
    if 1 in unique_vals:
        x1 = 1
        x0 = [val for val in unique_vals if val != 1][0]
    else:
        x1, x0 = unique_vals[0], unique_vals[1]  # fallback

    group1 = sub[sub[binary_col] == x1][num_col]
    group0 = sub[sub[binary_col] == x0][num_col]

    mean1, mean0 = group1.mean(), group0.mean()
    std1, std0 = group1.std(), group0.std()

    pooled_std = ((std1 ** 2 + std0 ** 2) / 2) ** 0.5
    effect_size = 0.0 if pooled_std == 0 else (mean1 - mean0) / pooled_std

    return {
        str(x1): mean1,
        str(x0): mean0,
        'Effect Size': effect_size
    }





def cohenEffectSize(group1, group2):
    # You need to implement this helper function
    # This should not be too hard...
   pass
    import numpy as np

    # Convert to numpy arrays and drop NaNs
    group1 = np.array(group1).astype(float)
    group2 = np.array(group2).astype(float)

    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    # Means and standard deviations
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)

    # Cohen's d
    if pooled_std == 0:

    return



def cohortCompare(df, cohorts, statistics=['mean', 'median', 'std', 'min', 'max']):
    """
    This function takes a dataframe and a list of cohort column names, and returns a dictionary
    where each key is a cohort name and each value is an object containing the specified statistics
    """
    metrics = CohortMetric('Full Dataset')

    num_columns = df.select_dtypes(include='number').columns
    cat_columns = [col for col in cat_columns if col in df.columns]

    for col in num_columns:
        metrics.add_numerical_stats(col, df[col])

    for col in cat_columns:
        metrics.add_categorical_counts(col,df[col])

    return{'Full Dataset': metrics}
  

class CohortMetric():
    # don't change this
    def __init__(self, cohort_name):
        self.cohort_name = cohort_name
        self.statistics = {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None
        }
    def setMean(self, new_mean):
        self.statistics["mean"] = new_mean
    def setMedian(self, new_median):
        self.statistics["median"] = new_median
    def setStd(self, new_std):
        self.statistics["std"] = new_std
    def setMin(self, new_min):
        self.statistics["min"] = new_min
    def setMax(self, new_max):
        self.statistics["max"] = new_max

    def compare_to(self, other):
        for stat in self.statistics:
            if not self.statistics[stat].equals(other.statistics[stat]):
                return False
        return True
    def __str__(self):
        output_string = f"\nCohort: {self.cohort_name}\n"
        for stat, value in self.statistics.items():
            output_string += f"\t{stat}:\n{value}\n"
            output_string += "\n"
        return output_string
