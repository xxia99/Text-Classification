# Author: Xin Xia
# Date: 3/26/2023
##########################
#  Code for NaN Filling  #
##########################
import pandas as pd
import random
from feature_distribution import *

def check_nan_rows(column, show=False):
    """
      :param column: The input raw column loaded from .csv.
      :param show: Default to be False. If need to check the rows with nans, turn it to be True.
      :return nan_rows, nan_rows_index: The content and index of rows with NaN values.
      Find out which rows have NaN values of a given feature.
    """
    col_value = df[column]
    nan_rows = df[col_value.isna()]
    nan_rows_index = list(nan_rows.index)
    if len(nan_rows) > 0:
        print(f"NaN value(s) found in column: {column}, in row(s): {nan_rows_index}")
    else:
        print(f"No NaN values found in column {column}")
    if show:
        show_nan_rows(2)
    return nan_rows, nan_rows_index


def fill_with_random_values(x):
    # return 0
    return random.choice([0, 1])


def fill_nan(column, show=False):
    """
      :param column: The column with NaN values required to be filled.
      :param show: Default to be False. If need to check the filled values, turn it to be True
      Fill the NaN values in the given column(AKA: feature), considering the feature types.
      For features in the binary list, fill the NaNs with a random value from {0,1}.
      For features in the categorical list, fill NaNs with the common value of the column.
      For features in the numerical list, fill NaNs with the mean value of the column.
    """
    if column in binary:  # fill NaNs in binary columns with a random value from {0,1}
        df[column] = df[column].apply(fill_with_random_values)
    elif column in categorical:  # fill NaNs in categorical columns with the common values
        column_mode = df[column].mode()[0]
        df[column].fillna(column_mode, inplace=True)
    else:  # fill NaNs in numerical columns with mean value of the column
        column_mean = df[column].mean()
        df[column].fillna(column_mean, inplace=True)
    if show:
        show_filled_value(column)


def show_nan_rows(num_row):
    print(f"NaN values exist in these rows, for example:")
    print(nan_rows[:num_row])


def show_filled_value(column):
    print(f"NaN values in column: {column} are filled with: ")
    print([df[column][i] for i in nan_rows_index])


def double_check(df):
    """
      :param df: Modified pandas dataframe of the csv
      To double check if all of the NaN values have been filled. If not, print a
      summary of NaN values in each column.
    """
    if not df.isna().any().any():
        print("All NaN Values Have Been Filled!")
    else:
        print(f"NaN Values Still Remain in: {df.isna().sum()}")


def caption_tabular(df):
  """
  :param df: the input tabular dataframe(preprocessed with filling NaN values)
  This function generate text descriptions for each row of a tabular dataframe.
  The descriptions depend on the feature types(columns except "target_label"):
  for numerical and categorical features, concatenate the column names and the
  cell values; for binary features, only keep the column when its value is 1.
  """
  text_descriptions = []
  features = list(df.columns)
  for index, row in df.iterrows():
    descriptions = []
    for feature in features[:-1]:
      if feature in numerical or feature in categorical:
        feature_description = feature + " "+ str(row[feature])
        descriptions.append(feature_description)
      elif row[feature]:
        feature_description = feature
        descriptions.append(feature_description)
    text_descriptions.append("; ".join(descriptions))
  return text_descriptions


if __name__ == "__main__":
    df = pd.read_csv('./tabular_patient_data/data_raw.csv')
    # The main scripts of filling NaN values.
    for column in df.columns:
        nan_rows, nan_rows_index = check_nan_rows(column, show=False)  # Find out which rows have NaN values of this column
        if len(nan_rows_index):
            fill_nan(column, show=False)  # Fill the NaN values, turn show to True if want to check the filled values

    double_check(df)  # Double check

    text_descriptions = caption_tabular(df)
    df_text = pd.DataFrame(text_descriptions, columns=['text_description'])
    labels = [int(label) for label in df["target_label"]]
    df_text["target_label"] = labels
    print(df_text)
    df_text.to_csv("preprocessed_data.csv")