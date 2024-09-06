import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def clean_data(dataset):
    """Perform data cleaning."""
    # Perform inner merge on common columns
    merged = pd.concat(dataset, axis=0, ignore_index=True)

    # Drop uncommon columns between datasets, remove Referee column also because it is unneeded
    drop_ref = "Referee"
    drop_div = "Div"
    drop_date = "Date"
    merged = merged.drop(drop_ref, axis=1, errors='ignore')
    merged = merged.drop(drop_div, axis=1, errors='ignore')
    merged = merged.drop(drop_date, axis=1, errors='ignore')
    merged = merged.dropna(axis=1)
    # replace H = Home win to 0, D = draw to 2, and A = Away win to 1
    merged['FTR'] = merged['FTR'].replace({'H': 0, 'D': 2, 'A': 1})
    merged['HTR'] = merged['HTR'].replace({'H': 0, 'D': 2, 'A': 1})

    return merged


def label_encode_data(data):
    """label-encode the data."""
    results_categorical_columns = ['HomeTeam', 'AwayTeam']

    # Create a copy of the DataFrame
    encoding = data.copy()

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    for column in results_categorical_columns:
        encoding[column] = label_encoder.fit_transform(encoding[column])

    return encoding


def hot_encode_data(data):
    """Hot-encode the data."""
    # Define the categorical columns to be one-hot encoded
    results_categorical_columns = ['HomeTeam', 'AwayTeam']

    # Create a copy of the DataFrame
    encoding = data.copy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    # Select the categorical columns for encoding
    categorical_data = encoding[results_categorical_columns]

    # Apply one-hot encoding
    encoded_data = encoder.fit_transform(categorical_data)
    encoded_columns = encoder.get_feature_names_out(results_categorical_columns)

    # Convert to a dense DataFrame
    encoding_encoded = pd.DataFrame(encoded_data, columns=encoded_columns)
    encoding = encoding.drop(columns=results_categorical_columns)

    encoding.reset_index(drop=True, inplace=True)
    encoding_encoded.reset_index(drop=True, inplace=True)
    encoding = pd.concat([encoding, encoding_encoded], axis=1)
    encoding.reset_index(drop=True, inplace=True)

    return encoding


def preprocess_data(data):
    data = label_encode_data(data)
    return data


def feature_engineering(merged):
    # merged['GoalDifference'] = merged['FTHG'] - merged['FTAG']
    # merged['TotalGoals'] = merged['FTHG'] + merged['FTAG']
    merged['FTGoalDiff'] = merged['FTHG'] - merged['FTAG']
    merged['HTGoalDiff'] = merged['HTHG'] - merged['HTAG']

    # merged['GoalRatio'] = merged['FTHG'] / (merged['FTHG'] + merged['FTAG'])
    # merged['WinningStreak'] = merged['FTR'].shift(1) == 'H'
    # merged['WinningStreak'] = merged['WinningStreak'].fillna(False).astype(int)
    # merged['ShotsOnTargetRatio'] = merged['HST'] / merged['HS']
    # merged['FoulRatio'] = merged['HF'] / (merged['HF'] + merged['AF'])
    merged['RedCardDifference'] = merged['HR'] - merged['AR']
    # merged['AvgGoals'] = merged['FTHG'].transform('mean')
    # merged['AvgShots'] = merged['HS'].transform('mean')
    # Calculate the ratio of yellow cards to total cards for both home and away teams
    # merged['HYellowRatio'] = merged['HY'] / np.where((merged['HY'] + merged['AY']) == 0, 1, (merged['HY'] + merged['AY']))
    # merged['AYellowRatio'] = merged['AY'] / np.where((merged['HY'] + merged['AY']) == 0, 1, (merged['HY'] + merged['AY']))

    # Calculate the ratio of fouls committed to total fouls for both home and away teams
    # merged['HFoulRatio'] = merged['HF'] / np.where((merged['HF'] + merged['AF']) == 0, 1, (merged['HF'] + merged['AF']))
    # merged['AFoulRatio'] = merged['AF'] / np.where((merged['HF'] + merged['AF']) == 0, 1, (merged['HF'] + merged['AF']))

    # Calculate the ratio of shots on target to total shots for both home and away teams
    #if (merged['HST'] == 0 or merged['AST']  == 0 or merged['HS'] == 0 or merged['AS'] == 0):
    #    merged['HShotRatio']=0
    #    merged['AShotRatio']=0
    #else:
    #    merged['HShotRatio'] = merged['HST'] / (merged['HS']) 
    #    merged['AShotRatio'] = merged['AST'] / (merged['AS'])
    

    # Calculate the ratio of corners won to total corners for both home and away teams

    condition = (
    (merged['HST'] == 0) | 
    (merged['AST'] == 0) | 
    (merged['HS'] == 0) | 
    (merged['AS'] == 0)
    )

    # Apply conditions
    merged['HShotRatio'] = 0
    merged['AShotRatio'] = 0

# Use loc to set values based on conditions
    merged.loc[~condition, 'HShotRatio'] = merged['HST'] / merged['HS']
    merged.loc[~condition, 'AShotRatio'] = merged['AST'] / merged['AS']


    
# Apply conditions
    merged['HCornersRatio'] = 0
    merged['ACornersRatio'] = 0

# Use loc to set values based on conditions

    corner_condition = (merged['HC'] == 0) | (merged['AC'] == 0)
    merged.loc[~corner_condition, 'HCornersRatio'] = merged['HC'] / (merged['HC'] + merged['AC'])
    merged.loc[~corner_condition, 'ACornersRatio'] = merged['AC'] / (merged['HC'] + merged['AC'])
    #if merged['HC'] == 0 or merged['AC'] == 0:
    #    merged['HCornersRatio']=0
    #    merged['AShotRatio']=0
    #else:
    #    merged['HCornersRatio'] = merged['HC'] / (merged['HC'] + merged['AC']) 
    #    merged['ACornersRatio'] = merged['AC'] / (merged['HC'] + merged['AC'])

    return merged


def save_data(data, output_file):
    """Save the preprocessed data to a new CSV file."""
    data.to_csv(output_file, index=False)


def normalize_data(data, columns):
    """Standardize the numeric columns in the data."""
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data


if __name__ == "__main__":
    # Define the paths
    original_premier_1 = "../../data/PremierRaw/2000-01.csv"
    original_premier_2 = "../../data/PremierRaw/2001-02.csv"
    original_premier_3 = "../../data/PremierRaw/2002-03.csv"
    original_premier_4 = "../../data/PremierRaw/2003-04.csv"
    original_premier_5 = "../../data/PremierRaw/2004-05.csv"
    original_premier_6 = "../../data/PremierRaw/2005-06.csv"
    original_premier_7 = "../../data/PremierRaw/2006-07.csv"
    original_premier_8 = "../../data/PremierRaw/2007-08.csv"
    original_premier_9 = "../../data/PremierRaw/2008-09.csv"
    original_premier_10 = "../../data/PremierRaw/2009-10.csv"
    original_premier_11 = "../../data/PremierRaw/2010-11.csv"
    original_premier_12 = "../../data/PremierRaw/2011-12.csv"
    original_premier_13 = "../../data/PremierRaw/2012-13.csv"
    original_premier_14 = "../../data/PremierRaw/2013-14.csv"
    original_premier_15 = "../../data/PremierRaw/2014-15.csv"
    original_premier_16 = "../../data/PremierRaw/2015-16.csv"
    original_premier_17 = "../../data/PremierRaw/2016-17.csv"
    original_premier_18 = "../../data/PremierRaw/2017-18.csv"
    original_premier_19 = "../../data/PremierRaw/2018-19.csv"
    original_premier_20 = "../../data/PremierRaw/2019-20.csv"
    original_premier_21 = "../../data/PremierRaw/2020-2021.csv"
    original_premier_22 = "../../data/PremierRaw/2021-2022.csv"

    # Load the data
    original_datasets = []

    for i in range(1, 23):
        dataset_name = f"original_premier_{i}"
        df = load_data(eval(dataset_name))
        original_datasets.append(df)

    merged_df = clean_data(original_datasets)
    encoded_merged_df = preprocess_data(merged_df)
    featured_df = feature_engineering(encoded_merged_df)
    numeric_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
                       'HF', 'AF', 'HY', 'AY', 'HR', 'AR','FTGoalDiff','HTGoalDiff','RedCardDifference','HShotRatio','AShotRatio','HCornersRatio','ACornersRatio']
    normalized_df = normalize_data(featured_df, numeric_columns)

    save_data(normalized_df, "../../data/PremierProcessed/mergedDataOriginal.csv")
    #save_data(merged_df, "../../data/PremierProcessed/mergedData.csv")