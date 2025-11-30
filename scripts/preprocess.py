# modules for data preprocessing
import pandas as pd
import numpy as np
import json

# module for data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.STRINGS = ["#DIV/0!", "#VALUE!"]

    # inner join on a given key
    def __merge_dataframes(self, left, right, key):
        return pd.merge(left, right, on=list(key), how='inner')

    # load and merge data 
    # from multiple sources
    def __load_and_merge_data(self):

        with open(self.metadata_path) as f:

            # metadata about datasets
            data_dir = json.load(f)

            # read relevant columns 
            # from each dataset
            relevant_data = []
            col_name_for_coordinate = ["COORDINATES", "coordinate", "COORDINATES", "COORDINATES"]
            col_name_for_depth = ["Depth (cm)", None, "Depth..cm.", "Depth..cm."]
            for dataset in data_dir.values():

                df = None
                relevant_cols = None if dataset["features"] == "all" else dataset["features"]

                # handle different file types
                if dataset["extension"] == "csv":
                    df = pd.read_csv(dataset["file_path"], usecols=relevant_cols)
                elif dataset["extension"] == "xlsx":
                    df = pd.read_excel(dataset["file_path"], usecols=relevant_cols)

                # use standard column names
                df = df.rename(columns={col_name_for_coordinate[dataset["id"]]: "ID",
                                col_name_for_depth[dataset["id"]]: "depth"})
                relevant_data.append(df)
            
            # merge all dataframes
            merged_df = relevant_data[0]
            for data in relevant_data[1:]:

                # find common keys to merge on
                key = set(merged_df.columns).intersection(["ID", "depth"])
                key = key.intersection(set(data.columns))
                merged_df = self.__merge_dataframes(merged_df, data, key)
            
            # handle edge cases
            merged_df.replace(self.STRINGS, np.nan, inplace=True)
            merged_df = merged_df.rename(columns={"Organic Matter (%)": "organic_matter"})

            return merged_df

    # add is_rocky_terrain column
    def __add_target_column(self, df):

        df["is_rocky_terrain"] = np.where(df["organic_matter"].isna(), "yes", "no")
        return df

    # perform quality check
    def __perform_quality_check(self, df):

        mask_rocky_0_10 = (df["depth"] == "0-10") & (df["is_rocky_terrain"] == "yes")
        rocky_coords = df.loc[mask_rocky_0_10, "ID"]

        mask_10_20 = (df["depth"] == "10-20") & (df["ID"].isin(rocky_coords))
        df.loc[mask_10_20, "is_rocky_terrain"] = "yes"

        return df
    
    def __standardize(self, train: pd.DataFrame, test: pd.DataFrame):
        
       self.scaler = StandardScaler()

       train_ = pd.DataFrame(self.scaler.fit_transform(train), 
                            columns=train.columns,
                            index = train.index)

       test_ = pd.DataFrame(self.scaler.transform(test), 
                           columns=test.columns,
                           index = test.index)
      
       return train_, test_
    
    def __encode_labels(self, x: pd.Series, y: pd.Series):
        
        depth_mapping = {"0-10": 0, "10-20": 1}
        terrain_mapping = {"yes": 1, "no": 0}

        x["depth_encoded"] = x["depth"].map(depth_mapping)
        y = y.map(terrain_mapping)

        return x.drop("depth", axis=1), y

    # driver function
    def preprocess(self, remove_correlated_features=False,
                   standardize_num: bool = True, encode_labels: bool = True):

        df = self.__load_and_merge_data()
        df = self.__add_target_column(df)
        df = self.__perform_quality_check(df)

        spatial_info = ["ID", "depth"]
        idx_spatial_mapping = df[spatial_info].reset_index(drop = False, inplace = False)

        important_features = [
            "NDVI_mean_year2024",
            "NDMI_mean_year2024",
            "aspect",
            "slope",
            "flow_length",
            "S2REP_mean_year2024",
            "twi",
            "depth"
        ]

        correlated_features = ["roughness", "profile_curvature", 
                               "bd2", "NDWI_mean_year2024",
                               "elevation", "sand2", "clay2"]
        
        if remove_correlated_features:
            df = df.drop(columns=correlated_features, errors='ignore')
        
        # drop columns
        df = df.drop(columns=["ID", "organic_matter"], errors='ignore')

        # CAT_FEATURES = ["depth", "is_rocky_terrain"]
        # NUM_FEATURES = df.columns.difference(CAT_FEATURES).to_list()

        # X_train, X_test, y_train, y_test = train_test_split(
        #     df.drop("is_rocky_terrain", axis=1), 
        #     df["is_rocky_terrain"], 
        #     test_size=0.2, 
        #     random_state=42, 
        #     stratify=df["is_rocky_terrain"])

        # if standardize_num:
        #     X_train[NUM_FEATURES], X_test[NUM_FEATURES] = self.__standardize(X_train[NUM_FEATURES], 
        #                                                                      X_test[NUM_FEATURES])
        
        # if encode_labels:
        #     X_train, y_train = self.__encode_labels(X_train, y_train)
        #     X_test, y_test = self.__encode_labels(X_test, y_test)

        #     X_train.drop("depth_encoded", axis=1, inplace=True)
        #     X_test.drop("depth_encoded", axis=1, inplace=True)

        return df[important_features + ["is_rocky_terrain"]], idx_spatial_mapping