import pandas as pd
import numpy as np

from pipeline.preprocess import Preprocessor

class SpatialFeatureGenerator:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.preprocessor = Preprocessor(metadata_path)
        self.feature_columns = ['aspect', 'NDVI_mean_year2024', 'NDMI_mean_year2024', 
                                'S2REP_mean_year2024', 'flow_length', 'slope', 'twi']

    def disect_data_by_depth(self):

        df = self.preprocessor.preprocess()

        df_lower_depth = df[df['depth'] == '0-10'].copy()
        df_higher_depth = df[df['depth'] == '10-20'].copy()

        return df_lower_depth, df_higher_depth
    
    def df_to_matrix(self, df: pd.DataFrame):
        # Create a copy to avoid modifying the original dataframe
        df_clean = df.drop('depth', axis=1)
    
        coords = df_clean['ID'].str.replace('[()]', '', regex=True).str.split(',', expand=True)
        coords.columns = ['x', 'y']
        
        coords['x'] = pd.to_numeric(coords['x'])
        coords['y'] = pd.to_numeric(coords['y'])
        
        unique_x = sorted(coords['x'].unique())
        unique_y = sorted(coords['y'].unique())
        
        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}
        
        num_features = len(self.feature_columns)
        matrix = np.full((len(unique_x), len(unique_y), num_features), np.nan)
        
        # Use enumerate to avoid index issues
        for row_idx, (idx, row) in enumerate(df_clean.iterrows()):
            x_coord = coords.iloc[row_idx]['x']
            y_coord = coords.iloc[row_idx]['y']
            x_idx = x_to_idx[x_coord]
            y_idx = y_to_idx[y_coord]
            
            for feat_idx, feature in enumerate(self.feature_columns):
                matrix[x_idx, y_idx, feat_idx] = row[feature]
        
        return matrix
    
    def compute_spatial_features_with_stride(self, matrix: np.ndarray, grid_size: int, stride: int):
        x_size, y_size, num_features = matrix.shape
        
        result_data = []
        coordinates = []
        
        for grid_start_i in range(0, x_size, stride):
            for grid_start_j in range(0, y_size, stride):
                grid_end_i = min(grid_start_i + grid_size, x_size)
                grid_end_j = min(grid_start_j + grid_size, y_size)
                
                grid = matrix[grid_start_i:grid_end_i, grid_start_j:grid_end_j, :]

                padded_grid = np.full((grid_size, grid_size, num_features), np.nan)
                
                actual_height = grid_end_i - grid_start_i
                actual_width = grid_end_j - grid_start_j
                padded_grid[:actual_height, :actual_width, :] = grid

                # Only process points that are actually in the matrix (not padding)
                for local_i in range(actual_height):
                    for local_j in range(actual_width):
                        global_i = grid_start_i + local_i
                        global_j = grid_start_j + local_j
                        
                        point_features = matrix[global_i, global_j, :].copy()
                        
                        spatial_features = []
                        
                        for feat_idx in range(num_features):
                            feature_values = padded_grid[:, :, feat_idx]
                            
                            # Mean and std using padded grid
                            mean_val = np.nanmean(feature_values)
                            spatial_features.append(mean_val)
                            
                            std_val = np.nanstd(feature_values)
                            spatial_features.append(std_val)
                            
                            # Weighted statistics excluding current point
                            include_mask = np.ones_like(feature_values, dtype=bool)
                            include_mask[local_i, local_j] = False
                            
                            valid_include_mask = include_mask & ~np.isnan(feature_values)
                            
                            # No need to check if sum > 0, we know there are other points in the padded grid
                            current_point = np.array([local_i, local_j])
                            other_points = np.column_stack(np.where(valid_include_mask))
                            
                            distances = np.sqrt(np.sum((other_points - current_point)**2, axis=1))
                            weights = 1.0 / (distances + 1e-8)
                            
                            weighted_mean = np.average(feature_values[valid_include_mask], weights=weights)
                            spatial_features.append(weighted_mean)
                            
                            weighted_var = np.average((feature_values[valid_include_mask] - weighted_mean)**2, weights=weights)
                            weighted_std = np.sqrt(weighted_var)
                            spatial_features.append(weighted_std)
                        
                        coordinates.append((global_i, global_j))
                        result_data.append(list(point_features) + spatial_features)
        
        original_cols = self.feature_columns
        spatial_cols = []
        for feat_name in self.feature_columns:
            spatial_cols.extend([
                f'{feat_name}_spatial_mean',
                f'{feat_name}_spatial_std',
                f'{feat_name}_spatial_weighted_mean',
                f'{feat_name}_spatial_weighted_std'
            ])
        
        all_cols = original_cols + spatial_cols
        
        result_df = pd.DataFrame(result_data, columns=all_cols)
        
        result_df['x_coord'] = [coord[0] for coord in coordinates]
        result_df['y_coord'] = [coord[1] for coord in coordinates]
                
        return result_df

    def compute_spatial_features(self):

        df_lower_depth, df_higher_depth = self.disect_data_by_depth()

        matrix_lower_depth = self.df_to_matrix(df_lower_depth)
        matrix_higher_depth = self.df_to_matrix(df_higher_depth)

        df_lower_depth_spatial = self.compute_spatial_features_with_stride(matrix_lower_depth, 3, 3)
        df_higher_depth_spatial = self.compute_spatial_features_with_stride(matrix_higher_depth, 3, 3)

        return df_lower_depth_spatial, df_higher_depth_spatial