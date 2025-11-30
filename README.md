<div align="center">
<h1>Grid-based Feature Engineering</h1>
</div>

## Aim

- To compute local spatial features to instill spatial awareness in the rocky terrain predictor.
  
- This implementation takes the point-based data as input, computes spatial features, and returns the dataset with spatial features.

## Environment Setup

- Clone this repository.
  
```bash
git clone https://github.com/Gaurav0502/grid-based-spatial-feature-engineer.git
```

- Install all packages in the ```requirements.txt``` file.

```bash
pip install -r requirements.txt
```

- The `scripts/spatialfeaturegenerator.py` contains the code for generating the spatial features. To get the updated dataset (with the spatial features), you need to execute the `compute_spatial_features()` function inside the class `SpatialFeatureGenerator`.

- After execution, you must get four new features for every remote sensing index (i.e., any column except the depth and coordinate):

1. `*_spatial_mean`
2. `*_spatial_std`
3. `*_spatial_weighted_mean` (excludes the point itself*)
4. `*_spatial_weighted_std` (excludes the point itself*)

\* prevent zero division error.

**<u>Notes:</u>** 
- The function only computes features for a grid of dimensions (3 x 3). To use any other dimensions, you must use the `compute_spatial_features_with_stride()` with the `grid_size` as input and the `stride` for the sliding window.

- Any values outside the bounds of the feature space are considered as NaN and are excluded from the computation.

- When using the output dataframe for modelling, ensure you make use of the `GroupKFold` from <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html">sklearn</a> to mitigate chances of data leakage. In other words, a grid (in its entirety) must either be in the train set or the test set.

- Preferably, keep the grids for both depths in the same set to instill spatial awareness and depth perception into the model.



