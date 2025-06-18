# Clustering Visualization App

This project is an interactive Dash app for performing and visualizing **K-Means** and **Agglomerative Clustering** on meteorological data. It allows users to explore clustering patterns and dynamically re-cluster selected data points.

---

## Dataset

We use the dataset from the Wolfgangpass weather station:  
[Download Dataset](https://www.envidat.ch/dataset/weather-station-wolfgangpass/resource/f0944b54-4ee9-496a-96a8-069ad5c73487)

The dataset contains multiple NaN values — these are removed before clustering.

The following variables are used for clustering:
- `T10m`: Temperature at 10m
- `RH10m`: Relative Humidity at 10m
- `p10m`: Pressure at 10m
- `p_NN10m`: Pressure reduced to sea level at 10m

---

### Step 1: K-Means Clustering

- Apply K-Means clustering using the four selected variables.
- Cluster data into three groups.
- Provide two dropdown menus to select any two of the four variables.
- Display two scatter plots:
  - X-axis: date
  - Y-axis: variable selected from dropdown
  - Color: cluster assignment

### Step 2: Agglomerative Clustering

- Apply Agglomerative Clustering using the same four variables.
- Cluster data into three groups.
- Use the same dropdowns for variable selection.
- Add two more scatter plots similar to Step 1.
- Arrange all four plots in a 2x2 grid:
  - Rows: Selected variables from the dropdowns
  - Columns: Clustering method (K-Means vs Agglomerative)

### Step 3: Interaction and Linking

- Allow lasso selection in any scatter plot.
- On selection:
  - Recluster only the selected points using both methods.
  - Color selected points by their new cluster assignment.
  - Color unselected points in grey.
- All four plots update simultaneously.
- Use `customdata` if necessary to track point indices across plots.

---

## ⚙️ Dependencies

Make sure you have the following Python libraries installed:

```bash
pip install dash pandas plotly scikit-learn
```

### Run the App
After installing the dependencies, you can run the app with:

```bash
python app.py
```

The app will open in your browser at http://127.0.0.1:5000/.

### Notes
- Be sure to handle missing values (NaN) before clustering.
- Cluster colors are consistent within method but not across methods.
- Customdata is helpful for managing cross-plot selection and interactivity.

