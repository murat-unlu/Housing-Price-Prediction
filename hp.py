import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load California Housing data
@st.cache_data
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target
    return df

data = load_data()

# Preprocess dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('MedHouseVal', axis=1))
X = data_scaled
y = data['MedHouseVal']

# Streamlit UI
st.title("California Housing Price Prediction")
st.sidebar.header("Navigation")

# Navigation
page = st.sidebar.selectbox("Choose a Page", ["Supervised Learning", "Unsupervised Learning", "Prediction","Model Comparison","Data Analysis"])

if page == "Supervised Learning":
    st.header(" Supervised Learning")

    # Select model
    model_name = st.sidebar.selectbox("Select Model",
                                      ["Linear Regression", "Decision Tree Regressor", "Gradient Boosting Regressor",
                                       "Support Vector Regressor"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model initialization
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "Support Vector Regressor":
        model = SVR()

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mpe = np.mean((y_test - y_pred) / y_test) * 100  # Mean Percentage Error
    r2 = r2_score(y_test, y_pred)

    # Display results
    st.write(f"### Model: {model_name}")
    st.write("#### Evaluation Metrics")
    st.write(f"- Cross-Validation R2 Score: {np.mean(scores):.4f}")
    st.write(f"- R2 Score: {r2:.4f}")
    st.write(f"- RMSE: {rmse:.4f}")
    st.write(f"- MAE: {mae:.4f}")
    st.write(f"- MPE: {mpe:.2f}%")

    # Results table
    results = {
        "Metric": ["Cross-Validation R2", "R2 Score", "RMSE", "MAE", "MPE"],
        "Value": [np.mean(scores), r2, rmse, mae, mpe]
    }
    results_df = pd.DataFrame(results)
    st.write("### Model Performance Summary")
    st.write(results_df)

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(results["Metric"], results["Value"], color='skyblue')
    ax.set_title(f"Performance Metrics for {model_name}")
    ax.set_ylabel("Metric Value")
    st.pyplot(fig)

    # Visualization: True vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, color='b')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('True vs Predicted Values')
    st.pyplot(fig)

    # Visualization: Residual Errors
    fig, ax = plt.subplots()
    residuals = y_test - y_pred
    ax.hist(residuals, bins=20, color='orange', edgecolor='black')
    ax.set_title('Residual Errors')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Hyperparameter tuning
    if st.sidebar.checkbox("Perform Hyperparameter Tuning (Gradient Boosting Only)"):
        if model_name == "Gradient Boosting Regressor":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }

            grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=kfold, scoring='r2')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_

            st.write(f"### Best Hyperparameters for Gradient Boosting Regressor")
            st.write(best_params)
            st.write(f"Best R2 Score from Cross-Validation: {grid_search.best_score_:.4f}")

            # Retrain with best parameters
            best_model = GradientBoostingRegressor(**best_params, random_state=42)
            best_model.fit(X_train, y_train)
            y_best_pred = best_model.predict(X_test)

            # Re-evaluate
            best_r2 = r2_score(y_test, y_best_pred)
            best_rmse = np.sqrt(mean_squared_error(y_test, y_best_pred))
            st.write(f"### Performance After Tuning")
            st.write(f"- R2 Score: {best_r2:.4f}")
            st.write(f"- RMSE: {best_rmse:.4f}")
elif page == "Prediction":
    st.header("Predict Median House Value")

    st.write("### Enter Feature Values")
    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.number_input("Median Income (in $10,000s)", value=5.0)
        HouseAge = st.number_input("Median House Age", value=20)
        AveRooms = st.number_input("Average Number of Rooms per Household", value=6.0)
        AveBedrms = st.number_input("Average Number of Bedrooms per Household", value=1.0)

    with col2:
        Population = st.number_input("Population in Block Group", value=300)
        AveOccup = st.number_input("Average Number of Occupants per Household", value=3.0)
        Latitude = st.number_input("Block Group Latitude", value=34.0)
        Longitude = st.number_input("Block Group Longitude", value=-118.0)

    # Prepare input data
    user_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    user_data_scaled = scaler.transform(user_data)

    # Predict with Gradient Boosting Regressor by default
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X, y)
    prediction = gbr.predict(user_data_scaled)

    # Display prediction
    st.write("### Predicted Median House Value")
    st.write(f"Predicted Value: ${prediction[0] * 100000:.2f}")

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.write("### Compare Performance of All Models")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "Support Vector Regressor": SVR()
    }

    # Initialize results dictionary
    comparison_results = {
        "Model": [],
        "R2 Score": [],
        "RMSE": [],
        "MAE": [],
        "MPE (%)": []
    }

    # Evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mpe = np.mean((y_test - y_pred) / y_test) * 100

        # Append results
        comparison_results["Model"].append(name)
        comparison_results["R2 Score"].append(r2)
        comparison_results["RMSE"].append(rmse)
        comparison_results["MAE"].append(mae)
        comparison_results["MPE (%)"].append(mpe)

    # Convert results to DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    st.write("### Model Comparison Table")
    st.write(comparison_df)

    # Visualize results
    st.write("### Visualization of Model Performance")
    metrics = ["R2 Score", "RMSE", "MAE", "MPE (%)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.bar(comparison_results["Model"], comparison_results[metric], color="skyblue")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(comparison_results["Model"], rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

elif page == "Data Analysis":
    st.header("Data Analysis and Visualization")

    # Histograms
    st.subheader("Feature Distributions (Histograms)")
    columns_to_plot = data.columns
    for column in columns_to_plot:
        fig, ax = plt.subplots()
        ax.hist(data[column], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    plt.colorbar(cax)
    ticks = np.arange(0, len(correlation_matrix.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='left')
    ax.set_yticklabels(correlation_matrix.columns)
    ax.set_title("Correlation Matrix", pad=20)
    st.pyplot(fig)

    # Option for Regression Residual Plot
    st.subheader("Residual Plot (Regression Models Only)")
    model_name = st.selectbox("Choose a Regression Model for Residuals",
                              ["Linear Regression", "Decision Tree Regressor", "Gradient Boosting Regressor", "Support Vector Regressor"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the selected model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "Support Vector Regressor":
        model = SVR()

    # Train the model and calculate residuals
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Plot residuals
    fig, ax = plt.subplots()
    ax.scatter(y_test, residuals, alpha=0.6, color='orange', edgecolor='k')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot for {model_name}")
    st.pyplot(fig)

elif page == "Unsupervised Learning":
    st.header("Unsupervised Learning for Exploratory Analysis")

    # PCA for dimensionality reduction
    st.write("### Principal Component Analysis (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
    ax.set_title("PCA - Dimensionality Reduction")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.colorbar(scatter, ax=ax, label='Median House Value')
    st.pyplot(fig)

    # KMeans clustering
    st.write("### KMeans Clustering")
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.5)
    ax.set_title("KMeans Clustering")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)