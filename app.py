
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# 1. Configure Streamlit page
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Prediction Using Python",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. Load and clean dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Advertising.csv")

    # Remove extra spaces and standardize column names
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "tv": "TV",
        "radio": "Radio",
        "newspaper": "Newspaper",
        "sales": "Sales"
    })

    return df


# --------------------------------------------------
# 3. Train Random Forest model
# --------------------------------------------------
@st.cache_resource
def train_model():
    df = load_data()

    required_columns = [
        "TV", "Radio", "Newspaper", "Sales"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    return model


# --------------------------------------------------
# 4. Main application
# --------------------------------------------------
st.title("📊 Sales Prediction Using Python")

st.write(
    "Predict product sales based on TV, Radio, "
    "and Newspaper advertising budgets."
)

st.divider()

try:
    df = load_data()
    model = train_model()

    # ----------------------------------------------
    # 5. Sales prediction inputs
    # ----------------------------------------------
    st.header("💰 Enter Advertising Budgets")

    col1, col2, col3 = st.columns(3)

    with col1:
        tv = st.number_input(
            "TV advertising budget",
            min_value=0.0,
            value=150.0,
            step=10.0
        )

    with col2:
        radio = st.number_input(
            "Radio advertising budget",
            min_value=0.0,
            value=25.0,
            step=5.0
        )

    with col3:
        newspaper = st.number_input(
            "Newspaper advertising budget",
            min_value=0.0,
            value=20.0,
            step=5.0
        )

    if st.button("Predict Sales", type="primary"):
        input_data = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Sales: {prediction:.2f}"
        )

        st.caption(
            "The prediction uses the same units as "
            "the Sales column in Advertising.csv."
        )

    st.divider()

    # ----------------------------------------------
    # 6. Advertising vs Sales graphs
    # ----------------------------------------------
    st.header("📈 Advertising Budget vs Sales")

    st.write(
        "Explore the relationship between advertising "
        "budgets and sales."
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4)
    )

    features = ["TV", "Radio", "Newspaper"]

    for ax, feature in zip(axes, features):
        sns.regplot(
            data=df,
            x=feature,
            y="Sales",
            ax=ax,
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"}
        )

        ax.set_title(f"{feature} vs Sales")
        ax.set_xlabel(f"{feature} Advertising")
        ax.set_ylabel("Sales")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ----------------------------------------------
    # 7. Correlation heatmap
    # ----------------------------------------------
    st.header("🔥 Correlation Heatmap")

    correlation = df[
        ["TV", "Radio", "Newspaper", "Sales"]
    ].corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title("Correlation Between Advertising and Sales")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Values closer to +1 indicate a stronger positive "
        "linear relationship; values closer to -1 indicate "
        "a stronger negative linear relationship."
    )

    st.divider()

    # ----------------------------------------------
    # 8. Feature importance graph
    # ----------------------------------------------
    st.header("📊 Advertising Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=["TV", "Radio", "Newspaper"]
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    importance.plot(
        kind="barh",
        ax=ax
    )

    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Advertising Channel")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Feature importance shows how the trained model "
        "uses each input for its predictions. It does not "
        "prove that an advertising channel causes sales."
    )

    st.divider()

    # ----------------------------------------------
    # 9. Dataset information
    # ----------------------------------------------
    with st.expander("View Dataset Information"):
        st.write("Dataset shape:", df.shape)
        st.write("Dataset columns:", df.columns.tolist())
        st.dataframe(df.head())

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

except FileNotFoundError:
    st.error(
        "Advertising.csv was not found. Please place "
        "it in the same folder as app.py."
    )

except Exception as error:
    st.error(f"Unable to run the application: {error}")
