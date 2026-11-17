
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Insider Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ AI Insider Threat Detection System")
st.write("Employee Security Monitoring Dashboard")

st.divider()

# Load dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "dataset" / "employee_activity.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(DATA_PATH)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Employee Activity", "Data Analysis"]
    )

    # Overview
    if page == "Overview":
        st.subheader("Security Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            st.metric("Total Features", len(df.columns))

        with col3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        st.divider()

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Dataset Information")
        st.write("Columns:", list(df.columns))

    # Employee Activity
    elif page == "Employee Activity":
        st.subheader("Employee Activity Monitoring")

        search = st.text_input("Search employee or activity")

        if search:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(
                    search, case=False, na=False
                )
            ).any(axis=1)

            filtered_df = df[mask]
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True)
        st.write("Records found:", len(filtered_df))

    # Data Analysis
    elif page == "Data Analysis":
        st.subheader("Activity Analysis")

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox(
                "Select a numerical feature",
                numeric_cols
            )

            st.subheader(f"Distribution of {selected_col}")

            fig, ax = plt.subplots()
            df[selected_col].dropna().hist(ax=ax, bins=20)
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info("No numerical columns found.")

        st.subheader("Missing Values")
        st.bar_chart(df.isnull().sum())

except FileNotFoundError:
    st.error("Dataset not found. Check dataset/employee_activity.csv")

except Exception as e:
    st.error(f"Something went wrong: {e}")
