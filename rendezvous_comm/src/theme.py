"""Polimi-branded theme constants and Streamlit CSS injection."""

# ── Politecnico di Milano brand colours ──────────────────────────────
POLIMI_DARK_BLUE = "#002F6C"
POLIMI_LIGHT_BLUE = "#009CDC"
POLIMI_WHITE = "#FFFFFF"
POLIMI_GRAY = "#333333"
POLIMI_GREEN = "#00A651"
POLIMI_RED = "#E4002B"
POLIMI_ORANGE = "#F39200"
POLIMI_LIGHT_GRAY = "#F0F2F6"


def apply_theme():
    """Inject Polimi CSS into Streamlit: hide chrome, apply brand."""
    import streamlit as st

    st.markdown(
        """
        <style>
        /* Hide hamburger menu, deploy button, footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header [data-testid="stToolbar"] {display: none;}

        /* Polimi header accent */
        .stApp > header {
            background-color: #002F6C;
        }

        /* Sidebar header */
        [data-testid="stSidebar"] > div:first-child {
            background-color: #F0F2F6;
        }

        /* Metric card styling */
        [data-testid="stMetric"] {
            background-color: #F0F2F6;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 4px solid #002F6C;
        }
        [data-testid="stMetric"] label {
            color: #333333;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #002F6C;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
