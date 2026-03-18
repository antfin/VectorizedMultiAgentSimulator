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


def apply_theme(title: str = ""):
    """Inject Polimi CSS + optional top bar into Streamlit.

    Args:
        title: if provided, renders a dark-blue top banner with this text.
    """
    import streamlit as st

    st.markdown(
        """
        <style>
        /* Hide deploy button, main menu, footer — but keep toolbar
           visible so sidebar expand/collapse button works */
        footer {visibility: hidden;}
        [data-testid="stAppDeployButton"] {display: none !important;}
        [data-testid="stMainMenuButton"] {display: none !important;}
        [data-testid="stToolbarActions"] {display: none !important;}

        /* Keep native header white, shrink to minimal height —
           but expand when running (spinner visible) */
        [data-testid="stHeader"],
        .stAppHeader {
            background-color: #FFFFFF !important;
            height: 15px !important;
            min-height: 15px !important;
        }
        [data-testid="stHeader"]:has([data-testid="stStatusWidget"]),
        .stAppHeader:has([data-testid="stStatusWidget"]) {
            height: auto !important;
            min-height: 40px !important;
        }

        /* Make spinner/status text visible (dark blue on white) */
        [data-testid="stStatusWidget"],
        [data-testid="stStatusWidget"] * {
            color: #002F6C !important;
        }
        [data-testid="stStatusWidget"] svg {
            fill: #002F6C !important;
            stroke: #002F6C !important;
        }

        /* Reduce top padding to match smaller header */
        [data-testid="stMainBlockContainer"],
        .stMainBlockContainer {
            padding-top: 1.5rem !important;
        }

        /* Sidebar — dark blue with white text */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #002F6C !important;
        }

        /* Sidebar nav links — white text */
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] a span,
        [data-testid="stSidebar"] a p,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p {
            color: #FFFFFF !important;
        }

        /* Sidebar active page highlight — light blue accent */
        [data-testid="stSidebar"] a[aria-current="page"],
        [data-testid="stSidebar"] li:has(a[aria-current="page"]) {
            background-color: rgba(0, 156, 220, 0.25) !important;
            border-radius: 6px;
        }

        /* Sidebar collapse button (<<) — always visible, white on dark blue */
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button span {
            color: #FFFFFF !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stSidebar"] button svg {
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        /* Expand button (>>) — always visible, dark blue on white header */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stExpandSidebarButton"] span {
            color: #002F6C !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stExpandSidebarButton"] svg {
            fill: #002F6C !important;
            stroke: #002F6C !important;
        }

        /* Sidebar widget labels (selectbox, radio, etc.) */
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] [data-testid="stMetric"] label {
            color: #FFFFFF !important;
        }

        /* Sidebar buttons — light blue bg with white text */
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
        [data-testid="stSidebar"] button[kind="secondary"],
        [data-testid="stSidebar"] .stButton button {
            background-color: #009CDC !important;
            color: #FFFFFF !important;
            border: none !important;
        }

        /* Sidebar selectbox dropdown arrow */
        [data-testid="stSidebar"] .stSelectbox svg {
            fill: #333333 !important;
        }

        /* Sidebar metric cards — adapt to dark bg */
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-left: 4px solid #009CDC !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }

        /* Top banner — sits inside content flow */
        .polimi-banner {
            background-color: #002F6C;
            color: white;
            padding: 18px 24px 14px 24px;
            margin: -1rem -1rem 1.5rem -1rem;
            font-size: 1.4rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            border-bottom: 3px solid #009CDC;
        }
        .polimi-banner .subtitle {
            font-size: 0.85rem;
            font-weight: 400;
            opacity: 0.8;
            margin-top: 4px;
        }

        /* Metric card styling (main content) */
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

    if title:
        st.markdown(
            f"""
            <div class="polimi-banner">
                {title}
                <div class="subtitle">
                    Politecnico di Milano — PhD Research Dashboard
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
