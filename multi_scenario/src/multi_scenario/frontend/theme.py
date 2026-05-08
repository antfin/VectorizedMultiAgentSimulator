"""Polimi-branded theme constants and Streamlit CSS injection.

Ported from ``rendezvous_comm/src/theme.py`` (F7.1) — same brand palette
and CSS rules; only the banner subtitle text differs (project-agnostic).
"""

# ── Politecnico di Milano brand colours ──────────────────────────────
POLIMI_DARK_BLUE = "#002F6C"
POLIMI_LIGHT_BLUE = "#009CDC"
POLIMI_WHITE = "#FFFFFF"
POLIMI_GRAY = "#333333"
POLIMI_GREEN = "#00A651"
POLIMI_RED = "#E4002B"
POLIMI_ORANGE = "#F39200"
POLIMI_LIGHT_GRAY = "#F0F2F6"


def apply_theme(title: str = "", subtitle: str = "Politecnico di Milano — PhD Research") -> None:
    """Inject Polimi CSS + optional top banner into Streamlit.

    Args:
        title: if provided, renders a dark-blue top banner with this text.
        subtitle: smaller text below the title; Polimi attribution by default.
    """
    # Local import keeps theme.py importable in non-Streamlit contexts (tests,
    # tools that just want the colour constants).
    # pylint: disable=import-outside-toplevel
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

        /* Sidebar text — white across the board (links, headings, captions) */
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] a span,
        [data-testid="stSidebar"] a p,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: #FFFFFF !important;
        }

        /* Sidebar inputs — Streamlit gives them a white wrapper; keep that
           and just darken the typed text so it's visible on white. */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            color: #002F6C !important;
        }

        /* Help "?" icon on sidebar widget labels — Streamlit's SVG uses
           ``stroke="currentColor"`` with ``fill="none"`` so the icon is an
           OUTLINED glyph, not a solid shape. Driving ``color`` is enough;
           do NOT force ``fill`` (would solid-fill the circle, hiding the
           "?" inside). */
        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
        [data-testid="stSidebar"] button[aria-label^="Help"] svg {
            color: #009CDC !important;
            stroke: #009CDC !important;
        }
        [data-testid="stSidebar"] button[aria-label^="Help"] {
            background-color: transparent !important;
            border: none !important;
        }

/* Multi-page nav (stSidebarNav) — keep visible with subtle hover */
        [data-testid="stSidebarNav"] {
            background-color: transparent !important;
            padding-top: 0.5rem;
        }
        /* "Pages" label above the nav — st.navigation always pins itself to
           the top of the sidebar, so widget calls before ``nav.run()`` end
           up *below* it. CSS ``::before`` is the cleanest way to put a
           heading above. Sized to match the per-page ``st.sidebar.header``
           h2s ("Scenario" / "Run") for visual consistency. */
        [data-testid="stSidebarNav"]::before {
            content: "Pages";
            display: block;
            color: #FFFFFF !important;
            font-size: 1.5rem;
            font-weight: 600;
            line-height: 1.2;
            padding: 0 8px 0.5rem 8px;
        }
        [data-testid="stSidebarNav"] li a:hover {
            background-color: rgba(0, 156, 220, 0.15) !important;
            border-radius: 6px;
        }
        /* Section header ("Experiments") — align its text with the page-icon
           column (8px nudge) and disable the default click-to-collapse so
           the user can't accidentally hide the children. */
        [data-testid="stNavSectionHeader"] {
            padding-left: 8px !important;
            pointer-events: none !important;
            cursor: default !important;
        }
        /* Embedded videos (st.video) — sensible min/max so they never dwarf
           the page on huge screens but stay legible on small ones. The
           ``stVideo`` testid wraps the actual ``<video>`` element. */
        [data-testid="stVideo"],
        [data-testid="stVideo"] video {
            max-width: 480px !important;
            min-width: 240px;
            width: 100%;
            height: auto;
            margin: 0 auto;
            display: block;
        }

        /* Matplotlib charts (st.pyplot) render as ``<img>`` inside
           ``[data-testid="stImage"]`` (or its newer rename ``stImageContainer``).
           min-width keeps tiny charts legible inside a narrow column;
           max-width caps how big they grow on huge screens (1440px+);
           ``object-fit: contain`` preserves the aspect ratio between bounds.
           ``!important`` is required on max-width — Streamlit's own default
           rule sets ``max-width: 100%`` at higher specificity, which lets the
           image fill the column on wide screens. */
        [data-testid="stImage"] img,
        [data-testid="stImageContainer"] img {
            min-width: 280px;
            max-width: 720px !important;
            object-fit: contain;
        }

        /* Path caption pinned at the bottom of the sidebar — single-line,
           ellipsised; HTML ``title`` attribute supplies the full path on hover. */
        .ms-path-caption {
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 0.72rem !important;
            color: rgba(255, 255, 255, 0.65) !important;
            padding: 0 8px;
            margin-top: 0.5rem;
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
                    {subtitle}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
