import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="INX Employee Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0f1117; color: #e2e8f0; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .header-icon { font-size: 3rem; }
    .header-title { font-size: 1.9rem; font-weight: 700; color: #e2e8f0; margin: 0; }
    .header-sub { font-size: 0.9rem; color: #718096; margin: 0.2rem 0 0; }

    /* Metric cards */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #63b3ed; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d3748;
    }

    /* Prediction result */
    .pred-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px solid;
    }
    .pred-low    { background: #2d1515; border-color: #e53e3e; }
    .pred-good   { background: #1a2d1a; border-color: #48bb78; }
    .pred-excel  { background: #1a2340; border-color: #63b3ed; }
    .pred-rating { font-size: 3.5rem; font-weight: 800; }
    .pred-label  { font-size: 1.1rem; font-weight: 600; margin-top: 0.3rem; }
    .pred-desc   { font-size: 0.85rem; color: #a0aec0; margin-top: 0.5rem; }

    /* Input group labels */
    .input-group-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #63b3ed;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.2rem 0 0.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        color: #718096;
    }
    .stTabs [aria-selected="true"] {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
    }

    /* Streamlit overrides */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.82rem !important;
        color: #a0aec0 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stForm"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .preset-btn > button {
        background: #2d3748 !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.8rem !important;
    }

    /* Info boxes */
    .info-box {
        background: #1a2340;
        border-left: 3px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #a0aec0;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #2d2315;
        border-left: 3px solid #ed8936;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #a0aec0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model Artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("inx_employee_performance_model.pkl")
    features = joblib.load("inx_feature_columns.pkl")
    le_dict  = joblib.load("inx_label_encoders.pkl")
    return model, features, le_dict

try:
    model, feature_cols, le_dict = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ─── Constants ───────────────────────────────────────────────────────────────
CAT_OPTIONS = {
    "Gender":                   ["Male", "Female"],
    "EducationBackground":      ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
    "MaritalStatus":            ["Single", "Married", "Divorced"],
    "EmpDepartment":            ["Sales", "Research & Development", "Human Resources", "Finance", "Data Science", "Development"],
    "EmpJobRole":               ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                  "Manufacturing Director", "Healthcare Representative", "Manager",
                                  "Sales Representative", "Research Director", "Human Resources"],
    "BusinessTravelFrequency":  ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    "OverTime":                 ["Yes", "No"],
    "Attrition":                ["Yes", "No"],
}

RATING_INFO = {
    2: ("⚠️", "Low Performer", "pred-low",   "#e53e3e", "Employee needs improvement and targeted support."),
    3: ("✅", "Good Performer", "pred-good",  "#48bb78", "Employee meets expectations and is on track."),
    4: ("🌟", "Excellent Performer", "pred-excel", "#63b3ed", "Top performer — high potential for growth."),
}

PRESETS = {
    "🌟 High Performer": dict(
        Age=34, Gender="Male", EducationBackground="Life Sciences",
        MaritalStatus="Married", EmpDepartment="Development",
        EmpJobRole="Research Scientist", BusinessTravelFrequency="Travel_Rarely",
        DistanceFromHome=5, EmpEducationLevel=4, EmpEnvironmentSatisfaction=4,
        EmpHourlyRate=75, EmpJobInvolvement=4, EmpJobLevel=3,
        EmpJobSatisfaction=4, NumCompaniesWorked=2, OverTime="No",
        EmpLastSalaryHikePercent=22, EmpRelationshipSatisfaction=4,
        TotalWorkExperienceInYears=10, TrainingTimesLastYear=3,
        EmpWorkLifeBalance=3, ExperienceYearsAtThisCompany=7,
        ExperienceYearsInCurrentRole=5, YearsSinceLastPromotion=1,
        YearsWithCurrManager=4, Attrition="No"
    ),
    "⚠️ At Risk": dict(
        Age=29, Gender="Female", EducationBackground="Marketing",
        MaritalStatus="Single", EmpDepartment="Sales",
        EmpJobRole="Sales Representative", BusinessTravelFrequency="Travel_Frequently",
        DistanceFromHome=25, EmpEducationLevel=2, EmpEnvironmentSatisfaction=1,
        EmpHourlyRate=45, EmpJobInvolvement=2, EmpJobLevel=1,
        EmpJobSatisfaction=2, NumCompaniesWorked=6, OverTime="Yes",
        EmpLastSalaryHikePercent=11, EmpRelationshipSatisfaction=2,
        TotalWorkExperienceInYears=5, TrainingTimesLastYear=1,
        EmpWorkLifeBalance=2, ExperienceYearsAtThisCompany=1,
        ExperienceYearsInCurrentRole=1, YearsSinceLastPromotion=4,
        YearsWithCurrManager=1, Attrition="Yes"
    ),
    "📊 Average Employee": dict(
        Age=38, Gender="Male", EducationBackground="Technical Degree",
        MaritalStatus="Married", EmpDepartment="Research & Development",
        EmpJobRole="Laboratory Technician", BusinessTravelFrequency="Travel_Rarely",
        DistanceFromHome=10, EmpEducationLevel=3, EmpEnvironmentSatisfaction=3,
        EmpHourlyRate=60, EmpJobInvolvement=3, EmpJobLevel=2,
        EmpJobSatisfaction=3, NumCompaniesWorked=3, OverTime="No",
        EmpLastSalaryHikePercent=14, EmpRelationshipSatisfaction=3,
        TotalWorkExperienceInYears=12, TrainingTimesLastYear=2,
        EmpWorkLifeBalance=3, ExperienceYearsAtThisCompany=5,
        ExperienceYearsInCurrentRole=3, YearsSinceLastPromotion=2,
        YearsWithCurrManager=3, Attrition="No"
    ),
}

# ─── Helper: encode + predict ────────────────────────────────────────────────
def encode_and_predict(inputs: dict):
    row = inputs.copy()
    for col, le in le_dict.items():
        if col in row:
            row[col] = le.transform([row[col]])[0]
    df_input = pd.DataFrame([row])[feature_cols]
    pred  = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    classes = model.classes_
    return pred, dict(zip(classes, proba))

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-icon">📊</div>
    <div>
        <div class="header-title">INX Future Inc — Employee Performance Intelligence</div>
        <div class="header-sub">Project 10281 · Predict · Analyse · Act · Powered by Tuned Random Forest</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error("⚠️ Model files not found. Upload `inx_employee_performance_model.pkl`, `inx_feature_columns.pkl`, and `inx_label_encoders.pkl` to the app directory.")
    st.stop()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Predict Performance", "📈  Analyse Dataset", "🔍  Model Insights"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # Preset buttons
    st.markdown('<div class="section-header">Quick Load Preset Profile</div>', unsafe_allow_html=True)
    p_col1, p_col2, p_col3, _ = st.columns([1, 1, 1, 2])
    preset_name = None
    with p_col1:
        if st.button("🌟 High Performer"): preset_name = "🌟 High Performer"
    with p_col2:
        if st.button("⚠️ At Risk"):        preset_name = "⚠️ At Risk"
    with p_col3:
        if st.button("📊 Average"):        preset_name = "📊 Average Employee"

    preset = PRESETS.get(preset_name, PRESETS["📊 Average Employee"])
    if "loaded_preset" not in st.session_state:
        st.session_state.loaded_preset = preset
    if preset_name:
        st.session_state.loaded_preset = preset
    P = st.session_state.loaded_preset

    st.markdown('<div class="section-header">Employee Profile</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        # ── Personal ────────────────────────────────────────────────────────
        st.markdown('<div class="input-group-label">👤 Personal</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age        = c1.slider("Age", 18, 60, P["Age"])
        gender     = c2.selectbox("Gender", CAT_OPTIONS["Gender"], index=CAT_OPTIONS["Gender"].index(P["Gender"]))
        marital    = c3.selectbox("Marital Status", CAT_OPTIONS["MaritalStatus"], index=CAT_OPTIONS["MaritalStatus"].index(P["MaritalStatus"]))

        c1, c2 = st.columns(2)
        edu_bg  = c1.selectbox("Education Background", CAT_OPTIONS["EducationBackground"], index=CAT_OPTIONS["EducationBackground"].index(P["EducationBackground"]))
        edu_lvl = c2.slider("Education Level (1–5)", 1, 5, P["EmpEducationLevel"])

        # ── Job ─────────────────────────────────────────────────────────────
        st.markdown('<div class="input-group-label">💼 Job</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        dept    = c1.selectbox("Department", CAT_OPTIONS["EmpDepartment"], index=CAT_OPTIONS["EmpDepartment"].index(P["EmpDepartment"]))
        role    = c2.selectbox("Job Role", CAT_OPTIONS["EmpJobRole"], index=CAT_OPTIONS["EmpJobRole"].index(P["EmpJobRole"]))

        c1, c2, c3 = st.columns(3)
        job_lvl   = c1.slider("Job Level (1–5)", 1, 5, P["EmpJobLevel"])
        travel    = c2.selectbox("Travel Frequency", CAT_OPTIONS["BusinessTravelFrequency"], index=CAT_OPTIONS["BusinessTravelFrequency"].index(P["BusinessTravelFrequency"]))
        overtime  = c3.selectbox("Overtime", CAT_OPTIONS["OverTime"], index=CAT_OPTIONS["OverTime"].index(P["OverTime"]))

        # ── Compensation ─────────────────────────────────────────────────────
        st.markdown('<div class="input-group-label">💰 Compensation</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        hike      = c1.slider("Last Salary Hike %", 10, 25, P["EmpLastSalaryHikePercent"])
        hourly    = c2.slider("Hourly Rate", 30, 100, P["EmpHourlyRate"])
        dist      = c3.slider("Distance From Home (km)", 1, 30, P["DistanceFromHome"])

        # ── Satisfaction ──────────────────────────────────────────────────────
        st.markdown('<div class="input-group-label">😊 Satisfaction Scores (1–4)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        env_sat   = c1.slider("Environment",    1, 4, P["EmpEnvironmentSatisfaction"])
        job_sat   = c2.slider("Job",            1, 4, P["EmpJobSatisfaction"])
        rel_sat   = c3.slider("Relationship",   1, 4, P["EmpRelationshipSatisfaction"])

        c1, c2 = st.columns(2)
        job_inv   = c1.slider("Job Involvement (1–4)", 1, 4, P["EmpJobInvolvement"])
        wlb       = c2.slider("Work-Life Balance (1–4)", 1, 4, P["EmpWorkLifeBalance"])

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown('<div class="input-group-label">📅 Experience</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        total_exp  = c1.slider("Total Experience (yrs)", 0, 40, P["TotalWorkExperienceInYears"])
        co_exp     = c2.slider("Years at INX", 0, 40, P["ExperienceYearsAtThisCompany"])
        role_exp   = c3.slider("Years in Current Role", 0, 20, P["ExperienceYearsInCurrentRole"])

        c1, c2, c3 = st.columns(3)
        promo_yrs  = c1.slider("Years Since Promotion", 0, 15, P["YearsSinceLastPromotion"])
        mgr_yrs    = c2.slider("Years with Manager", 0, 17, P["YearsWithCurrManager"])
        num_co     = c3.slider("Num Companies Worked", 0, 9, P["NumCompaniesWorked"])

        c1, c2 = st.columns(2)
        training   = c1.slider("Training Times Last Year", 0, 6, P["TrainingTimesLastYear"])
        attrition  = c2.selectbox("Attrition", CAT_OPTIONS["Attrition"], index=CAT_OPTIONS["Attrition"].index(P["Attrition"]))

    # ── Prediction Panel ─────────────────────────────────────────────────────
    with right:
        inputs = dict(
            Age=age, Gender=gender, EducationBackground=edu_bg,
            MaritalStatus=marital, EmpDepartment=dept, EmpJobRole=role,
            BusinessTravelFrequency=travel, DistanceFromHome=dist,
            EmpEducationLevel=edu_lvl, EmpEnvironmentSatisfaction=env_sat,
            EmpHourlyRate=hourly, EmpJobInvolvement=job_inv, EmpJobLevel=job_lvl,
            EmpJobSatisfaction=job_sat, NumCompaniesWorked=num_co,
            OverTime=overtime, EmpLastSalaryHikePercent=hike,
            EmpRelationshipSatisfaction=rel_sat, TotalWorkExperienceInYears=total_exp,
            TrainingTimesLastYear=training, EmpWorkLifeBalance=wlb,
            ExperienceYearsAtThisCompany=co_exp, ExperienceYearsInCurrentRole=role_exp,
            YearsSinceLastPromotion=promo_yrs, YearsWithCurrManager=mgr_yrs,
            Attrition=attrition
        )

        pred, proba = encode_and_predict(inputs)
        icon, label, css_class, color, desc = RATING_INFO[pred]

        st.markdown(f"""
        <div class="pred-card {css_class}">
            <div class="pred-rating" style="color:{color}">{icon} {pred}</div>
            <div class="pred-label" style="color:{color}">{label}</div>
            <div class="pred-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence gauge
        confidence = proba[pred] * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
            title={"text": "Prediction Confidence", "font": {"size": 13, "color": "#718096"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4a5568"},
                "bar": {"color": color},
                "bgcolor": "#2d3748",
                "bordercolor": "#4a5568",
                "steps": [
                    {"range": [0, 40],  "color": "#1a1f2e"},
                    {"range": [40, 70], "color": "#1a2340"},
                    {"range": [70, 100],"color": "#1a2d1a"},
                ],
            }
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="#0f1117", font_color="#e2e8f0"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Class probability bars
        st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)
        rating_labels = {2: "Rating 2 — Low", 3: "Rating 3 — Good", 4: "Rating 4 — Excellent"}
        bar_colors    = {2: "#e53e3e", 3: "#48bb78", 4: "#63b3ed"}

        for r in [2, 3, 4]:
            pct = proba.get(r, 0) * 100
            highlight = "font-weight:700;" if r == pred else "opacity:0.6;"
            st.markdown(f"""
            <div style="margin-bottom:0.6rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; {highlight}">
                    <span>{rating_labels[r]}</span><span>{pct:.1f}%</span>
                </div>
                <div style="background:#2d3748; border-radius:4px; height:8px; margin-top:3px;">
                    <div style="width:{pct}%; background:{bar_colors[r]}; height:8px; border-radius:4px; transition:width 0.5s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Key flags
        st.markdown('<div class="section-header">Key Signals</div>', unsafe_allow_html=True)
        if hike >= 20:
            st.markdown('<div class="info-box">🟢 High salary hike — strong performance indicator</div>', unsafe_allow_html=True)
        elif hike <= 12:
            st.markdown('<div class="warn-box">🔴 Low salary hike — linked to lower performance</div>', unsafe_allow_html=True)
        if promo_yrs >= 4:
            st.markdown('<div class="warn-box">🔴 No promotion in 4+ years — stagnation risk</div>', unsafe_allow_html=True)
        if env_sat >= 3:
            st.markdown('<div class="info-box">🟢 Good environment satisfaction</div>', unsafe_allow_html=True)
        elif env_sat == 1:
            st.markdown('<div class="warn-box">🔴 Very low environment satisfaction</div>', unsafe_allow_html=True)
        if overtime == "Yes" and job_sat <= 2:
            st.markdown('<div class="warn-box">🟡 Overtime + low job satisfaction — burnout risk</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Upload Dataset for Live Analysis</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload `INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1_8.xls`",
                                type=["xls", "xlsx", "csv"])

    @st.cache_data
    def load_data(file_bytes, fname):
        if fname.endswith(".csv"):
            return pd.read_csv(file_bytes)
        return pd.read_excel(file_bytes, engine="xlrd" if fname.endswith(".xls") else "openpyxl")

    if uploaded:
        df = load_data(uploaded, uploaded.name)
        df_clean = df.drop(columns=["EmpNumber"], errors="ignore")

        # ── KPI Row ──────────────────────────────────────────────────────────
        total     = len(df)
        avg_hike  = df["EmpLastSalaryHikePercent"].mean()
        avg_age   = df["Age"].mean()
        pct_ot    = (df["OverTime"] == "Yes").mean() * 100
        pct_attr  = (df["Attrition"] == "Yes").mean() * 100

        k1, k2, k3, k4, k5 = st.columns(5)
        for col, lbl, val, fmt in [
            (k1, "Total Employees", total, "{:.0f}"),
            (k2, "Avg Salary Hike %", avg_hike, "{:.1f}%"),
            (k3, "Avg Age", avg_age, "{:.1f}"),
            (k4, "Overtime Rate", pct_ot, "{:.1f}%"),
            (k5, "Attrition Rate", pct_attr, "{:.1f}%"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{fmt.format(val)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Department + Rating Distribution ───────────────────────────
        c1, c2 = st.columns(2)

        with c1:
            dept_perf = df.groupby(["EmpDepartment", "PerformanceRating"]).size().reset_index(name="Count")
            fig1 = px.bar(dept_perf, x="EmpDepartment", y="Count", color="PerformanceRating",
                          barmode="group", color_discrete_map={2: "#e53e3e", 3: "#48bb78", 4: "#63b3ed"},
                          title="Department-wise Performance Rating",
                          labels={"EmpDepartment": "", "PerformanceRating": "Rating"})
            fig1.update_layout(
                paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                font_color="#a0aec0", title_font_color="#e2e8f0",
                legend_title_text="Rating", height=340,
                xaxis=dict(tickangle=-20),
                margin=dict(t=45, b=10, l=10, r=10)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            rating_counts = df["PerformanceRating"].value_counts().reset_index()
            rating_counts.columns = ["Rating", "Count"]
            rating_counts["Label"] = rating_counts["Rating"].map({2: "Low", 3: "Good", 4: "Excellent"})
            fig2 = px.pie(rating_counts, values="Count", names="Label",
                          title="Overall Performance Distribution",
                          color_discrete_sequence=["#e53e3e", "#48bb78", "#63b3ed"],
                          hole=0.5)
            fig2.update_layout(
                paper_bgcolor="#1a1f2e", font_color="#a0aec0",
                title_font_color="#e2e8f0", height=340,
                margin=dict(t=45, b=10, l=10, r=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Row 2: Hike vs Performance + Avg Dept Rating ──────────────────────
        c1, c2 = st.columns(2)

        with c1:
            fig3 = px.box(df, x="PerformanceRating", y="EmpLastSalaryHikePercent",
                          color="PerformanceRating",
                          color_discrete_map={2: "#e53e3e", 3: "#48bb78", 4: "#63b3ed"},
                          title="Salary Hike % by Performance Rating",
                          labels={"PerformanceRating": "Rating", "EmpLastSalaryHikePercent": "Salary Hike %"})
            fig3.update_layout(
                paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                font_color="#a0aec0", title_font_color="#e2e8f0",
                showlegend=False, height=320,
                margin=dict(t=45, b=10, l=10, r=10)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            dept_avg = df.groupby("EmpDepartment")["PerformanceRating"].mean().sort_values().reset_index()
            fig4 = px.bar(dept_avg, x="PerformanceRating", y="EmpDepartment",
                          orientation="h", title="Avg Performance Rating by Department",
                          labels={"EmpDepartment": "", "PerformanceRating": "Avg Rating"},
                          color="PerformanceRating",
                          color_continuous_scale=["#e53e3e", "#48bb78", "#63b3ed"])
            fig4.update_layout(
                paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                font_color="#a0aec0", title_font_color="#e2e8f0",
                coloraxis_showscale=False, height=320,
                margin=dict(t=45, b=10, l=10, r=10)
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── Row 3: Satisfaction heatmap ───────────────────────────────────────
        sat_cols = ["EmpEnvironmentSatisfaction", "EmpJobSatisfaction",
                    "EmpRelationshipSatisfaction", "EmpWorkLifeBalance", "EmpJobInvolvement"]
        sat_by_rating = df.groupby("PerformanceRating")[sat_cols].mean().round(2)

        fig5 = px.imshow(sat_by_rating,
                         title="Avg Satisfaction Scores by Performance Rating",
                         color_continuous_scale="Blues",
                         text_auto=True,
                         labels={"x": "Satisfaction Metric", "y": "Performance Rating", "color": "Score"})
        fig5.update_layout(
            paper_bgcolor="#1a1f2e", font_color="#a0aec0",
            title_font_color="#e2e8f0", height=280,
            margin=dict(t=45, b=10, l=10, r=10)
        )
        st.plotly_chart(fig5, use_container_width=True)

    else:
        st.markdown("""
        <div class="warn-box">
            Upload the original <code>.xls</code> dataset to unlock live charts:
            department breakdowns, salary hike distributions, satisfaction heatmaps, and more.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:

    # Feature importance from saved model
    rf_inner = model.named_steps["model"]
    importances = pd.Series(rf_inner.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)

    fig_fi = px.bar(importances, orientation="h",
                    title="Top 15 Feature Importances — Tuned Random Forest",
                    labels={"value": "Importance Score", "index": "Feature"},
                    color=importances.values,
                    color_continuous_scale=["#2d3748", "#3182ce", "#63b3ed"])
    fig_fi.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#a0aec0", title_font_color="#e2e8f0",
        coloraxis_showscale=False, showlegend=False,
        height=480, margin=dict(t=45, b=10, l=180, r=20)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Rating guide
    st.markdown('<div class="section-header">Performance Rating Guide</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    for col, r in zip([g1, g2, g3], [2, 3, 4]):
        icon, label, css_class, color, desc = RATING_INFO[r]
        col.markdown(f"""
        <div class="pred-card {css_class}" style="padding:1.2rem;">
            <div style="font-size:2rem; color:{color}">{icon} {r}</div>
            <div style="font-weight:600; color:{color}; margin-top:0.3rem">{label}</div>
            <div style="font-size:0.8rem; color:#a0aec0; margin-top:0.4rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # Top 3 factors
    st.markdown('<div class="section-header">Top 3 Factors — Deliverable 2</div>', unsafe_allow_html=True)
    factors = [
        ("1", "EmpLastSalaryHikePercent", "#63b3ed",
         "Highest importance feature. Employees with ≥20% hike are 3× more likely to be Rating 4."),
        ("2", "EmpEnvironmentSatisfaction", "#48bb78",
         "Second most predictive. Low satisfaction (score 1) strongly predicts Rating 2."),
        ("3", "YearsSinceLastPromotion", "#ed8936",
         "Employees not promoted in 4+ years show significant performance decline."),
    ]
    for rank, name, color, insight in factors:
        st.markdown(f"""
        <div style="background:#1a1f2e; border:1px solid #2d3748; border-left:4px solid {color};
                    border-radius:0 10px 10px 0; padding:1rem 1.2rem; margin-bottom:0.8rem;">
            <div style="display:flex; align-items:center; gap:0.8rem;">
                <div style="font-size:1.5rem; font-weight:800; color:{color};">#{rank}</div>
                <div>
                    <div style="font-weight:600; color:#e2e8f0; font-family:monospace;">{name}</div>
                    <div style="font-size:0.82rem; color:#a0aec0; margin-top:0.2rem;">{insight}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Model info
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
    params = rf_inner.get_params()
    info_items = {
        "Algorithm": "Random Forest Classifier (Tuned)",
        "Estimators": str(params.get("n_estimators", "—")),
        "Max Depth":  str(params.get("max_depth", "None (unlimited)")),
        "Min Samples Split": str(params.get("min_samples_split", "—")),
        "Features Used": str(len(feature_cols)),
        "Target Classes": "2 (Low), 3 (Good), 4 (Excellent)",
        "Evaluation Metric": "F1 Macro (handles class imbalance)",
    }
    m1, m2 = st.columns(2)
    items = list(info_items.items())
    for i, (k, v) in enumerate(items):
        col = m1 if i % 2 == 0 else m2
        col.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:0.5rem 0;
                    border-bottom:1px solid #2d3748; font-size:0.85rem;">
            <span style="color:#718096">{k}</span>
            <span style="color:#e2e8f0; font-weight:500">{v}</span>
        </div>""", unsafe_allow_html=True)