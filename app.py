import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import os

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditSense · Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ───────────────────────────────────────────────
DM_TO_INR = 35          # 1 DM ≈ ₹35 (modern purchasing-power equivalent)
INR_MIN   = 5_000
INR_MAX   = 700_000
INR_STEP  = 5_000

def inr_to_dm(inr: float) -> float:
    return inr / DM_TO_INR

def dm_to_inr(dm: float) -> float:
    return dm * DM_TO_INR

def fmt_inr(n: float) -> str:
    return f"₹{int(n):,}"

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0c0f1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111520 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #c8cfe0 !important; }

/* ── Main container ── */
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* ── Typography ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: #f0f4ff; letter-spacing: -.02em;
    line-height: 1.1; margin: 0 0 .2rem;
}
.hero-sub  { font-size:.85rem; color:#6b7fa3; font-weight:300; margin-bottom:1.2rem; }
.hero-accent { color: #5b7cfa; }
h1,h2,h3  { color:#e8eeff !important; font-family:'Syne',sans-serif !important; }
p, li      { color: #8a9bbf; }

/* ── Section labels ── */
.section-label {
    font-family:'Syne',sans-serif; font-size:.65rem; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase; color:#5b7cfa;
    display:block; margin:.8rem 0 .4rem;
}

/* ── Card wrappers ── */
.form-card {
    background:#161b2e; border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:1rem;
}
.result-panel {
    background:#111827; border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:1.2rem 1.4rem; height:100%;
}

/* ── Divider ── */
.styled-divider {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(91,124,250,.35),transparent);
    margin:.8rem 0;
}

/* ── Verdict cards ── */
.result-good {
    background:linear-gradient(135deg,#0d2b1f,#12352a);
    border:1px solid rgba(62,207,142,.3); border-radius:14px;
    padding:1.2rem; text-align:center; margin-bottom:.8rem;
}
.result-bad {
    background:linear-gradient(135deg,#2b0d0d,#351212);
    border:1px solid rgba(255,99,99,.3); border-radius:14px;
    padding:1.2rem; text-align:center; margin-bottom:.8rem;
}
.result-label { font-family:'Syne',sans-serif; font-size:.75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.3rem; }
.result-prob  { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800; line-height:1; }

/* ── Risk bar ── */
.risk-bar-container { background:#1e2538; border-radius:100px; height:7px; margin:.4rem 0 .15rem; overflow:hidden; }
.risk-bar-fill      { height:100%; border-radius:100px; }

/* ── Factor pills ── */
.factor-pill {
    display:flex; align-items:center; gap:.5rem;
    padding:.35rem .6rem; background:#1e2538;
    border-radius:7px; margin-bottom:.3rem;
    border-left:3px solid transparent;
}


/* ── Streamlit widgets ── */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    color:#8a9bbf !important; font-size:.78rem !important; font-weight:400 !important;
}
input[type="number"], .stSelectbox > div > div {
    background:#1e2538 !important; border:1px solid rgba(255,255,255,.1) !important;
    border-radius:9px !important; color:#e8eeff !important;
}
.stFormSubmitButton > button {
    background:linear-gradient(135deg,#5b7cfa,#3b5bdb) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:.88rem !important; padding:.55rem 1.5rem !important;
    box-shadow:0 4px 18px rgba(91,124,250,.32) !important;
}
.stFormSubmitButton > button:hover { opacity:.9 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#161b2e; border-radius:10px; padding:3px; gap:3px;
    border:1px solid rgba(255,255,255,.06);
}
.stTabs [data-baseweb="tab"] {
    background:transparent; border-radius:7px;
    color:#6b7fa3 !important; font-size:.8rem; padding:.4rem 1rem;
}
.stTabs [aria-selected="true"] { background:#5b7cfa !important; color:#fff !important; }

/* ── Metric cards ── */
.metric-card {
    background:#161b2e; border:1px solid rgba(255,255,255,.07);
    border-radius:12px; padding:.9rem 1.1rem;
}
.metric-card .mc-label { font-size:.62rem; font-weight:500; letter-spacing:.11em; text-transform:uppercase; color:#4d5e80; margin-bottom:.3rem; }
.metric-card .mc-value { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:#e8eeff; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0c0f1a; }
::-webkit-scrollbar-thumb { background:#2a3352; border-radius:3px; }
.stAlert { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("outputs/rf_model.pkl")
    scaler        = joblib.load("outputs/scaler.pkl")
    le_dict       = joblib.load("outputs/le_dict.pkl")
    feature_names = joblib.load("outputs/feature_names.pkl")
    return model, scaler, le_dict, feature_names

@st.cache_data
def load_data():
    return pd.read_csv("german_credit_data.csv", index_col=0)

try:
    model, scaler, le_dict, feature_names = load_artifacts()
    df_raw = load_data()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.warning(f"⚠️ Demo mode – model artifacts not found. ({e})")


# ─── Helpers ─────────────────────────────────────────────────
def preprocess_and_predict(df_input):
    """Accepts a DataFrame whose 'Credit amount' column is already in DM."""
    df_enc = df_input.copy()
    for col in ["Saving accounts"]:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].fillna("little")

    clip_map = {
        "Age":           (4.5,     64.5),
        "Credit amount": (-2544.6, 7882.4),
        "Duration":      (-6.0,    42.0),
    }
    for col, (lo, hi) in clip_map.items():
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].clip(lo, hi)

    cat_cols = ["Sex", "Housing", "Saving accounts", "Purpose"]
    for col in cat_cols:
        if col in df_enc.columns:
            le = le_dict[col]
            df_enc[col] = df_enc[col].apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else 0
            )

    df_enc  = df_enc[feature_names]
    scaled  = scaler.transform(df_enc)
    preds   = model.predict(scaled)
    probs   = model.predict_proba(scaled)
    labels  = le_dict["Risk"].inverse_transform(preds)
    good_p  = (probs[:, 1] * 100).round(2)
    bad_p   = (probs[:, 0] * 100).round(2)
    return labels, good_p, bad_p


def mock_score(age, dm, dur, saving, job, housing):
    """Rule-based approximation when model files are absent."""
    import random
    s = 60
    if 30 <= age <= 55: s += 8
    elif age < 25:       s -= 10
    if dm < 2000:        s += 10
    elif dm > 7000:      s -= 14
    if dur <= 18:        s += 8
    elif dur > 48:       s -= 12
    if saving == "rich":        s += 15
    elif saving == "quite rich": s += 8
    elif saving == "little":     s -= 8
    if job == 3: s += 6
    elif job == 0: s -= 6
    if housing == "own":  s += 4
    elif housing == "rent": s -= 3
    s += random.uniform(-3, 3)
    return round(min(97, max(5, s)), 1)


def grade(gp):
    if gp >= 85: return "A+", "#3ecf8e"
    if gp >= 75: return "A",  "#5bcfa0"
    if gp >= 65: return "B",  "#5b7cfa"
    if gp >= 55: return "C",  "#f5a623"
    if gp >= 45: return "D",  "#ff8c42"
    return          "F",  "#ff6363"


def risk_factors(age, dm, dur, saving, job):
    f = []
    if dm > 6000:
        f.append(("⚠", "High loan amount increases default risk", "warn"))
    if dur > 36:
        f.append(("⚠", "Long duration adds repayment exposure", "warn"))
    if saving == "little":
        f.append(("⚠", "Low savings reduce repayment confidence", "warn"))
    if age < 25:
        f.append(("⚠", "Young applicant — limited credit history", "warn"))
    if saving in ("quite rich", "rich"):
        f.append(("✓", "Strong savings buffer improves outlook", "good"))
    if job >= 2:
        f.append(("✓", "Skilled employment signals stable income", "good"))
    if dm < 2000:
        f.append(("✓", "Small loan amount reduces default risk", "good"))
    if age > 35:
        f.append(("✓", "Experienced age bracket — lower risk", "good"))
    return f


def gauge_fig(gp):
    fig, ax = plt.subplots(figsize=(3.6, 2.0), facecolor="none")
    ax.set_facecolor("none")
    th = np.linspace(np.pi, 0, 300)
    ro, ri = 1.0, 0.62

    segs = [(0, 0.40, "#2b1515"), (0.40, 0.65, "#2b1f15"), (0.65, 1.0, "#1a2b1a")]
    lin  = np.linspace(0, 1, 300)
    for lo, hi, c in segs:
        idx = np.where((lin >= lo) & (lin <= hi))[0]
        if len(idx):
            t = th[idx]
            ax.fill(np.concatenate([np.cos(t)*ro, np.cos(t[::-1])*ri]),
                    np.concatenate([np.sin(t)*ro, np.sin(t[::-1])*ri]),
                    color=c, zorder=1)

    fe   = np.pi * (1 - gp / 100)
    tf   = np.linspace(np.pi, fe, 300)
    col  = "#3ecf8e" if gp >= 65 else "#f5a623" if gp >= 45 else "#ff6363"
    ax.fill(np.concatenate([np.cos(tf)*ro, np.cos(tf[::-1])*ri]),
            np.concatenate([np.sin(tf)*ro, np.sin(tf[::-1])*ri]),
            color=col, zorder=2, alpha=0.9)

    ang = np.pi * (1 - gp / 100)
    ax.plot([0.32*np.cos(ang), 0.88*np.cos(ang)],
            [0.32*np.sin(ang), 0.88*np.sin(ang)],
            color="white", lw=2.2, zorder=5)
    ax.add_patch(plt.Circle((0,0), 0.07, color="white", zorder=6))
    ax.add_patch(plt.Circle((0,0), 0.04, color="#0c0f1a", zorder=7))
    ax.text(0, 0.17, f"{gp:.0f}%", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white", zorder=8)
    ax.text(-0.9, -0.1, "High Risk", ha="center", fontsize=6.5, color="#6b7fa3")
    ax.text( 0.9, -0.1, "Low Risk",  ha="center", fontsize=6.5, color="#6b7fa3")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.3, 1.15); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def prob_bar_fig(gp, bp):
    fig, ax = plt.subplots(figsize=(4.5, 1.8), facecolor="none")
    ax.set_facecolor("none")
    bars = ax.barh(["Good Credit", "Bad Credit"], [gp, bp],
                   color=["#3ecf8e", "#ff6363"], height=0.42, edgecolor="none")
    for b, v in zip(bars, [gp, bp]):
        ax.text(v+1, b.get_y()+b.get_height()/2, f"{v:.1f}%",
                va="center", ha="left", fontsize=11, fontweight="bold", color="white")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Probability (%)", color="#6b7fa3", fontsize=8)
    ax.tick_params(colors="#6b7fa3", labelsize=9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#2a3352")
    ax.spines["left"].set_color("#2a3352")
    ax.xaxis.grid(True, ls="--", alpha=0.12, color="white"); ax.set_axisbelow(True)
    plt.tight_layout(pad=0.4)
    return fig


def render_result(label, gp, bp, age, dm, dur, saving, job,
                  housing, sex, purpose, inr_amount):
    g, gc = grade(gp)
    col_g, col_r = st.columns([1, 1.2])

    with col_g:
        st.markdown("**Creditworthiness Gauge**")
        fig = gauge_fig(gp); st.pyplot(fig, use_container_width=True); plt.close()

    with col_r:
        if label == "good":
            st.markdown(f"""
            <div class="result-good">
                <div class="result-label" style="color:#3ecf8e;">✓ Approved · Good Credit</div>
                <div class="result-prob" style="color:#3ecf8e;">{gp:.1f}%</div>
                <div style="font-size:.75rem;color:#5ecfa0;margin-top:.2rem;">Creditworthiness probability</div>
                <div style="margin-top:.7rem;display:inline-block;background:#3ecf8e22;border:1px solid #3ecf8e44;
                            border-radius:7px;padding:.25rem .7rem;font-family:'Syne',sans-serif;
                            font-size:1.3rem;font-weight:800;color:{gc};">Grade: {g}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-bad">
                <div class="result-label" style="color:#ff6363;">✕ Declined · High Risk</div>
                <div class="result-prob" style="color:#ff6363;">{bp:.1f}%</div>
                <div style="font-size:.75rem;color:#ff9090;margin-top:.2rem;">Default probability</div>
                <div style="margin-top:.7rem;display:inline-block;background:#ff636322;border:1px solid #ff636344;
                            border-radius:7px;padding:.25rem .7rem;font-family:'Syne',sans-serif;
                            font-size:1.3rem;font-weight:800;color:{gc};">Grade: {g}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    col_bar, col_fac = st.columns([1.1, 1])
    with col_bar:
        st.markdown("**Prediction Probabilities**")
        fig2 = prob_bar_fig(gp, bp); st.pyplot(fig2, use_container_width=True); plt.close()

    with col_fac:
        st.markdown("**Key Risk Factors**")
        for icon, text, kind in risk_factors(age, dm, dur, saving, job):
            color = "#3ecf8e" if kind == "good" else "#f5a623"
            st.markdown(f"""
            <div class="factor-pill" style="border-left-color:{color};">
                <span style="color:{color};font-size:.85rem;">{icon}</span>
                <span style="font-size:.73rem;color:#a0b0cc;">{text}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
  
    

    st.markdown("")
    if label == "good":
        st.success(f"✅ **Recommendation:** Grade **{g}** · {gp:.1f}% creditworthiness. "
                   f"Consider proceeding with standard loan terms.")
    else:
        st.error(f"🚨 **Recommendation:** Elevated default risk ({bp:.1f}%). "
                 f"Consider requiring additional collateral, reducing amount, or declining.")


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.8rem 0 1.5rem;">
        <p style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                  color:#e8eeff;margin:0;letter-spacing:-.01em;">💳 CreditSense</p>
        <p style="font-size:.68rem;color:#4d5e80;margin:.15rem 0 0;
                  letter-spacing:.08em;text-transform:uppercase;">Risk Intelligence Platform</p>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation",
                    ["🔮 Predict Risk", "⚖️ Compare Applicants", "📋 Batch Score"],
                    label_visibility="collapsed")

    
    


# ══════════════════════════════════════════════════════════════
# PAGE — PREDICT RISK  (form + result on same page, side-by-side)
# ══════════════════════════════════════════════════════════════
if page == "🔮 Predict Risk":
    st.markdown('<h1 class="hero-title">Credit <span class="hero-accent">Risk</span> Assessment</h1>',
                unsafe_allow_html=True)
    

    left_col, right_col = st.columns([1, 1.35], gap="large")

    # ── LEFT: input form ──────────────────────────────────────
    with left_col:
        with st.form("predict_form"):
            st.markdown('<span class="section-label">Personal Information</span>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 18, 80, 35)
            with c2:
                sex = st.selectbox("Gender", ["male", "female"])
            job = st.selectbox("Employment Level", [0, 1, 2, 3],
                               format_func=lambda x: {
                                   0: "0 — Unskilled (non-resident)",
                                   1: "1 — Unskilled (resident)",
                                   2: "2 — Skilled",
                                   3: "3 — Highly Skilled"}[x])

            st.markdown('<span class="section-label">Financial Profile</span>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                housing = st.selectbox("Housing", ["own", "free", "rent"],
                                       format_func=lambda x: {"own":"🏠 Own","free":"🆓 Free","rent":"🏢 Rent"}[x])
            with c2:
                saving_acc = st.selectbox("Savings Account",
                                          ["little", "moderate", "quite rich", "rich"],
                                          format_func=lambda x: {
                                              "little":"💸 Little","moderate":"💰 Moderate",
                                              "quite rich":"💎 Quite Rich","rich":"🤑 Rich"}[x])

            # ── INR loan amount ──
            loan_inr = st.number_input(
                f"Loan Amount (₹ INR)  [min ₹{INR_MIN:,} · max ₹{INR_MAX:,}]",
                min_value=INR_MIN, max_value=INR_MAX,
                value=105_000, step=INR_STEP)
            loan_dm  = inr_to_dm(loan_inr)
            

            st.markdown('<span class="section-label">Loan Details</span>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                duration = st.slider("Duration (months)", 4, 72, 24)
            with c2:
                purpose = st.selectbox("Purpose", [
                    "car", "furniture/equipment", "radio/TV",
                    "domestic appliances", "repairs", "education",
                    "business", "vacation/others"],
                    format_func=lambda x: {
                        "car":"🚗 Car","furniture/equipment":"🛋️ Furniture",
                        "radio/TV":"📺 Radio/TV","domestic appliances":"🏠 Appliances",
                        "repairs":"🔧 Repairs","education":"🎓 Education",
                        "business":"💼 Business","vacation/others":"✈️ Vacation"}[x])

            submitted = st.form_submit_button("⚡ Generate Risk Assessment",
                                              use_container_width=True)

    # ── RIGHT: result panel ───────────────────────────────────
    with right_col:
        if submitted:
            raw = pd.DataFrame([{
                "Age": age, "Sex": sex, "Job": job, "Housing": housing,
                "Saving accounts": saving_acc,
                "Credit amount": loan_dm,          # DM for the model
                "Duration": duration, "Purpose": purpose
            }])

            if artifacts_loaded:
                labels, gps, bps = preprocess_and_predict(raw)
                lbl, gp, bp = labels[0], gps[0], bps[0]
            else:
                gp  = mock_score(age, loan_dm, duration, saving_acc, job, housing)
                bp  = round(100 - gp, 1)
                lbl = "good" if gp >= 50 else "bad"

            st.session_state["last"] = dict(lbl=lbl, gp=gp, bp=bp,
                age=age, dm=loan_dm, dur=duration, saving=saving_acc,
                job=job, housing=housing, sex=sex, purpose=purpose, inr=loan_inr)

        if "last" in st.session_state:
            r = st.session_state["last"]
            render_result(r["lbl"], r["gp"], r["bp"],
                          r["age"], r["dm"], r["dur"], r["saving"],
                          r["job"], r["housing"], r["sex"], r["purpose"], r["inr"])
        else:
            st.markdown("""
            <div style="text-align:center;padding:5rem 1rem;color:#4d5e80;">
                <div style="font-size:2.5rem;margin-bottom:.8rem;">📊</div>
                <p style="font-family:'Syne',sans-serif;font-size:1rem;color:#6b7fa3;">
                    Fill the form and click<br>
                    <strong style="color:#5b7cfa;">Generate Risk Assessment</strong>
                </p>
                <p style="font-size:.75rem;margin-top:.4rem;">Results appear here instantly</p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE — COMPARE APPLICANTS
# ══════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Applicants":
    st.markdown('<h1 class="hero-title">Compare <span class="hero-accent">Applicants</span></h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Evaluate two applicants side-by-side to make informed lending decisions</p>',
                unsafe_allow_html=True)

    def applicant_inputs(prefix, defaults):
        age     = st.slider("Age", 18, 80, defaults["age"], key=f"{prefix}_age")
        sex     = st.selectbox("Gender", ["male","female"], key=f"{prefix}_sex")
        job     = st.selectbox("Job Level", [0,1,2,3], index=defaults["job"], key=f"{prefix}_job")
        housing = st.selectbox("Housing", ["own","free","rent"], key=f"{prefix}_housing")
        saving  = st.selectbox("Savings", ["little","moderate","quite rich","rich"],
                               key=f"{prefix}_saving")
        inr     = st.number_input(f"Loan Amount (₹ INR)", INR_MIN, INR_MAX,
                                  defaults["inr"], INR_STEP, key=f"{prefix}_inr")
        dm      = inr_to_dm(inr)
        
        dur     = st.slider("Duration (months)", 4, 72, defaults["dur"], key=f"{prefix}_dur")
        pur     = st.selectbox("Purpose",
                               ["car","furniture/equipment","radio/TV","domestic appliances",
                                "repairs","education","business","vacation/others"],
                               key=f"{prefix}_purpose")
        return dict(Age=age,Sex=sex,Job=job,Housing=housing,
                    **{"Saving accounts":saving},
                    **{"Credit amount":dm}, Duration=dur, Purpose=pur,
                    _inr=inr)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<span class="section-label">👤 Applicant A</span>', unsafe_allow_html=True)
        da = applicant_inputs("a", {"age":30,"job":2,"inr":87_500,"dur":18})
    with col_b:
        st.markdown('<span class="section-label">👤 Applicant B</span>', unsafe_allow_html=True)
        db = applicant_inputs("b", {"age":45,"job":1,"inr":175_000,"dur":36})

    if st.button("⚡ Compare Both Applicants", use_container_width=True):
        inr_a, inr_b = da.pop("_inr"), db.pop("_inr")
        df_both = pd.DataFrame([da, db])

        if artifacts_loaded:
            labels, gps, bps = preprocess_and_predict(df_both)
        else:
            gps    = [mock_score(da["Age"],da["Credit amount"],da["Duration"],
                                 da["Saving accounts"],da["Job"],da["Housing"]),
                      mock_score(db["Age"],db["Credit amount"],db["Duration"],
                                 db["Saving accounts"],db["Job"],db["Housing"])]
            bps    = [round(100-g,1) for g in gps]
            labels = ["good" if g>=50 else "bad" for g in gps]

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Comparison Results")
        r1, r2 = st.columns(2, gap="large")
        for col, i, name, inr_amt in [(r1,0,"Applicant A",inr_a),(r2,1,"Applicant B",inr_b)]:
            with col:
                gp  = gps[i]; bp = bps[i]; lbl = labels[i]
                g, gc = grade(gp)
                vc = "#3ecf8e" if lbl=="good" else "#ff6363"
                bg = "#0d2b1f" if lbl=="good" else "#2b0d0d"
                vt = "✓ Good Credit" if lbl=="good" else "✕ High Risk"
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {vc}33;border-radius:14px;
                            padding:1.2rem;text-align:center;">
                    <p style="font-size:.65rem;font-weight:700;letter-spacing:.14em;
                              text-transform:uppercase;color:#6b7fa3;margin:0 0 .2rem;">{name}</p>
                    <p style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                              color:{vc};margin:0;">{gp:.1f}%</p>
                    <p style="font-size:.75rem;color:{vc};margin:.15rem 0 .6rem;">{vt}</p>
                    <p style="font-size:.72rem;color:#6b7fa3;margin:0 0 .4rem;">
                        {fmt_inr(inr_amt)}</p>
                    <span style="background:{vc}22;border:1px solid {vc}44;border-radius:7px;
                                 padding:.2rem .9rem;font-family:'Syne',sans-serif;
                                 font-weight:800;font-size:1.1rem;color:{gc};">Grade {g}</span>
                </div>""", unsafe_allow_html=True)

        wi   = 0 if gps[0] > gps[1] else 1
        diff = abs(gps[0]-gps[1])
        st.info(f"💡 **{'Applicant A' if wi==0 else 'Applicant B'}** shows "
                f"{diff:.1f}% higher creditworthiness and is the stronger candidate.")


# ══════════════════════════════════════════════════════════════
# PAGE — BATCH SCORE
# ══════════════════════════════════════════════════════════════
elif page == "📋 Batch Score":
    st.markdown('<h1 class="hero-title">Batch <span class="hero-accent">Scoring</span></h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Upload a CSV to score multiple applicants at once</p>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#161b2e;border:1px dashed rgba(91,124,250,.4);
                border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
        <p style="color:#6b7fa3;font-size:.82rem;margin:0 0 .4rem;">
            📁 CSV columns required:
        </p>
        <code style="background:#1e2538;color:#a0cfff;padding:.25rem .6rem;
                     border-radius:5px;font-size:.77rem;">
            Age, Sex, Job, Housing, Saving accounts, Credit amount (₹ INR), Duration, Purpose
        </code>
        <p style="color:#4d5e80;font-size:.72rem;margin:.5rem 0 0;">
            <em>Credit amount column should be in INR</em>
        </p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df_up)} applicants")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("⚡ Score All Applicants", use_container_width=True):
                # Convert INR → DM in-place for the model
                df_model = df_up.copy()
                if "Credit amount" in df_model.columns:
                    df_model["Credit amount"] = df_model["Credit amount"].apply(inr_to_dm)

                if artifacts_loaded:
                    labels, gps, bps = preprocess_and_predict(df_model)
                else:
                    import random
                    gps    = [round(random.uniform(30,90),1) for _ in range(len(df_up))]
                    bps    = [round(100-g,1) for g in gps]
                    labels = ["good" if g>=50 else "bad" for g in gps]

                df_up["Predicted Risk"]  = labels
                df_up["Good Prob (%)"]   = gps
                df_up["Bad Prob (%)"]    = bps
                df_up["Grade"]           = [grade(g)[0] for g in gps]
                # Show INR values in output
                if "Credit amount" in df_up.columns:
                    df_up["Credit amount (INR)"] = df_up["Credit amount"]

                n_good = sum(1 for l in labels if l=="good")
                n_bad  = len(labels)-n_good
                avg_gp = sum(gps)/len(gps)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Applicants", len(df_up))
                c2.metric("Approved", n_good, f"{n_good/len(df_up)*100:.0f}%")
                c3.metric("Declined", n_bad,  f"{n_bad/len(df_up)*100:.0f}%")
                c4.metric("Avg Creditworthiness", f"{avg_gp:.1f}%")

                st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
                st.dataframe(df_up, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download Scored CSV",
                                   df_up.to_csv(index=False),
                                   "scored_applicants.csv", "text/csv",
                                   use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        tmpl = pd.DataFrame([{
            "Age":35,"Sex":"male","Job":2,"Housing":"own",
            "Saving accounts":"moderate",
            "Credit amount (INR)": 105_000,   # INR in template
            "Duration":24,"Purpose":"car"
        }])
        st.download_button("📥 Download CSV Template",
                           tmpl.to_csv(index=False),
                           "credit_template.csv","text/csv")

