"""
LeukNet AML Detector - Streamlit app (uses your trained Keras ResNet50)

Install:  pip install streamlit tensorflow pillow pandas numpy
Run:      streamlit run app.py

Works without a model file (DEMO mode, simulated results).
Add best_resnet50.keras next to this file to use the real model.
"""

import os
import zlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# SETTINGS
# ==========================================
MODEL_PATH = os.environ.get("MODEL_PATH", "best_resnet50.keras")

# Same order as training: 0 = monocyte, 1 = myeloblast
CLASS_LABELS = ["Monocyte (Non-AML)", "Myeloblast (AML-positive)"]

st.set_page_config(page_title="LeukNet AML Detector", layout="centered")

st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #1f2937; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #111827; color: white; }
    .report-box { border-left: 4px solid #22c55e; background-color: #f0fdf4;
                  padding: 15px; border-radius: 5px; margin-top: 20px; color: #111827; }
    .report-box.blue { border-left-color: #3b82f6; background-color: #eff6ff; }
    .report-box.red { border-left-color: #ef4444; background-color: #fef2f2; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE (keeps data between clicks)
# ==========================================
if "patients" not in st.session_state:
    st.session_state.patients = []
if "reports" not in st.session_state:
    st.session_state.reports = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


def get_model():
    """Try to load the model. If it is missing or broken, return None (demo mode)."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")

    if os.path.isfile(MODEL_PATH):
        try:
            m = load_model(MODEL_PATH)
            st.sidebar.success(f"Loaded: {MODEL_PATH}")
            return m
        except Exception:
            st.sidebar.warning("Model file is invalid. Running in DEMO mode.")
            return None

    st.sidebar.info("No model file found. Running in DEMO mode.")
    return None


def predict(model, image, raw_bytes=b""):
    """Return (class_index, confidence %) for one PIL image.
    With no model, return a SIMULATED result (demo only)."""
    if model is None:
        rng = np.random.default_rng(zlib.crc32(raw_bytes))   # same image = same result
        idx = int(rng.integers(0, 2))
        return idx, float(rng.uniform(70, 99))

    size = model.input_shape[1]                      # e.g. 128
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    probs = model.predict(x[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx] * 100)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("LeukNet AML")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Patient Register", "AML Detector", "Reports Manager"],
)
model = get_model()

# ==========================================
# PAGES
# ==========================================
if page == "Dashboard":
    st.title("Dashboard")
    st.write("Welcome to the LeukNet AML Diagnostic Platform.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(st.session_state.patients))
    c2.metric("Reports", len(st.session_state.reports))
    c3.metric("Mode", "Live model" if model is not None else "Demo")

    col1, col2 = st.columns(2)
    with col1:
        st.info("**Patient Register**\n\nManage profiles of registered patients")
        st.info("**Reports Manager**\n\nView records and download result reports")
    with col2:
        st.success("**AML Detector**\n\nResNet50-powered myeloblast detection")

elif page == "Patient Register":
    st.title("Patient Register")

    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("Patient Name")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=0, max_value=120, value=12)
        with col2:
            address = st.text_input("Address")
            contact = st.text_input("Contact")
        submit = st.form_submit_button("Add Patient")

    if submit:
        if not name.strip():
            st.error("Please enter the patient name.")
        else:
            st.session_state.patients.append({
                "Name": name.strip().upper(), "Gender": gender, "Age": age,
                "Address": address, "Contact": contact,
            })
            st.success(f"Patient {name.strip().upper()} added successfully!")

    if st.session_state.patients:
        st.subheader("Registered Patients")
        st.dataframe(pd.DataFrame(st.session_state.patients), use_container_width=True)

elif page == "AML Detector":
    st.title("AML Detector (ResNet50)")
    st.write("Upload blood smear cell images to detect myeloblasts (AML).")

    if model is None:
        st.warning("DEMO MODE: no model loaded. Results below are SIMULATED "
                   "and are not a real diagnosis.")

    patient_names = [p["Name"] for p in st.session_state.patients]
    if patient_names:
        patient = st.selectbox("For Which Patient", patient_names)
    else:
        st.info("No patients registered yet. You can still run a test as 'Unregistered'.")
        patient = "UNREGISTERED"

    files = st.file_uploader(
        "Click or drag and drop smear images here",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if files:
        st.write(f"Uploaded {len(files)} sample(s).")
        st.image([Image.open(f).convert("RGB") for f in files],
                 caption=[f"Sample {i+1}" for i in range(len(files))], width=140)

        if st.button("Process Samples"):
            with st.spinner("Processing images through ResNet50..."):
                rows = []
                for f in files:
                    f.seek(0)
                    raw = f.read()
                    f.seek(0)
                    idx, conf = predict(model, Image.open(f), raw)
                    rows.append({
                        "File": f.name,
                        "Prediction": CLASS_LABELS[idx],
                        "Confidence (%)": round(conf, 1),
                        "Blast": idx == 1,
                    })
                st.session_state.last_result = {
                    "patient": patient,
                    "rows": rows,
                    "time": datetime.now().strftime("%b %d, %Y %I:%M %p"),
                }

    res = st.session_state.last_result
    if res:
        rows = res["rows"]
        blasts = sum(r["Blast"] for r in rows)

        st.subheader("Results" + (" (DEMO - simulated)" if model is None else ""))
        st.write(f"**Patient:** {res['patient']}")
        st.write(f"**Uploaded sample size:** {len(rows)}")
        st.write(f"**Cells classified as myeloblast:** {blasts}")
        st.dataframe(
            pd.DataFrame(rows).drop(columns="Blast"), use_container_width=True
        )

        if blasts > 0:
            st.markdown(
                f"""<div class="report-box red"><h4>Results Report</h4>
                <p>{blasts} of {len(rows)} cell(s) classified as myeloblast.
                Acute Myeloid Leukaemia (AML) may be indicated. Please confirm
                with a haematologist.</p></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="report-box blue"><h4>Results Report</h4>
                <p>No myeloblasts detected in the uploaded cells.</p></div>""",
                unsafe_allow_html=True,
            )

        if st.button("Save Results to Patient Record"):
            st.session_state.reports.append({
                "Date": res["time"],
                "Patient Name": res["patient"],
                "Result": (f"AML indicated ({blasts} blast(s))"
                           if blasts else "No blasts detected"),
                "Samples": len(rows),
            })
            st.success("Saved to Reports Manager.")

    st.caption("Research prototype. Not a certified medical device. "
               "Results must be confirmed by a qualified professional.")

elif page == "Reports Manager":
    st.title("Reports Manager")
    st.write("View records and download result reports.")

    if st.session_state.reports:
        df = pd.DataFrame(st.session_state.reports)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download Reports (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="aml_reports.csv",
            mime="text/csv",
        )
    else:
        st.info("No saved reports yet. Run a test and click "
                "'Save Results to Patient Record'.")
