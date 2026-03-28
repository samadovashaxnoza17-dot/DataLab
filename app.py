"""
AI-Assisted Data Wrangler & Visualizer
Data Wrangling and Visualization (5COSC038C) - Coursework 2025-26
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import io
import json
import re
import datetime
from scipy import stats as scipy_stats
import csv
import gspread
from google.oauth2 import service_account


def clean_ascii(text):
    """Remove non-ASCII characters to prevent encoding errors with AI API."""
    return str(text).encode('ascii', 'ignore').decode()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Wrangler & Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────
DEFAULTS = {
    "raw_df": None,
    "working_df": None,
    "filename": None,
    "transform_log": [],
    "api_key": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────
def reset_session():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def log_step(operation: str, params: dict):
    st.session_state.transform_log.append({
        "step": len(st.session_state.transform_log) + 1,
        "operation": operation,
        "params": params,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "shape_after": list(st.session_state.working_df.shape),
    })


def undo_last():
    if st.session_state.transform_log:
        st.session_state.transform_log.pop()
        st.toast("Last step removed from log. Re-apply from raw data if needed.", icon="↩️")


def wdf() -> pd.DataFrame:
    return st.session_state.working_df


def set_wdf(df: pd.DataFrame):
    st.session_state.working_df = df


@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, filename: str):
    ext = filename.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(file_bytes)
    warnings = []

    if ext == "csv":
        bad_lines = []
        good_rows = []
        decoded = file_bytes.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(decoded))
        header = [col.strip() for col in next(reader)]

        # Fix duplicate column names
        seen = {}
        clean_header = []
        for col in header:
            if col in seen:
                seen[col] += 1
                new_name = f"{col}_{seen[col]}"
                clean_header.append(new_name)
                warnings.append(f"Duplicate column '{col}' renamed to '{new_name}'")
            else:
                seen[col] = 0
                clean_header.append(col)

        # Check corrupted rows
        expected_len = len(clean_header)
        for i, row in enumerate(reader, start=2):
            if len(row) != expected_len:
                bad_lines.append(i)
            else:
                good_rows.append(row)

        if bad_lines:
            warnings.append(f"Skipped {len(bad_lines)} corrupted row(s): lines {bad_lines[:10]}")

        if not good_rows:
            raise ValueError("No valid rows found. Please check your CSV.")

        result_df = pd.DataFrame(good_rows, columns=clean_header)
        # Replace string "None" and "nan" with actual NaN
        result_df = result_df.replace({"None": None, "nan": None, "": None})
        result_df = result_df.convert_dtypes()
        return result_df, warnings
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(buf), warnings   # ← warnings here too

    elif ext == "json":
        import json as json_lib
        buf.seek(0)
        raw = json_lib.load(buf)
        if isinstance(raw, list):
            return pd.DataFrame(raw), warnings
        elif isinstance(raw, dict):
            try:
                return pd.DataFrame(raw), warnings
            except ValueError:
                return pd.DataFrame([raw]), warnings
        else:
            raise ValueError("Unsupported JSON structure. Must be an array or object.")

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"Missing Count": miss, "Missing %": pct})[miss > 0].sort_values("Missing %", ascending=False)


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
PAGES = ["📂 Upload & Overview", "🧹 Cleaning Studio", "📊 Visualization Builder", "💾 Export & Report", "🤖 AI Assistant"]

with st.sidebar:
    st.title("🔬 Data Wrangler")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()
        st.rerun()
    if st.session_state.working_df is not None:
        st.caption(f"📄 **{st.session_state.filename}**")
        df_shape = st.session_state.working_df.shape
        st.caption(f"Shape: {df_shape[0]:,} rows x {df_shape[1]} cols")
        st.caption(f"🗂️ {len(st.session_state.transform_log)} transform(s) applied")
    st.divider()
    st.markdown("### 🤖 AI Assistant")
    st.markdown("[![Get Free Groq Key](https://img.shields.io/badge/Get%20Free%20API%20Key-Groq-orange?style=for-the-badge)](https://console.groq.com)")
    st.caption(
        "**How to get your free key:**\n"
        "1. Click the button above\n"
        "2. Sign up for free (no card needed)\n"
        "3. Go to **API Keys** -> **Create API Key**\n"
        "4. Copy and paste it below 👇"
    )
    st.session_state.api_key = st.text_input(
        "Paste your Groq API Key here",
        type="password",
        value=st.session_state.get("api_key", ""),
        placeholder="gsk_...",
        key="sidebar_api_key"
    )
    if st.session_state.api_key:
        key = st.session_state.api_key
        # Check for non-ASCII characters in the key
        if key != key.encode('ascii', 'ignore').decode():
            st.error("❌ Your API key contains invalid characters. "
                     "Please copy it again directly from console.groq.com")
        elif not key.startswith("gsk_"):
            st.warning("⚠️ This doesn't look like a Groq key. "
                       "Get yours free at console.groq.com")
        else:
            st.success("✅ AI enabled")
    else:
        st.info("⬆️ Paste your key to enable AI features")
# ═══════════════════════════════════════════════════════════════════
# PAGE A - Upload & Overview
# ═══════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("📂 Upload & Overview")

    uploaded = st.file_uploader(
        "Upload your dataset (CSV, Excel, JSON)",
        type=["csv", "xlsx", "xls", "json"],
        help="Datasets should have ≥ 1,000 rows and ≥ 8 columns for full feature coverage.",
    )

    if uploaded:
        try:
            raw, load_warnings = load_file(uploaded.read(), uploaded.name)
            for w in load_warnings:
                st.warning(f"⚠️ {w}")
            # Only reset working_df when a new file is uploaded
            if st.session_state.filename != uploaded.name:
                st.session_state.raw_df = raw.copy()
                st.session_state.working_df = raw.copy()
                st.session_state.filename = uploaded.name
                st.session_state.transform_log = []
                st.success(f"✅ Loaded **{uploaded.name}** - {raw.shape[0]:,} rows x {raw.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # ── Google Sheets ─────────────────────────────────────────
    st.divider()
    st.subheader("📊 Or load from Google Sheets (optional)")
    method = st.radio("Connection method", ["None", "Public link", "Service Account"],
                      horizontal=True)

    if method == "Public link":
        sheet_url = st.text_input("Paste public Google Sheets URL")
        if sheet_url and "docs.google.com" in sheet_url:
            try:
                sheet_id = sheet_url.split("/d/")[1].split("/")[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                raw = pd.read_csv(csv_url)
                st.session_state.raw_df = raw.copy()
                st.session_state.working_df = raw.copy()
                st.session_state.filename = "google_sheet.csv"
                st.session_state.transform_log = []
                st.success(f"✅ Google Sheet loaded! {raw.shape[0]:,} rows x {raw.shape[1]} cols")
            except Exception as e:
                st.error(f"Error loading sheet: {e}")

    elif method == "Service Account":
        uploaded_key = st.file_uploader("Upload service account JSON key", type="json",
                                         key="gsheets_key")
        sheet_url = st.text_input("Paste Google Sheets URL")
        if uploaded_key and sheet_url:
            try:
                key_data = json.load(uploaded_key)
                creds = service_account.Credentials.from_service_account_info(
                    key_data,
                    scopes=["https://spreadsheets.google.com/feeds",
                            "https://www.googleapis.com/auth/drive"]
                )
                client = gspread.authorize(creds)
                sheet_id = sheet_url.split("/d/")[1].split("/")[0]
                spreadsheet = client.open_by_key(sheet_id)
                worksheets = [ws.title for ws in spreadsheet.worksheets()]
                selected_tab = st.selectbox("Select worksheet", worksheets)
                worksheet = spreadsheet.worksheet(selected_tab)
                raw = pd.DataFrame(worksheet.get_all_records())
                st.session_state.raw_df = raw.copy()
                st.session_state.working_df = raw.copy()
                st.session_state.filename = "google_sheet.csv"
                st.session_state.transform_log = []
                st.success(f"✅ Private Google Sheet loaded! {raw.shape[0]:,} rows x {raw.shape[1]} cols")
            except Exception as e:
                st.error(f"Error: {e}")

    if wdf() is None:
        st.info("👆 Upload a file above to get started. Sample datasets are included in `sample_data/`.")
        st.stop()

    df = wdf()

    # ── Dataset quality check ─────────────────────────────────────
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    raw_unnamed = [c for c in st.session_state.raw_df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""] if st.session_state.raw_df is not None else []
    
    if unnamed_cols or raw_unnamed:
        if unnamed_cols:
            st.warning(f"⚠️ Your dataset has **{len(unnamed_cols)} unnamed column(s)** — this usually means the file has extra header rows or a formatting issue.")
        else:
            st.success("✅ Unnamed columns have been cleaned!")
        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        with col_opt1:
            if st.button("🧹 Auto-clean"):
                cleaned = df.drop(columns=unnamed_cols)
                if cleaned.shape[1] == 0:
                    st.error("No valid columns remain after cleaning.")
                else:
                    st.session_state.working_df = cleaned
                    st.session_state["pre_clean_df"] = df.copy()
                    st.success(f"✅ Dropped {len(unnamed_cols)} unnamed columns.")
                    st.rerun()
        with col_opt2:
            if st.button("➡️ Continue anyway"):
                st.info("Proceeding with unnamed columns.")
        with col_opt3:
            if st.button("🛑 Stop & fix"):
                st.error("Please fix your file and re-upload.\n"
                         "- Open in Excel\n"
                         "- Make Row 1 the header\n"
                         "- Save as CSV and re-upload")
                st.stop()
        with col_opt4:
            if st.button("↩️ Undo clean"):
                if "pre_clean_df" in st.session_state:
                    st.session_state.working_df = st.session_state["pre_clean_df"].copy()
                    st.success("✅ Reverted to original upload!")
                    st.rerun()
                else:
                    st.info("Nothing to undo yet.")
            

    st.subheader("📋 Dataset Validation Check")


    row_count = df.shape[0]
    col_count = df.shape[1]
    missing_cells = df.isnull().sum().sum()

    num_cols = sum(1 for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.5)
    cat_cols = df.shape[1] - num_cols
    for c in df.select_dtypes(include="object").columns:
        if pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.5:
            num_cols += 1
            cat_cols -= 1
    

    if row_count >= 1000:
        st.success(f"✔ Rows: {row_count} (≥ 1000)")
    else:
        st.warning(f"⚠ Rows: {row_count} (< 1000)")

    if col_count >= 8:
        st.success(f"✔ Columns: {col_count} (≥ 8)")
    else:
        st.warning(f"⚠ Columns: {col_count} (< 8)")

    if num_cols > 0 and cat_cols > 0:
        st.success("✔ Mixed types (numeric + categorical)")
    else:
        st.warning("⚠ Need mixed data types")

    if missing_cells > 0:
        st.success(f"✔ Missing values present ({missing_cells})")
    else:
        st.warning("⚠ No missing values found")
          
    # ── Shape & column types ──────────────────────────────────────
    st.subheader("📐 Dataset Shape")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    c4.metric("Duplicate Rows", f"{duplicate_count(df):,}")

    st.subheader("🗂️ Column Info")
    col_info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non-null count": df.notnull().sum(),
        "null count": df.isnull().sum(),
        "null %": (df.isnull().sum() / len(df) * 100).round(2),
        "unique values": df.nunique(),
        "sample value": [df[c].dropna().iloc[0] if df[c].notnull().any() else "N/A" for c in df.columns],
    })
    st.dataframe(col_info, use_container_width=True)

# ── Summary statistics ────────────────────────────────────────
    st.subheader("📊 Summary Statistics")
    tab_num, tab_cat = st.tabs(["Numeric", "Categorical"])

    with tab_num:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab_cat:
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if cat_cols:
            rows = []
            for c in cat_cols:
                vc = df[c].value_counts()
                rows.append({
                    "column": c,
                    "dtype": str(df[c].dtype),
                    "unique": df[c].nunique(),
                    "top value": vc.index[0] if len(vc) else "N/A",
                    "top freq": vc.iloc[0] if len(vc) else 0,
                    "top %": f"{vc.iloc[0]/len(df)*100:.1f}%" if len(vc) else "N/A",
                })
            st.dataframe(pd.DataFrame(rows).set_index("column"), use_container_width=True)
        else:
            st.info("No categorical columns found.")

    # ── Missing values ────────────────────────────────────────────
    st.subheader("❓ Missing Values")
    ms = missing_summary(df)
    if ms.empty:
        st.success("No missing values found! 🎉")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(ms, use_container_width=True)
        with c2:
            ms_display = ms.head(15)
            fig, ax = plt.subplots(figsize=(6, max(3, len(ms_display) * 0.4)))
            ax.barh(ms_display.index, ms_display["Missing %"], color="#ef4444")
            ax.tick_params(axis='y', labelsize=8)
            # Truncate long labels
            truncated_labels = [label[:25] + '...' if len(label) > 25 else label 
                               for label in ms_display.index]
            ax.set_yticklabels(truncated_labels)
            ax.set_xlabel("Missing %")
            if len(ms) > 15:
                ax.set_title(f"Missing % per column (top 15 of {len(ms)})")
            else:
                ax.set_title("Missing % per column")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Outlier Overview ──────────────────────────────────────────
    st.subheader("📈 Outlier Overview (IQR method)")
    num_cols_out = df.select_dtypes(include="number").columns.tolist()
    if num_cols_out:
        out_rows = []
        for col in num_cols_out:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            out_rows.append({
                "column": col,
                "outlier_count": n_out,
                "outlier_%": round(n_out / len(s) * 100, 2)
            })
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    # ── Data preview ─────────────────────────────────────────────
    st.subheader("👁️ Data Preview")
    n_rows = st.slider("Rows to preview", 5, 100, 20)
    st.dataframe(df.head(n_rows), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE B - Cleaning & Preparation Studio
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("🧹 Cleaning & Preparation Studio")

    if wdf() is None:
        st.warning("Please upload a dataset on the Upload page first.")
        st.stop()

    df = wdf()

    # Transformation log panel (collapsible)
    # ── Inline AI Cleaning Advisor ────────────────────────────────
    api_key = st.session_state.get("api_key", "")
    if api_key:
        with st.expander("🤖 AI Cleaning Assistant", expanded=False):
            st.caption("Type a cleaning instruction and AI will suggest what to do.")
            nl_command = st.text_input("Your instruction",
                                       placeholder="e.g. fill missing age with median",
                                       key="page_b_nl")
            if st.button("🤖 Get Suggestion", key="page_b_nl_btn") and nl_command:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
                num_cols_b = df.select_dtypes(include="number").columns.tolist()
                cat_cols_b = df.select_dtypes(exclude="number").columns.tolist()
                miss_cols_b = df.isnull().sum()
                miss_cols_b = miss_cols_b[miss_cols_b > 0].index.tolist()
                prompt = (
                    f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols\n"
                    f"Numeric columns: {clean_ascii(num_cols_b)}\n"
                    f"Categorical columns: {clean_ascii(cat_cols_b)}\n"
                    f"Columns with missing values: {clean_ascii(miss_cols_b)}\n\n"
                    f"User instruction: \"{nl_command}\"\n\n"
                    f"Suggest the exact cleaning operation. Format as:\n"
                    f"OPERATION: ...\n"
                    f"COLUMNS: ...\n"
                    f"PARAMETERS: ...\n"
                    f"EXPLANATION: ..."
                )
                prompt = prompt.encode('ascii', 'ignore').decode('ascii')
                with st.spinner("🤖 Thinking..."):
                    try:
                        prompt = prompt.encode('ascii', 'ignore').decode('ascii')
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,
                            temperature=0.3,
                        )
                        ai_suggestion = response.choices[0].message.content.strip()
                        st.info(ai_suggestion)
                        st.warning("⚠️ This is an AI suggestion — review before applying.")
                        st.success("👇 Go to the section below to apply this operation manually.")
                    except Exception as e:
                        st.warning(f"⚠️ AI error: {e}")
    else:
        st.info("🔑 Add Groq API key in sidebar to enable AI cleaning suggestions.")
        if st.session_state.transform_log:
            log_df = pd.DataFrame(st.session_state.transform_log)
            st.dataframe(log_df[["step", "operation", "params", "timestamp", "shape_after"]], use_container_width=True)
            if st.button("↩️ Undo Last Step"):
                undo_last()
                st.rerun()
        else:
            st.info("No transformations applied yet.")

    st.divider()

    sections = [
        "4.1 Missing Values",
        "4.2 Duplicates",
        "4.3 Data Types & Parsing",
        "4.4 Categorical Tools",
        "4.5 Numeric Cleaning (Outliers)",
        "4.6 Normalization / Scaling",
        "4.7 Column Operations",
        "4.8 Data Validation Rules",
    ]
    section = st.selectbox("Select operation:", sections)

    # ── 4.1 Missing Values ────────────────────────────────────────
    if section == sections[0]:
        st.subheader("❓ Missing Values Handler")
        ms = missing_summary(df)
        if ms.empty:
            st.success("No missing values found!")
        else:
            st.dataframe(ms, use_container_width=True)

            cols_with_missing = ms.index.tolist()
            sel_cols = st.multiselect("Select columns to treat", cols_with_missing, default=cols_with_missing[:1])

            action = st.selectbox("Action", [
                "Drop rows with missing in selected columns",
                "Drop columns above missing % threshold",
                "Fill - constant value",
                "Fill - mean (numeric)",
                "Fill - median (numeric)",
                "Fill - mode / most frequent",
                "Fill - forward fill",
                "Fill - backward fill",
            ])

            fill_value = None
            threshold = 50.0
            if action == "Fill - constant value":
                fill_value = st.text_input("Constant value", "0")
            if action == "Drop columns above missing % threshold":
                threshold = st.slider("Threshold (%)", 1, 100, 50)

            if st.button("▶ Apply"):
                new_df = df.copy()
                try:
                    if action == "Drop rows with missing in selected columns":
                        before = len(new_df)
                        new_df = new_df.dropna(subset=sel_cols)
                        after = len(new_df)
                        st.info(f"Dropped {before - after:,} rows.")
                    elif action == "Drop columns above missing % threshold":
                        drop_cols = ms[ms["Missing %"] >= threshold].index.tolist()
                        new_df = new_df.drop(columns=drop_cols)
                        st.info(f"Dropped columns: {drop_cols}")
                    elif action == "Fill - constant value":
                        for c in sel_cols:
                            try:
                                val = float(fill_value) if pd.api.types.is_numeric_dtype(new_df[c]) else fill_value
                            except ValueError:
                                val = fill_value
                            new_df[c] = new_df[c].fillna(val)
                    elif action == "Fill - mean (numeric)":
                        for c in sel_cols:
                            if pd.api.types.is_numeric_dtype(new_df[c]):
                                new_df[c] = new_df[c].fillna(new_df[c].mean())
                    elif action == "Fill - median (numeric)":
                        for c in sel_cols:
                            if pd.api.types.is_numeric_dtype(new_df[c]):
                                new_df[c] = new_df[c].fillna(new_df[c].median())
                    elif action == "Fill - mode / most frequent":
                        for c in sel_cols:
                            mode_val = new_df[c].mode()
                            if not mode_val.empty:
                                new_df[c] = new_df[c].fillna(mode_val.iloc[0])
                    elif action == "Fill - forward fill":
                        new_df[sel_cols] = new_df[sel_cols].ffill()
                    elif action == "Fill - backward fill":
                        new_df[sel_cols] = new_df[sel_cols].bfill()

                    log_step("Missing Values", {"action": action, "columns": sel_cols})
                    set_wdf(new_df)
                    st.success("✅ Applied!")

                    # Before/after
                    col1, col2 = st.columns(2)
                    col1.metric("Rows before", df.shape[0])
                    col2.metric("Rows after", new_df.shape[0])
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── 4.2 Duplicates ────────────────────────────────────────────
    elif section == sections[1]:
        st.subheader("♻️ Duplicate Handler")
        dup_mode = st.radio("Check mode", ["Full-row duplicates", "Subset of columns"])
        subset = None
        if dup_mode == "Subset of columns":
            subset = st.multiselect("Select key columns", df.columns.tolist(), default=df.columns.tolist()[:2])

        total_dups = int(df.duplicated(subset=subset).sum())
        st.metric("Duplicate rows found", total_dups)

        if total_dups > 0:
            show_dups = st.toggle("Show duplicate groups")
            if show_dups:
                dup_df = df[df.duplicated(subset=subset, keep=False)].sort_values(by=subset if subset else df.columns.tolist())
                st.dataframe(dup_df.head(200), use_container_width=True)

            keep = st.selectbox("Keep which occurrence", ["first", "last"])
            if st.button("▶ Remove Duplicates"):
                new_df = df.drop_duplicates(subset=subset, keep=keep)
                log_step("Remove Duplicates", {"mode": dup_mode, "keep": keep, "removed": total_dups})
                set_wdf(new_df)
                st.success(f"✅ Removed {total_dups} duplicate rows.")
                st.rerun()
        else:
            st.success("No duplicates found! 🎉")

    # ── 4.3 Data Types & Parsing ──────────────────────────────────
    elif section == sections[2]:
        st.subheader("🔢 Data Types & Parsing")

        col = st.selectbox("Column to convert", df.columns.tolist())
        current_dtype = str(df[col].dtype)
        st.caption(f"Current dtype: `{current_dtype}`")
        st.dataframe(df[[col]].head(5), use_container_width=True)

        target = st.selectbox("Convert to", ["numeric", "categorical (category)", "string", "datetime"])

        dt_format = None
        if target == "datetime":
            dt_format = st.text_input("Datetime format (leave blank for auto)", placeholder="%Y-%m-%d")

        col1_btn, col2_btn = st.columns(2)
        with col1_btn:
            if st.button("▶ Convert"):
                new_df = df.copy()
                try:
                    if target == "numeric":
                        if new_df[col].dtype == object:
                            new_df[col] = new_df[col].astype(str).str.replace(r"[,$€£%\s]", "", regex=True)
                        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
                    elif target == "categorical (category)":
                        new_df[col] = new_df[col].astype("category")
                    elif target == "string":
                        new_df[col] = new_df[col].astype(str)
                    elif target == "datetime":
                        fmt = dt_format if dt_format else None
                        new_df[col] = pd.to_datetime(new_df[col], format=fmt, errors="coerce")
                    st.session_state["pre_convert_df"] = df.copy()
                    log_step("Type Conversion", {"column": col, "from": current_dtype, "to": target})
                    set_wdf(new_df)
                    st.success(f"✅ Converted `{col}` to {target}. New dtype: `{new_df[col].dtype}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
        with col2_btn:
            if st.button("↩️ Undo Conversion"):
                if "pre_convert_df" in st.session_state:
                    set_wdf(st.session_state["pre_convert_df"])
                    st.success("✅ Conversion undone!")
                    st.rerun()
                else:
                    st.info("Nothing to undo yet.")

    # ── 4.4 Categorical Tools ─────────────────────────────────────
    elif section == sections[3]:
        st.subheader("🏷️ Categorical Tools")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            st.warning("No categorical/string columns found.")
            st.stop()

        col = st.selectbox("Column", cat_cols)
        tool = st.selectbox("Tool", [
            "Trim whitespace & fix casing",
            "Value mapping / replacement",
            "Rare category grouping -> Other",
            "One-hot encoding",
        ])

        if tool == "Trim whitespace & fix casing":
            case_opt = st.selectbox("Case", ["lower", "UPPER", "Title", "No change"])
            if st.button("▶ Apply"):
                new_df = df.copy()
                new_df[col] = new_df[col].astype(str).str.strip()
                if case_opt == "lower":
                    new_df[col] = new_df[col].str.lower()
                elif case_opt == "UPPER":
                    new_df[col] = new_df[col].str.upper()
                elif case_opt == "Title":
                    new_df[col] = new_df[col].str.title()
                log_step("Categorical - Trim/Case", {"column": col, "case": case_opt})
                set_wdf(new_df)
                st.success("✅ Applied!")
                st.rerun()

        elif tool == "Value mapping / replacement":
            unique_vals = df[col].dropna().unique().tolist()[:30]
            st.caption(f"Sample values (up to 30): {unique_vals}")
            mapping_input = st.text_area(
                "Enter mapping as JSON (e.g. `{\"old\": \"new\"}`)",
                placeholder='{"Yes": "1", "No": "0"}',
                height=150,
            )
            unmatched = st.selectbox("Unmatched values", ["Keep unchanged", "Set to Other"])
            if st.button("▶ Apply Mapping"):
                try:
                    mapping = json.loads(mapping_input)
                    new_df = df.copy()
                    new_df[col] = new_df[col].map(lambda x: mapping.get(str(x), ("Other" if unmatched == "Set to Other" else x)))
                    log_step("Categorical - Mapping", {"column": col, "mapping": mapping})
                    set_wdf(new_df)
                    st.success("✅ Mapping applied!")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON. Please check the format.")

        elif tool == "Rare category grouping -> Other":
            threshold_pct = st.slider("Group categories below this frequency (%)", 0.1, 20.0, 2.0)
            vc = df[col].value_counts(normalize=True) * 100
            rare = vc[vc < threshold_pct].index.tolist()
            st.info(f"Will group {len(rare)} rare categories into 'Other': {rare[:10]}{'...' if len(rare) > 10 else ''}")
            if st.button("▶ Apply"):
                new_df = df.copy()
                new_df[col] = new_df[col].apply(lambda x: "Other" if x in rare else x)
                log_step("Categorical - Rare Grouping", {"column": col, "threshold_pct": threshold_pct, "grouped": len(rare)})
                set_wdf(new_df)
                st.success("✅ Applied!")
                st.rerun()

        elif tool == "One-hot encoding":
            prefix = st.text_input("Column prefix (leave blank for column name)", "")
            drop_orig = st.toggle("Drop original column", value=True)
            if st.button("▶ Apply One-Hot Encoding"):
                new_df = df.copy()
                dummies = pd.get_dummies(new_df[col], prefix=(prefix or col), dtype=int)
                if drop_orig:
                    new_df = new_df.drop(columns=[col])
                new_df = pd.concat([new_df, dummies], axis=1)
                log_step("One-Hot Encoding", {"column": col, "new_columns": dummies.columns.tolist()})
                set_wdf(new_df)
                st.success(f"✅ Created {len(dummies.columns)} new columns.")
                st.rerun()

    # ── 4.5 Numeric Cleaning ──────────────────────────────────────
    elif section == sections[4]:
        st.subheader("🔭 Numeric Cleaning & Outlier Detection")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.warning("No numeric columns found.")
            st.stop()

        col = st.selectbox("Column", num_cols)
        method = st.radio("Detection method", ["IQR (Interquartile Range)", "Z-Score"])

        series = df[col].dropna()

        if method == "IQR (Interquartile Range)":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            z = np.abs(scipy_stats.zscore(series))
            z_thresh = st.slider("Z-score threshold", 1.0, 5.0, 3.0, step=0.1)
            lower = series[z < z_thresh].min()
            upper = series[z < z_thresh].max()

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(outlier_mask.sum())
        st.metric("Outliers detected", n_outliers)
        st.caption(f"Lower bound: {lower:.4f} | Upper bound: {upper:.4f}")

        # Box plot preview
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.boxplot(series, vert=False, patch_artist=True, boxprops=dict(facecolor="#93c5fd"))
        ax.set_title(f"Boxplot: {col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        action = st.selectbox("Action", ["Do nothing", "Cap / Winsorize at bounds", "Remove outlier rows"])
        if st.button("▶ Apply") and action != "Do nothing":
            new_df = df.copy()
            if action == "Cap / Winsorize at bounds":
                new_df[col] = new_df[col].clip(lower=lower, upper=upper)
                st.success(f"✅ Capped {n_outliers} values in `{col}`.")
            elif action == "Remove outlier rows":
                new_df = new_df[~outlier_mask]
                st.success(f"✅ Removed {n_outliers} outlier rows.")
            log_step("Outlier Treatment", {"column": col, "method": method, "action": action, "n_outliers": n_outliers})
            set_wdf(new_df)
            st.rerun()

    # ── 4.6 Normalization / Scaling ───────────────────────────────
    elif section == sections[5]:
        st.subheader("📏 Normalization & Scaling")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.warning("No numeric columns.")
            st.stop()

        sel_cols = st.multiselect("Columns to scale", num_cols, default=num_cols[:2])
        method = st.selectbox("Scaling method", ["Min-Max (0 to 1)", "Z-Score Standardization"])

        if sel_cols:
            before_stats = df[sel_cols].describe().loc[["mean", "std", "min", "max"]].T
            st.write("**Before:**")
            st.dataframe(before_stats.round(4), use_container_width=True)

        if st.button("▶ Apply Scaling"):
            new_df = df.copy()
            for c in sel_cols:
                col_series = new_df[c]
                if method == "Min-Max (0 to 1)":
                    cmin, cmax = col_series.min(), col_series.max()
                    new_df[c] = (col_series - cmin) / (cmax - cmin) if cmax != cmin else 0.0
                else:
                    cmean, cstd = col_series.mean(), col_series.std()
                    new_df[c] = (col_series - cmean) / cstd if cstd != 0 else 0.0

            log_step("Scaling", {"method": method, "columns": sel_cols})
            set_wdf(new_df)
            st.success("✅ Scaling applied!")
            st.rerun()            

    # ── 4.7 Column Operations ─────────────────────────────────────
    elif section == sections[6]:
        st.subheader("🔧 Column Operations")

        op = st.selectbox("Operation", ["Rename column", "Drop columns", "Create new column", "Bin numeric column"])

        if op == "Rename column":
            col = st.selectbox("Column to rename", df.columns.tolist())
            new_name = st.text_input("New name", value=col)
            if st.button("▶ Rename") and new_name:
                new_df = df.rename(columns={col: new_name})
                log_step("Rename Column", {"from": col, "to": new_name})
                set_wdf(new_df)
                st.success(f"✅ Renamed `{col}` -> `{new_name}`")
                st.rerun()

        elif op == "Drop columns":
            drop_cols = st.multiselect("Columns to drop", df.columns.tolist())
            if drop_cols and st.button("▶ Drop"):
                new_df = df.drop(columns=drop_cols)
                log_step("Drop Columns", {"columns": drop_cols})
                set_wdf(new_df)
                st.success(f"✅ Dropped: {drop_cols}")
                st.rerun()

        elif op == "Create new column":
            new_col_name = st.text_input("New column name", "new_col")
            st.markdown("**Formula** - use column names directly. Examples:")
            st.code("col_a / col_b\nnp.log(col_a + 1)\ncol_a - col_a.mean()")
            formula = st.text_area("Formula (Python expression)", height=80)
            if st.button("▶ Create") and formula and new_col_name:
                try:
                    local_vars = {c: df[c] for c in df.columns}
                    local_vars["np"] = np
                    result = eval(formula, {"__builtins__": {}}, local_vars)
                    new_df = df.copy()
                    new_df[new_col_name] = result
                    log_step("Create Column", {"new_column": new_col_name, "formula": formula})
                    set_wdf(new_df)
                    st.success(f"✅ Created column `{new_col_name}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Formula error: {e}")

        elif op == "Bin numeric column":
            num_cols = df.select_dtypes(include="number").columns.tolist()
            col = st.selectbox("Numeric column to bin", num_cols)
            n_bins = st.slider("Number of bins", 2, 20, 5)
            strategy = st.radio("Binning strategy", ["Equal-width", "Quantile (equal-frequency)"])
            bin_col_name = st.text_input("New binned column name", f"{col}_bin")
            if st.button("▶ Bin"):
                new_df = df.copy()
                if strategy == "Equal-width":
                    new_df[bin_col_name] = pd.cut(new_df[col], bins=n_bins, labels=False)
                else:
                    new_df[bin_col_name] = pd.qcut(new_df[col], q=n_bins, labels=False, duplicates="drop")
                log_step("Bin Column", {"column": col, "bins": n_bins, "strategy": strategy, "new_column": bin_col_name})
                set_wdf(new_df)
                st.success(f"✅ Created `{bin_col_name}`")
                st.rerun()

    # ── 4.8 Data Validation ───────────────────────────────────────
    elif section == sections[7]:
        st.subheader("✅ Data Validation Rules")
        rule_type = st.selectbox("Rule type", [
            "Numeric range check (min/max)",
            "Allowed categories list",
            "Non-null constraint",
        ])

        if rule_type == "Numeric range check (min/max)":
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                st.warning("⚠️ No numeric columns found in the current dataset.")
                st.stop()
            col = st.selectbox("Column", num_cols)
            if col:
                col_min = float(df[col].min())
                col_max = float(df[col].max())

            vmin = st.number_input("Minimum allowed", value=col_min)
            vmax = st.number_input("Maximum allowed", value=col_max)
            if st.button("▶ Check"):
                mask = (df[col] < vmin) | (df[col] > vmax)
                violations = df[mask]
                st.metric("Violations found", len(violations))
                if not violations.empty:
                    st.dataframe(violations, use_container_width=True)
                    buf = io.BytesIO()
                    violations.to_csv(buf, index=False)
                    st.download_button("⬇️ Export violations CSV", buf.getvalue(), file_name="violations.csv")

        elif rule_type == "Allowed categories list":
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            col = st.selectbox("Column", cat_cols)
            unique_vals = df[col].dropna().unique().tolist()
            st.caption(f"Current unique values: {unique_vals[:20]}")
            allowed_input = st.text_area("Allowed values (one per line)")
            if st.button("▶ Check") and allowed_input:
                allowed = [v.strip() for v in allowed_input.strip().split("\n") if v.strip()]
                mask = ~df[col].isin(allowed) & df[col].notna()
                violations = df[mask]
                st.metric("Violations found", len(violations))
                if not violations.empty:
                    st.dataframe(violations[[col]], use_container_width=True)
                    buf = io.BytesIO()
                    violations.to_csv(buf, index=False)
                    st.download_button(
                        "⬇️ Export violations CSV",
                        buf.getvalue(),
                        file_name="category_violations.csv",
                        mime="text/csv",
                    )        
        elif rule_type == "Non-null constraint":
            sel_cols = st.multiselect("Columns that must be non-null", df.columns.tolist())
            if st.button("▶ Check") and sel_cols:
                mask = df[sel_cols].isnull().any(axis=1)
                violations = df[mask]
                st.metric("Rows with nulls in selected columns", len(violations))
                if not violations.empty:
                    st.dataframe(violations[sel_cols], use_container_width=True)
                    buf = io.BytesIO()
                    violations.to_csv(buf, index=False)
                    st.download_button(
                        "⬇️ Export violations CSV",
                        buf.getvalue(),
                        file_name="null_violations.csv",
                        mime="text/csv",
                    )        


# ═══════════════════════════════════════════════════════════════════
# PAGE C - Visualization Builder
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("📊 Visualization Builder")

    if wdf() is None:
        st.warning("Please upload a dataset on the Upload page first.")
        st.stop()

    df = wdf()

    missing = df.isnull().sum()

    if missing.sum() > 0:
        st.warning("⚠️ Dataset contains missing values. Visualizations may not work properly.")

        st.write("Missing values per column:")
        st.dataframe(missing[missing > 0])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧹 Drop missing values"):
                df = df.dropna()
                st.success("Missing values removed!")
    
        with col2:
            if st.button("➡️ Continue anyway"):
                st.info("Proceeding with missing data...")
    # ── Filters sidebar ───────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=False):
        # Only show truly categorical columns (low unique values)
        all_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in all_cat if df[c].nunique() <= 30]
        num_cols = df.select_dtypes(include="number").columns.tolist()

        filters = {}
        if cat_cols:
            cat_filter_col = st.selectbox("Filter by category column", ["(none)"] + cat_cols)
            if cat_filter_col != "(none)":
                unique_cats = df[cat_filter_col].dropna().unique().tolist()
                sel_cats = st.multiselect("Include categories", unique_cats, default=unique_cats)
                filters["cat"] = (cat_filter_col, sel_cats)

        if num_cols:
            num_filter_col = st.selectbox("Filter by numeric range", ["(none)"] + num_cols)
            if num_filter_col != "(none)":
                col_min = float(df[num_filter_col].min())
                col_max = float(df[num_filter_col].max())
                rng = st.slider("Range", col_min, col_max, (col_min, col_max))
                filters["num"] = (num_filter_col, rng)

        # Apply filters
        filtered_df = df.copy()
        if "cat" in filters:
            c, cats = filters["cat"]
            filtered_df = filtered_df[filtered_df[c].isin(cats)]
        if "num" in filters:
            c, (lo, hi) = filters["num"]
            filtered_df = filtered_df[(filtered_df[c] >= lo) & (filtered_df[c] <= hi)]

        st.caption(f"Filtered dataset: {len(filtered_df):,} rows")

    # ── Chart builder ─────────────────────────────────────────────
    CHART_TYPES = [
        "Histogram",
        "Box Plot",
        "Scatter Plot",
        "Line Chart",
        "Bar Chart",
        "Heatmap / Correlation Matrix",
    ]
 
    chart_type = st.selectbox("Chart type", CHART_TYPES)
    st.caption("💡 Not sure what chart to use? Try the **🤖 AI Assistant** page for suggestions!")

    # ── Inline AI advisor ─────────────────────────────────────────
    api_key = st.session_state.get("api_key", "")
    if api_key:
        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            if st.button("🤖 Suggest best columns for this chart", key="inline_ai_btn"):
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
                num_cols_info = df.select_dtypes(include="number").columns.tolist()
                cat_cols_info = df.select_dtypes(exclude="number").columns.tolist()
                prompt = (
                    f"Dataset has these columns:\n"
                    f"Numeric: {clean_ascii(num_cols_info)}\n"
                    f"Categorical: {clean_ascii(cat_cols_info)}\n\n"
                    f"The user wants to create a {chart_type}.\n"
                    f"Suggest the BEST combination of columns to use. "
                    f"Give 2-3 specific examples like:\n"
                    f"Example 1: X=quantity, Y=revenue, Color=region - Why: ...\n"
                    f"Keep it short and practical."
                )
                prompt = prompt.encode('ascii', 'ignore').decode('ascii')
                with st.spinner("🤖 AI is thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,
                            temperature=0.3,
                        )
                        suggestion = response.choices[0].message.content.strip()
                        st.info(f"💡 **AI Column Advisor:**\n\n{suggestion}")
                    except Exception as e:
                        st.warning(f"⚠️ AI error: {e}")
        with col_ai2:
            if st.button("🤖 Suggest & Draw Charts automatically", key="ai_draw_btn"):
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
                num_cols_info = df.select_dtypes(include="number").columns.tolist()
                cat_cols_info = df.select_dtypes(exclude="number").columns.tolist()
                prompt = (
                    f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols\n"
                    f"Numeric columns: {clean_ascii(num_cols_info)}\n"
                    f"Categorical columns: {clean_ascii(cat_cols_info)}\n\n"
                    f"Suggest exactly 3 charts. Reply ONLY with a valid JSON array, no explanation.\n"
                    f"Each object must have: chart_type (Histogram/Bar Chart/Scatter Plot/Line Chart/Box Plot), "
                    f"x (column name), y (column name or null), title (short title).\n"
                    f"Only use column names that exist."
                )
                prompt = prompt.encode('ascii', 'ignore').decode('ascii')
                with st.spinner("🤖 AI is drawing charts..."):
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500,
                            temperature=0.3,
                        )
                        result = response.choices[0].message.content.strip()
                        json_match = re.search(r'\[.*\]', result, re.DOTALL)
                        if not json_match:
                            raise ValueError("No JSON found")
                        charts = json.loads(json_match.group())
                        st.subheader("💡 AI Recommendations")
                        for c in charts:
                            st.info(f"**{c.get('title')}** - {c.get('chart_type')} using `{c.get('x')}`" +
                                    (f" vs `{c.get('y')}`" if c.get('y') else ""))
                        st.divider()
                        st.subheader("📊 AI Generated Charts")
                        for i, chart in enumerate(charts):
                            ct = chart.get("chart_type", "")
                            xc = chart.get("x")
                            yc = chart.get("y")
                            title = chart.get("title", f"Chart {i+1}")
                            if xc not in df.columns:
                                st.warning(f"⚠️ Column '{xc}' not found - skipping.")
                                continue
                            if yc and yc not in df.columns:
                                yc = None
                            try:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                if ct == "Histogram":
                                    num_data = pd.to_numeric(df[xc], errors="coerce").dropna()
                                    ax.hist(num_data, bins=30, color="#6366f1", edgecolor="white", alpha=0.85)
                                    ax.set_xlabel(xc); ax.set_ylabel("Count")
                                elif ct == "Bar Chart":
                                    vc = df[xc].value_counts().head(10)
                                    bars = ax.bar(vc.index.astype(str), vc.values, color="#6366f1", edgecolor="white")
                                    ax.bar_label(bars, padding=3, fontsize=9)
                                    ax.set_xlabel(xc); ax.set_ylabel("Count")
                                    plt.xticks(rotation=30, ha="right")
                                elif ct == "Scatter Plot" and yc:
                                    x_num = pd.to_numeric(df[xc], errors="coerce")
                                    y_num = pd.to_numeric(df[yc], errors="coerce")
                                    pdf = pd.DataFrame({"x": x_num, "y": y_num}).dropna().sample(min(500, len(df)))
                                    ax.scatter(pdf["x"], pdf["y"], alpha=0.5, color="#6366f1", s=20)
                                    ax.set_xlabel(xc); ax.set_ylabel(yc)
                                elif ct == "Line Chart" and yc:
                                    x_num = pd.to_numeric(df[xc], errors="coerce")
                                    y_num = pd.to_numeric(df[yc], errors="coerce")
                                    pdf = pd.DataFrame({"x": x_num, "y": y_num}).dropna()
                                    pdf = pdf.groupby("x")["y"].mean().reset_index()
                                    ax.plot(pdf["x"], pdf["y"], color="#6366f1", linewidth=2)
                                    ax.fill_between(pdf["x"], pdf["y"], alpha=0.1, color="#6366f1")
                                    ax.set_xlabel(xc); ax.set_ylabel(yc)
                                elif ct == "Box Plot":
                                    num_data = pd.to_numeric(df[xc], errors="coerce").dropna()
                                    ax.boxplot(num_data, patch_artist=True,
                                               boxprops=dict(facecolor="#93c5fd", alpha=0.7),
                                               medianprops=dict(color="#1e40af", linewidth=2))
                                    ax.set_ylabel(xc)
                                else:
                                    plt.close(); continue
                                ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
                                ax.spines["top"].set_visible(False)
                                ax.spines["right"].set_visible(False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                st.caption("⚠️ AI-generated chart. Results may not always be accurate.")
                                plt.close()
                            except Exception as chart_err:
                                st.warning(f"⚠️ Could not draw '{title}': {chart_err}")
                                plt.close()
                    except Exception as e:
                        st.warning(f"⚠️ AI error: {e}")
    else:
        st.caption("🔑 Add Groq API key in sidebar to enable AI chart features")
    use_plotly = st.toggle("🔍 Interactive mode (Plotly)", value=False,
                           help="Enable zoomable, hoverable charts using Plotly")
    
    all_cols = filtered_df.columns.tolist()
    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    # Also include object columns that contain numeric values
    for c in filtered_df.select_dtypes(include="object").columns:
        if pd.to_numeric(filtered_df[c], errors="coerce").notna().sum() > len(filtered_df) * 0.5:
            if c not in num_cols:
                num_cols.append(c)
    cat_cols = [c for c in filtered_df.select_dtypes(include=["object", "category"]).columns
                if c not in num_cols and filtered_df[c].nunique() <= 30]
    date_cols = filtered_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
 
    fig = None
 
    # ── Histogram ────────────────────────────────────────────────
    if chart_type == "Histogram":
        hist_cols = num_cols + [c for c in filtered_df.select_dtypes(include="object").columns
                                if pd.to_numeric(filtered_df[c], errors="coerce").notna().sum() > len(filtered_df) * 0.3]
        col = st.selectbox("Column", hist_cols if hist_cols else filtered_df.columns.tolist())
        bins = st.slider("Bins", 5, 100, 30)
        hue = st.selectbox("Group by (color)", ["(none)"] + cat_cols)
 
        if col:
            # 🔴 ADD THIS HERE
            if hue != "(none)":
                top_n = 5
                top_categories = filtered_df[hue].value_counts().head(top_n).index
                filtered_df = filtered_df[filtered_df[hue].isin(top_categories)]
 
        
            fig, ax = plt.subplots(figsize=(9, 5))
            col_data = pd.to_numeric(filtered_df[col], errors="coerce").dropna()
            is_categorical = len(col_data) < len(filtered_df) * 0.5

            if is_categorical:
                # Show as bar chart for categorical columns
                vc = filtered_df[col].value_counts().head(15)
                ax.bar(vc.index.astype(str), vc.values, color="#6366f1", edgecolor="white")
                ax.set_ylabel("Count")
                plt.xticks(rotation=30, ha="right")
            elif hue == "(none)":
                ax.hist(col_data, bins=bins, color="#6366f1", edgecolor="white")
            else:
                for cat, group in filtered_df.groupby(hue):
                    ax.hist(pd.to_numeric(group[col], errors="coerce").dropna(),
                            bins=bins, alpha=0.6, label=str(cat), edgecolor="white")
                ax.legend(title=hue)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            plt.xticks(rotation=45, ha="right")
            # Remove extreme outliers from display range
            col_clean = col_data[np.isfinite(col_data)]
            if len(col_clean) > 0:
                q99 = col_clean.quantile(0.99)
                q01 = col_clean.quantile(0.01)
                if q99 > q01:
                    ax.set_xlim(q01 - (q99-q01)*0.1, q99 + (q99-q01)*0.1)
            plt.xticks(rotation=45, ha="right")
            
            ax.set_title(f"Histogram: {col}")
 
    # ── Box Plot ─────────────────────────────────────────────────
    elif chart_type == "Box Plot":
        col = st.selectbox("Numeric column (Y)", num_cols if num_cols else all_cols)
        group = st.selectbox("Group by (X)", ["(none)"] + cat_cols)
 
        if col:
            fig, ax = plt.subplots(figsize=(9, 5))
            col_data = pd.to_numeric(filtered_df[col], errors="coerce").dropna()
            if group == "(none)":
                ax.boxplot(col_data, patch_artist=True, boxprops=dict(facecolor="#93c5fd"))
                ax.set_xticklabels([col])
            else:
                top_n = st.slider("Top N categories", 2, 20, 8)
                top_cats = filtered_df[group].value_counts().head(top_n).index.tolist()
                data = [filtered_df[filtered_df[group] == c][col].dropna() for c in top_cats]
                ax.boxplot(data, patch_artist=True, labels=top_cats)
                ax.set_xlabel(group)
                plt.xticks(rotation=30, ha="right")
            ax.set_ylabel(col)
            ax.set_title(f"Box Plot: {col}")
 
    # ── Scatter Plot ─────────────────────────────────────────────
    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("X axis", num_cols if num_cols else all_cols, index=0)
        y_col = st.selectbox("Y axis", num_cols if num_cols else all_cols,
                             index=min(1, len(num_cols) - 1) if len(num_cols) > 1 else 0)
        color = st.selectbox("Color by", ["(none)"] + cat_cols + num_cols)
        alpha = st.slider("Opacity", 0.1, 1.0, 0.6)
 
        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(9, 6))
            x_data = pd.to_numeric(filtered_df[x_col], errors="coerce")
            y_data = pd.to_numeric(filtered_df[y_col], errors="coerce")
            if color == "(none)":
                sc = ax.scatter(x_data, y_data, alpha=alpha, color="#6366f1", s=15)
            else:
                if color in cat_cols:
                    cats = filtered_df[color].astype("category").cat.codes
                    sc = ax.scatter(x_data, y_data, c=cats, alpha=alpha, cmap="tab10", s=15)
                else:
                    sc = ax.scatter(x_data, y_data, c=pd.to_numeric(filtered_df[color], errors="coerce"),
                                   alpha=alpha, cmap="viridis", s=15)
                    plt.colorbar(sc, ax=ax, label=color)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter: {x_col} vs {y_col}")
 
    # ── Line Chart ────────────────────────────────────────────────
    elif chart_type == "Line Chart":
        all_sortable = date_cols + num_cols
        x_col = st.selectbox("X axis (time/index)", all_sortable if all_sortable else all_cols)
        y_col = st.selectbox("Y axis (numeric)", num_cols if num_cols else all_cols)
        group = st.selectbox("Group by (color)", ["(none)"] + cat_cols)
        agg = st.selectbox("Aggregation", ["mean", "sum", "count", "median"])

        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(11, 5))
            plot_df = filtered_df.copy()
            # Only convert Y to numeric, keep X as-is
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")

            if x_col == y_col:
                st.warning("⚠️ X and Y columns must be different.")
                fig = None
            elif group == "(none)":
                tmp = plot_df[[x_col, y_col]].dropna()
                # If X is numeric with too many values, bucket it
                x_numeric = pd.to_numeric(tmp[x_col], errors="coerce")
                if x_numeric.notna().sum() > len(tmp) * 0.5 and tmp[x_col].nunique() > 50:
                    n_buckets = st.select_slider(
                        "Too many X values — group into buckets",
                        options=[10, 20, 30, 50],
                        value=20,
                        key="line_buckets"
                    )
                    tmp[x_col] = pd.cut(x_numeric, bins=n_buckets, labels=False).astype(float)
                tmp = tmp.groupby(x_col)[y_col].agg(agg).reset_index().sort_values(x_col)
                ax.plot(tmp[x_col], tmp[y_col], color="#6366f1", linewidth=2, marker='o', markersize=4)
            elif group != "(none)":
                top_cats = plot_df[group].value_counts().head(8).index.tolist()
                for cat in top_cats:
                    sub = plot_df[plot_df[group] == cat][[x_col, y_col]].dropna()
                    sub = sub.groupby(x_col)[y_col].agg(agg).reset_index().sort_values(x_col)
                    ax.plot(sub[x_col], sub[y_col], label=str(cat), linewidth=2, marker='o', markersize=4)
                ax.legend(title=group)
            if fig is not None and x_col != y_col:
                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg}({y_col})")
                ax.set_title(f"Line Chart: {y_col} by {x_col}")
                plt.xticks(rotation=30, ha="right")
                
    # ── Bar Chart ─────────────────────────────────────────────────
    elif chart_type == "Bar Chart":
        x_col = st.selectbox("Category column (X)", cat_cols + num_cols if cat_cols else all_cols)
        y_col = st.selectbox("Value column (Y)", num_cols if num_cols else all_cols)
        group = st.selectbox("Group by (stacked/grouped)", ["(none)"] + cat_cols)
        agg = st.selectbox("Aggregation", ["mean", "sum", "count", "median"])
        top_n = st.slider("Show top N categories", 2, 30, 10)
 
        if x_col and y_col:
            if x_col == y_col:
                st.warning("⚠️ X and Y columns must be different.")
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                # Convert y column to numeric safely
                filtered_df = filtered_df.copy()
                filtered_df[y_col] = pd.to_numeric(filtered_df[y_col], errors="coerce")
                if group == "(none)":
                    plot_df = filtered_df.groupby(x_col)[y_col].agg(agg).nlargest(top_n).reset_index()
                    ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color="#6366f1")
                else:
                    top_cats = filtered_df[x_col].value_counts().head(top_n).index.tolist()
                    sub = filtered_df[filtered_df[x_col].isin(top_cats)]
                    pivot = sub.groupby([x_col, group])[y_col].agg(agg).unstack(fill_value=0)
                    pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
                    ax.legend(title=group, bbox_to_anchor=(1.01, 1), loc="upper left")
                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg}({y_col})")
                ax.set_title(f"Bar Chart: {agg}({y_col}) by {x_col}")
                plt.xticks(rotation=30, ha="right")
 
    # ── Heatmap / Correlation Matrix ──────────────────────────────
    elif chart_type == "Heatmap / Correlation Matrix":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation matrix. Try converting columns to numeric in Page B -> 4.3 first.")
        else:
            sel_cols = st.multiselect("Columns to include", num_cols, default=num_cols[:min(8, len(num_cols))])
            cmap = st.selectbox("Color map", ["coolwarm", "viridis", "RdYlGn", "Blues"])
            if sel_cols and len(sel_cols) >= 2:
                heat_df = filtered_df[sel_cols].copy()
                for c in heat_df.columns:
                    heat_df[c] = pd.to_numeric(heat_df[c], errors="coerce")
                heat_df = heat_df.dropna()
                if len(heat_df) == 0:
                    st.warning("⚠️ No valid numeric data after conversion. Check your columns.")
                else:
                    corr = heat_df.corr()
                    fig, ax = plt.subplots(figsize=(max(6, len(sel_cols)), max(5, len(sel_cols) - 1)))
                    sns.heatmap(
                        corr, annot=True, fmt=".2f", cmap=cmap,
                        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                    )
                    ax.set_title("Correlation Matrix")
 
    # ── Render chart ─────────────────────────────────────────────
    if use_plotly:
        import plotly.express as px
        pfig = None
        try:
            if chart_type == "Histogram":
                hist_df = filtered_df.copy()
                col_numeric = pd.to_numeric(hist_df[col], errors="coerce")
                if col_numeric.notna().sum() < len(hist_df) * 0.5:
                    # Categorical — show as bar chart
                    vc = hist_df[col].value_counts().head(15).reset_index()
                    vc.columns = [col, "count"]
                    pfig = px.bar(vc, x=col, y="count",
                                  title=f"Distribution: {col}",
                                  template="plotly_white")
                else:
                    # Numeric — show as histogram
                    hist_df[col] = col_numeric
                    hist_df = hist_df.dropna(subset=[col])
                    pfig = px.histogram(hist_df, x=col,
                                        color=hue if hue != "(none)" else None,
                                        nbins=bins,
                                        barmode="overlay",
                                        title=f"Histogram: {col}",
                                        template="plotly_white",
                                        opacity=0.75)
                    col_clean = hist_df[col]
                    if len(col_clean) > 0 and col_clean.max() > col_clean.min():
                        bin_size = (col_clean.max() - col_clean.min()) / bins
                        pfig.update_traces(xbins=dict(size=bin_size),
                                           marker=dict(line=dict(width=1, color="white")))
            elif chart_type == "Box Plot":
                pfig = px.box(filtered_df, y=col, x=group if group != "(none)" else None,
                              title=f"Box Plot: {col}", template="plotly_white")
            elif chart_type == "Scatter Plot":
                pfig = px.scatter(filtered_df, x=x_col, y=y_col,
                                  color=color if color != "(none)" else None,
                                  opacity=alpha, title=f"Scatter: {x_col} vs {y_col}",
                                  template="plotly_white")

            elif chart_type == "Line Chart" and x_col != y_col:
                line_df = filtered_df.copy()
                line_df[y_col] = pd.to_numeric(line_df[y_col], errors="coerce")
                line_df = line_df.dropna(subset=[y_col])
                x_numeric = pd.to_numeric(line_df[x_col], errors="coerce")
                if x_numeric.notna().sum() > len(line_df) * 0.5 and line_df[x_col].nunique() > 50:
                    line_df[x_col] = pd.cut(x_numeric, bins=20, labels=False).astype(float)
                line_df = line_df.groupby(x_col)[y_col].agg(agg).reset_index()
                pfig = px.line(line_df, x=x_col, y=y_col,
                               color_discrete_sequence=["#6366f1"],
                               title=f"Line Chart: {agg}({y_col}) by {x_col}",
                               template="plotly_white",
                               markers=True)

            elif chart_type == "Bar Chart":
                # Warn if Y column looks like an ID
                if any(keyword in y_col.lower() for keyword in ["id", "code", "key", "index", "num"]):
                    st.warning(f"⚠️ '{y_col}' looks like an ID column - aggregating it may not be meaningful. "
                               f"Try selecting a numeric measure column instead (e.g. price, score, amount).")
                plot_df = filtered_df.copy()
                plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
                plot_df = plot_df.groupby(x_col)[y_col].agg(agg).nlargest(top_n).reset_index()
                plot_df = plot_df.sort_values(y_col, ascending=False)
                pfig = px.bar(plot_df, x=x_col, y=y_col,
                              title=f"Bar Chart: {agg}({y_col}) by {x_col} (Top {top_n})",
                              template="plotly_white",
                              color=y_col,
                              color_continuous_scale="Blues")
                pfig.update_layout(showlegend=False)
            elif chart_type == "Heatmap / Correlation Matrix" and len(num_cols) >= 2:
                heat_df = filtered_df[sel_cols].copy()
                for c in heat_df.columns:
                    heat_df[c] = pd.to_numeric(heat_df[c], errors="coerce")
                corr = heat_df.dropna().corr()
                # Map matplotlib cmap names to Plotly scale names
                cmap_map = {
                    "coolwarm": "RdBu_r",
                    "viridis": "Viridis",
                    "RdYlGn": "RdYlGn",
                    "Blues": "Blues"
                }
                plotly_cmap = cmap_map.get(cmap, "RdBu_r")
                pfig = px.imshow(corr, text_auto=".2f",
                                 color_continuous_scale=plotly_cmap,
                                 zmin=-1, zmax=1,
                                 title="Correlation Matrix",
                                 template="plotly_white")
            if pfig:
                st.plotly_chart(pfig, use_container_width=True)
                st.caption("💡 Interactive chart - zoom, pan and hover for details.")
            else:
                st.info("Please select valid columns to generate a chart.")
        except Exception as e:
            st.warning(f"⚠️ Could not render interactive chart: {e}. Try matplotlib mode.")

    elif fig is not None:
        plt.tight_layout()
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        st.download_button(
            "⬇️ Download Chart (PNG)",
            buf.getvalue(),
            file_name=f"{chart_type.lower().replace(' ', '_')}.png",
            mime="image/png",
        )
    else:
        st.info("Please select valid columns to generate a chart.") 
# ═══════════════════════════════════════════════════════════════════
# PAGE D - Export & Report
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("💾 Export & Report")

    if wdf() is None:
        st.warning("Please upload a dataset on the Upload page first.")
        st.stop()

    df = wdf()
    log = st.session_state.transform_log

    # ── Dataset export ────────────────────────────────────────────
    st.subheader("📤 Export Cleaned Dataset")
    c1, c2 = st.columns(2)
    with c1:
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv_buf.getvalue(),
            file_name=f"cleaned_{st.session_state.filename or 'dataset'}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            "⬇️ Download Excel",
            xlsx_buf.getvalue(),
            file_name=f"cleaned_{st.session_state.filename or 'dataset'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ── Transformation report ────────────────────────────────────
    st.subheader("📋 Transformation Report")
    if not log:
        st.info("No transformations have been applied yet.")
    else:
        st.dataframe(
            pd.DataFrame(log)[["step", "operation", "params", "timestamp", "shape_after"]],
            use_container_width=True,
        )

        # JSON recipe
        report = {
            "filename": st.session_state.filename,
            "generated_at": datetime.datetime.now().isoformat(),
            "original_shape": list(st.session_state.raw_df.shape) if st.session_state.raw_df is not None else None,
            "final_shape": list(df.shape),
            "steps": log,
        }
        report_json = json.dumps(report, indent=2, default=str)

        st.download_button(
            "⬇️ Download Transformation Report (JSON)",
            report_json,
            file_name="transformation_report.json",
            mime="application/json",
            use_container_width=True,
        )

        # Python script snippet
        st.subheader("🐍 Python Pipeline Snippet")
        st.caption("A runnable pandas script that replays your transformations.")
        code_lines = ["import pandas as pd", "import numpy as np", "", f'df = pd.read_csv("{st.session_state.filename}")', ""]
        for step in log:
            op = step["operation"]
            p = step["params"]
            code_lines.append(f"# Step {step['step']}: {op}")
            if op == "Missing Values":
                action = p.get("action", "")
                cols = p.get("columns", [])
                if "Drop rows" in action:
                    code_lines.append(f"df = df.dropna(subset={cols})")
                elif "mean" in action.lower():
                    for c in cols:
                        code_lines.append(f"df['{c}'] = df['{c}'].fillna(df['{c}'].mean())")
                elif "median" in action.lower():
                    for c in cols:
                        code_lines.append(f"df['{c}'] = df['{c}'].fillna(df['{c}'].median())")
                elif "mode" in action.lower():
                    for c in cols:
                        code_lines.append(f"df['{c}'] = df['{c}'].fillna(df['{c}'].mode().iloc[0])")
                elif "forward" in action.lower():
                    code_lines.append(f"df[{cols}] = df[{cols}].ffill()")
                elif "backward" in action.lower():
                    code_lines.append(f"df[{cols}] = df[{cols}].bfill()")
            elif op == "Remove Duplicates":
                code_lines.append(f"df = df.drop_duplicates(keep='{p.get('keep', 'first')}')")
            elif op == "Type Conversion":
                code_lines.append(f"df['{p['column']}'] = pd.to_numeric(df['{p['column']}'], errors='coerce')")
            elif op == "Categorical - Trim/Case":
                case_map = {"lower": ".str.lower()", "UPPER": ".str.upper()", "Title": ".str.title()", "No change": ""}
                suffix = case_map.get(p.get("case", "No change"), "")
                code_lines.append(f"df['{p['column']}'] = df['{p['column']}'].astype(str).str.strip(){suffix}")
            elif op == "Scaling":
                for c in p.get("columns", []):
                    if "Min-Max" in p["method"]:
                        code_lines.append(f"df['{c}'] = (df['{c}'] - df['{c}'].min()) / (df['{c}'].max() - df['{c}'].min())")
                    else:
                        code_lines.append(f"df['{c}'] = (df['{c}'] - df['{c}'].mean()) / df['{c}'].std()")
            elif op == "Drop Columns":
                code_lines.append(f"df = df.drop(columns={p.get('columns', [])})")
            elif op == "Rename Column":
                code_lines.append(f"df = df.rename(columns={{'{p['from']}': '{p['to']}'}})")
            elif op == "Create Column":
                code_lines.append(f"df['{p['new_column']}'] = {p['formula']}")
            elif op == "Outlier Treatment":
                col = p.get("column")
                action = p.get("action", "")
                if "Cap" in action:
                    code_lines.append(f"_q1, _q3 = df['{col}'].quantile(0.25), df['{col}'].quantile(0.75)")
                    code_lines.append(f"_iqr = _q3 - _q1")
                    code_lines.append(f"df['{col}'] = df['{col}'].clip(lower=_q1-1.5*_iqr, upper=_q3+1.5*_iqr)")
                elif "Remove" in action:
                    code_lines.append(f"_q1, _q3 = df['{col}'].quantile(0.25), df['{col}'].quantile(0.75)")
                    code_lines.append(f"_iqr = _q3 - _q1")
                    code_lines.append(f"df = df[(df['{col}'] >= _q1-1.5*_iqr) & (df['{col}'] <= _q3+1.5*_iqr)]")
            code_lines.append("")

        code_lines.append('df.to_csv("cleaned_output.csv", index=False)')
        snippet = "\n".join(code_lines)
        st.code(snippet, language="python")
        st.download_button(
            "⬇️ Download Pipeline Script (.py)",
            snippet,
            file_name="pipeline.py",
            mime="text/plain",
        )

    # ── Current dataset preview ───────────────────────────────────
    st.subheader("👁️ Final Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ═══════════════════════════════════════════════════════════════════
# PAGE E - AI Assistant (Optional Feature)
# ═══════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("🤖 AI Assistant (Optional Feature)")
    st.caption("⚠️ AI outputs may be imperfect. Always review suggestions before applying.")

    # ── API Key input ─────────────────────────────────────────────
    api_key = st.session_state.get("api_key", "")
    ai_enabled = bool(api_key)    

    api_key = st.session_state.get("api_key", "")
    ai_enabled = bool(api_key)

    if not ai_enabled:
        st.info("👆 Enter your Groq API key in the sidebar to enable AI features.")
        st.stop()

    if wdf() is None:
        st.warning("Please upload a dataset on the Upload page first.")
        st.stop()

    df = wdf()

    # ── Helper functions ──────────────────────────────────────────
    def ask_openai(prompt: str, system: str = "You are a helpful data science assistant.") -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            # Clean non-ASCII characters from prompt
            prompt = prompt.encode('ascii', 'ignore').decode('ascii')
            system = system.encode('ascii', 'ignore').decode('ascii')
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=800,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ AI error: {e}"

    def get_dataset_context(df: pd.DataFrame) -> str:
        """Build a short text summary of the dataset for AI context."""
        num_cols  = df.select_dtypes(include="number").columns.tolist()
        cat_cols  = df.select_dtypes(exclude="number").columns.tolist()
        miss      = df.isnull().sum()
        miss_cols = miss[miss > 0].index.tolist()
        return (
            f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns.\n"
            f"Numeric columns: {clean_ascii(num_cols)}\n"
            f"Categorical columns: {clean_ascii(cat_cols)}\n"
            f"Columns with missing values: {clean_ascii(miss_cols)}\n"
            f"Sample data (first 3 rows):\n{clean_ascii(df.head(3).to_string())}"
        )

    # ── Four AI feature tabs ──────────────────────────────────────
    ai_tab3, ai_tab4 = st.tabs([
        "🐍 Code Snippet Generator",
        "📖 Data Dictionary",
    ])
    # ──────────────────────────────────────────────────────────────
    # FEATURE 3 - Code Snippet Generator
    # ──────────────────────────────────────────────────────────────
    with ai_tab3:
        st.subheader("🐍 Code Snippet Generator")
        st.markdown("Generates a clean pandas Python script based on your transformation log.")

        log = st.session_state.transform_log

        if not log:
            st.info("No transformations applied yet. Apply some cleaning steps in Page B first.")
        else:
            st.write(f"**{len(log)} transformation(s)** recorded in your log.")

            if st.button("🤖 Generate Code", key="code_btn"):
                log_text = "\n".join([
                    f"Step {s['step']}: {s['operation']} - params: {s['params']}"
                    for s in log
                ])
                context = get_dataset_context(df)
                prompt = (
                    f"{context}\n\n"
                    f"The user applied these transformations:\n{log_text}\n\n"
                    f"Generate a clean, well-commented pandas Python script that reproduces "
                    f"all these steps from scratch. Start with pd.read_csv('filename.csv'). "
                    f"Use only pandas and numpy. Add a comment before each step."
                )
                prompt = prompt.encode('ascii', 'ignore').decode('ascii')
                with st.spinner("Generating code..."):
                    result = ask_openai(prompt)

                st.subheader("💡 Generated Python Script")
                st.code(result, language="python")

                st.download_button(
                    "⬇️ Download AI Script (.py)",
                    result,
                    file_name="ai_pipeline.py",
                    mime="text/plain",
                )
                st.caption("⚠️ Always test AI-generated code before using in production.")

    # ──────────────────────────────────────────────────────────────
    # FEATURE 4 - Data Dictionary Generator
    # ──────────────────────────────────────────────────────────────
    with ai_tab4:
        st.subheader("📖 Data Dictionary Generator")
        st.markdown("The AI will infer the likely meaning of each column and flag potential data quality issues.")

        if st.button("🤖 Generate Dictionary", key="dict_btn"):
            context = get_dataset_context(df)
            prompt = (
                f"{context}\n\n"
                f"Generate a data dictionary for this dataset. For each column provide:\n"
                f"- Column name\n"
                f"- Likely meaning / description\n"
                f"- Data type\n"
                f"- Potential issues (e.g. missing values, outliers, wrong type)\n"
                f"- Suggested cleaning action if needed\n\n"
                f"Format as a clear table."
            )
            prompt = prompt.encode('ascii', 'ignore').decode('ascii')
            with st.spinner("Generating data dictionary..."):
                result = ask_openai(prompt)

            st.subheader("💡 AI Data Dictionary")
            st.info(result)

            st.download_button(
                "⬇️ Download Data Dictionary (.txt)",
                result,
                file_name="data_dictionary.txt",
                mime="text/plain",
            )
            st.caption("⚠️ AI-inferred meanings may not be accurate. Review and edit as needed.")