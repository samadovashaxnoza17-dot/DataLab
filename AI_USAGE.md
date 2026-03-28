# AI Usage Documentation

**Module:** Data Wrangling and Visualization (5COSC038C)  
**Project:** AI-Assisted Data Wrangler & Visualizer

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Claude (Anthropic) | Code generation, architecture planning, debugging |
| GitHub Copilot | Inline code suggestions |

---

## How AI Was Used

### Architecture Design
- Used Claude to plan the 4-page Streamlit structure before writing any code.
- Prompt: *"Design a Streamlit app structure for a data wrangling tool with upload, cleaning, visualization, and export pages."*

### Feature Implementation
- Individual cleaning modules (outlier detection, normalization, one-hot encoding) were prompted separately and reviewed before integration.
- All AI-generated code was manually tested against both sample datasets.

### Debugging
- Used Claude to help diagnose `st.session_state` management issues and caching behaviour.

---

## What We Verified Manually

- [x] All file upload types (CSV, Excel, JSON) tested with real datasets
- [x] Missing value fill methods verified against expected pandas behaviour
- [x] Outlier IQR and Z-score bounds cross-checked with manual calculations
- [x] Normalization output range verified (min-max → [0, 1]; z-score → mean≈0, std≈1)
- [x] Transformation log records correct operations and parameters
- [x] Export: CSV and JSON downloads tested for completeness and correctness
- [x] Python pipeline snippet runs end-to-end independently of the app
- [x] All 6 chart types render without error on both sample datasets
- [x] Session reset clears all state correctly
- [x] Error messages appear for invalid inputs (bad formula, invalid JSON mapping)

---

## Features Implemented

- Google Sheets integration (public link + service account via gspread)
- LLM assistant (Groq/Llama-3.3-70b) with 4 features: natural language cleaning suggestions, AI chart suggestion, code snippet generator, and data dictionary generator

---

## Limitations Noted

- The formula input in "Create column" uses `eval()` with restricted builtins — complex expressions may not work as expected.
- LLM outputs may be imperfect; all suggestions require user confirmation before being applied.

---

## Prompts Used (Summary Log)

All chat transcripts are included in the `prompts/` folder submitted in the ZIP.

1. Initial architecture prompt
2. Page A — Upload & profiling
3. Page B — Each of the 8 cleaning sections
4. Page C — Visualization builder (each chart type)
5. Page D — Export and Python snippet generation
6. Debugging session: session state + caching
7. Sample dataset generation prompts
