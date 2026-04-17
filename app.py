import streamlit as st
import tempfile, os, json
from pathlib import Path
from engine import build_index, load_existing_index, run_audit
from config import LLM_PROVIDER, GROQ_MODEL, OLLAMA_LLM_MODEL

st.set_page_config(
    page_title="Assumption Auditor",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Assumption Auditor")
st.caption(
    f"Running on: **{LLM_PROVIDER.upper()}** → "
    f"`{GROQ_MODEL if LLM_PROVIDER == 'groq' else OLLAMA_LLM_MODEL}`"
)

for key in ["index", "audit_result", "paper_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("📄 Upload Paper")
    uploaded = st.file_uploader("Upload a research PDF", type=["pdf"])

    if uploaded:
        if st.session_state.paper_name != uploaded.name:
            with st.spinner("Indexing paper..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    st.session_state.index = build_index(tmp_path)
                    st.session_state.paper_name = uploaded.name
                    st.session_state.audit_result = None
                    st.success(f"✅ Indexed: {uploaded.name}")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                finally:
                    os.unlink(tmp_path)

    elif Path("./chromadb_store").exists():
        if st.button("🔄 Load Previous Index"):
            try:
                st.session_state.index = load_existing_index()
                st.success("Loaded existing index.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.divider()
    st.markdown("**Provider Status**")
    if LLM_PROVIDER == "groq":
        from config import GROQ_API_KEY
        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            st.success("Groq ✓ Key set")
        else:
            st.error("Groq ✗ Set GROQ_API_KEY in .env")

if st.session_state.index is None:
    st.info("⬅️ Upload a research PDF from the sidebar to begin.")
    st.stop()

if st.button("🚀 Run Assumption Audit", type="primary", use_container_width=True):
    progress_box = st.empty()
    with st.spinner("Auditing paper..."):
        def update_progress(msg):
            progress_box.info(f"⏳ {msg}")
        try:
            result = run_audit(st.session_state.index, progress_callback=update_progress)
            st.session_state.audit_result = result
            progress_box.empty()
        except Exception as e:
            progress_box.empty()
            st.error(f"Audit failed: {e}")
            st.stop()

if st.session_state.audit_result:
    result = st.session_state.audit_result

    st.subheader("📊 Audit Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Assumptions", result["total"])
    c2.metric("Explicit", result["explicit_count"])
    c3.metric("Hidden (Implicit)", result["implicit_count"])
    c4.metric("🔴 Fatal", result["collapse_count"])
    c5.metric("🟡 Weakens", result["weaken_count"])

    st.divider()
    st.subheader("📌 Paper's Main Conclusion")
    st.info(result["conclusion"])

    st.divider()
    st.subheader("🔍 Assumptions Found")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_type = st.selectbox(
            "Filter by type",
            ["All", "Explicit only", "Implicit (Hidden) only"]
        )
    with col_f2:
        filter_crit = st.selectbox(
            "Filter by criticality",
            ["All", "🔴 Collapse", "🟡 Weaken", "🟢 Survive"]
        )

    CRITICALITY_MAP = {
        "collapse": ("🔴", "FATAL — conclusion collapses"),
        "weaken":   ("🟡", "WEAKENS — result loses strength"),
        "survive":  ("🟢", "SURVIVES — minor degradation"),
    }
    CATEGORY_COLORS = {
        "data": "🟦", "mathematical": "🟪", "scope": "🟧",
        "computational": "🟨", "worldview": "⬜", "experimental": "🟫"
    }

    assumptions = result["assumptions"]
    if filter_type == "Explicit only":
        assumptions = [a for a in assumptions if a.explicit]
    elif filter_type == "Implicit (Hidden) only":
        assumptions = [a for a in assumptions if not a.explicit]

    crit_map = {"🔴 Collapse": "collapse", "🟡 Weaken": "weaken", "🟢 Survive": "survive"}
    if filter_crit != "All":
        assumptions = [a for a in assumptions if a.criticality == crit_map.get(filter_crit)]

    if not assumptions:
        st.warning("No assumptions match the current filters.")
    else:
        for i, a in enumerate(assumptions):
            crit_icon, crit_label = CRITICALITY_MAP.get(a.criticality, ("⚪", "Unknown"))
            cat_icon = CATEGORY_COLORS.get(a.category, "⬜")
            explicit_tag = "**[EXPLICIT]**" if a.explicit else "**[HIDDEN]**"

            with st.expander(
                f"{crit_icon} {explicit_tag}  {a.assumption[:90]}{'...' if len(a.assumption) > 90 else ''}",
                expanded=(i < 3)
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Full Assumption:**  \n{a.assumption}")
                    if a.explicit and a.quote:
                        st.markdown(f"**Found in paper:**  \n> _{a.quote}_")
                    elif not a.explicit and a.evidence:
                        st.markdown(f"**Relies on:**  \n_{a.evidence}_")
                        if a.detection_reasoning:
                            st.markdown(f"**Why it's hidden:**  \n_{a.detection_reasoning}_")
                    if a.layman_explanation:
                        st.markdown("---")
                        st.markdown(f"💬 **Plain English:**  \n{a.layman_explanation}")
                with col2:
                    st.markdown(f"{cat_icon} **Category:** `{a.category}`")
                    st.markdown(f"{crit_icon} **Criticality:** {crit_label}")
                    if a.criticality_reasoning:
                        st.caption(a.criticality_reasoning)
                    st.markdown("---")
                    if a.real_world_bridge:
                        st.markdown(f"🌍 **Real-world failure:**  \n_{a.real_world_bridge}_")

    st.divider()
    export_data = {
        "conclusion": result["conclusion"],
        "summary": {
            "total": result["total"],
            "explicit": result["explicit_count"],
            "implicit": result["implicit_count"],
            "fatal": result["collapse_count"],
        },
        "assumptions": [a.dict() for a in result["assumptions"]]
    }
    st.download_button(
        label="⬇️ Export Full Audit as JSON",
        data=json.dumps(export_data, indent=2),
        file_name="assumption_audit.json",
        mime="application/json"
    )