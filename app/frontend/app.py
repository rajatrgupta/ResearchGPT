import streamlit as st
import requests
import time
import markdown
from fpdf import FPDF

def generate_pdf(md_text):
    # Sanitize common unicode characters that trip up fpdf2 default fonts
    safe_md_text = md_text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    html = markdown.markdown(safe_md_text)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    try:
        pdf.write_html(html)
    except Exception as e:
        print(f"Warning: PDF HTML parsing failed: {e}. Falling back to plain text.")
        pdf = FPDF() # Reset
        pdf.add_page()
        pdf.set_font("Courier", size=10)
        safe_text = safe_md_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 5, safe_text)
    
    return bytes(pdf.output())

# Backend API Configuration
API_BASE_URL = "https://researchgpt-backend-ygh8.onrender.com/api/v1/research"

st.set_page_config(
    page_title="ResearchGPT | AI Research Workspace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .report-header { font-size: 2rem; font-weight: 600; color: #FFFFFF; margin-bottom: 1rem; }
    .status-box { padding: 1rem; border-radius: 0.5rem; background-color: #1E2127; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "status" not in st.session_state:
    st.session_state.status = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None

# ==========================================
# SIDEBAR: Control Panel
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8212/8212611.png", width=60) # Placeholder generic AI logo
    st.title("ResearchGPT")
    st.caption("Autonomous Multi-Agent Intelligence Engine")
    st.divider()
    
    st.markdown("**Research Parameters**")
    query = st.text_area("Topic / Directive", placeholder="e.g., Future of Quantum Computing in Healthcare...", height=120)
    
    st.divider()
    if st.button("🚀 Initialize Research", type="primary", use_container_width=True):
        if not query:
            st.warning("⚠️ Please provide a research topic.")
        else:
            # Reset state for new run
            st.session_state.run_id = None
            st.session_state.final_report = None
            
            with st.spinner("Connecting to ResearchGPT Engine..."):
                try:
                    response = requests.post(API_BASE_URL, json={"query": query})
                    if response.status_code == 202:
                        data = response.json()
                        st.session_state.run_id = data["run_id"]
                        st.session_state.status = "pending"
                    else:
                        st.error(f"Engine Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

# ==========================================
# MAIN AREA: Workspace & Output
# ==========================================
if not st.session_state.run_id:
    # Empty State
    st.markdown("<div class='report-header'>Welcome to the ResearchGPT Workspace</div>", unsafe_allow_html=True)
    st.info("👈 Enter a research directive in the sidebar to begin autonomous synthesis.")
    
    # Example prompts to guide the user
    st.markdown("### Suggested Directives:")
    st.button("Analyze the economic impact of AGI by 2030", disabled=True)
    st.button("Compare Solid-State vs Lithium-ion battery architectures", disabled=True)

else:
    # Polling & Display State
    if st.session_state.status not in ["completed", "failed"]:
        st.markdown("### 🧠 Engine Execution in Progress...")
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        while st.session_state.status not in ["completed", "failed"]:
            try:
                poll_res = requests.get(f"{API_BASE_URL}/{st.session_state.run_id}")
                if poll_res.status_code == 200:
                    poll_data = poll_res.json()
                    current_status = poll_data.get("status")
                    current_stage = poll_data.get("current_stage", "initialized")
                    
                    # Visual feedback of agent stages
                    stages = {"initialized": 10, "planner": 30, "search": 50, "retriever": 70, "critic": 85, "writer": 95, "completed": 100}
                    progress_bar.progress(stages.get(current_stage, 50))
                    
                    with status_container.container():
                        st.markdown(f"<div class='status-box'><b>Active Agent:</b> <code>{current_stage.upper()}</code><br><small>Run ID: {st.session_state.run_id}</small></div>", unsafe_allow_html=True)
                    
                    if current_status in ["completed", "failed"]:
                        st.session_state.status = current_status
                        if current_status == "completed":
                            st.session_state.final_report = poll_data.get("final_report")
                        else:
                            st.error(f"Execution Failed: {poll_data.get('error_message')}")
                        st.rerun() # Force UI refresh to show tabs
                        break
                time.sleep(2)
            except Exception as e:
                st.error(f"Telemetry Error: {e}")
                break

    # Final Output State (Using Tabs for clean UX)
    if st.session_state.status == "completed" and st.session_state.final_report:
        st.success(f"✅ Research Synthesis Complete (ID: {st.session_state.run_id[:8]})")
        
        tab_report, tab_logs = st.tabs(["📄 Final Report", "⚙️ System Logs"])
        
        with tab_report:
            # Export Buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Download Markdown (.md)",
                    data=st.session_state.final_report,
                    file_name=f"ResearchGPT_Report_{st.session_state.run_id[:8]}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                try:
                    pdf_bytes = generate_pdf(st.session_state.final_report)
                    st.download_button(
                        label="📄 Download PDF (.pdf)",
                        data=pdf_bytes,
                        file_name=f"ResearchGPT_Report_{st.session_state.run_id[:8]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation unavailable: {str(e)}")
            
            st.divider()
            st.markdown(st.session_state.final_report)
            
        with tab_logs:
            st.code(f"RUN ID: {st.session_state.run_id}\nSTATUS: {st.session_state.status.upper()}\nMODE: AUTONOMOUS_MULTI_AGENT", language="yaml")
