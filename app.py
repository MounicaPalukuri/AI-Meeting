"""
AI Meeting Intelligence System — Gradio UI
==========================================
Upload a meeting audio file and get:
  • Full transcript with timestamps
  • Concise meeting summary
  • Action items (who, what, priority)
  • Deadlines detected from discussion

Run:  python app.py
"""

import os
import sys
import logging
import gradio as gr

# Add project root to path for clean imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.meeting_pipeline import MeetingPipeline
from utils.formatters import format_analysis_report, format_transcript_display, format_error_display
from utils.validators import check_dependencies

# ─── Logging Setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MeetingIntelligence")

# ─── Configuration ───────────────────────────────────────────────────
# Change these defaults as needed
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", "faster-whisper")
WHISPER_CPP_PATH = os.environ.get("WHISPER_CPP_PATH", None)
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEVICE = os.environ.get("DEVICE", "auto")
LANGUAGE = os.environ.get("LANGUAGE", None)  # None = auto-detect

# ─── Pipeline Initialization ─────────────────────────────────────────
pipeline = None


def initialize_pipeline():
    """Initialize the meeting pipeline (lazy loading)."""
    global pipeline
    if pipeline is None:
        try:
            pipeline = MeetingPipeline(
                whisper_model=WHISPER_MODEL,
                whisper_backend=WHISPER_BACKEND,
                whisper_cpp_path=WHISPER_CPP_PATH,
                llm_model=LLM_MODEL,
                ollama_host=OLLAMA_HOST,
                device=DEVICE,
                language=LANGUAGE,
            )
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise
    return pipeline


# ─── Main Processing Function ────────────────────────────────────────
def process_meeting(audio_file, progress=gr.Progress()):
    """
    Process an uploaded meeting audio file.
    
    Returns:
        (transcript_md, summary_md, action_items_md, deadlines_md, status_md)
    """
    # Handle no file uploaded
    if audio_file is None:
        empty_msg = "_Upload an audio file to get started._"
        return empty_msg, empty_msg, empty_msg, empty_msg, "⚠️ No file uploaded."

    try:
        # Initialize pipeline on first use
        progress(0.05, desc="🔧 Initializing pipeline...")
        pipe = initialize_pipeline()

        # Progress wrapper for the pipeline
        def progress_callback(pct, message):
            progress(pct, desc=message)

        # Run the pipeline
        analysis = pipe.process(audio_file, progress_callback=progress_callback)

        # Check for errors
        if analysis.error:
            error_display = format_error_display(analysis.error)
            return error_display, error_display, "_—_", "_—_", f"❌ {analysis.error}"

        # Format outputs
        progress(0.95, desc="📄 Formatting results...")

        # Transcript
        transcript_md = format_transcript_display(analysis)

        # Summary
        summary_md = f"## 📝 Meeting Summary\n\n{analysis.summary}"

        # Action Items
        if analysis.action_items:
            action_lines = ["## ✅ Action Items\n"]
            for item in analysis.action_items:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    item.priority.lower(), "⚪"
                )
                action_lines.append(
                    f"| {priority_emoji} | **{item.assignee}** | {item.task} | `{item.priority.upper()}` |"
                )
            action_items_md = (
                "## ✅ Action Items\n\n"
                "| Priority | Assignee | Task | Level |\n"
                "|:--------:|----------|------|:-----:|\n"
                + "\n".join(
                    f"| {({'high': '🔴', 'medium': '🟡', 'low': '🟢'}).get(item.priority.lower(), '⚪')} "
                    f"| **{item.assignee}** | {item.task} | `{item.priority.upper()}` |"
                    for item in analysis.action_items
                )
            )
        else:
            action_items_md = "## ✅ Action Items\n\n_No action items detected in this meeting._"

        # Deadlines
        if analysis.deadlines:
            deadlines_md = (
                "## ⏰ Deadlines\n\n"
                "| Task | Due Date | Assignee |\n"
                "|------|----------|----------|\n"
                + "\n".join(
                    f"| {dl.task} | **{dl.date}** | {dl.assignee or '—'} |"
                    for dl in analysis.deadlines
                )
            )
        else:
            deadlines_md = "## ⏰ Deadlines\n\n_No deadlines detected in this meeting._"

        # Status
        status_md = (
            f"✅ **Processing complete!**  \n"
            f"📁 File: `{analysis.audio_file}`  \n"
            f"⏱️ Duration: {analysis.duration_formatted}  \n"
            f"📊 {len(analysis.segments)} segments | "
            f"{len(analysis.action_items)} action items | "
            f"{len(analysis.deadlines)} deadlines"
        )

        progress(1.0, desc="✅ Done!")
        return transcript_md, summary_md, action_items_md, deadlines_md, status_md

    except Exception as e:
        logger.exception("Processing failed")
        error_msg = format_error_display(str(e))
        return error_msg, error_msg, "_—_", "_—_", f"❌ Error: {e}"


def check_system_status():
    """Check system dependencies and return a status report."""
    deps = check_dependencies()

    lines = ["## 🔧 System Status\n"]

    # FFmpeg
    ff = deps.get("ffmpeg", {})
    if ff.get("installed"):
        lines.append(f"✅ **FFmpeg**: Installed  \n`{ff.get('version', 'unknown')}`\n")
    else:
        lines.append(f"❌ **FFmpeg**: Not found  \n{ff.get('message', '')}\n")

    # Ollama
    ol = deps.get("ollama", {})
    if ol.get("installed"):
        lines.append(f"✅ **Ollama**: Installed  \nModels:\n```\n{ol.get('models', 'unknown')}\n```\n")
    else:
        lines.append(f"❌ **Ollama**: Not found  \n{ol.get('message', '')}\n")

    # Config
    lines.append("---\n## ⚙️ Configuration\n")
    lines.append(f"- **Whisper Model**: `{WHISPER_MODEL}`")
    lines.append(f"- **Whisper Backend**: `{WHISPER_BACKEND}`")
    lines.append(f"- **LLM Model**: `{LLM_MODEL}`")
    lines.append(f"- **Ollama Host**: `{OLLAMA_HOST}`")
    lines.append(f"- **Device**: `{DEVICE}`")
    lines.append(f"- **Language**: `{LANGUAGE or 'auto-detect'}`")

    return "\n".join(lines)


# ─── Gradio UI ────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""

    # Custom CSS for premium look
    custom_css = """
    /* ── Global ── */
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    
    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        margin-bottom: 20px;
        color: white;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .app-header p {
        color: #a0aec0;
        margin-top: 8px;
        font-size: 0.95rem;
    }
    
    /* ── Tabs ── */
    .tab-nav button {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* ── Status bar ── */
    .status-bar {
        padding: 12px 16px;
        border-radius: 10px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2d3748;
        color: #e2e8f0;
    }
    
    /* ── Result panels ── */
    .result-panel {
        min-height: 250px;
    }
    """

    with gr.Blocks(
        title="AI Meeting Intelligence",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=custom_css,
    ) as app:

        # ── Header ──
        gr.HTML("""
        <div class="app-header">
            <h1>🧠 AI Meeting Intelligence</h1>
            <p>Upload a meeting recording • Get transcript, summary, action items & deadlines</p>
        </div>
        """)

        with gr.Row():
            # ── Left Column: Upload & Controls ──
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 🎤 Upload Meeting Audio")

                audio_input = gr.Audio(
                    label="Meeting Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                    elem_id="audio-input",
                )

                process_btn = gr.Button(
                    "🚀 Analyze Meeting",
                    variant="primary",
                    size="lg",
                    elem_id="process-btn",
                )

                status_output = gr.Markdown(
                    value="_Upload an audio file and click **Analyze Meeting** to begin._",
                    label="Status",
                    elem_classes=["status-bar"],
                )

                with gr.Accordion("⚙️ System Status", open=False):
                    status_check_btn = gr.Button("🔍 Check Dependencies", size="sm")
                    system_status = gr.Markdown("_Click to check system status._")

            # ── Right Column: Results ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📝 Transcript", id="transcript-tab"):
                        transcript_output = gr.Markdown(
                            value="_Transcript will appear here after processing._",
                            label="Transcript",
                            elem_classes=["result-panel"],
                        )

                    with gr.Tab("📋 Summary", id="summary-tab"):
                        summary_output = gr.Markdown(
                            value="_Summary will appear here after processing._",
                            label="Summary",
                            elem_classes=["result-panel"],
                        )

                    with gr.Tab("✅ Action Items", id="actions-tab"):
                        actions_output = gr.Markdown(
                            value="_Action items will appear here after processing._",
                            label="Action Items",
                            elem_classes=["result-panel"],
                        )

                    with gr.Tab("⏰ Deadlines", id="deadlines-tab"):
                        deadlines_output = gr.Markdown(
                            value="_Deadlines will appear here after processing._",
                            label="Deadlines",
                            elem_classes=["result-panel"],
                        )

        # ── Footer ──
        gr.HTML("""
        <div style="text-align:center; padding:16px; color:#718096; font-size:0.85rem; margin-top:20px;">
            <p>🧠 AI Meeting Intelligence System • Powered by Whisper + Ollama • 100% Local & Private</p>
        </div>
        """)

        # ── Event Handlers ──
        process_btn.click(
            fn=process_meeting,
            inputs=[audio_input],
            outputs=[transcript_output, summary_output, actions_output, deadlines_output, status_output],
            show_progress="full",
        )

        status_check_btn.click(
            fn=check_system_status,
            inputs=[],
            outputs=[system_status],
        )

    return app


# ─── Main Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  AI Meeting Intelligence System")
    logger.info("=" * 60)

    # Check dependencies on startup
    deps = check_dependencies()

    if not deps["ffmpeg"]["installed"]:
        logger.warning("⚠️  FFmpeg not found — audio preprocessing will fail.")
        logger.warning("   Install: https://ffmpeg.org/download.html")

    if not deps["ollama"]["installed"]:
        logger.warning("⚠️  Ollama not found — LLM analysis will fail.")
        logger.warning("   Install: https://ollama.com/download")
    
    logger.info(f"Whisper model: {WHISPER_MODEL} ({WHISPER_BACKEND})")
    logger.info(f"LLM model: {LLM_MODEL} via {OLLAMA_HOST}")
    logger.info("-" * 60)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
