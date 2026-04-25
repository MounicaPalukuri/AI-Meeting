"""
Output formatting utilities for the Meeting Intelligence System.
Formats analysis results for display in the Gradio UI.
"""

from models.schemas import MeetingAnalysis


def format_analysis_report(analysis: MeetingAnalysis) -> str:
    """
    Generate a complete formatted report from meeting analysis.
    
    Args:
        analysis: The complete meeting analysis results.
    
    Returns:
        Formatted markdown report string.
    """
    sections = []

    # Header
    sections.append("# 📋 Meeting Intelligence Report\n")
    sections.append(f"**Audio File:** {analysis.audio_file}")
    sections.append(f"**Duration:** {analysis.duration_formatted}")
    sections.append("---\n")

    # Summary
    sections.append("## 📝 Summary\n")
    sections.append(analysis.summary if analysis.summary else "_No summary generated._")
    sections.append("\n---\n")

    # Action Items
    sections.append("## ✅ Action Items\n")
    if analysis.action_items:
        for item in analysis.action_items:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                item.priority.lower(), "⚪"
            )
            sections.append(
                f"- {priority_emoji} **{item.assignee}**: {item.task} "
                f"_[{item.priority.upper()}]_"
            )
    else:
        sections.append("_No action items detected._")
    sections.append("\n---\n")

    # Deadlines
    sections.append("## ⏰ Deadlines\n")
    if analysis.deadlines:
        for dl in analysis.deadlines:
            assignee_str = f" (**{dl.assignee}**)" if dl.assignee else ""
            sections.append(f"- 📅 {dl.task}{assignee_str} — **Due: {dl.date}**")
    else:
        sections.append("_No deadlines detected._")

    return "\n".join(sections)


def format_transcript_display(analysis: MeetingAnalysis) -> str:
    """
    Format transcript for Gradio display with timestamps.
    
    Args:
        analysis: The meeting analysis containing transcript data.
    
    Returns:
        Formatted transcript string.
    """
    if not analysis.transcript and not analysis.segments:
        return "_No transcript available._"

    lines = ["## 🎙️ Meeting Transcript\n"]
    lines.append(f"**Duration:** {analysis.duration_formatted}\n")
    lines.append("---\n")

    if analysis.segments:
        for seg in analysis.segments:
            timestamp = f"`{seg.format_timestamp(seg.start)} → {seg.format_timestamp(seg.end)}`"
            speaker = f"**{seg.speaker}:** " if seg.speaker else ""
            lines.append(f"{timestamp}  {speaker}{seg.text}\n")
    else:
        # Plain transcript without segments
        lines.append(analysis.transcript)

    return "\n".join(lines)


def format_error_display(error_message: str) -> str:
    """Format an error message for UI display."""
    return (
        "## ❌ Error\n\n"
        f"{error_message}\n\n"
        "---\n"
        "_Please check the error details above and try again._"
    )
