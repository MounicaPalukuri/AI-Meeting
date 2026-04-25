"""
LLM Analyzer Module
Uses Ollama to analyze meeting transcripts and extract:
  - Concise meeting summary
  - Action items (who, what, priority)
  - Deadlines (task, date, assignee)

All prompts are production-quality with structured output instructions.
"""

import json
import logging
import re
from typing import List, Optional

from models.schemas import ActionItem, Deadline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# PRODUCTION-QUALITY PROMPTS
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert meeting analyst AI. Your role is to analyze meeting transcripts 
and extract structured, actionable intelligence. Be precise, concise, and professional.
Always base your analysis strictly on the transcript content — never hallucinate or invent information."""

SUMMARY_PROMPT = """Analyze the following meeting transcript and generate a concise, professional summary.

## Instructions:
1. Identify the main topics discussed in the meeting.
2. Highlight key decisions that were made.
3. Note any important points of agreement or disagreement.
4. Keep the summary between 3-8 sentences depending on meeting length.
5. Use clear, professional language.
6. Structure the summary with the most important information first.

## Meeting Transcript:
{transcript}

## Your Summary:
Provide a concise summary of the meeting below. Do NOT add any labels or prefixes — just write the summary directly."""

ACTION_ITEMS_PROMPT = """Analyze the following meeting transcript and extract ALL action items.

## Instructions:
1. Identify every task, to-do, or commitment mentioned in the meeting.
2. For each action item, determine:
   - **assignee**: The person responsible (use the name as mentioned; if unclear, use "Unassigned")
   - **task**: Clear, specific description of what needs to be done
   - **priority**: "high", "medium", or "low" based on urgency/importance conveyed in the meeting
3. Only extract items that are actual tasks or commitments — not general discussion points.
4. If no action items are found, return an empty list.

## Meeting Transcript:
{transcript}

## Output Format:
Respond with a valid JSON array ONLY. No explanations, no markdown formatting, no code blocks.
Each element must have: "assignee", "task", "priority"

Example format:
[
  {{"assignee": "John", "task": "Send the Q3 report to the team", "priority": "high"}},
  {{"assignee": "Sarah", "task": "Schedule follow-up meeting for next week", "priority": "medium"}}
]

If no action items are found, respond with: []"""

DEADLINES_PROMPT = """Analyze the following meeting transcript and extract ALL deadlines and time-bound commitments.

## Instructions:
1. Identify every deadline, due date, or time-bound commitment mentioned.
2. For each deadline, determine:
   - **task**: What needs to be completed by the deadline
   - **date**: The deadline date/time as mentioned (e.g., "Friday", "end of week", "March 15", "by next Monday")
   - **assignee**: Who is responsible (use "Unassigned" if unclear)
3. Include both explicit deadlines ("due by Friday") and implicit ones ("we need this before the launch next week").
4. If no deadlines are found, return an empty list.

## Meeting Transcript:
{transcript}

## Output Format:
Respond with a valid JSON array ONLY. No explanations, no markdown formatting, no code blocks.
Each element must have: "task", "date", "assignee"

Example format:
[
  {{"task": "Complete the design mockups", "date": "Friday, March 15", "assignee": "Alice"}},
  {{"task": "Submit budget proposal", "date": "end of this week", "assignee": "Bob"}}
]

If no deadlines are found, respond with: []"""


class LLMAnalyzer:
    """
    Analyzes meeting transcripts using an LLM via Ollama.
    Extracts summaries, action items, and deadlines.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        ollama_host: str = "http://localhost:11434",
        timeout: int = 600,
    ):
        """
        Initialize the LLM analyzer.
        
        Args:
            model:       Ollama model name (e.g., "llama3.2", "mistral", "phi3").
            ollama_host: Ollama server URL.
            timeout:     Request timeout in seconds.
        """
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout = timeout
        self._verify_ollama()

    def _verify_ollama(self):
        """Verify Ollama is running and the model is available."""
        import requests

        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if resp.status_code != 200:
                raise ConnectionError("Ollama server returned non-200 status.")

            available_models = [m["name"] for m in resp.json().get("models", [])]
            
            # Check if model is available (handle tag variants like "llama3.2:latest")
            model_found = any(
                self.model in m or m.startswith(self.model) 
                for m in available_models
            )

            if not model_found:
                model_list = "\n  ".join(available_models) if available_models else "(none)"
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models:\n  {model_list}\n"
                    f"Pull it with: ollama pull {self.model}"
                )
                # Don't raise — the model may auto-pull on first use
            else:
                logger.info(f"Ollama model '{self.model}' is available.")

        except requests.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama server.\n"
                "Make sure Ollama is running: ollama serve\n"
                f"Expected URL: {self.ollama_host}"
            )

    def _query_ollama(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        """
        Send a prompt to Ollama and return the response text.
        
        Args:
            prompt: The user prompt.
            system: The system prompt.
        
        Returns:
            Response text from the LLM.
        """
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.3,     # Low temperature for consistent, factual output
                "top_p": 0.9,
                "num_predict": 2048,    # Max output tokens
            },
        }

        try:
            logger.info(f"Querying Ollama ({self.model})...")
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error (HTTP {resp.status_code}): {resp.text}"
                )

            result = resp.json()
            response_text = result.get("response", "").strip()

            if not response_text:
                raise RuntimeError("Ollama returned an empty response.")

            logger.info(f"Ollama response received ({len(response_text)} chars)")
            return response_text

        except requests.ConnectionError:
            raise ConnectionError(
                "Lost connection to Ollama server during request.\n"
                "Make sure Ollama is running: ollama serve"
            )
        except requests.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s.\n"
                "The model may be too slow or the transcript too long.\n"
                "Try a smaller model (e.g., phi3) or increase timeout."
            )

    def generate_summary(self, transcript: str) -> str:
        """
        Generate a concise meeting summary.
        
        Args:
            transcript: Full meeting transcript text.
        
        Returns:
            Summary string.
        """
        if not transcript or transcript.strip() == "[No speech detected in the audio]":
            return "No meaningful speech was detected in the audio to summarize."

        try:
            prompt = SUMMARY_PROMPT.format(transcript=transcript)
            summary = self._query_ollama(prompt)
            return summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"⚠️ Summary generation failed: {e}"

    def extract_action_items(self, transcript: str) -> List[ActionItem]:
        """
        Extract action items from the transcript.
        
        Args:
            transcript: Full meeting transcript text.
        
        Returns:
            List of ActionItem objects.
        """
        if not transcript or transcript.strip() == "[No speech detected in the audio]":
            return []

        try:
            prompt = ACTION_ITEMS_PROMPT.format(transcript=transcript)
            response = self._query_ollama(prompt)
            return self._parse_action_items(response)
        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            return [ActionItem(
                assignee="System",
                task=f"⚠️ Action item extraction failed: {e}",
                priority="high",
            )]

    def extract_deadlines(self, transcript: str) -> List[Deadline]:
        """
        Extract deadlines from the transcript.
        
        Args:
            transcript: Full meeting transcript text.
        
        Returns:
            List of Deadline objects.
        """
        if not transcript or transcript.strip() == "[No speech detected in the audio]":
            return []

        try:
            prompt = DEADLINES_PROMPT.format(transcript=transcript)
            response = self._query_ollama(prompt)
            return self._parse_deadlines(response)
        except Exception as e:
            logger.error(f"Deadline extraction failed: {e}")
            return [Deadline(
                task=f"⚠️ Deadline extraction failed: {e}",
                date="N/A",
            )]

    def _parse_action_items(self, response: str) -> List[ActionItem]:
        """Parse LLM response into ActionItem objects with robust JSON extraction."""
        try:
            items_data = self._extract_json_array(response)
            items = []
            for item in items_data:
                items.append(ActionItem(
                    assignee=str(item.get("assignee", "Unassigned")),
                    task=str(item.get("task", "")),
                    priority=str(item.get("priority", "medium")).lower(),
                ))
            return items
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse action items JSON, falling back to text parsing: {e}")
            return self._fallback_parse_action_items(response)

    def _parse_deadlines(self, response: str) -> List[Deadline]:
        """Parse LLM response into Deadline objects with robust JSON extraction."""
        try:
            items_data = self._extract_json_array(response)
            deadlines = []
            for item in items_data:
                deadlines.append(Deadline(
                    task=str(item.get("task", "")),
                    date=str(item.get("date", "Not specified")),
                    assignee=item.get("assignee"),
                ))
            return deadlines
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse deadlines JSON, falling back to text parsing: {e}")
            return self._fallback_parse_deadlines(response)

    def _extract_json_array(self, text: str) -> list:
        """
        Robustly extract a JSON array from LLM output.
        Handles cases where LLM wraps JSON in markdown code blocks or adds extra text.
        """
        # Strip markdown code blocks if present
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON array from response: {text[:200]}...")

    def _fallback_parse_action_items(self, text: str) -> List[ActionItem]:
        """Fallback text-based parsing when JSON extraction fails."""
        items = []
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip().lstrip("•-*0123456789.) ")
            if not line or len(line) < 5:
                continue
            # Try to extract assignee from patterns like "John: do something" or "John - do something"
            match = re.match(r"(?:\*\*)?(.+?)(?:\*\*)?[\s]*[:–—-]\s*(.+)", line)
            if match:
                items.append(ActionItem(
                    assignee=match.group(1).strip(),
                    task=match.group(2).strip(),
                    priority="medium",
                ))
            else:
                items.append(ActionItem(
                    assignee="Unassigned",
                    task=line,
                    priority="medium",
                ))
        return items

    def _fallback_parse_deadlines(self, text: str) -> List[Deadline]:
        """Fallback text-based parsing when JSON extraction fails."""
        deadlines = []
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip().lstrip("•-*0123456789.) ")
            if not line or len(line) < 5:
                continue
            # Look for date patterns
            date_match = re.search(
                r"(?:by|due|before|until|deadline:?)\s+(.+?)(?:\s*[-–—]\s*|$)", 
                line, re.IGNORECASE
            )
            if date_match:
                date_str = date_match.group(1).strip()
                task_str = line[:date_match.start()].strip() or line
                deadlines.append(Deadline(task=task_str, date=date_str))
            elif any(word in line.lower() for word in ["monday", "tuesday", "wednesday", 
                      "thursday", "friday", "saturday", "sunday", "week", "month", "tomorrow"]):
                deadlines.append(Deadline(task=line, date="See description"))
        return deadlines
