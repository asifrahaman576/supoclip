"""
AI-related functions for transcript analysis with enhanced precision and virality scoring.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
import asyncio
import logging
import re
import traceback
import json
import httpx

logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import AliasChoices, BaseModel, Field, field_validator

from .config import Config, get_config
from .runtime_settings import apply_settings_to_process_env

logger = logging.getLogger(__name__)

IDEAL_CLIP_MIN_SECONDS = 25
IDEAL_CLIP_MAX_SECONDS = 50
MIN_ACCEPTED_CLIP_SECONDS = 15
MAX_ACCEPTED_CLIP_SECONDS = 60
TRANSCRIPT_ANALYSIS_CACHE_VERSION = "longer-clips-v3-duration-repair"
TRANSCRIPT_SPAN_RE = re.compile(
    r"^\[(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<text>.*)$"
)


class ViralityAnalysis(BaseModel):
    """Detailed virality breakdown for a segment."""

    hook_score: int = Field(default=15, description="How strong is the opening hook (0-25)")
    engagement_score: int = Field(default=15, description="How engaging/entertaining is the content (0-25)")
    value_score: int = Field(default=15, description="Educational/informational value (0-25)")
    shareability_score: int = Field(default=15, description="Likelihood of being shared (0-25)")
    total_score: int = Field(default=60, description="Combined virality score (0-100)")
    hook_type: Optional[str] = Field(default="none", description="Type of hook")
    virality_reasoning: str = Field(
        default="The model did not provide a detailed virality breakdown.",
        description="Explanation of the virality score",
    )

    @field_validator("hook_score", "engagement_score", "value_score", "shareability_score", mode="before")
    @classmethod
    def _normalize_score(cls, value: Any) -> int:
        if value is None:
            return 0
        try:
            val = float(value)
            if val > 25:
                val = val / 4
            return int(round(val))
        except (ValueError, TypeError):
            return 0

    @field_validator("total_score", mode="before")
    @classmethod
    def _normalize_total(cls, value: Any) -> int:
        if value is None:
            return 0
        try:
            val = float(value)
            if val > 100:
                val = 100
            return int(round(val))
        except (ValueError, TypeError):
            return 0


def _default_virality_analysis() -> ViralityAnalysis:
    return ViralityAnalysis()


class TranscriptSegment(BaseModel):
    """Represents a relevant segment of transcript with precise timing and virality analysis."""

    start_time: str = Field(description="Start timestamp in MM:SS format")
    end_time: str = Field(description="End timestamp in MM:SS format")
    text: str = Field(
        validation_alias=AliasChoices("text", "segment"),
        description=(
            "Transcript text taken only from the selected timestamp range. "
            "Keep it verbatim or near-verbatim, and do not paraphrase or merge non-contiguous lines."
        )
    )
    relevance_score: float = Field(
        default=0.75,
        description="Relevance score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        default="Selected by the AI model as a clip candidate.",
        description=(
            "Brief factual explanation of why this exact segment works as a clip. "
            "Base it only on the provided transcript content."
        )
    )
    virality: Optional[ViralityAnalysis] = Field(
        default=None,
        description="Detailed virality score breakdown (optional)",
    )

    @field_validator("virality", mode="before")
    @classmethod
    def _normalize_virality(cls, value: Any) -> Optional[ViralityAnalysis]:
        if value is None:
            return None
        if isinstance(value, str):
            return ViralityAnalysis(virality_reasoning=f"Virality level: {value}")
        if isinstance(value, dict):
            return ViralityAnalysis(**value)
        return value

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _coerce_percent_relevance_score(cls, value: Any) -> Any:
        if value is None:
            return 0.0
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric_value > 1 and numeric_value <= 100:
            return numeric_value / 100
        if numeric_value > 100:
            return 1.0
        if numeric_value < 0:
            return 0.0
        return numeric_value

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _validate_timestamp(cls, value: Any) -> str:
        if value is None:
            return "00:00"
        return str(value)


class BRollOpportunity(BaseModel):
    """Identifies an opportunity to insert B-roll footage."""

    timestamp: str = Field(
        default="00:00",
        validation_alias=AliasChoices("timestamp", "segment_start_time", "start_time"),
        description="When to insert B-roll (MM:SS format)",
    )
    duration: float = Field(
        default=3.0,
        description="How long to show B-roll (2-5 seconds)",
        ge=2.0,
        le=5.0,
    )
    search_term: str = Field(
        default="related visual",
        validation_alias=AliasChoices("search_term", "broll", "visual", "query"),
        description="Keyword to search for B-roll footage",
    )
    context: str = Field(
        default="Suggested B-roll opportunity from the model.",
        validation_alias=AliasChoices("context", "description"),
        description="What's being discussed at this point",
    )

    @field_validator("search_term", "context", mode="before")
    @classmethod
    def _coerce_textish_value(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item is not None)
        return str(value)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value: Any) -> str:
        if value is None:
            return "00:00"
        return str(value)


class TranscriptAnalysis(BaseModel):
    """Analysis result for transcript segments with virality and B-roll opportunities."""

    most_relevant_segments: List[TranscriptSegment]
    summary: str = Field(
        default="No summary provided.",
        description="Brief summary of the video content"
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="List of main topics discussed"
    )
    broll_opportunities: Optional[List[BRollOpportunity]] = Field(
        default=None, description="Opportunities to insert B-roll footage"
    )

    @field_validator("key_topics", mode="before")
    @classmethod
    def _normalize_key_topics(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    if "topic" in item:
                        result.append(str(item["topic"]))
                    elif "name" in item:
                        result.append(str(item["name"]))
                    elif "title" in item:
                        result.append(str(item["title"]))
                    else:
                        result.append(str(item))
                elif isinstance(item, str):
                    result.append(item)
                else:
                    result.append(str(item))
            return result
        return []

    @field_validator("summary", mode="before")
    @classmethod
    def _normalize_summary(cls, value: Any) -> str:
        if value is None:
            return "No summary provided."
        if isinstance(value, dict):
            for key in ["summary", "text", "content", "description"]:
                if key in value:
                    return str(value[key])
            return str(value)
        return str(value)

    @field_validator("most_relevant_segments", mode="before")
    @classmethod
    def _validate_segments(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return []


# SIMPLIFIED SYSTEM PROMPT - Very short and direct
transcript_analysis_system_prompt = """You are a transcript analyzer that outputs ONLY valid JSON.

Your response must be a JSON object with this exact structure:
{
  "most_relevant_segments": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "text": "exact transcript text",
      "relevance_score": 0.75,
      "reasoning": "brief explanation",
      "virality": {
        "hook_score": 15,
        "engagement_score": 15,
        "value_score": 15,
        "shareability_score": 15,
        "total_score": 60,
        "hook_type": "statement",
        "virality_reasoning": "explanation"
      }
    }
  ],
  "summary": "brief summary",
  "key_topics": ["topic1", "topic2"]
}

Rules:
- Output ONLY JSON, no other text
- Start with { and end with }
- No markdown, no explanations, no prose
- Find 2-5 segments from the transcript
- Each segment must be 15-60 seconds long"""

_transcript_agent: Optional[Agent[None, str]] = None
_transcript_agent_signature: Optional[tuple[str | None, ...]] = None

SUPPORTED_LLM_PROVIDERS = {"google", "google-gla", "openai", "anthropic", "ollama"}


def _split_llm_name(model_name: str) -> tuple[str, str | None]:
    if ":" not in model_name:
        return model_name.strip().lower(), None

    provider, provider_model_name = model_name.split(":", 1)
    return provider.strip().lower(), provider_model_name.strip() or None


def _get_missing_llm_key_error(model_name: str, runtime_config: Config) -> Optional[str]:
    provider, provider_model_name = _split_llm_name(model_name)

    if provider not in SUPPORTED_LLM_PROVIDERS:
        return (
            f"Unsupported LLM provider '{provider}'. "
            "Use google-gla:*, openai:*, anthropic:*, or ollama:*."
        )

    if not provider_model_name:
        return (
            "Selected LLM is missing a model name. "
            "Use the format provider:model, for example ollama:gpt-oss:20b."
        )

    if provider in {"google", "google-gla"} and not runtime_config.google_api_key:
        return (
            "Selected LLM provider is Google, but GOOGLE_API_KEY is not set. "
            "Set GOOGLE_API_KEY or set LLM to openai:* / anthropic:* / ollama:* with the matching API key."
        )

    if provider == "openai" and not runtime_config.openai_api_key:
        return (
            "Selected LLM provider is OpenAI, but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY or choose another provider with a matching API key."
        )

    if provider == "anthropic" and not runtime_config.anthropic_api_key:
        return (
            "Selected LLM provider is Anthropic, but ANTHROPIC_API_KEY is not set. "
            "Set ANTHROPIC_API_KEY or choose another provider with a matching API key."
        )

    return None


def _build_transcript_model(runtime_config: Config) -> Model | str:
    provider, provider_model_name = _split_llm_name(runtime_config.llm)
    
    if provider == "ollama" or (provider == "openai" and "qwen" in (provider_model_name or "").lower()):
        model_name = provider_model_name or "qwen3:8b"
        logger.info("Routing request through native PydanticAI Ollama provider for %s", model_name)
        return OllamaModel(
            model_name,
            provider=OllamaProvider(
                base_url=runtime_config.resolve_ollama_base_url(),
                api_key=runtime_config.ollama_api_key,
            ),
        )

    return runtime_config.llm


def get_transcript_agent() -> Agent[None, str]:
    global _transcript_agent, _transcript_agent_signature
    runtime_config = get_config()
    signature = (
        runtime_config.llm,
        runtime_config.openai_api_key,
        runtime_config.google_api_key,
        runtime_config.anthropic_api_key,
        runtime_config.ollama_base_url,
        runtime_config.ollama_api_key,
    )
    if _transcript_agent is None or _transcript_agent_signature != signature:
        apply_settings_to_process_env(runtime_config.as_runtime_settings())
        config_error = _get_missing_llm_key_error(runtime_config.llm, runtime_config)
        if config_error:
            raise RuntimeError(config_error)

        _transcript_agent = Agent[None, str](
            model=_build_transcript_model(runtime_config),
            system_prompt=transcript_analysis_system_prompt,
        )
        _transcript_agent_signature = signature
    return _transcript_agent


def build_transcript_analysis_prompt(
    transcript: str, include_broll: bool = False, clip_signals: str | None = None
) -> str:
    """Build the prompt for transcript analysis."""
    return f"""Analyze this transcript and return JSON.

Transcript:
{transcript}

Return JSON with:
- most_relevant_segments: 2-5 segments (start_time, end_time, text, relevance_score, reasoning, virality)
- summary: brief summary
- key_topics: list of topics

JSON only, no explanations:"""


def _parse_transcript_timestamp_seconds(timestamp: str) -> int:
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp format: {timestamp}")


def _format_transcript_timestamp(seconds: int) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _parse_transcript_spans(transcript: str) -> list[dict[str, Any]]:
    spans = []
    for line in transcript.splitlines():
        match = TRANSCRIPT_SPAN_RE.match(line.strip())
        if not match:
            continue
        try:
            start_seconds = _parse_transcript_timestamp_seconds(match.group("start"))
            end_seconds = _parse_transcript_timestamp_seconds(match.group("end"))
        except ValueError:
            continue
        if end_seconds <= start_seconds:
            continue
        spans.append(
            {
                "start": start_seconds,
                "end": end_seconds,
                "text": match.group("text").strip(),
            }
        )
    return spans


def _extract_transcript_text(
    transcript_spans: list[dict[str, Any]], start_seconds: int, end_seconds: int
) -> str:
    selected_text = [
        span["text"]
        for span in transcript_spans
        if span["text"]
        and span["end"] > start_seconds
        and span["start"] < end_seconds
    ]
    return " ".join(selected_text).strip()


def _choose_repaired_bounds(
    transcript_spans: list[dict[str, Any]], start_seconds: int, end_seconds: int
) -> tuple[int, int] | None:
    if not transcript_spans:
        return None

    starts = sorted({span["start"] for span in transcript_spans})
    ends = sorted({span["end"] for span in transcript_spans})
    current_duration = end_seconds - start_seconds

    if current_duration > MAX_ACCEPTED_CLIP_SECONDS:
        target_end = start_seconds + IDEAL_CLIP_MAX_SECONDS
        candidate_ends = [
            candidate
            for candidate in ends
            if start_seconds + MIN_ACCEPTED_CLIP_SECONDS
            <= candidate
            <= min(target_end, end_seconds)
        ]
        if candidate_ends:
            return start_seconds, max(candidate_ends)
        if start_seconds + MIN_ACCEPTED_CLIP_SECONDS <= target_end:
            return start_seconds, target_end
        return None

    if current_duration < MIN_ACCEPTED_CLIP_SECONDS:
        candidate_ranges: list[tuple[int, int, int]] = []
        for candidate_start in starts:
            if candidate_start > start_seconds:
                continue
            for candidate_end in ends:
                if candidate_end < end_seconds:
                    continue
                duration = candidate_end - candidate_start
                if MIN_ACCEPTED_CLIP_SECONDS <= duration <= MAX_ACCEPTED_CLIP_SECONDS:
                    extra_context = (start_seconds - candidate_start) + (
                        candidate_end - end_seconds
                    )
                    ideal_penalty = 0
                    if duration < IDEAL_CLIP_MIN_SECONDS:
                        ideal_penalty = IDEAL_CLIP_MIN_SECONDS - duration
                    elif duration > IDEAL_CLIP_MAX_SECONDS:
                        ideal_penalty = duration - IDEAL_CLIP_MAX_SECONDS
                    candidate_ranges.append(
                        (ideal_penalty * 1000 + extra_context, candidate_start, candidate_end)
                    )
        if candidate_ranges:
            _, repaired_start, repaired_end = min(candidate_ranges)
            return repaired_start, repaired_end

    return None


def _repair_segment_bounds(
    segment: TranscriptSegment,
    transcript_spans: list[dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
) -> tuple[int, int] | None:
    """Adjust near-miss model ranges to usable transcript-aligned bounds."""
    repaired_bounds = _choose_repaired_bounds(
        transcript_spans,
        start_seconds,
        end_seconds,
    )
    if not repaired_bounds:
        return None

    repaired_start, repaired_end = repaired_bounds
    segment.start_time = _format_transcript_timestamp(repaired_start)
    segment.end_time = _format_transcript_timestamp(repaired_end)
    segment.text = _extract_transcript_text(
        transcript_spans,
        repaired_start,
        repaired_end,
    )
    return repaired_bounds


def _clean_markdown_json(raw_text: str) -> str:
    """Extract JSON from response, handling markdown code blocks."""
    if not raw_text:
        return ""
    
    text = raw_text.strip()
    
    # Handle markdown code blocks
    if "```" in text:
        lines = text.split("\n")
        in_code = False
        code_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        if code_lines:
            text = "\n".join(code_lines).strip()
    
    # If no code blocks, try to find JSON directly
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start != -1 and json_end != -1 and json_start < json_end:
        return text[json_start:json_end + 1]
    
    # Try to find JSON array
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_start < array_end:
        return text[array_start:array_end + 1]
    
    return ""


async def get_most_relevant_parts_by_transcript(
    transcript_text: str,
    include_broll: bool = False,
    clip_signals: str | None = None,
) -> TranscriptAnalysis:
    """
    Get most relevant parts from transcript using PydanticAI with Ollama.
    """
    # Validate input
    if not transcript_text or not transcript_text.strip():
        raise ValueError("Transcript text is empty or contains only whitespace")
    
    # Clean transcript
    transcript_text = transcript_text.strip()
    
    # Build prompt
    prompt = build_transcript_analysis_prompt(
        transcript_text, 
        include_broll=include_broll,
        clip_signals=clip_signals
    )
    
    if not prompt or not prompt.strip():
        raise ValueError("Generated prompt is empty")
    
    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.info(f"First 200 chars: {prompt[:200]}...")
    
    # Get runtime config
    runtime_config = get_config()
    
    # Build model using PydanticAI
    model = _build_transcript_model(runtime_config)
    
    # Create a fresh agent
    agent = Agent[None, str](
        model=model,
        system_prompt=transcript_analysis_system_prompt,
    )
    
    # Run the agent
    result = await agent.run(prompt)
    
    # For pydantic-ai 1.89.1, the response is in result.output
    if not result or not hasattr(result, 'output'):
        logger.error(f"Result object: {type(result)}")
        logger.error(f"Result attributes: {dir(result) if result else 'None'}")
        raise ValueError("Invalid result object from LLM")
    
    response_text = result.output
    if not response_text or not response_text.strip():
        raise ValueError("Empty response from LLM")
    
    # Clean response
    response_text = response_text.strip()
    
    logger.info(f"Response length: {len(response_text)} characters")
    logger.info(f"First 200 chars: {response_text[:200]}...")
    
    # Clean markdown if present
    cleaned_text = _clean_markdown_json(response_text)
    if not cleaned_text:
        logger.error(f"Could not extract JSON. Response: {response_text[:500]}")
        raise ValueError("No JSON content found in response")
    
    # Parse into dictionary
    data = json.loads(cleaned_text)
    
    # Validate required fields
    if "most_relevant_segments" not in data:
        raise ValueError("Response missing 'most_relevant_segments' field")
    
    if not data["most_relevant_segments"]:
        raise ValueError("Response has empty segments list")
    
    # Parse into TranscriptAnalysis with relaxed validation
    analysis = TranscriptAnalysis(**data)
    
    # Set default virality for segments that don't have it
    for segment in analysis.most_relevant_segments:
        if segment.virality is None:
            segment.virality = _default_virality_analysis()
    
    # Post-process and validate segments
    transcript_spans = _parse_transcript_spans(transcript_text)
    if transcript_spans:
        repaired_count = 0
        for segment in analysis.most_relevant_segments:
            try:
                start_seconds = _parse_transcript_timestamp_seconds(segment.start_time)
                end_seconds = _parse_transcript_timestamp_seconds(segment.end_time)
                
                duration = end_seconds - start_seconds
                if duration < MIN_ACCEPTED_CLIP_SECONDS or duration > MAX_ACCEPTED_CLIP_SECONDS:
                    repaired = _repair_segment_bounds(
                        segment, transcript_spans, start_seconds, end_seconds
                    )
                    if repaired:
                        repaired_count += 1
            except Exception as e:
                logger.warning(f"Could not repair segment bounds: {e}")
        
        if repaired_count > 0:
            logger.info(f"Repaired {repaired_count} segment bounds")
    
    logger.info(f"Successfully parsed {len(analysis.most_relevant_segments)} segments")
    return analysis