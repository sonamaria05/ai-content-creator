"""
Multi-agent content pipeline.
Three agents work in sequence, each handing off to the next:
  Strategist  -> plans the content (hook, key points, structure)
  Writer      -> turns the plan into a full draft
  Editor      -> polishes, formats, adds hashtags/CTA
"""

from crewai import Agent, Task, Crew, Process, LLM


MODEL_MAP = {
    "groq": "openai/llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "anthropic/claude-3-5-sonnet-20241022",
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Concrete, enforceable format rules per platform. Vague instructions like
# "match platform conventions" get ignored by the model — hard constraints don't.
PLATFORM_SPECS = {
    "LinkedIn Post": (
        "150-300 words. NO markdown syntax at all (no #headers, no ##, no ===, no * or - bullet markers, "
        "no numbered lists with periods) because LinkedIn does not render markdown - it will show up as "
        "literal symbols in the body text. "
        "Use short paragraphs (1-3 sentences), with a blank line between each for scannability. "
        "Start with a one-line hook that stops the scroll. End with a question or CTA. "
        "Then, on the very last line only, add EXACTLY 3-5 hashtags in true hashtag format: "
        "the # symbol directly attached to a single CamelCase word with no spaces, like "
        "'#ComputerScience #JobSearch #TechCareers'. Do not write out topic phrases with spaces instead of hashtags."
    ),
    "X/Twitter Thread": (
        "A numbered thread of 5-8 tweets, each under 280 characters. "
        "Format as 'Tweet 1/, Tweet 2/,' etc. Tweet 1 is the hook alone. "
        "No markdown headers. Short punchy sentences. Last tweet is a CTA "
        "(follow, reply, or link) plus 2-3 hashtags."
    ),
    "Blog Article": (
        "600-900 words. Markdown formatting IS appropriate here: use a title, "
        "H2/H3 subheadings, short paragraphs, and bullet lists where useful. "
        "Include an intro hook, body sections, and a conclusion with a CTA."
    ),
    "YouTube Script": (
        "A spoken-word script with timestamps/section labels like [HOOK - 0:00], "
        "[INTRO], [MAIN POINTS], [OUTRO/CTA]. Written to be read aloud naturally, "
        "not as prose to be read on a page. No markdown headers, just section labels in brackets."
    ),
}


def build_crew(topic: str, platform: str, tone: str, audience: str, api_key: str, provider: str = "groq") -> Crew:
    if provider == "groq":
        # Groq exposes an OpenAI-compatible endpoint, so we use CrewAI's native
        # OpenAI client pointed at Groq's base_url. This avoids needing litellm,
        # which needs a Rust toolchain to build on some Windows setups.
        llm = LLM(model=MODEL_MAP[provider], base_url=GROQ_BASE_URL, api_key=api_key, temperature=0.7)
    else:
        llm = LLM(model=MODEL_MAP[provider], api_key=api_key, temperature=0.7)

    format_spec = PLATFORM_SPECS.get(platform, "Follow standard conventions for this platform.")

    planner = Agent(
        role="Content Strategist",
        goal=f"Create a clear, engaging content plan for a {platform} post about '{topic}', targeted at {audience}.",
        backstory=(
            "You are a veteran content strategist who has planned viral content for top creators. "
            "You know exactly what structure keeps readers hooked on each platform."
        ),
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Content Writer",
        goal=f"Write a complete, ready-to-publish {platform} post in a {tone} tone, following the strategist's plan exactly.",
        backstory=(
            "You are a skilled ghostwriter known for turning outlines into polished, "
            "platform-native content that reads naturally and drives engagement."
        ),
        llm=llm,
        verbose=True,
    )

    editor = Agent(
        role="Editor",
        goal=(
            "Proofread and tighten the draft, fix any grammar issues, strengthen the hook and CTA, "
            "and enforce the exact platform format rules given below - no exceptions."
        ),
        backstory="You are a meticulous editor with an eye for clarity, flow, and platform-specific formatting conventions.",
        llm=llm,
        verbose=True,
    )

    plan_task = Task(
        description=(
            f"Create a content outline for a {platform} post about: '{topic}'. "
            f"Audience: {audience}. Include: a hook idea, 3-5 key points, and a suggested structure. "
            f"FORMAT RULES for this platform: {format_spec}"
        ),
        expected_output="A clear bullet-point outline with a hook, key points, and structure.",
        agent=planner,
    )

    write_task = Task(
        description=(
            f"Using the outline above, write the full {platform} post about '{topic}' in a {tone} tone. "
            f"FORMAT RULES you MUST follow exactly: {format_spec}"
        ),
        expected_output="A complete first draft of the post, following the format rules exactly.",
        agent=writer,
        context=[plan_task],
    )

    edit_task = Task(
        description=(
            "Polish the draft: fix grammar, strengthen the hook, add a call-to-action. "
            f"Then verify and enforce these FORMAT RULES exactly, rewriting anything that violates them: {format_spec}"
        ),
        expected_output="The final, publish-ready version of the content, strictly matching the format rules. Return ONLY the final content, no commentary.",
        agent=editor,
        context=[write_task],
    )

    return Crew(
        agents=[planner, writer, editor],
        tasks=[plan_task, write_task, edit_task],
        process=Process.sequential,
        verbose=True,
    )