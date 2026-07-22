import streamlit as st
from agents import build_crew

st.set_page_config(page_title="AI Content Creator", page_icon="✍️", layout="centered")

st.title("✍️ Personalized AI Content Creator")
st.caption("Multi-agent pipeline: Strategist → Writer → Editor")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["groq", "openai", "anthropic"])
    api_key = st.text_input(f"{provider.title()} API Key", type="password")
    st.markdown("---")
    st.markdown("**Free option:** get a Groq key at [console.groq.com/keys](https://console.groq.com/keys)")

topic = st.text_input("Topic", placeholder="e.g. Why most ML resumes fail interviews")
platform = st.selectbox("Platform", ["LinkedIn Post", "X/Twitter Thread", "Blog Article", "YouTube Script"])
tone = st.selectbox("Tone", ["Professional", "Casual", "Witty", "Persuasive", "Educational"])
audience = st.text_input("Target Audience", placeholder="e.g. final-year CS students")

generate = st.button("Generate Content", type="primary", disabled=not (topic and api_key))

if generate:
    with st.spinner("Agents are working... (planning → writing → editing)"):
        try:
            crew = build_crew(topic, platform, tone, audience, api_key, provider)
            result = crew.kickoff()
            st.success("Done!")
            st.markdown("### Final Output")
            st.markdown(str(result))
            st.download_button("Download as .txt", str(result), file_name="content.txt")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if not api_key:
    st.info("Enter your API key in the sidebar to get started.")