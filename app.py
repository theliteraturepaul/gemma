import streamlit as st
from normalizer import NormalizerError, normalize


EXAMPLES = [
    "kal tui ashbi na? amar bday ache, cake niye aye",
    "yaar kal office nhi jaana, chal ghumne chalte hain",
    "bro ei weekend e plan ki? Ghurte jaabi naaki?",
]


st.set_page_config(
    page_title="Benglish/Hinglish Chat Normalizer",
    page_icon="📝",
)

if "message" not in st.session_state:
    st.session_state.message = ""


st.title("Benglish/Hinglish Chat Normalizer")
st.caption(
    "Make Roman-script Bengali/Hindi/English Kolkata chat easier to read, translate, and index."
)

example_columns = st.columns(3)
for index, example in enumerate(EXAMPLES):
    with example_columns[index]:
        if st.button(f"Example {index + 1}", use_container_width=True):
            st.session_state.message = example

message = st.text_area(
    "Chat message",
    key="message",
    height=140,
    placeholder="Paste one Benglish/Hinglish chat message...",
)

if st.button("Normalize", type="primary"):
    if not message.strip():
        st.warning("Please enter a chat message first.")
    else:
        try:
            # normalize() is the cached, sync-safe entry point defined in
            # normalizer.py — it owns its own event loop (asyncio.run) and
            # caches repeated inputs (e.g. the example buttons above).
            # Do NOT call normalize_async()/asyncio.run() directly here;
            # that duplicates loop management that already lives in
            # normalizer.py and bypasses the cache entirely.
            with st.spinner("Calling Gemma 4... this can take up to 30 seconds"):
                result = normalize(message)
        except NormalizerError as exc:
            st.error(str(exc))
        except Exception as e:
            # Per design.md/Phase 3: never surface a raw traceback in the UI.
            # Full details still go to the server logs for debugging.
            print(f"Unexpected error in normalize(): {e!r}")
            st.error("Something went wrong while normalizing this message. Please try again.")
        else:
            native_col, english_col = st.columns(2)
            with native_col:
                st.subheader("Native script")
                st.write(result["native_script"])
            with english_col:
                st.subheader("English translation")
                st.write(result["english_translation"])

            ratio = result["language_ratio"]
            st.subheader("Language mix")
            st.write(
                f"Bengali: {ratio['bengali_pct']}% | "
                f"Hindi: {ratio['hindi_pct']}% | "
                f"English: {ratio['english_pct']}%"
            )
            st.progress(min(max(int(ratio["bengali_pct"]), 0), 100), text="Bengali")
            st.progress(min(max(int(ratio["hindi_pct"]), 0), 100), text="Hindi")
            st.progress(min(max(int(ratio["english_pct"]), 0), 100), text="English")
