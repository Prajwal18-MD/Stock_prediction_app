import streamlit as st

def display_news_ticker(news_headlines):
    """
    Display a scrolling news ticker.
    """
    st.markdown(
        f'<marquee>{ " | ".join(news_headlines) }</marquee>',
        unsafe_allow_html=True
    )

