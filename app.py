import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title='Zuma — by.Tara',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# CSS agar iframe responsif penuh
st.markdown("""
    <style>
    header, footer {visibility: hidden;}
    .css-18e3th9 {padding-top: 0rem;}
    iframe {
        width: 100% !important;
        height: 100vh !important;
        border: none;
    }
    body, html {
        margin: 0;
        padding: 0;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Judul
st.markdown(
    '<h2 style="text-align:center">Zuma — by.Tara (Frog Shooter)</h2>',
    unsafe_allow_html=True
)

# Load game HTML
html_path = Path(__file__).parent / 'web' / 'game.html'
html_code = html_path.read_text(encoding='utf-8')

# Render game HTML dengan scrolling agar tombol tetap bisa dipakai
components.html(html_code, height=0, scrolling=True)
