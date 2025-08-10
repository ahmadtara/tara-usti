import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title='Zuma — by.Tara',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# CSS untuk sembunyikan header/footer Streamlit + responsif
hide_streamlit_style = """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .css-18e3th9 {padding-top: 0rem;}
    iframe {
        width: 100% !important;
        height: 100vh !important;
        border: none;
    }
    @media (max-width: 768px) {
        iframe {
            height: 100vh !important;
        }
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Judul di tengah
st.markdown(
    '<h2 style="text-align:center">Zuma — by.Tara (Frog Shooter)</h2>',
    unsafe_allow_html=True
)

# Load HTML game
html_path = Path(__file__).parent / 'web' / 'game.html'
html_code = html_path.read_text(encoding='utf-8')

# Render game HTML responsif
components.html(html_code, height=800, scrolling=False)
