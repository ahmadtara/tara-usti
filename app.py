import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title='Zuma — by.Tara', layout='wide', initial_sidebar_state='collapsed')
hide_streamlit_style = """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .css-18e3th9 {padding-top: 0rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown('<h2 style="text-align:center">Zuma — by.Tara (HD)</h2>', unsafe_allow_html=True)

# load the bundled game html
html_path = Path(__file__).parent / 'web' / 'game.html'
html_code = html_path.read_text(encoding='utf-8')
# set a responsive height - large so canvas can be HD
components.html(html_code, height=960, scrolling=False)
