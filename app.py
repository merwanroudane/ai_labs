"""
Hands-On Machine Learning — Interactive Teaching Platform
=========================================================
Entry point.  Run with:

    streamlit run app.py

The whole course is declared here as a `st.navigation` tree so the sidebar
shows Part I / Part II / Labs & Reference exactly like a real syllabus.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ML Platform · Dr Merwan Roudane",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core import theme  # noqa: E402

theme.inject()

P = "views"


def page(script: str, title: str, icon: str, url: str):
    return st.Page(os.path.join(P, script), title=title, icon=icon, url_path=url)


NAV = {
    "Start here": [
        st.Page(os.path.join(P, "p00_home.py"), title="Course home",
                icon="🏠", url_path="home", default=True),
        page("p25_foundations.py", "Foundations · what learning is",
             "🧭", "foundations"),
        page("p90_syllabus.py", "Full syllabus & search", "🗺️", "syllabus"),
        page("p91_setup.py", "Environment & setup", "⚙️", "setup"),
    ],
    "Part I · The Fundamentals of Machine Learning": [
        page("p01_landscape.py", "1 · The ML Landscape", "🌍", "ch01"),
        page("p02_endtoend.py", "2 · End-to-End ML Project", "🏗️", "ch02"),
        page("p03_classification.py", "3 · Classification", "🎯", "ch03"),
        page("p04_training.py", "4 · Training Models", "📉", "ch04"),
        page("p05_svm.py", "5 · Support Vector Machines", "🛡️", "ch05"),
        page("p06_trees.py", "6 · Decision Trees", "🌳", "ch06"),
        page("p07_ensembles.py", "7 · Ensembles & Random Forests", "🧩", "ch07"),
        page("p08_dimred.py", "8 · Dimensionality Reduction", "🗜️", "ch08"),
        page("p09_unsupervised.py", "9 · Unsupervised Learning", "🔮", "ch09"),
    ],
    "Part II · Neural Networks and Deep Learning": [
        page("p10_ann.py", "10 · Intro to ANNs with Keras", "🧠", "ch10"),
        page("p11_deep.py", "11 · Training Deep Nets", "🏔️", "ch11"),
        page("p12_custom_tf.py", "12 · Custom Models & Training", "🔧", "ch12"),
        page("p13_data_tf.py", "13 · Loading & Preprocessing Data", "🚰", "ch13"),
        page("p14_cnn.py", "14 · Deep Computer Vision (CNNs)", "👁️", "ch14"),
        page("p15_rnn.py", "15 · Sequences with RNNs & CNNs", "🔁", "ch15"),
        page("p16_nlp.py", "16 · NLP, Attention & Transformers", "💬", "ch16"),
        page("p17_generative.py", "17 · Autoencoders, GANs, Diffusion", "🎨", "ch17"),
        page("p18_rl.py", "18 · Reinforcement Learning", "🕹️", "ch18"),
        page("p19_scale.py", "19 · Training & Deploying at Scale", "🚀", "ch19"),
    ],
    "Labs & Reference": [
        page("p20_ai_lab.py", "AI Lab · live model workbench", "🧪", "ai-lab"),
        page("p21_math.py", "Math appendix", "📐", "math"),
        page("p22_checklist.py", "A · ML project checklist", "✅", "checklist"),
        page("p23_autodiff.py", "B · Autodiff", "➗", "autodiff"),
        page("p24_glossary.py", "Glossary & symbol table", "📖", "glossary"),
    ],
}

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:11px;margin-bottom:4px">
          <div style="width:38px;height:38px;border-radius:11px;
                      background:linear-gradient(135deg,#6C4DF6,#00C2A8);
                      display:flex;align-items:center;justify-content:center;
                      font-size:20px">🧠</div>
          <div>
            <div style="font-weight:800;font-size:1.0rem;line-height:1.15;color:#0E1428">
              ML Platform</div>
            <div style="font-size:.72rem;color:#7A8199;letter-spacing:.05em">
              19 chapters · animated · executable</div>
          </div>
        </div>
        <div style="margin:2px 0 0 49px;font-size:.72rem;color:#7A8199">
          developed by
          <a href="https://github.com/merwanroudane" target="_blank"
             style="color:#6C4DF6;font-weight:650;text-decoration:none">
            Dr Merwan Roudane</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

nav = st.navigation(NAV, position="sidebar")
nav.run()

with st.sidebar:
    st.markdown("<hr style='margin:18px 0 10px 0'/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:.74rem;line-height:1.5;color:#7A8199">
          <div style="color:#0E1428;font-weight:700;margin-bottom:2px">
            Dr Merwan Roudane</div>
          <div>Developer &amp; author of this platform</div>
          <div style="margin-top:5px">
            <a href="https://github.com/merwanroudane" target="_blank"
               style="color:#6C4DF6;text-decoration:none">github.com/merwanroudane</a>
          </div>
          <div style="margin-top:2px">
            <a href="https://ailabo.streamlit.app/" target="_blank"
               style="color:#6C4DF6;text-decoration:none">ailabo.streamlit.app</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Original lecture material written for this platform. "
        "Chapter ordering follows *Hands-On Machine Learning with "
        "Scikit-Learn, Keras & TensorFlow* (A. Géron, 3rd ed.) as a syllabus; "
        "no text or figures are reproduced from the book."
    )
