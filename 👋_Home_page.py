import streamlit as st

st.set_page_config(
    page_title="Home page",
    page_icon="👋",
    layout="centered")

st.write("## Welcome to the Deblending Forecast plotting platform! 👋")

st.sidebar.success("Select the type of analysis you want to explore")

st.markdown(
    """
    Select on the left panel what you want to explore:
    - With 📈 forecast, you will be able to explore the Euclid Blending Forecast, with interactive histograms regarding the various blending parameters and thresholds

    - With 👁 visualisation, you will explore examples of the different cases of blended situations.
    """
)
