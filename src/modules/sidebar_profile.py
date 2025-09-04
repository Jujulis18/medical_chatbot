# modules/sidebar_profile.py
import streamlit as st
import pandas as pd

def show_filter_profile():
    st.sidebar.header("My profile")

    health_issues = ["Type 2 diabete", "sleep apnea", "mental health issues", "chronic back pain"]
    selection_pays = st.sidebar.multiselect(
        "Select your health issues",
        options=health_issues,
        default=None
    )

    gender = st.sidebar.radio(
        "Gender",
        options=["Other", "Homme", "Femme"],
        index=0
    )

    medication_list = ["Ibuprofen", "somnifer"]
    medication = st.sidebar.multiselect(
        "Prescribed medication",
        options=medication_list,
        default=None
    )
       

    return {
        "health_issues": health_issues,
        "gender": gender,
        "medication": medication
    }
