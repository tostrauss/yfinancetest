#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Launching Streamlit..."
streamlit run streamlit_app.py
