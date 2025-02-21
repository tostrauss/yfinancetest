#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
echo "Launching Streamlit..."
streamlit run app.py

#dash
echo "Creating virtual environment..."
python3 -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
echo "Launching Dash..."
python dashTry.py