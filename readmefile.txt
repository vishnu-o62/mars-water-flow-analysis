TESTATHON PROJECT README

PROJECT TITLE AI-Based Identification of Past Water-Flow Patterns on
Mars Using NASA Imagery

PROJECT OVERVIEW--This project is a web-based system for analysing Mars
surface imagery and identifying terrain patterns that may be associated
with ancient water-flow features. The application processes an uploaded
image through the project’s image-analysis pipeline and presents visual
outputs to help users inspect detected valley-like features and
estimated flow paths.

The results are intended as an image-analysis aid. They do not
independently prove that water flowed at a location or provide a
complete scientific reconstruction of Mars’ ancient hydrology.

MAIN FEATURES - Upload a Mars surface image through the web interface. -
Process the image using the project’s configured analysis pipeline. -
Display analysis panels and visual results, including detected features
and flow-path outputs when available. - Present evaluation metrics where
supported by the application. - Display an informative result when no
acceptable valley or flow direction is identified. - Reject ZIP files;
the frontend is configured to enforce a 5 MB upload limit.

TECHNOLOGIES USED Frontend: - React - TypeScript - Vite

Backend and image analysis: - Python - Flask - PyTorch - OpenCV - NumPy
and other dependencies listed in requirements.txt - U-Net-based image
segmentation and the project’s flow-analysis pipeline - SAM2 is used in
the project’s mask-generation workflow

PROJECT STRUCTURE - app.py Flask application entry point - src/ Project
processing and pipeline code - frontend/ React + TypeScript web
interface - models_saved/ Saved model files - data/ Project image data -
outputs/ Generated prediction and analysis outputs - requirements.txt
Python dependencies

HOW TO RUN THE APPLICATION LOCALLY Prerequisites: - Python installed -
Node.js and npm installed

1.  Start the backend:

    a.  Open a terminal in the main project folder.
    b.  Install Python dependencies if they are not already installed:
        pip install -r requirements.txt
    c.  Start the Flask application: python app.py

    Keep this terminal open. The backend is expected to run at:
    http://127.0.0.1:5000

2.  Start the frontend:

    a.  Open a second terminal.
    b.  Change to the frontend folder: cd frontend
    c.  Install frontend dependencies if required: npm install
    d.  Start the development server: npm run dev

    Open the local URL printed by Vite in the terminal. The usual
    development URL is: http://localhost:5173

3.  Use the application:

    a.  Open the frontend in a browser.
    b.  Upload a supported Mars image within the 5 MB limit.
    c.  Start the analysis using the interface.
    d.  Review the generated visual outputs and available evaluation
        metrics.

IMPORTANT NOTES - The local URLs above work on the computer running the
application; they are not public deployment URLs. - The frontend and
backend must both be running for the complete application workflow. -
Exact output panels depend on the input image and the analysis
results. - Ensure the required model weights and project data are
present before running the pipeline. - If a command fails, check the
terminal error and verify that dependencies, model files, and configured
paths are available.

PROJECT PURPOSE: The system supports exploratory analysis of Mars imagery
by detecting terrain patterns that may resemble ancient water-flow
features. Outputs should be interpreted carefully and validated against
suitable scientific reference data before drawing scientific
conclusions.
