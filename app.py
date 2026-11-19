
import gradio as gr
import pandas as pd
import numpy as np
import pickle

with open("students_rf_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

def predict_gpa(gender, age, address, famsize, Pstatus,M_Edu, F_Edu, M_Job, F_Job, relationship,
                smoker, tuition_fee, time_friends, ssc_result):
    input_df = pd.DataFrame([[
        gender, age, address, famsize, Pstatus, M_Edu,F_Edu, M_Job, F_Job, relationship, smoker,
        tuition_fee, time_friends, ssc_result
    ]],
    columns=[
        "gender", "age", "address", "famsize", "Pstatus","M_Edu", "F_Edu", "M_Job", "F_Job", "relationship",
        "smoker", "tuition_fee", "time_friends", "ssc_result"
    ])
    prediction = model.predict(input_df)[0]
    return f"Predicted HSC Result: {np.clip(prediction, 0, 5):.2f}"

inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Age", value=18),
    gr.Radio(["Urban", "Rural"], label="Current Address"),
    gr.Radio(
        ["Family Member > 3", "3 <= Family Member"],
        label="Family Size"
    ),
    gr.Radio(["Together", "Apart"], label="Parents Status"),
    gr.Slider(0, 5, step=1, label="Mother's Education Level"),
    gr.Slider(0, 5, step=1, label="Father's Education Level"),
    gr.Dropdown(
        ["Housewife", "Health Care", "Public Services", "Teacher","others"],
        label="Mother's Job"
    ),
    gr.Dropdown(
        ["Doctor", "Public Service", "Teacher","Businessman","Farmer"],
        label="Father's Job"
    ),
    gr.Radio(["Yes", "No"], label="Relationship Status"),
    gr.Radio(["Yes", "No"], label="Smoker?"),
    gr.Number(label="Yearly Tuition Fee"),
    gr.Slider(1, 5, step=1, label="Time Spent with Friends"),
    gr.Number(label="SSC Result", value=4.0)
]


demo = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction"),
    title="HSC Result Predictor",
    description="Enter student information to predict HSC results."
)

demo.launch(share=True)
