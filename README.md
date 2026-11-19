🎓 HSC GPA Predictor
A machine learning web app that predicts a student's HSC result (GPA out of 5.00) from background information such as family details, SSC result, tuition spending, and lifestyle factors.
Built with Python, Scikit-learn, and Gradio.
🔗 Live Demo: https://8edc3434538e2ac7a4.gradio.live
---
✨ Features
Predicts HSC GPA (clipped to the valid 0 to 5 range)
Clean web interface built with Gradio, with no coding needed to use it
End-to-end Scikit-learn pipeline (imputation, scaling, encoding, model) so raw inputs go straight in
Handles missing values and unseen categories safely
🧠 How It Works
Data: `bangladesh_student_performance.csv` contains student records with the target column `hsc_result`.
Preprocessing (`train.py`):
Numeric features: median imputation, then `StandardScaler`
Categorical features: most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`
Model: `RandomForestRegressor` (200 trees, `max_depth=10`) inside a single `Pipeline`.
Training: 80/20 train/test split with `random_state=42`. The trained pipeline is saved as `students_rf_pipeline.pkl`.
App (`app.py`): loads the pipeline, builds a DataFrame from the form inputs, and returns the predicted GPA.
📥 Input Features
Feature	Type	Description
Gender	Categorical	Male / Female
Age	Numeric	Student's age
Current Address	Categorical	Urban / Rural
Family Size	Categorical	Family Member > 3 / Family Member <= 3
Parents Status	Categorical	Together / Apart
Mother's Education	Numeric	Education level
Father's Education	Numeric	Education level
Mother's Job	Categorical	Occupation category
Father's Job	Categorical	Occupation category
Relationship Status	Categorical	Yes / No
Smoker	Categorical	Yes / No
Yearly Tuition Fee	Numeric	Tuition spending per year
Time Spent with Friends	Numeric	Scale of 1 to 5
SSC Result	Numeric	SSC GPA (out of 5.00)
📊 Model Performance
Evaluated on the 20% held-out test set:
Metric	Value
R²	0.951
RMSE	0.135
📁 Project Structure
```
HSC_GPA_Predictor/
├── app.py                              # Gradio web app
├── train.py                            # Data cleaning, training, model export
├── students_rf_pipeline.pkl            # Trained Random Forest pipeline
├── bangladesh_student_performance.csv  # Dataset
├── requirements.txt                    # Dependencies
└── README.md
```
🚀 Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/farhanazad/HSC_GPA_Predictor.git
cd HSC_GPA_Predictor

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```
Then open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).
Retrain the model (optional)
```bash
python train.py
```
This regenerates `students_rf_pipeline.pkl`. Use the same scikit-learn version for training and running the app, or the pickle may not load.
🛠️ Tech Stack
Python
Pandas / NumPy for data handling
Scikit-learn for the ML pipeline and Random Forest
Gradio for the web interface
⚠️ Disclaimer
This is an educational project. Predictions are statistical estimates based on a limited dataset and should not be used to make real decisions about any student.
🔮 Future Improvements
Compare more models (Gradient Boosting, Stacking) with cross-validation
Hyperparameter tuning
Add feature importance visualisation
Collect a larger, more representative dataset
👤 Author
Farhan Azad
GitHub: @farhanazad
