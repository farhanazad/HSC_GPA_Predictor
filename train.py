import numpy as np 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder 
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor,StackingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("bangladesh_student_performance.csv")
df=df.drop('date',axis=1)

df['famsize']=df['famsize'].map({
    "GT3": "Family Member > 3",
    "LE3": "Family Member <= 3"
})
df['gender']=df['gender'].map({
    "M": "Male",
    "F": "Female"
})
df['M_Job']=df['M_Job'].map({
    "At_home": "Housewife",
    "Services": "Public Services",
    "Health": "Health Care",
    "Other": "Others",
    "Teacher": "Teacher",
})
df['F_Job']=df['F_Job'].map({
    "Services": "Public Services",
    "Health": "Doctor",
    "Business": "Businessman",
    "Teacher": "Teacher",
    "Farmer": "Farmer",
})
x=df.drop('hsc_result',axis=1)
y=df['hsc_result']

numeric_features=x.select_dtypes(include=np.number).columns.tolist()
categorical_features=x.select_dtypes(exclude=np.number).columns.tolist()

nums_transformer=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('scaler',StandardScaler())
])

cat_transformer=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',nums_transformer,numeric_features), 
        ('cat',cat_transformer,categorical_features)
    ]
)

rf_model=RandomForestRegressor(
    n_estimators=200,
    max_depth=10, 
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)


rf_pipeline=Pipeline(
    [
        ('preprocessor',preprocessor), 
        ('model',rf_model )
    ]
)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)
rf_pipeline.fit(x_train,y_train)
y_pred=rf_pipeline.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)

print("RMSE : ",rmse)
print("R2 Score : ",r2)
with open("students_rf_pipeline.pkl","wb") as f:
    pickle.dump(rf_pipeline,f)



