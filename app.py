import pickle
from flask import request, Flask, render_template, redirect, url_for, flash #type:ignore
import pandas as pd #type:ignore

from dashboard.utils import reading_cleaning

# ======================================Create app=================================================
app = Flask(__name__)

# ======================================loading models and datasets================================
df = pd.read_csv('recruitment_system/dataset/HR_comma_sep.csv.crdownload')

model = pickle.load(open('recruitment_system/models/model.pkl','rb'))
scaler = pickle.load(open('recruitment_system/models/scaler.pkl','rb'))

df = reading_cleaning(df)



# routes===================================================================

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/job')
def job():
    return render_template('job.html')



if __name__ == "__main__":
    app.run(debug = True)