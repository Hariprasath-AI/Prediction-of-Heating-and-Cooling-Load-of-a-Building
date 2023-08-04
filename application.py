# Importing all packages for appliction file.
# These are all the important packages to run the project(Flask and numpy)
# Flask and numpy is enough to run this project(already model build and ready to use)
from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
from src.components.model_loader import ModelLoader
from src.logger import logging
# Prediction-of-Heating-and-Cooling-Load-of-a-Building
application=Flask(__name__)

'''
For this application, we created only one page. That one page acts as a home page as well as predict page.
@application.route('/') refers to the homepage of our project i.e., home.html located in templates folder of this project.
@application.route('/predict') also refers to the same page. 
'''
@application.route('/')
def home():
    return render_template('home.html')

@application.route('/predict', methods=['GET', 'POST'])
def make_predictions():
    if request.method=='GET':
        return render_template('home.html')
    elif request.method=='POST':
        model_y1 = ModelLoader.loader_y1()
        model_y2 = ModelLoader.loader_y2()
        try:
            data = [float(x) for x in request.form.values()]
            data = np.array(data)
            pred_y1 = model_y1.predict(data)
            data = [float(x) for x in request.form.values()]
            data.append(pred_y1)
            data = np.array(data)
            pred_y2 = model_y2.predict(data)
        except:
            pred = "--Input Error--"
            logging.info("[app.py] Error Occured while getting data from FORM-HTML[home.html]")
        return render_template('home.html', y1_pred=pred_y1, y2_pred=pred_y2)

# This main method is called while running this application
if __name__ == '__main__':
    application.run(host='0.0.0.0')