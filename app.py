import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
from pgmpy.models import BayesianNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from pgmpy.estimators import MaximumLikelihoodEstimator
import seaborn as sns
import base64
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
data = None
processed_data = None
model = None
metrics = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the dataset
        data = pd.read_csv(filepath)
        return render_template('index.html', columns=data.columns.tolist(), data_preview=data.head().to_html())
    return "Invalid file format. Please upload a CSV file.", 400

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    global data, processed_data
    if data is not None:
        processed_data = data.copy()
        preprocess_step = request.form.get('preprocess_step')

        if preprocess_step == "missing_values":
            for column in processed_data.select_dtypes(include=['float64', 'int64']):
                processed_data[column].fillna(processed_data[column].mean(), inplace=True)
        elif preprocess_step == "normalization":
            for column in processed_data.select_dtypes(include=['float64', 'int64']):
                processed_data[column] = (processed_data[column] - processed_data[column].min()) / (
                    processed_data[column].max() - processed_data[column].min()
                )
        elif preprocess_step == "outliers":
            for column in processed_data.select_dtypes(include=['float64', 'int64']):
                Q1 = processed_data[column].quantile(0.25)
                Q3 = processed_data[column].quantile(0.75)
                IQR = Q3 - Q1
                processed_data = processed_data[
                    (processed_data[column] >= (Q1 - 1.5 * IQR)) & (processed_data[column] <= (Q3 + 1.5 * IQR))
                ]

        return render_template('index.html', columns=processed_data.columns.tolist(), data_preview=processed_data.head().to_html())
    return "No data available for preprocessing.", 400
@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    global processed_data
    if processed_data is not None:
        # Generate the statistical summary
        statistical_summary = processed_data.describe().to_string()
        
        # Render the summary in the HTML page
        return render_template('index.html', 
                               columns=processed_data.columns.tolist(), 
                               data_preview=processed_data.head().to_html(), 
                               statistical_summary=statistical_summary)
    return "No data available for statistical summary.", 400

@app.route('/visualize', methods=['POST'])
def visualize_data():
    global processed_data
    if processed_data is not None:
        chart_type = request.form.get('chart_type')
        x_axis = request.form.get('x_axis')
        y_axis = request.form.get('y_axis')

        if chart_type and x_axis and y_axis:
            plt.figure(figsize=(10, 6))
            if chart_type == 'bar':
                processed_data.groupby(x_axis)[y_axis].mean().plot(kind='bar')
            elif chart_type == 'line':
                processed_data.plot(x=x_axis, y=y_axis, kind='line')
            elif chart_type == 'scatter':
                plt.scatter(processed_data[x_axis], processed_data[y_axis])
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
            else:
                return "Invalid chart type.", 400

            plt.title(f'{chart_type.capitalize()} Chart: {x_axis} vs {y_axis}')
            plot_path = os.path.join('static', 'plot.png')
            plt.savefig(plot_path)
            plt.close()
            return render_template('index.html', plot_url=f'/{plot_path}', columns=processed_data.columns.tolist(), data_preview=processed_data.head().to_html())
    return "No data available for visualization.", 400

def evaluate_model(data, model):
    bic = BicScore(data).score(model)
    k2 = K2Score(data).score(model)
    return {
        'BIC Score': bic,
        'K2 Score': k2,
        'Edges': len(model.edges())
    }

@app.route('/train', methods=['POST'])
def train_model():
    global processed_data, model, metrics
    if processed_data is not None:
        optimization_method = request.form.get('optimization_method')
        no_optimization = request.form.get('no_optimization')

        hc = HillClimbSearch(processed_data)
        if no_optimization:
            best_model = hc.estimate(scoring_method=BicScore(processed_data))  # Use a valid score function here
        elif optimization_method == "hill_climb":
            best_model = hc.estimate(scoring_method=BicScore(processed_data))
        elif optimization_method == "k2":
            best_model = hc.estimate(scoring_method=K2Score(processed_data))
        else:
            return "Invalid optimization method.", 400

        model = BayesianNetwork(best_model.edges())
        model.fit(processed_data, estimator=MaximumLikelihoodEstimator)

        metrics = evaluate_model(processed_data, model)

        return render_template('index.html', metrics=metrics, columns=processed_data.columns.tolist(), data_preview=processed_data.head().to_html(), post_visualization=True)
    return "No data available for training.", 400

@app.route('/download_report', methods=['POST'])
def download_report():
    global processed_data, model

    if processed_data is not None:
        file_format = request.form.get('file_format', 'csv')

        if file_format == 'csv':
            file_path = 'model_report.csv'
            processed_data.to_csv(file_path, index=False)
        elif file_format == 'excel':
            file_path = 'model_report.xlsx'
            processed_data.to_excel(file_path, index=False)
        elif file_format == 'json':
            file_path = 'model_report.json'
            processed_data.to_json(file_path, orient='records')

        return send_file(file_path, as_attachment=True)

    return "No model available for report generation.", 400

if __name__ == '_main_':
    app.run(debug=True)