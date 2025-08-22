from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Define the SimpleNaiveBayes class (same as the one used for training the model)
class SimpleNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_prior = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Calculate class prior probabilities
        for c in self.classes:
            self.class_prior[c] = np.sum(y == c) / n_samples

        # Calculate feature probabilities (handling both categorical and numerical)
        for feature in X.columns:
            self.feature_probs[feature] = {}
            if X[feature].dtype == 'object':  # Categorical
                for c in self.classes:
                    feature_values = X[feature][y == c]
                    self.feature_probs[feature][c] = feature_values.value_counts(normalize=True).to_dict()
            else:  # Numerical - using mean and variance for simplicity
                for c in self.classes:
                    feature_values = X[feature][y == c]
                    self.feature_probs[feature][c] = {
                        'mean': np.mean(feature_values),
                        'std': np.std(feature_values) + 1e-9  # Add small value to avoid division by zero
                    }

    def predict(self, X):
        predictions = []
        for _, sample in X.iterrows():
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.class_prior[c])
                likelihood = 0
                for feature in X.columns:
                    if X[feature].dtype == 'object':  # Categorical
                        if sample[feature] in self.feature_probs[feature][c]:
                            likelihood += np.log(self.feature_probs[feature][c][sample[feature]])
                        else:
                            likelihood += np.log(1e-9)  # Handle unseen values
                    else:  # Numerical (Gaussian Naive Bayes)
                        mean = self.feature_probs[feature][c]['mean']
                        std = self.feature_probs[feature][c]['std']
                        exponent = -((sample[feature] - mean)**2 / (2 * std**2))
                        likelihood += exponent - np.log(np.sqrt(2 * np.pi * std**2))

                posteriors[c] = prior + likelihood

            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Load the Naive Bayes model from the pickle file
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset for column references
df = pd.read_csv('osteoporosis.csv')

# Route to Home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    gender = request.form['gender']
    hormonal_changes = request.form['hormonal_changes']
    family_history = request.form['family_history']
    race = request.form['race']
    body_weight = request.form['body_weight']
    calcium_intake = request.form['calcium_intake']
    vitamin_d_intake = request.form['vitamin_d_intake']
    physical_activity = request.form['physical_activity']
    smoking = request.form['smoking']
    alcohol_consumption = request.form['alcohol_consumption']
    medical_conditions = request.form['medical_conditions']
    medications = request.form['medications']
    prior_fractures = request.form['prior_fractures']

    # Create a DataFrame for the user input (similar to X)
    input_data = pd.DataFrame([[gender, hormonal_changes, family_history, race, body_weight, 
                                calcium_intake, vitamin_d_intake, physical_activity, smoking,
                                alcohol_consumption, medical_conditions, medications, prior_fractures]],
                              columns=['Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity', 
                                       'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
                                       'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications', 
                                       'Prior Fractures'])

    # Use the model to predict the output
    prediction = model.predict(input_data)
    osteoporosis_result = "Yes" if prediction[0] == 1 else "No"

    return render_template('index.html', prediction_text=f"Osteoporosis Risk: {osteoporosis_result}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
