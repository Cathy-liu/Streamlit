from flask import Flask, request
import pandas as pd
import os
import mlflow.pyfunc

# Step 2: Instantiate the Flask API
api = Flask('ModelEndpoint')

# Step 3: Load the model
model = mlflow.pyfunc.load_model(model_uri="./best_estimator")

# Step 4: Create the routes
## route 1: Health check. Just return success if the API is running
@api.route('/')
def home():
    # return a simple string
    return {"message": "Hi there!", "success": True}, 200

# route 2: accept input data
# Post method is used when we want to receive some data from the user
@api.route('/predict', methods = ['POST'])
def make_predictions():
    # Get the data sent over the API
    user_input = request.get_json(force=True)
    
    # Convert user inputs to pandas dataframe
    df_schema = {"gre":float, "gpa": float} # To ensure the columns get the correct datatype
    user_input_df = pd.read_json(user_input, lines=True, dtype=df_schema) # Convert JSONL to dataframe
    
    # Run predictions and convert to list
    predictions = model.predict(user_input_df).tolist()
    
    return {'predictions': predictions}
    

# Step 5: Main function that actually runs the API!
if __name__ == '__main__':
    api.run(host='0.0.0.0', 
            debug=True, # Debug=True ensures any changes to inference.py automatically updates the running API
            port=int(os.environ.get("PORT", 8080))
           ) 
