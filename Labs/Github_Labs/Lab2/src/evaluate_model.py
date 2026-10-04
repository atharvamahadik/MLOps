import pickle, os, json, random
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib, glob, sys
import argparse
from sklearn.datasets import load_breast_cancer

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
        model = joblib.load(f'{model_version}.joblib')
    except:
        raise ValueError('Failed to catching the latest model')
        
    try:
        # Check if the file exists within the folder
        X, y = load_breast_cancer(return_X_y=True)
    except:
        raise ValueError('Failed to catching the data')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_predict = model.predict(X_test)
    metrics = {"F1_Score": f1_score(y_test, y_predict)}
    
    # Save metrics to a JSON file

    if not os.path.exists('metrics/'): 
        # then create it.
        os.makedirs("metrics/")
        
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    
    print(f"Metrics: {metrics}")
               
    
