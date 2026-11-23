
import os
import sys
import joblib
import sklearn

MODEL_FILE = "WeatherWise_RandomForest_Model_fixed.pkl"

print("=" * 45)
print("       WEATHERWISE MODEL TEST")
print("=" * 45)

print("\nPython version:", sys.version)
print("Scikit-learn version:", sklearn.__version__)

# Step 1: Check if model exists
print("\nChecking model file...")

if not os.path.exists(MODEL_FILE):
    print("ERROR: Model file not found!")
    print("Expected filename:", MODEL_FILE)
    print("\nFiles in current folder:")
    print(os.listdir("."))
    sys.exit(1)

# Step 2: Check file size
file_size = os.path.getsize(MODEL_FILE)

print("Model file found!")
print("File size:", file_size, "bytes")

if file_size == 0:
    print("\nERROR: Model file is empty!")
    print("Download the valid model again from Google Colab.")
    sys.exit(1)

# Step 3: Load model
print("\nLoading trained Random Forest model...")

try:
    model = joblib.load(MODEL_FILE)
    print("SUCCESS! Model loaded successfully.")

except Exception as error:
    print("\nMODEL LOADING FAILED!")
    print("Error:", repr(error))
    sys.exit(1)

# Step 4: Display model information
print("\nModel type:", type(model))

if hasattr(model, "n_features_in_"):
    print("Number of input features:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\nInput feature names:")
    print(list(model.feature_names_in_))

if hasattr(model, "classes_"):
    print("\nPrediction classes:")
    print(model.classes_)

# Step 5: Final result
print("\n" + "=" * 45)
print("       WEATHERWISE MODEL READY")
print("=" * 45)
