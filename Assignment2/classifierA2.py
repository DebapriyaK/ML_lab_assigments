import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_processing import load_classification_data

def train_classifier():
    #Load data for classification
    X,y=load_classification_data("Lab Session Data.xlsx")

    #Initialize and train logistic regression model
    model=LogisticRegression()
    model.fit(X,y)

    #Predict on training data
    y_pred=model.predict(X)

    #Print classification report
    print("Classification Report:")
    print(classification_report(y,y_pred,target_names=["poor","rich"]))

    # Print actual classification for each customer
    print("\nCustomer Classification:")
    for idx, label in enumerate(y_pred, start=1):
        status = "rich" if label == 1 else "poor"
        print(f"Customer {idx}: {status}")

if __name__ == "__main__":
    train_classifier()
