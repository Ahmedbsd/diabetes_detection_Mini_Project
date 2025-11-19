**Diabetes Classification – Mini Machine Learning Project :**

This project demonstrates a complete machine learning workflow to build a diabetes classification model using a provided dataset.
It includes data loading, preprocessing, model training, evaluation, and exporting the final model for use in another program.

***Workflow Overview :***

1) Data Importation :
   
      Load the diabetes dataset from the folder.

3) Data Preprocessing :

      - Handle missing values
      
      - Scale numerical features
        
      - Prepare data for model training
      
      - The preprocessor is saved as preprocessor in diabetes_model.pkl.

4) Training Three Models :
      - Logistic Regression
      - Decision Tree
      - KNN

5) Model Testing & Comparison :
   
      Evaluate all three models based on accuracy and choose the best one.

7) Selecting the Best Model :
   
      The top-performing model is saved as model in diabetes_model.pkl.

9) Using the Model in Another Program :
    
      test.py loads both the preprocessor and the saved model to make predictions on new data.

***Outputs :***

*After training, one file will be generated :*

diabetes_model.pkl Inside it you will find :

  - preprocessor— saved preprocessing pipeline
    
  - model — best-performing trained model

This file allow predictions in any external script without retraining.
