


# Disaster Response Pipeline Project Deployment


## Project Motivation 

In this project, I applied data engineering skills analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages using Natural Language Processing.

The project use data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events and simulated send the messages to an appropriate disaster relief agency.
This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


## Project Components
1. ETL Pipeline - data Folder  
The process_data.py is a data cleaning pipeline that:  
•	Loads the messages and categories datasets  
•	Merges the two datasets  
•	Cleans the data  
•	Stores it in a SQLite database  
2. ML Pipeline - model folder
The train_classifier.py is a machine learning pipeline that:  
•	Loads data from the SQLite database  
•	Splits the dataset into training and test sets  
•	Builds a text processing and machine learning NLP pipeline 
•	Trains and tunes a model using GridSearchCV  
•	Outputs results on the test set  
•	Exports the final model as a pickle file  
3. Flask Web App  - app folder
The run.py contains a flask web app, the folder includes the html, css and javascript to vizualize the webapp.  


## How to Interact with the project
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

## Licensing, Authors, Acknowledgements, etc.
MIT

