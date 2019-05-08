import sys
# import libraries

#General numeric and dataframes
import numpy as np
import pandas as pd

# Data base related
from sqlalchemy import create_engine

#Regex
import re

#Object serialization
import pickle

#Natural Language
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

#Machine learning models
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, log_loss, recall_score
    
    
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesCategories', engine)
    X = df["message"]
    Y = df.loc[:, "related":"direct_report"]
    return X, Y, Y.columns.tolist()

def tokenize(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    #remove punctation
    text = re.sub(r"[^a-zA-Z0-9]",' ',text.lower())
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]    
    
    return (tokens)


def build_model():
    steps = [('tfidf',TfidfVectorizer(tokenizer=tokenize)),
             ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 1)))]
    
    pipeline = Pipeline(steps)
    
    parameters = {'tfidf__max_df': [1.0],
                  'tfidf__smooth_idf': [False],
                  'clf__estimator__max_features': ['auto']}
    
    cv = GridSearchCV(pipeline,param_grid=parameters,cv=5)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names): 
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    
    print("\n Classification Report")
    for i in range(36):
        print("\n" + category_names[i] +"\n", classification_report(Y_test.iloc[:,i], y_pred.iloc[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()