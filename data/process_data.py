import sys
import pandas as pd

# # ETL Pipeline Preparation

def load_data(messages_filepath, categories_filepath):
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # ## Merge datasets.
    # - Merge the messages and categories datasets using the common id
    # - Assign this combined dataset to `df`, which will be cleaned in 
    # the following steps
    
    df = pd.merge(messages, categories, on=['id'])
    
    return df


def clean_data(df):
    
    ### Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character so     
    # that each value becomes a separate column. 
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names.
    
    categories = df.categories.str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [item.split("-")[0] for item in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # ### Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # ### Replace categories column in df with new category columns.
    # Drop the categories column from the df dataframe since it is no longer needed.
    # Concatenate df and categories data frames.
    
    # drop the original categories column from `df`
    df = df.drop(labels='categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])
    
    # ### Remove duplicates.
    # Check how many duplicates are in this dataset.
    # Drop the duplicates.
    # Confirm duplicates were removed.
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    from sqlalchemy import create_engine
    
    engine = create_engine('sqlite:///'+ database_filename)
    conn = engine.connect()
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')
    conn.close()
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()