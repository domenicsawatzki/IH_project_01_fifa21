
import pandas as pd 
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
import seaborn as sns 



# show shape, df overview, and dtypes for dataframe 
def firstLook(df):
    
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df is not a valid DataFrame.")
    
    print(df.shape)
    display(df)
    print(df.dtypes)

# shows and count unique values
def showUnique(df):
     # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df is not a valid DataFrame.")
    
    for col in df:
        print('values of column:',col)
        l = 0
        print(df[col].unique())
        l = len(df[col].unique())
        print('number of unique values:', l, '\n')
        
def nullTable(dataset):
    round(dataset.isna().sum()/len(dataset),4)*100  # shows the percentage of null values in a column
    nulls_df = pd.DataFrame(round(dataset.isna().sum()/len(dataset),4)*100)
    nulls_df
    nulls_df = nulls_df.reset_index()
    nulls_df
    nulls_df.columns = ['header_name', 'percent_nulls']
    display(nulls_df)
    display(dataset.isna().sum())
    
        
# transform to snake_case, lower column names and strip
def clean_column_names(df):
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df is not a valid DataFrame.")
    
    # create copy
    clean_df = df.copy()
    
    #cleaning
    clean_df.columns= [col.strip() for col in clean_df.columns]
    clean_df.columns= [col.lower() for col in clean_df.columns]
    clean_df.columns= [col.replace(' ', '_') for col in clean_df.columns]
    
    return clean_df

# change the column names by a given mapping
def change_column_names(df, mapping):
    try:

        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df is not a valid DataFrame.")
        
        # Check if 'mapping' is a dictionary
        if not isinstance(mapping, dict):
            raise ValueError("'mapping' argument must be a dictionary.")
        
        # create copy
        clean_df = df.copy()
        print(mapping)
        
        #cleaning
        clean_df.rename(columns=mapping, inplace=True)
    
        return clean_df
    
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")



# Replace inconsistent values with their correct counterparts
def replace_inconsistent_values(df, mapping):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")

        # Check if 'mapping' is a dictionary
        if not isinstance(mapping, dict):
            raise ValueError("'mapping' argument must be a dictionary.")
        
        for key in mapping:
            # Check if 'column' is a valid column name in 'data'
            if key not in mapping:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")        
            print(key , '// ', mapping[key])
           
            # Replace inconsistent values in the specified column with their correct counterparts
            df[key] = df[key].replace([mapping[key]])
            
        return df

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")
    
# Split a dataset into numerical and categorical 

def num_cat_split(df):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")
        
        df_num = df.select_dtypes(include = np.number)
        df_cat = df.select_dtypes(include = object)
        
        
        display(df_num)
        display(df_cat)

    
        return df_num, df_cat
    
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")
 
 
def X_y_split(df, y_columname):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")
                
        if not isinstance(y_columname, str):
            raise ValueError("Input df must be a string.")
        
        y = df[y_columname]
        X = df.drop([y_columname], axis=1)
        
        return X, y
            
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  

# normalize data by using MinMaxScaler
def min_max_scaler(df): 
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")

        from sklearn.preprocessing import MinMaxScaler   
        display(df.describe().T)

        MinMaxtransformer = MinMaxScaler().fit(df)
        normalized_df = MinMaxtransformer.transform(df)

        normalized_df = pd.DataFrame(normalized_df,columns=df.columns)


        display(normalized_df.describe().T)
        
        return normalized_df
        
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  
 
     
# normalize data by using StandardScaler
def standard_scaler(df): 
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")
        
        from sklearn.preprocessing import StandardScaler
        
        display(df.describe().T)

        StandardTransformer = StandardScaler().fit(df)
        normalized_df = StandardTransformer.transform(df)

        normalized_df = pd.DataFrame(normalized_df,columns=df.columns)


        display(normalized_df.describe().T)
        
        return normalized_df
        
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  
      
# drop columns from list   
def drop_columns(df, columns_to_drop):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")
                
        if not isinstance(columns_to_drop, list):
            raise ValueError("Input df must be a list.")    
        
        df = df.drop(columns_to_drop, axis=1)
        
        return df
    
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")   
    
    
# creating histograms for all columns by using matplotlib function  
def hist_plot(df, bin):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")   
                
        for col in df:
            plt.hist(df[col], bins = bin)
            headertext = 'Histograms for '
            plt.title(headertext + col)
            plt.show()

    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  
 
# create a correlation matrix 
def cor_matrix(df):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")   
        
        correlations_matrix = df.corr().round(2)
        correlations_matrix
        
        return correlations_matrix
                
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  


def remove_outlier(df, parameter, col, bin):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input dfmust be a pandas DataFrame.")  
        
        if not isinstance(parameter, dict):
            raise ValueError("Input parameter must be a dictionary.")
        
        if not isinstance(col, str):
            raise ValueError("Input col must be a string.")
        
        if not isinstance(bin, int):
            raise ValueError("Input bin must be a int.")             
        
        print(df.shape)
        plt.hist(df[col], bins = bin)
        headertext = 'Histograms for '
        plt.title(headertext + col)
        plt.show()
        
        upper_l = parameter['upper_l']  
        lower_l = parameter['lower_l']
        upper_factor = parameter['upper_factor']
        lower_factor = parameter['lower_factor']
        
        
        iqr = np.percentile(df[col],upper_l) - np.percentile(df[col],lower_l)
        upper_limit = np.percentile(df[col],upper_l) + upper_factor*iqr
        lower_limit = np.percentile(df[col],lower_l) - lower_factor*iqr
        
        df_removed_outlier = df[(df[col]>lower_limit) & (df[col]<upper_limit)].copy()
        
        print(df_removed_outlier.shape)
        plt.hist(df_removed_outlier[col], bins = bin)
        headertext = 'Histograms for '
        headertext2 = 'after removing outlier'
        plt.title(headertext + col + headertext2)
        plt.show()
        
        return df_removed_outlier
        
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  

def log_transform(x):
    x = np.log10(x)
    if np.isfinite(x):
        return x
    else:
        return 0

def log_transform_for_rows(df, columns, bin_size):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")  
        
        if not isinstance(columns, list):
            raise ValueError("Input col must be a list.")
        
        if not isinstance(bin_size, int):
            raise ValueError("Input bin must be a int.")             
        
        

        df_removed_outlier = df.copy()
        
        for col in columns:
            print(df.shape)
            plt.hist(df[col], bins = bin_size)
            headertext = 'Histograms for '
            plt.title(headertext + col)
            plt.show()
        
            
            
            df_removed_outlier[col] = df[col].apply(log_transform)

            print(df_removed_outlier.shape)
            plt.hist(df_removed_outlier[col], bins = bin_size)
            headertext = 'Histograms for '
            headertext2 = 'after log'
            plt.title(headertext + col + headertext2)
            plt.show()
        
        return df_removed_outlier
        
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  



def OneHot(df):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")  
        

        from sklearn.preprocessing import OneHotEncoder

        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(drop='first')

        # Fit the encoder on the categorical data
        encoder.fit(df)

        # Transform the categorical data using the fitted encoder
        encoded_df = encoder.transform(df)

        # Convert the encoded data to a pandas DataFrame
        encoded_df = pd.DataFrame(encoded_df.toarray(), columns=encoder.get_feature_names_out(df.columns))

        encoded_df

        return encoded_df
        
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  

def split_the_data_into_train_test_datasets(X, y, factor):
    try:
        # Check if 'data' is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")  
        
        # train-test-split
        from sklearn.model_selection import train_test_split as tts

        X_train, X_test, y_train, y_test=tts(X, y, test_size=factor)
        
        print('X_train', X_train.shape)
        print('y_train', y_train.shape)
        print('X_test',X_test.shape)
        print('y_test',y_test.shape)
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        # Handle any exceptions and provide informative error messages
        raise ValueError(f"Error occurred: {e}")  




    
def run_and_test_linear_model(X_train, X_test, y_train, y_test):
    
    from sklearn.linear_model import LinearRegression as LinReg
    
    linreg=LinReg()    # model
    linreg.fit(X_train, y_train)   # model train
    y_pred_linreg=linreg.predict(X_test)   # model prediction
    

    train_r2=linreg.score(X_train, y_train)
    test_r2=linreg.score(X_test, y_test)

    print ('train r2 score: {} -- test r2 score: {}'.format(train_r2, test_r2))
    
    from sklearn.metrics import mean_squared_error as mse

    train_mse=mse(linreg.predict(X_train), y_train)
    test_mse=mse(linreg.predict(X_test), y_test)


    print ('train MSE: {} -- test MSE: {}'.format(train_mse, test_mse))
    print ('train RMSE: {} -- test RMSE: {}'.format(train_mse**.5, test_mse**.5))
    
    from sklearn.metrics import mean_absolute_error as mae

    train_mae=mae(linreg.predict(X_train), y_train) #MAE
    test_mae=mae(linreg.predict(X_test), y_test)


    print ('Model: {}, train MAE: {} -- test MAE: {}'.format('linear', train_mae, test_mae))
    
    return y_pred_linreg

def use_minMaxTransformer_and_oneHotEncoder(X, MinMaxtransformer, encoder):
    
    # split into num and object
    X_num = X.select_dtypes(include = np.number)
    X_cat = X.select_dtypes(include = object)
   
    
    from sklearn.preprocessing import MinMaxScaler

    # minMaxTransform
    X_normalized = MinMaxtransformer.transform(X_num)
    X_normalized = pd.DataFrame(X_normalized,columns=X_num.columns)
  
    from sklearn.preprocessing import OneHotEncoder


    # Transform the categorical data using the fitted encoder
    encoded_X_cat = encoder.transform(X_cat)

    # Convert the encoded data to a pandas DataFrame
    encoded_X_cat = pd.DataFrame(encoded_X_cat.toarray(), columns=encoder.get_feature_names_out(X_cat.columns))

    encoded_X_cat
    
    X = pd.concat([X_normalized ,encoded_X_cat], axis=1)
    
    return X


def predict_data_and_validate_model(X, y, model):
    y_predicted = model.predict(X)
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    
    r2 = round(model.score(X, y), 3)
    mse = round(mse(y_predicted, y), 3)
    mae = round(mae(y_predicted, y), 3)
    
    print ('The r2 score is: {}.'.format(r2))
    print ('The MSE is: {}.'.format(mse))
    print ('The RMSE is: {:.3f}.'.format(mse**.5))
    print ('The MAE is: {}.'.format(mae))
    print ('\n')
    
    return 

    # for i in range(len(models)):
        
    #     preds[i] = models[i].predict(X_test)
        
    #     train_score=models[i].score(X_train, y_train) #R2
    #     test_score=models[i].score(X_test, y_test)

    #     print ('Model: {}, train R2: {} -- test R2: {}'.format(model_names[i], train_score, test_score))
        
        
    #     from sklearn.metrics import mean_squared_error as mse
    #     train_mse=mse(models[i].predict(X_train), y_train) #MSE
    #     test_mse=mse(preds[i], y_test)

    #     print ('Model: {}, train MSE: {} -- test MSE: {}'.format(model_names[i], train_mse, test_mse))

    #     train_rmse=mse(models[i].predict(X_train), y_train)**0.5 #RMSE
    #     test_rmse=mse(preds[i], y_test)**0.5

    #     print ('Model: {}, train RMSE: {} -- test RMSE: {}'.format(model_names[i], train_rmse, test_rmse))
        
    #     from sklearn.metrics import mean_absolute_error as mae
            
    #     train_mae=mae(models[i].predict(X_train), y_train) #MAE
    #     test_mae=mae(preds[i], y_test)

    #     print ('Model: {}, train MAE: {} -- test MAE: {}'.format(model_names[i], train_mae, test_mae))
    #     print('\n')