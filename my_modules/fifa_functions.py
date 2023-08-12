from .analysis_functions import *



def cleaning_process_for_fifa_dataset(df): 
    columns_to_drop = ['Name', 'Age', 'Nationality', 'Club', 'Position', 'Team & Contract', 'Height', 'Weight', 'foot', 'Joined', 'Loan Date End', 'Wage', 'Release Clause', 'Contract', 'W/F', 'SM', 'A/W', 'D/W', 'ID',  'LS', 'ST', 'RS', 'LW', 'LF', 'CF',  'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LWB', 'LDM', 'LCM', 'CM', 'RCM', 'RM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'GK']

    # drop columns
    df = drop_columns(df, columns_to_drop)

    # clean column names 
    df = clean_column_names(df)

    # dropna
    df = df.dropna(subset=['composure'])
    def clean_value(x):
            
        unit = x[-1] 
            
        if unit == 'M':
            x = float(x[:-1])
                
        elif unit == 'K':
            x = float(float(x[:-1]) / 1000)
            
        else: x = 0
            
        return x 
    # cleaning 'value' column 
    df['value'] = df['value'].str.replace('€', '')
    df['value'] = list(map(clean_value, df['value']))

    # cleaning 'ir' column
    df['ir'] = df['ir'].replace('[^0-9]', '', regex=True)

    #clean hits
    df['ir'] = pd.to_numeric(df['ir'])

    def clean_hits(x):
        
        unit = x[-1] 
        
        if unit == 'K':
            x = float(x[:-1])
            x = x * 1000
        
        else: x     
        
        return x 


    df['hits'] = list(map(clean_hits, df['hits']))
    df['hits'].unique()

    df['hits'] = pd.to_numeric(df['hits'])

    return df 