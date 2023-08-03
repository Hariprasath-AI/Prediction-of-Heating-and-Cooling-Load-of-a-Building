# Necessary modules/packages imported here for the Data Transformation Phase.
import pandas as pd
from src.logger import logging 
from src.exceptions import CustomException 
from src.components.data_validation import DataValidation

# The class 'DataTransformation' is responsible for all data transformation operations.
class DataTransformation:
    # The 'handling_duplicates' function removes the duplicate record from the dataframe(data) and returns the data.
    def handling_duplicates():
        data=DataValidation.validate()
        if data.duplicated().sum() > 0:
            logging.info(f"[data_transformation.py] There's a {data.duplicated().sum()} duplicated record in the dataset and removed successfully.")
            logging.info("[data_transformation.py] The Data passed the 'duplicates_handling()' and moved to check datatypes of the features")
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
        else:
            logging.info("[data_transformation.py] While Handling the data, there's no duplicates. The Data passed the Duplicates Handling phase") 
        return data

    '''    
    The 'get_check_dtypes' function gets the data from 'handling_duplicates' and check whether the datatype of each column is in numeric type or not.
    If the data holds any non-numeric feature, then the data is not moved further in this project.
    '''
    def get_check_dtypes():
        data=DataTransformation.handling_duplicates()
        df_types=pd.DataFrame(data.dtypes)
        df_types.reset_index(inplace=True)
        df_types.rename(columns={'index': 'col_name', 0: 'data_type'}, inplace=True)
        logging.info("[data_transformation.py] Got Datatypes of each column successfully")
        problamatic_column = []
        for i in range(len(df_types)):
            if str(df_types['data_type'][i]).__contains__('int') or str(df_types['data_type'][i]).__contains__('float'):
                pass 
            else:
                problamatic_column.append(df_types['col_name'][i])
        if len(problamatic_column) == 0:
            logging.info("[data_transformation.py] There is no problem with the datatype of each column. The data passed 'get_check_dtypes()' successfully.")
            return data
        else:
            logging.info(f"[data_transformation.py] There is a problem with the datatype of column -> {problamatic_column}")
            logging.info(f"[data_transformation.py] The data holds non-numeric feature, then the data is not moved further in this project. Please resolve this!!")

    
