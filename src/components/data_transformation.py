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

    '''
    The function 'dimensionality_reduction' removes the any one of the feature having high correlation to each other with respect to target variable. Here, X1 and X2 are higly correlated. 
    X1 having less correlation on target varibales compared to X2. So, we're going to remove X1 only.
    '''
    def dimensionality_reduction(threshold, data):
        temp = pd.DataFrame(data.corr())
        lst = []
        for x in temp.index:
            for y in temp.columns:
                if x!=y:
                    if abs(temp[x][y]) > threshold:
                        lst.append([x,y])
        
        for i in range(len(lst)):
            first,second = lst[i][0],lst[i][1]
            try:
                if (abs(temp['Y1'][first]) > abs(temp['Y1'][second])) & (first != 'Y1') & (first != 'Y2') & (second != 'Y1') & (second != 'Y2'):
                    data.drop([second], axis=1, inplace=True)
                elif abs(temp['Y1'][first]) < abs(temp['Y1'][second]) & (first != 'Y1') & (first != 'Y2') & (second != 'Y1') & (second != 'Y2'):
                    data.drop([first], axis=1, inplace=True)
            except:
                pass
        # We, know that X6 feature is not useful for Y1 and Y2 as well as other independent features. So we czn remove directly.
        data.drop(['X6'],axis=1,inplace=True)
        return data

    def final():
        data = DataTransformation.get_check_dtypes()
        final_data = DataTransformation.dimensionality_reduction(0.95, data)
        final_data.drop_duplicates(inplace=True)
        logging.info(f"[data_transformation.py] The data successfully passed 'final' function")
        return final_data