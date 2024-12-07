import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load model and preprocessor
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Fill missing values with default values for categorical features
            features['gender'] = features['gender'].fillna('unknown')
            features['race_ethnicity'] = features['race_ethnicity'].fillna('unknown')
            features['parental_level_of_education'] = features['parental_level_of_education'].fillna('unknown')
            features['lunch'] = features['lunch'].fillna('unknown')
            features['test_preparation_course'] = features['test_preparation_course'].fillna('none')

            # Ensure that features are passed correctly to the preprocessor
            data_scale = preprocessor.transform(features)

            # Predict using the loaded model
            preds = model.predict(data_scale)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_prepration_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_prepration_course = test_prepration_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Creating a dictionary from the input features
            Custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_prepration_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            
            # Create the dataframe
            df = pd.DataFrame(Custom_data_input_dict)

            # Handle missing categorical data in the DataFrame (same as in predict method)
            df['gender'] = df['gender'].fillna('unknown')
            df['race_ethnicity'] = df['race_ethnicity'].fillna('unknown')
            df['parental_level_of_education'] = df['parental_level_of_education'].fillna('unknown')
            df['lunch'] = df['lunch'].fillna('unknown')
            df['test_preparation_course'] = df['test_preparation_course'].fillna('none')

            return df

        except Exception as e:
            raise CustomException(e, sys)

        
