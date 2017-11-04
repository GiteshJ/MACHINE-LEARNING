"""----------------------------------- RANDOM FOREST -----------------------------------------"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# File path for datasets
INPUT_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
#output file path on local device
OUTPUT_PATH = 'C:/Users/ANJALI/Desktop/cancer.csv'
# Global headers array
headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
               "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses",
               "CancerType"]
def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return:
    """
    data = pd.read_csv(path)
    return data
 
 
def get_headers(dataset):
    """
    dataset headers
    :param dataset:
    :return:
    """
    return dataset.columns.values

 
def data_file_to_csv():
     # Load the dataset into Pandas data frame
    dataset =read_data(INPUT_PATH)
    # Add the headers to the loaded dataset
    dataset = add_headers(dataset, headers) 
    # Save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH, index=False)
    print( "File saved ...!")


def add_headers(dataset, headers):
    """
    Add the headers to the dataset
    :param dataset:
    :param headers:
    :return:
    """
    dataset.columns = headers
    return dataset


def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print (dataset.describe())


def handel_missing_values(dataset, missing_values_header, missing_label):
    """
    Filter missing values from the dataset
    :param dataset:
    :param missing_values_header:
    :param missing_label:
    :return:
    """
 
    return dataset[dataset[missing_values_header] != missing_label]


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """
 
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


def main():
    """
    Main function
    :return:
    """
    # Read the file and convert the same to a csv file.
    data_file_to_csv()
    # Read the csv file into pandas dataframe
    dataset= pd.read_csv(OUTPUT_PATH)
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)
    # Filter missing values
    dataset = handel_missing_values(dataset, headers[6], '?')
    # Split the data into training and test sets
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, headers[1:-1], headers[-1])
    # Train and Test dataset size details for veification
    print ("Train_x Shape :: ", train_x.shape)
    print ("Train_y Shape :: ", train_y.shape)
    print ("Test_x Shape :: ", test_x.shape)
    print ("Test_y Shape :: ", test_y.shape)
    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print ("Trained model :: ", trained_model)
    #Run the classifier to predict the target variables for the test dataset
    predictions = trained_model.predict(test_x)
    # view the predicted target variable and actual target variable for the first five test datasets
    for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
    # print the test accuracy , train accuracy and the confusion matrix
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print (" Confusion matrix ", confusion_matrix(test_y, predictions))
    
    
# Calling the main()  function
if __name__ == "__main__":
    main()