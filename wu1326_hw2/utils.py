from random import randrange, seed
import numpy as np
import pandas as pd


def load_data(path):
    """
    Read the data from dating_train.csv. 
    Preprocess the data by performing the following:
    (i) Remove the quotes present in columns "race","race_o","field"
    (ii) Convert all the values in the column "field" to lowercase
    (iii) Perform Label Encoding for the columns 'gender','race','race_o','field'.


    Args:
        path: path to the dataset

    Returns:
        X: data
        y: labels
    """

    dataset = pd.read_csv(path) #dataset = pd.read_csv('dataset/dating_train.csv')
    df1 = dataset.copy()
  
    # Removal of quotes
    def _remove_quotes():
        columns = ["race","race_o","field"]
        count = 0
        ## >>> YOUR CODE HERE >>>
        for colName in columns:
            for text1 in dataset[colName]:
                if text1.find('\'') != -1:
                    count = count + 1
            dataset[colName] = dataset[colName].str.replace('\'','')
            
        ## <<< END OF YOUR CODE <<<
        print('Quotes removed from ' + str(count) + ' cells')
                
  
    # Convert all the values in the column field to lowercase
    def _to_lower_case():
        count = 0
        ## >>> YOUR CODE HERE >>>
        
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                if isinstance(dataset.iat[i,j], str):
                    dataset.iat[i,j] = dataset.iat[i,j].lower()
                    count = count + 1

        ## <<< END OF YOUR CODE <<<
        print('Standardized '+ str(count) + ' cells to lower case')


    # Label Encoding
    def _label_encoding():
        cols = ['gender','race','race_o','field']

        ## >>> YOUR CODE HERE >>>
        for colName in cols: # select each cols name
            colSetList = list(set(dataset[colName]))
            colSetList.sort()
            numList = list(range(len(colSetList)))
            if ('male' in colSetList) and (colName == 'gender'):
                gender_male_val = colSetList.index('male')
            elif ('european/caucasian-american' in colSetList) and (colName == 'race'):
                race_eu_ca_val = colSetList.index('european/caucasian-american')
            elif ('latino/hispanic american' in colSetList) and (colName == 'race_o'):
                race_o_la_his_val = colSetList.index('latino/hispanic american')
            elif ('law' in colSetList) and (colName == 'field'):
                field_law_val = colSetList.index('law')
            dataset[colName].replace(colSetList, numList, inplace=True)

        # gender_male_val = ...
        # race_eu_ca_val = ...
        # race_o_la_his_val = ...
        # field_law_val = ...
        ## <<< END OF YOUR CODE <<<

        print("Value assigned for male in column gender: ",gender_male_val)
        print("Value assigned for European/Caucasian-American in column race: ", race_eu_ca_val)
        print("Value assigned for Latino/Hispanic American in column race_o:", race_o_la_his_val)
        print("Value assigned for law in column field: ", field_law_val)

    def _normalize():
        preference_partner_parameters = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
        preference_participant_parameters = ['attractive_important', 'sincere_important', 'intelligence_important','funny_important', 'ambition_important', 'shared_interests_important']

        sum_preference_partner_scores=[]
        sum_preference_participant_scores = []
        preference_scores_of_participant_mean = []
        preference_scores_of_partner_mean = []

        # Calculating mean for preference_scores_of_partner
        for j in preference_partner_parameters:
            for i in range(len(dataset)):
                ## >>> YOUR CODE HERE >>>
                if j == 'pref_o_attractive':
                    sum_preference_partner_scores.append(dataset[j].values[i])
                else:
                    sum_preference_partner_scores[i] = sum_preference_partner_scores[i] + dataset[j].values[i]
        for j in preference_partner_parameters:
            for i in range(len(dataset)):
                dataset[j].values[i] = dataset[j].values[i] / sum_preference_partner_scores[i]

                # dataset[j].values[i] = ...
                ## <<< END OF YOUR CODE <<<

        # Calculating mean for preference_scores_of_participant
        for k in preference_participant_parameters:
            for i in range(len(dataset)):
                ## >>> YOUR CODE HERE >>>
                if k == 'attractive_important':
                    sum_preference_participant_scores.append(dataset[k].values[i])
                else:
                    sum_preference_participant_scores[i] = sum_preference_participant_scores[i] + dataset[k].values[i]
        for k in preference_participant_parameters:
            for i in range(len(dataset)):
                dataset[k].values[i] = dataset[k].values[i] / sum_preference_participant_scores[i]
                
                # dataset[k].values[i] = ...
                ## <<< END OF YOUR CODE <<<

        #Calculating the mean values of each columns in  preference_scores_of_partner      
        for x in preference_partner_parameters:
            preference_scores_of_partner_mean.append(dataset[x].mean())

        #Calculating the mean values of each columns in  preference_scores_of_participant     
        for y in preference_participant_parameters:
            preference_scores_of_participant_mean.append(dataset[y].mean())

        #Print the values of mean of each column
        print("Mean of attractive_important:" + str(round(preference_scores_of_participant_mean[0],2)))
        print("Mean of sincere_important:" + str(round(preference_scores_of_participant_mean[1],2)))
        print("Mean of intelligence_important:" + str(round(preference_scores_of_participant_mean[2],2)))
        print("Mean of funny_important:" + str(round(preference_scores_of_participant_mean[3],2)))
        print("Mean of ambition_important:" + str(round(preference_scores_of_participant_mean[4],2)))
        print("Mean of shared_interests_important:" + str(round(preference_scores_of_participant_mean[5],2)))

        print('Mean of pref_o_attractive:' + str(round(preference_scores_of_partner_mean[0],2)))
        print('Mean of pref_o_sincere:' + str(round(preference_scores_of_partner_mean[1],2)))
        print('Mean of pref_o_intelligence:' + str(round(preference_scores_of_partner_mean[2],2)))
        print('Mean of pref_o_funny:' + str(round(preference_scores_of_partner_mean[3],2)))
        print('Mean of pref_o_ambition:' + str(round(preference_scores_of_partner_mean[4],2)))
        print('Mean of pref_o_shared_interests:' + str(round(preference_scores_of_partner_mean[5],2)))

    _remove_quotes()
    _to_lower_case()
    _label_encoding()
    _normalize()
    
    X = dataset.drop(columns=['decision']).values
    y = dataset['decision'].values

    return X, y


def shuffle_data(X, y, random_state=None):
    """
    Shuffle the data.

    Args:
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )
        seed: int or None

    Returns:
        X: shuffled data
        y: shuffled labels
    """
    if random_state:
        np.random.seed(random_state)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def my_train_test_split(X, y, test_pct=0.2, shuffle=True, random_state=42):
    """
    Split the data into training and test sets.

    Args:
        X: numpy array of shape (N, D)
        y: numpy array of shape (N, )
        test_size: float, percentage of data to use as test set
        shuffle: bool, whether to shuffle the data or not
        seed: int or None

    Returns:
        X_train: numpy array of shape (N_train, D)
        X_val: numpy array of shape (N_val, D)
        y_train: numpy array of shape (N_train, )
        y_val: numpy array of shape (N_val, )
    """

    if shuffle:
        X, y = shuffle_data(X, y, random_state)

    n_train_samples = int(X.shape[0] * (1-test_pct))
    X_train, X_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    """
    Compute accuracy.
    Args:
        y_true: The true labels of shape (N,).
        y_pred: Predictions for the features of shape (N,).
    Raises:
        ValueError: If the shape of y_true or y_pred is not matched.
    Returns:
        The accuracy of the logistic regression.
    """
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred have mismatched lengths.')

    return np.mean(y_pred == y_true)


def evaluate(o_train, p_train, o_valid, p_valid, o_test=None, p_test=None):
    """
    Calculate the accuracy of given predictions on the given labels.

    Args:
        o_train: The original labels of the training set of shape (N,).
        p_train: Predictions for the training set of shape (N,).
        o_valid: The original labels of the validation set of shape (N,).
        p_valid: Predictions for the validation set of shape (N,).
        o_test: The original labels of the test set of shape (N,), optional.
        p_test: Predictions for the test set of shape (N,), optional.

    Returns:
        None
    """

    print('\tTraining Accuracy:', accuracy(o_train, p_train))
    print('\tValidation Accuracy:', accuracy(o_valid, p_valid))

    if o_test is not None and p_test is not None:
        print('\tTest Accuracy:', accuracy(o_test, p_test))
    else:
        print('\tTest Accuracy: Not available')