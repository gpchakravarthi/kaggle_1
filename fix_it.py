# Part 2 - Build a machine learning pipeline that will run autonomously with the csv file and return bestperforming model

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier# Part 2 - Build a machine learning pipeline that will run autonomously with the csv file and return bestperforming model

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


# Function to preprocess the data
def preprocess_data(path, target_feature, filetype='csv'):
    if filetype == 'csv':
        df = pd.read_csv(path)
    elif filetype == 'tsv':
        df = pd.read_csv(path, sep='\t')

    for cl in df.columns:
        if df[cl].isnull().values.any() > 0:
            # nan value replaced with mode for objects
            if str(df[cl].dtype) in ['object']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mode')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
            # nan value replaced with mean for numbers
            elif str(df[cl].dtype) in ['int64', 'float64']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
    # object must be converted as categorical
    for cl in df.columns:
        if str(df[cl].dtype) == 'object':
            df[cl] = pd.Categorical(df[cl])

    # drop duplicate values
    df = df.drop_duplicates(keep=False)

    # Dropping unnecessary features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(df[to_drop], axis=1)

    return df


def split_data_scaled(df, target_feature):
    x, y = df.loc[:, df.columns != target_feature], df[target_feature]
    # outliers handling
    x[x.select_dtypes(include=[np.number]).columns].apply(zscore)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
    return x_train, x_test, y_train, y_test


def training_models(x_train, x_test, y_train, y_test):
    # decision tree
    dtree = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=3).fit(x_train, y_train)
    btree = BaggingClassifier(dtree,random_state=1).fit(x_train, y_train)
    atree = AdaBoostClassifier(random_state=1).fit(x_train, y_train)
    gtree = GradientBoostingClassifier(random_state=1).fit(x_train, y_train)
    rtree = RandomForestClassifier(random_state=1).fit(x_train, y_train)

    return dtree, btree, atree, gtree, rtree


def measuringmodel_performance(df, target_feature, model, x_train, x_test, y_train, y_test):
    # measuring the model performence
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m = accuracy_score(y_test,
                                                                                     y_predict), precision_score(y_test,
                                                                                                                 y_predict), recall_score(
        y_test, y_predict), f1_score(y_test, y_predict)
    return accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score


def pickling_model(model):
    with open('filename.pkl', 'wb') as f:
        pickle.dump(model, f)




def main_combined(path,target_feature):
    df = preprocess_data(path,target_feature)
    x_train, x_test, y_train, y_test = split_data_scaled(df, target_feature)
    dtree, btree, atree, gtree, rtree = training_models(x_train, x_test, y_train, y_test)
    tree_type = ['decision_tree', 'BaggingClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier']
    mds = [dtree, btree, atree, gtree, rtree]
    f_test_score = 0
    f_model_number = 0
    for m in range(len(mds)):
        accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[m], x_train, x_test, y_train, y_test)
        if f_test_score < test_score:
            f_model_number = m

    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[f_model_number], x_train, x_test, y_train, y_test)
    pickling_model(mds[f_model_number])
    print('We need to select',tree_type[f_model_number], '. Since it is having higher accuracy for test data.' )
    print('Model performance details mentioned below')
    print('Train data accuracy:', train_score)
    print('Test data accuracy:',test_score )
    print('Accuracy score:', accuracy_score_m)
    print('precision score:', precision_score_m)
    print('recall score:', recall_score_m)
    print('f1 score:', f1_score_m)


main_combined('/Users/pradeep/Downloads/diabetes.csv','Outcome')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


# Function to preprocess the data
def preprocess_data(path, target_feature, filetype='csv'):
    if filetype == 'csv':
        df = pd.read_csv(path)
    elif filetype == 'tsv':
        df = pd.read_csv(path, sep='\t')

    for cl in df.columns:
        if df[cl].isnull().values.any() > 0:
            # nan value replaced with mode for objects
            if str(df[cl].dtype) in ['object']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mode')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
            # nan value replaced with mean for numbers
            elif str(df[cl].dtype) in ['int64', 'float64']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
    # object must be converted as categorical
    for cl in df.columns:
        if str(df[cl].dtype) == 'object':
            df[cl] = pd.Categorical(df[cl])

    # drop duplicate values
    df = df.drop_duplicates(keep=False)

    # Dropping unnecessary features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(df[to_drop], axis=1)

    return df


def split_data_scaled(df, target_feature):
    x, y = df.loc[:, df.columns != target_feature], df[target_feature]
    # outliers handling
    x[x.select_dtypes(include=[np.number]).columns].apply(zscore)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
    return x_train, x_test, y_train, y_test


def training_models(x_train, x_test, y_train, y_test):
    # decision tree
    dtree = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=3).fit(x_train, y_train)
    btree = BaggingClassifier(dtree,random_state=1).fit(x_train, y_train)
    atree = AdaBoostClassifier(random_state=1).fit(x_train, y_train)
    gtree = GradientBoostingClassifier(random_state=1).fit(x_train, y_train)
    rtree = RandomForestClassifier(random_state=1).fit(x_train, y_train)

    return dtree, btree, atree, gtree, rtree


def measuringmodel_performance(df, target_feature, model, x_train, x_test, y_train, y_test):
    # measuring the model performence
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m = accuracy_score(y_test,
                                                                                     y_predict), precision_score(y_test,
                                                                                                                 y_predict), recall_score(
        y_test, y_predict), f1_score(y_test, y_predict)
    return accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score


def pickling_model(model):
    with open('filename.pkl', 'wb') as f:
        pickle.dump(model, f)




def main_combined(path,target_feature):
    df = preprocess_data(path,target_feature)
    x_train, x_test, y_train, y_test = split_data_scaled(df, target_feature)
    dtree, btree, atree, gtree, rtree = training_models(x_train, x_test, y_train, y_test)
    tree_type = ['decision_tree', 'BaggingClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier']
    mds = [dtree, btree, atree, gtree, rtree]
    f_test_score = 0
    f_model_number = 0
    for m in range(len(mds)):
        accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[m], x_train, x_test, y_train, y_test)
        if f_test_score < test_score:
            f_model_number = m

    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[f_model_number], x_train, x_test, y_train, y_test)
    pickling_model(mds[f_model_number])
    print('We need to select',tree_type[f_model_number], '. Since it is having higher accuracy for test data.' )
    print('Model performance details mentioned below')
    print('Train data accuracy:', train_score)
    print('Test data accuracy:',test_score )
    print('Accuracy score:', accuracy_score_m)
    print('precision score:', precision_score_m)
    print('recall score:', recall_score_m)
    print('f1 score:', f1_score_m)
# Part 2 - Build a machine learning pipeline that will run autonomously with the csv file and return bestperforming model

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier# Part 2 - Build a machine learning pipeline that will run autonomously with the csv file and return bestperforming model

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


# Function to preprocess the data
def preprocess_data(path, target_feature, filetype='csv'):
    if filetype == 'csv':
        df = pd.read_csv(path)
    elif filetype == 'tsv':
        df = pd.read_csv(path, sep='\t')

    for cl in df.columns:
        if df[cl].isnull().values.any() > 0:
            # nan value replaced with mode for objects
            if str(df[cl].dtype) in ['object']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mode')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
            # nan value replaced with mean for numbers
            elif str(df[cl].dtype) in ['int64', 'float64']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
    # object must be converted as categorical
    for cl in df.columns:
        if str(df[cl].dtype) == 'object':
            df[cl] = pd.Categorical(df[cl])

    # drop duplicate values
    df = df.drop_duplicates(keep=False)

    # Dropping unnecessary features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(df[to_drop], axis=1)

    return df


def split_data_scaled(df, target_feature):
    x, y = df.loc[:, df.columns != target_feature], df[target_feature]
    # outliers handling
    x[x.select_dtypes(include=[np.number]).columns].apply(zscore)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
    return x_train, x_test, y_train, y_test


def training_models(x_train, x_test, y_train, y_test):
    # decision tree
    dtree = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=3).fit(x_train, y_train)
    btree = BaggingClassifier(dtree,random_state=1).fit(x_train, y_train)
    atree = AdaBoostClassifier(random_state=1).fit(x_train, y_train)
    gtree = GradientBoostingClassifier(random_state=1).fit(x_train, y_train)
    rtree = RandomForestClassifier(random_state=1).fit(x_train, y_train)

    return dtree, btree, atree, gtree, rtree


def measuringmodel_performance(df, target_feature, model, x_train, x_test, y_train, y_test):
    # measuring the model performence
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m = accuracy_score(y_test,
                                                                                     y_predict), precision_score(y_test,
                                                                                                                 y_predict), recall_score(
        y_test, y_predict), f1_score(y_test, y_predict)
    return accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score


def pickling_model(model):
    with open('filename.pkl', 'wb') as f:
        pickle.dump(model, f)




def main_combined(path,target_feature):
    df = preprocess_data(path,target_feature)
    x_train, x_test, y_train, y_test = split_data_scaled(df, target_feature)
    dtree, btree, atree, gtree, rtree = training_models(x_train, x_test, y_train, y_test)
    tree_type = ['decision_tree', 'BaggingClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier']
    mds = [dtree, btree, atree, gtree, rtree]
    f_test_score = 0
    f_model_number = 0
    for m in range(len(mds)):
        accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[m], x_train, x_test, y_train, y_test)
        if f_test_score < test_score:
            f_model_number = m

    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[f_model_number], x_train, x_test, y_train, y_test)
    pickling_model(mds[f_model_number])
    print('We need to select',tree_type[f_model_number], '. Since it is having higher accuracy for test data.' )
    print('Model performance details mentioned below')
    print('Train data accuracy:', train_score)
    print('Test data accuracy:',test_score )
    print('Accuracy score:', accuracy_score_m)
    print('precision score:', precision_score_m)
    print('recall score:', recall_score_m)
    print('f1 score:', f1_score_m)


main_combined('/Users/pradeep/Downloads/diabetes.csv','Outcome')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


# Function to preprocess the data
def preprocess_data(path, target_feature, filetype='csv'):
    if filetype == 'csv':
        df = pd.read_csv(path)
    elif filetype == 'tsv':
        df = pd.read_csv(path, sep='\t')

    for cl in df.columns:
        if df[cl].isnull().values.any() > 0:
            # nan value replaced with mode for objects
            if str(df[cl].dtype) in ['object']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mode')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
            # nan value replaced with mean for numbers
            elif str(df[cl].dtype) in ['int64', 'float64']:
                imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
                df[cl] = imputer.fit_transform(df[cl].values.reshape(-1, 1))[:, 0]
    # object must be converted as categorical
    for cl in df.columns:
        if str(df[cl].dtype) == 'object':
            df[cl] = pd.Categorical(df[cl])

    # drop duplicate values
    df = df.drop_duplicates(keep=False)

    # Dropping unnecessary features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(df[to_drop], axis=1)

    return df


def split_data_scaled(df, target_feature):
    x, y = df.loc[:, df.columns != target_feature], df[target_feature]
    # outliers handling
    x[x.select_dtypes(include=[np.number]).columns].apply(zscore)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
    return x_train, x_test, y_train, y_test


def training_models(x_train, x_test, y_train, y_test):
    # decision tree
    dtree = DecisionTreeClassifier(criterion='gini', random_state=1, max_depth=3).fit(x_train, y_train)
    btree = BaggingClassifier(dtree,random_state=1).fit(x_train, y_train)
    atree = AdaBoostClassifier(random_state=1).fit(x_train, y_train)
    gtree = GradientBoostingClassifier(random_state=1).fit(x_train, y_train)
    rtree = RandomForestClassifier(random_state=1).fit(x_train, y_train)

    return dtree, btree, atree, gtree, rtree


def measuringmodel_performance(df, target_feature, model, x_train, x_test, y_train, y_test):
    # measuring the model performence
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m = accuracy_score(y_test,
                                                                                     y_predict), precision_score(y_test,
                                                                                                                 y_predict), recall_score(
        y_test, y_predict), f1_score(y_test, y_predict)
    return accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score


def pickling_model(model):
    with open('filename.pkl', 'wb') as f:
        pickle.dump(model, f)




def main_combined(path,target_feature):
    df = preprocess_data(path,target_feature)
    x_train, x_test, y_train, y_test = split_data_scaled(df, target_feature)
    dtree, btree, atree, gtree, rtree = training_models(x_train, x_test, y_train, y_test)
    tree_type = ['decision_tree', 'BaggingClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier']
    mds = [dtree, btree, atree, gtree, rtree]
    f_test_score = 0
    f_model_number = 0
    for m in range(len(mds)):
        accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[m], x_train, x_test, y_train, y_test)
        if f_test_score < test_score:
            f_model_number = m

    accuracy_score_m, precision_score_m, recall_score_m, f1_score_m, train_score, test_score = measuringmodel_performance(df, target_feature, mds[f_model_number], x_train, x_test, y_train, y_test)
    pickling_model(mds[f_model_number])
    print('We need to select',tree_type[f_model_number], '. Since it is having higher accuracy for test data.' )
    print('Model performance details mentioned below')
    print('Train data accuracy:', train_score)
    print('Test data accuracy:',test_score )
    print('Accuracy score:', accuracy_score_m)
    print('precision score:', precision_score_m)
    print('recall score:', recall_score_m)
    print('f1 score:', f1_score_m)


main_combined('/Users/pradeep/Downloads/diabetes.csv','Outcome')

main_combined('/Users/pradeep/Downloads/diabetes.csv','Outcome')