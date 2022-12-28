
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV, RFE
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, roc_auc_score, roc_curve


IncepModel_train = pd.read_csv('DLR_IncepModel_train.csv',
                             header=None)
IncepModel_test = pd.read_csv('DLR_IncepModel_test.csv',
                            header=None)

RF_df_train = pd.read_csv(
    r'E:\ich\FinalMasterProject-AHHM-master\(Radiomics+Clinical)_Training.csv')
print(RF_df_train.shape)
RF_df_test = pd.read_csv(
    r'E:\ich\FinalMasterProject-AHHM-master\(Radiomics+Clinical)_testing.csv')
print(RF_df_test.shape)


RF_df_train = RF_df_train.loc[:, ~
                              RF_df_train.columns.str.startswith('diagnostics')]
RF_df_test = RF_df_test.loc[:, ~
                            RF_df_test.columns.str.startswith('diagnostics')]
print(RF_df_train.shape)
print(RF_df_test.shape)

radiomics_train = RF_df_train.filter(regex='train')


#POS_test_df = test_df.filter(regex='positive')
#NEG_test_df = test_df.filter(regex='negitive')


#POS_test_df = test_df.filter(regex='positive')
#NEG_test_df = test_df.filter(regex='negitive')


radiomics_test = RF_df_test.filter(regex='test')

y_train = RF_df_train['class']
y_test = RF_df_test['class']


#path = r'C:\Users\zy\Desktop\FinalMasterProject-AHHM-master\ACDC Dataset\2. Deeply Learned Features'
print(y_train.value_counts())
class_list = list(y_train.unique())


def reshapematrix(inputmatrix):

    x = pd.DataFrame(inputmatrix)

    y = x.values

    a = x.shape[0]
    b = x.shape[1]

    print(a/3)
    print(b*3)

    y = y.reshape(int(a/3), int(b*3))  # 603/3 4096*3

    y = pd.DataFrame(y)

    return y


IncepModel_train = reshapematrix(IncepModel_train)
IncepModel_test = reshapematrix(IncepModel_test)


#all_data_train = pd.concat([rad_train, med_info_train], axis=1)  
#all_data_test = pd.concat([rad_test, med_info_test], axis=1)
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


#all_data_train = rad_train
#all_data_test = rad_train

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# #### Models

#K-Best

def KBest_GS(X_train, y_train, X_test, y_test, model, param_grid, x_df):

    selector = SelectKBest()

    ### Pipeline

    ### we would need to adapt the "NUMBER OF FEATURES PARAMETER OF THE GRID"

    pipe = Pipeline([('selector', selector),
                     ('model', model)])

    featss = np.array(range(2, X_train.shape[1]))

    #featss = np.array(range(5,8))

    dict_1 = {'selector__score_func': [f_classif, chi2],
              'selector__k': featss}  # para pruebas

    dict_1.update(param_grid)

    gs = GridSearchCV(estimator=pipe,
                      param_grid=dict_1,
                      scoring=['accuracy', 'roc_auc',
                               'f1', 'precision', 'recall'],
                      n_jobs=1,
                      cv=StratifiedKFold(3, shuffle=True, random_state=42),
                      refit='accuracy',
                      verbose=1)

    print(pipe.get_params().keys())

    gs = gs.fit(X_train, y_train)

    print("Best Model", gs.best_params_)

    print('Best score:', gs.best_score_)

    y_test_pred = gs.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    probs = gs.predict_proba(X_test_p)
    probs = probs[:, 1]
    ROC_AUC = roc_auc_score(y_test, probs)
    precison_scor = precision_score(y_test, y_test_pred)
    recall_scor = recall_score(y_test, y_test_pred)

    fpr, tpr, thresholds = roc_curve(y_testp, probs)

    print("\n Test Accuracy with best estimator: ", test_acc)
    print("\n Roc AUC with best estimator: ", ROC_AUC)
    print("\n Precision with best estimator: ", precison_scor)
    print("\n Recall with best estimator: ", recall_scor, "\n")

    l = [[test_acc, ROC_AUC, precison_scor, recall_scor]]
    table = tabulate(
        l, headers=['Accuracy', 'ROC_AUC', 'Precision', 'Recall'], tablefmt='orgtbl')

    print(table)

    cm = confusion_matrix(y_test, y_test_pred)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(10, 4))
    fig.add_subplot(121)
#     plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.tight_layout()

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks+0.5, class_list, rotation=45)
    plt.yticks(tick_marks+0.5, class_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig.add_subplot(122)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.tight_layout()

    plt.show()

    print(classification_report(y_test, y_test_pred, target_names=class_list))

    cols = gs.best_estimator_.steps[0][1].k_feature_idx_
    cols =list(cols)
#  K_best = list(features_df_new.columns)

    print(K_best)

    return gs, K_best
    


# ---
# 
# #### Sequential Foward Selection
def SFS_GS(X_train, y_train, X_test, y_test, model, param_grid, df):
    #Setting up the SFS
    sfs1 = SFS(estimator=model,
               k_features=X_train.shape[1],
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=StratifiedKFold(3, shuffle=True, random_state=42))

    ### Pipeline

    ### we would need to adapt the "NUMBER OF FEATURES PARAMETER OF THE GRID"

    pipe = Pipeline([('sfs', sfs1),
                     ('model', model)])

    #dict_1 = {'sfs__k_features':list(range(1,X_train.shape[1]))}   #### para pruebas

    dict_1 = {'sfs__k_features': [20,25,30,50]}  # Testing

    dict_1.update(param_grid)

    gs = GridSearchCV(estimator=pipe,
                      param_grid=dict_1,
                      scoring='accuracy',
                      n_jobs=-1,
                      cv=StratifiedKFold(3, shuffle=True, random_state=42),
                      verbose=3,
                      refit=True)

    print(pipe.get_params().keys())

    gs = gs.fit(X_train, y_train)

#     print(gs.best_estimator_.steps)

    print("Best Model", gs.best_params_)

    print('Best score:', gs.best_score_)

    y_test_pred = gs.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    probs = gs.predict_proba(X_test_p)
    probs = probs[:, 1]
    ROC_AUC = roc_auc_score(y_test, probs)
    precison_scor = precision_score(y_test, y_test_pred)
    recall_scor = recall_score(y_test, y_test_pred)

    fpr, tpr, thresholds = roc_curve(y_testp, probs)

    print("\n Test Accuracy with best estimator: ", test_acc)
    print("\n Roc AUC with best estimator: ", ROC_AUC)
    print("\n Precision with best estimator: ", precison_scor)
    print("\n Recall with best estimator: ", recall_scor, "\n")

    l = [[test_acc, ROC_AUC, precison_scor, recall_scor]]
    table = tabulate(
        l, headers=['Accuracy', 'ROC_AUC', 'Precision', 'Recall'], tablefmt='orgtbl')

    print(table)

    cm = confusion_matrix(y_test, y_test_pred)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(10, 4))
    fig.add_subplot(121)
#     plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.tight_layout()

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks+0.5, class_list, rotation=45)
    plt.yticks(tick_marks+0.5, class_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig.add_subplot(122)
    plt.plot(fpr, tpr, color='C1', lw=3,
             label='ROC curve of fusion radiomics and incepModel (AUC = %0.2f)' % ROC_AUC)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.tight_layout()

    plt.show()

    print(classification_report(y_test, y_test_pred, target_names=class_list))

    cols = gs.best_estimator_.steps[0][1].k_feature_idx_
    cols = list(cols)
    print(cols)
    pipe = df.iloc[:, cols].columns
    return gs, pipe


def processing(X_train, y_train, X_test, y_test):
    #tools scaling and labelling
    scaler = MinMaxScaler()
    encoder = LabelEncoder()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    return X_train, y_train, X_test, y_test

#Support Vector Classifier


model_SVC = SVC(gamma='scale', max_iter=5000, random_state=42)

param_grid_SVC = {'model__kernel': ('linear', 'rbf'),
                  'model__C': [5, 10]}

param_grid_SVC_nested_2 = {'selector__estimator__kernel': ['linear', 'rbf'],
                           'selector__estimator__C': [15]}

param_grid_SVC_test_2 = {
    'estimator__model__C': [0.5, 1, 5, 10]}


#-------------------------------------------------------

#Random Forest

# Number of trees in random forest
n_estimators = [10, 100, 1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2, 4, 6, 8, 10]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid

param_grid_RF = {'model__n_estimators': n_estimators,
                 'model__max_features': max_features,
                 'model__max_depth': max_depth,
                 'model__min_samples_split': min_samples_split,
                 'model__min_samples_leaf': min_samples_leaf,
                 'model__bootstrap': bootstrap}

param_grid_RF_2 = {'estimator__n_estimators': n_estimators,
                   'estimator__bootstrap': bootstrap}

model_RF = RandomForestClassifier(random_state=42)

#--------------------------------------------------------------

# Logistic Regression

param_grid_LR_nested = {'model__penalty': ['l1', 'l2'],
                        'model__C': [0.1, 1, 10, 100, 200]}

param_grid_LR = {'penalty': ['l1', 'l2'],
                 'C': [0.1, 1, 10, 100, 200]}

model_LR = LogisticRegression(multi_class='auto', random_state=42)

model_SVC = SVC(gamma='scale', probability=True,
                max_iter=5000, random_state=42)

param_grid_SVC = {'model__kernel': ('linear', 'rbf'),
                  'model__C': [0.3, 0.5, 1, 5, 10]}

scaler = MinMaxScaler()
X_sc = scaler.fit_transform(IncepModel_train)

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y_train)

X_train_p, X_test_p, y_trainp, y_testp = train_test_split(X_sc, y_enc, test_size=100,shuffle=True,stratify=y_train,random_state=42)

gs, K_best =SFS_GS(X_train_p, y_trainp, X_test_p,
                   y_testp, model_SVC, param_grid_SVC, IncepModel_train)


top15 = pd.Series(abs(gs.best_estimator_.named_steps.model.coef_[0]), index=K_best).nlargest(15)


top15.sort_values(ascending=True, inplace=True)
plt.figure(figsize=(12, 6))
top15.plot(kind='barh', title='Top 15 Features Selected')
plt.ylabel('Feature name')
plt.xlabel('Coefficient')
plt.show()
#ED Alone
#X_train_POS, y_train, X_test_POS, y_test = processing(POS_train_df, y_train, POS_test_df, y_test)

#ES Alon

#X_train_NEG, y_train, X_test_NEG, y_test = processing(NEG_train_df, y_train, NEG_test_df, y_test)

#Radiomics 

#X_train_rad, y_train, X_test_rad, y_test = processing(rad_train, y_train, rad_test, y_test)

#All data

#X_train_all, y_train, X_test_all, y_test = processing(all_data_train, y_train, all_data_test, y_test)


# ### 3. ML Algorithms


#-------------------------------------------------------------------

#gs = KBest_GS(X_train_rad, y_train, X_test_rad,
#             y_test, model_SVC, param_grid_SVC)
#gs = KBest_GS(X_train_ED, y_train, X_test_ED, y_test, model_RF, param_grid_RF)
#gs, K_best = KBest_GS(X_train_rad, y_train, X_test_rad, y_test, model_SVC, param_grid_SVC, rad_train)
#gs, K_best = SFS_GS(X_train_rad, y_train, X_test_rad,y_test, model_SVC, param_grid_SVC)

#gs = KBest_GS(X_train_rad, y_train, X_test_rad, y_test, model_SVC, param_grid_SVC, rad_train)

