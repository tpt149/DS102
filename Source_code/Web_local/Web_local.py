import streamlit as st
import hydralit_components as hc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import openpyxl
from sklearn.naive_bayes import MultinomialNB
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xlsxwriter
from io import BytesIO
import joblib

#make it look nice from the start
st.set_page_config(page_title='Gender Predict', page_icon='🚀', layout='wide',initial_sidebar_state='collapsed')

# specify the primary menu definition
menu_data = [
    {'icon': "🦊", 'label':"Naive Bayes"},
    {'icon':"🐻",'label':"SVM"},
    {'icon': "🐼",'label':"Logistic Regression"},
    {'icon': "🐨", 'label':"KNN"},#no tooltip message
    {'icon': "🐯", 'label':"Decision Tree"},
    {'icon': "🐥", 'label':"RandomForest"}, #can add a tooltip message
    {'icon': "🐳", 'label':"VotingClassifier"},
    {'icon': "📗",'label':"Test_FullName", 'submenu':[{'icon': "🌤️", 'label':"Enter Your Name"},{'icon': "🌻", 'label':"Enter Your File (Excel)"}]},
]

#over_theme = {'txc_inactive': '#FFFFFF'}
over_theme = {'menu_background':'#22e0dd','txc_active':'black','option_active':'white'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Dataset',
    sticky_nav=True, 
    sticky_mode='pinned', 
    #hide_streamlit_markers=False
)
#Preprocessing Data
def Preprocessing(name):
  name = ' '.join(name.split())
  name = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), name, flags=re.I)
  name = name.lower()
  name = re.sub(r"[-()\\\"#/@;:<>{}`+=~|.!?,%/0123456789]", "", name)
  name = re.sub('\n', ' ', name)
  return name

def Train_model(model, name_model):
    if name_model[-2:]=='cv':
        model.fit(X_train_cv, y_train)
        y_pred = model.predict(X_test_cv)
    else: 
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
    return y_pred

def Measure_model(y_test, y_pred):
    
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1_avg = f1_score(y_test, y_pred, average='macro')
    
    return acc, pre, recall, f1_avg

def Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf):
    acc_cv, pre_cv, recall_cv, f1_cv_avg = Measure_model(y_test, y_pred_cv)
    acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg = Measure_model(y_test, y_pred_tfidf)
    col_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    idx_names = ['CountVectorizer', 'TfidfVectorizer']
    values = [[acc_cv, pre_cv, recall_cv, f1_cv_avg],
            [acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg]]
    table = pd.DataFrame(values, columns=col_names, index=idx_names)
    st.table(table)
    return [acc_cv, pre_cv, recall_cv, f1_cv_avg], [acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg]
    
def Plot_confusion_matrix(model_cv, model_tfidf, model_name):
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
    plot_confusion_matrix(model_cv, X_test_cv, y_test, ax=ax1,cmap='Blues')
    ax1.title.set_text(f'Count Vector + {model_name}')
    plot_confusion_matrix(model_tfidf, X_test_tfidf, y_test, ax=ax2,cmap='Blues')
    ax2.title.set_text(f'TF-IDF + {model_name}')
    for _ in range(7):
        st.write('\n')
    st.pyplot(fig)
   
def Plot_bar_chart(measure_cv, measure_tfidf):
    x_label = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    countvector = go.Bar(x=x_label, y=measure_cv, name='CountVector')
    tfidfvector = go.Bar(x=x_label, y=measure_tfidf, name='TfidfVector')
    fig = go.Figure(data=[countvector, tfidfvector], layout=go.Layout(
            template='plotly_white',
            barmode = 'group'))
    st.plotly_chart(fig)
st.title('Gender Prediction Based on Vietnamese Names with Machine Learning')

#@st.experimental_memo
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def Data_initialize():
    ethnic = pd.read_excel('Datasets/Data_ethnic.xlsx')
    data = pd.read_csv('Datasets/UIT-ViNames.csv')
    df = pd.concat((data, ethnic), ignore_index=True)

    data_female = pd.concat((data[data['Gender']==0].reset_index(drop=True), ethnic[ethnic['Gender']==0].reset_index(drop=True)), axis=0, ignore_index=True)
    data_male = pd.concat((data[data['Gender']==1].reset_index(drop=True), ethnic[ethnic['Gender']==1].reset_index(drop=True)), axis=0, ignore_index=True)

    X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(data_female['Full_Names'], data_female['Gender'], test_size=0.3, random_state=42)
    X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(data_male['Full_Names'], data_male['Gender'], test_size=0.3, random_state=42)

    X_train = pd.concat((X_train_female, X_train_male), ignore_index=True)
    X_test = pd.concat((X_test_female, X_test_male), ignore_index=True)
    y_train = pd.concat((y_train_female, y_train_male), ignore_index=True)
    y_test = pd.concat((y_test_female, y_test_male), ignore_index=True)
        
    # preprocessing for training set
    for i in range(len(X_train)):
        X_train[i] = Preprocessing(X_train[i])
            
    # preprocessing for test set
    for i in range(len(X_test)):
        X_test[i] = Preprocessing(X_test[i])
            
    # Use CountVectorizer to encode data
    st.session_state.encode_cv = CountVectorizer()
    X_train_cv = st.session_state.encode_cv.fit_transform(X_train)
    X_test_cv = st.session_state.encode_cv.transform(X_test)
        
    # Use TfidfVectorizer to encode data
    st.session_state.encode_tfidf = TfidfVectorizer()
    X_train_tfidf = st.session_state.encode_tfidf.fit_transform(X_train)
    X_test_tfidf = st.session_state.encode_tfidf.transform(X_test)
    
    return df, data_female, data_male, X_train_cv, X_test_cv, X_train_tfidf, X_test_tfidf, y_train, y_test

# Dataset
if 'encode_cv' not in st.session_state:
    st.session_state['encode_cv'] = CountVectorizer()
if 'encode_tfidf' not in st.session_state:
    st.session_state['encode_tfidf'] = TfidfVectorizer()
df, data_female, data_male, X_train_cv, X_test_cv, X_train_tfidf, X_test_tfidf, y_train, y_test = Data_initialize()

#Initialize model
#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
@st.experimental_memo
def Initialize_model():
    NB_model_cv, NB_model_tfidf = joblib.load('Model_package/NB_model_cv.h5'), joblib.load('Model_package/NB_model_tfidf.h5')
    LR_model_cv, LR_model_tfidf = joblib.load('Model_package/LR_model_cv.h5'), joblib.load('Model_package/LR_model_tfidf.h5')
    SVM_model_cv, SVM_model_tfidf = joblib.load('Model_package/SVM_model_cv.h5'), joblib.load('Model_package/SVM_model_tfidf.h5')
    KNN_model_cv, KNN_model_tfidf = joblib.load('Model_package/KNN_model_cv.h5'), joblib.load('Model_package/KNN_model_tfidf.h5')
    DT_model_cv, DT_model_tfidf = joblib.load('Model_package/DT_model_cv.h5'), joblib.load('Model_package/DT_model_tfidf.h5')
    RF_model_cv, RF_model_tfidf = joblib.load('Model_package/RF_model_cv.h5'), joblib.load('Model_package/RF_model_tfidf.h5')
    Voting_clf_cv, Voting_clf_tfidf = joblib.load('Model_package/Voting_clf_cv.h5'), joblib.load('Model_package/Voting_clf_tfidf.h5')
    return NB_model_cv, NB_model_tfidf, LR_model_cv, LR_model_tfidf, SVM_model_cv, SVM_model_tfidf, KNN_model_cv, KNN_model_tfidf, DT_model_cv, DT_model_tfidf, RF_model_cv, RF_model_tfidf, Voting_clf_cv, Voting_clf_tfidf

NB_model_cv, NB_model_tfidf, LR_model_cv, LR_model_tfidf, SVM_model_cv, SVM_model_tfidf, KNN_model_cv, KNN_model_tfidf, DT_model_cv, DT_model_tfidf, RF_model_cv, RF_model_tfidf, Voting_clf_cv, Voting_clf_tfidf = Initialize_model()

if menu_id == 'Dataset':
    col1, col2, col3 = st.columns((1,2,1))
    with col1:
        st.dataframe(df)
    with col3:
        for i in range(13):
            st.write('\n')
        st.write(f'Giới tính nữ: {len(data_female)} ({round(len(data_female)/len(df) *100,2)}%)')
        st.write(f'Giới tính nam: {len(data_male)} ({round(len(data_male)/len(df) *100,2)}%)')
        st.write(f'Điểm dữ liệu: {len(df)}')
    with col2:
        gender_female = go.Bar(x=['Female', 'Male'], y=[len(data_female)], name='Female')
        gender_male = go.Bar(x=['Male'], y=[len(data_male)], name='Male')
        fig = go.Figure(data=[gender_female, gender_male], layout=go.Layout(barmode = 'group'))
        st.plotly_chart(fig)

_,center,_ = st.columns(3)
# Naive Bayes model
if menu_id == 'Naive Bayes':
    with center:
        st.header('Naive Bayes Model')
        # List parameters of Naive Bayes model
        st.subheader('Select parameter')
        alpha = st.number_input('alpha : float, default=1.0', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        fit_prior = st.radio('fit_prior : bool, default=True', ['True', 'False'])
        class_prior = st.text_input('class_prior : array-like of shape (n_classes,), default=None', value=None)
        if class_prior != 'None':
            class_prior = float(class_prior)
        else: class_prior = None

        button2 = st.button('Run')
    if button2:
        with hc.HyLoader('Wait for it...😅',hc.Loaders.standard_loaders,index=[3,0,5]):
            if [alpha, fit_prior, class_prior] == [1.0, True, None]:
                y_pred_cv = NB_model_cv.predict(X_test_cv)
                y_pred_tfidf = NB_model_tfidf.predict(X_test_tfidf)
            else:
                model_cv = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
                model_tfidf = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
            
                y_pred_cv = Train_model(model_cv, 'NB_model_cv.h5')
                y_pred_tfidf = Train_model(model_tfidf, 'NB_model_tfidf.h5')

            measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
            
            col1, col2 = st.columns(2)
            with col1:
                Plot_confusion_matrix(NB_model_cv, NB_model_tfidf, model_name='Naive Bayes')
            with col2:
                Plot_bar_chart(measure_cv, measure_tfidf)

# SVM model
if menu_id == 'SVM':
    with center:
        st.header('Support Machine Vector Model')
        # List parameters of SVM model
        st.subheader('Select parameter')
        C = st.number_input('C : float, default=1.0', min_value=0.0, max_value=100.0, step=0.1, value=1.0)
        kernel = st.radio('kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
        if kernel == 'poly':
            degree = st.number_input('degree : int, default=3', value=3)
        else: degree = 3
        gamma = st.radio('gamma : {‘scale’, ‘auto’} or float, default=’scale’', ['scale', 'auto', 'float'])
        if gamma == 'float':
            gamma = st.number_input('Enter value of gamma', min_value=0.0, max_value=1.0)
        max_iter = st.number_input('max_iter : int, default=-1', min_value=-1, max_value=1000, step=1, value=-1)
        random_state = st.text_input('random_state : int, RandomState instance or None, default=None', value='None')
        if random_state != 'None':
            random_state = int(random_state)
        else: random_state = None
            
        button = st.button('Run SVM model')
    if button: 
        with hc.HyLoader('Wait for it...😅',hc.Loaders.standard_loaders,index=[3,0,5]):
            if [C, kernel, degree, gamma, max_iter, random_state] == [1.0, 'rbf', 3, 'scale', -1, None]:
                y_pred_cv = SVM_model_cv.predict(X_test_cv)
                y_pred_tfidf = SVM_model_tfidf.predict(X_test_tfidf)
            else:
            
                model_cv = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=max_iter, random_state=random_state)
                model_tfidf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=max_iter, random_state=random_state)

                y_pred_cv = Train_model(model_cv, 'SVM_model_cv')
                y_pred_tfidf = Train_model(model_tfidf, 'SVM_model_tfidf')

            measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
            
            col1, col2 = st.columns(2)
            with col1:
                Plot_confusion_matrix(SVM_model_cv, SVM_model_tfidf, model_name='SVM')
            with col2:
                Plot_bar_chart(measure_cv, measure_tfidf)

# Logistic Regression model
if menu_id == 'Logistic Regression':
    with center:
        st.header('Logistic Regression Model')
        # List parameters of Logistic Regression model
        st.subheader('Select parameter')
        penalty = st.radio('penalty : {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’', ['l1', 'l2', 'elasticnet', 'none'], index=1)
        C = st.number_input('C : float, default=1.0', min_value=0.0, max_value=1000.0, step=0.1, value=1.0)
        fit_intercept = st.radio('fit_interceptbool, default=True', [True, False])
        random_state = st.text_input('random_state : int, RandomState instance, default=None', value='None')
        if random_state != 'None':
            random_state = int(random_state)
        else: random_state = None
        solver = st.radio('solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], index=1)
        max_iter = st.number_input('max_iter : int, default=100', min_value=0, max_value=1000, step=1, value=100)
        button = st.button('Run Logistic Regression model')
    
    if button:
        with hc.HyLoader('Wait for it...😅',hc.Loaders.standard_loaders,index=[3,0,5]):
            if [penalty, C, fit_intercept, random_state, solver, max_iter] == ['l2', 1.0, True, None, 'lbfgs', 100]:
                y_pred_cv = LR_model_cv.predict(X_test_cv)
                y_pred_tfidf = LR_model_tfidf.predict(X_test_tfidf)
            else:
                model_cv = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, random_state=random_state, solver=solver ,max_iter=max_iter)
                model_tfidf = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, random_state=random_state, solver=solver, max_iter=max_iter)
                
                y_pred_cv = Train_model(model_cv, 'LR_model_cv')
                y_pred_tfidf = Train_model(model_tfidf, 'LR_model_tfidf')

            measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
            
            col1, col2 = st.columns(2)
            with col1:
                Plot_confusion_matrix(LR_model_cv, LR_model_tfidf, model_name='Logistic Regression')
            with col2:
                Plot_bar_chart(measure_cv, measure_tfidf)
        
# KNN model
if menu_id == 'KNN':
    with center:
        st.header('K-Nearest Neighbors Model')
        # List parameters of K-Nearest Neighbors model
        st.subheader('Select parameter')
        n_neighbors = st.number_input('n_neighbors : int, default=5', value=5)
        weight = st.radio('weights{‘uniform’, ‘distance’} or callable, default=’uniform’', ['uniform', 'distance'])
        algorithm = st.radio('algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf = st.number_input('leaf_sizeint, default=30', value=30)
        p = st.number_input('p : int, default=2', value=2)
        n_job = st.text_input('n_jobs : int, default=None', value=None)
        if n_job != 'None':
            n_job = int(n_job)
        else: n_job = None
        
        button = st.button('Run KNN model')
    if button:
        with hc.HyLoader('Wait for it...😅',hc.Loaders.standard_loaders,index=[3,0,5]):
            if [n_neighbors, weight, algorithm, leaf, p, n_job] == [5, 'uniform', 'auto', 30, 2, None]:
                y_pred_cv = KNN_model_cv.predict(X_test_cv)
                y_pred_tfidf = KNN_model_tfidf.predict(X_test_tfidf)
            else:
                model_cv = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight, algorithm=algorithm, leaf_size=leaf, p=p, n_jobs=n_job)
                model_tfidf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight, algorithm=algorithm, leaf_size=leaf, p=p, n_jobs=n_job)
                
                y_pred_cv = Train_model(model_cv, 'KNN_model_cv')
                y_pred_tfidf = Train_model(model_tfidf, 'KNN_model_tfidf')

            measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
            
            col1, col2 = st.columns(2)
            with col1:
                Plot_confusion_matrix(KNN_model_cv, KNN_model_tfidf, model_name='KNN')
            with col2:
                Plot_bar_chart(measure_cv, measure_tfidf)

# Decision Tree model
if menu_id == 'Decision Tree':
    with center:
        st.header('Decision Tree Model')
        # List parameters of Decision Tree model
        st.subheader('Select parameter')
        criterion = st.radio('criterion{“gini”, “entropy”, “log_loss”}, default=”gini”', ['gini', 'entropy', 'log_loss'])
        splitter = st.radio('splitter : {“best”, “random”}, default=”best”', ['best', 'random'])
        max_depth = st.text_input('max_dept : hint, default=None', value=None)
        if max_depth != 'None':
            max_depth = int(max_depth)
        else: max_depth = None
        min_samples_split = st.number_input('min_samples_split : int or float, default=2', value=2)
        min_samples_leaf = st.number_input('min_samples_leaf : int or float, default=1', value=1)
        min_weight_fraction_leaf = st.number_input('min_weight_fraction_leaf : float, default=0.0', min_value=0.0, max_value=1.0, value=0.0)
        max_features = st.radio('max_features : int, float or {“auto”, “sqrt”, “log2”}, default=None', ['int', 'float', 'auto', 'sqrt', 'log2', None], index=5)
        if max_features == 'int' or max_features == 'float':
            max_features = st.number_input('Enter value for max_features')
        random_state = st.text_input('random_state : int, RandomState instance or None, default=None', value=0, key='tab6_random_state')
        if random_state != 'None':
            random_state = int(random_state)
        else: random_state = None
        max_leaf_nodes = st.text_input('max_leaf_nodes : int, default=None', value=None)
        if max_leaf_nodes != 'None':
            max_leaf_nodes = int(random_state)
        else: max_leaf_nodes = None
        
        button = st.button('Run Decision Tree model')
    if button:
        with hc.HyLoader('Wait for it...😅',hc.Loaders.standard_loaders,index=[3,0,5]):
            if [criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes] == ['gini', 'best', None, 2, 1, 0.0, None, 0, None]:
                y_pred_cv = DT_model_cv.predict(X_test_cv)
                y_pred_tfidf = DT_model_tfidf.predict(X_test_tfidf)
            else:
                model_cv = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                    random_state=random_state, max_leaf_nodes=max_leaf_nodes)
                model_tfidf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                    random_state=random_state, max_leaf_nodes=max_leaf_nodes)
                y_pred_cv = Train_model(model_cv, 'DT_model_cv')
                y_pred_tfidf = Train_model(model_tfidf, 'DT_model_tfidf')

            measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
            
            col1, col2 = st.columns(2)
            with col1:
                Plot_confusion_matrix(DT_model_cv, DT_model_tfidf, model_name='Decision Tree')
            with col2:
                Plot_bar_chart(measure_cv, measure_tfidf)
        
# Random Forest model
if menu_id == 'RandomForest':
    with center:
        st.header('Random Forest Model')
        # List parameters of Random Forest model
        st.subheader('Select parameter')
        n_estimators = st.number_input('n_estimators : int, default=100', value=100)
        criterion = st.radio('criterion{“gini”, “entropy”, “log_loss”}, default=”gini”', ['gini', 'entropy', 'log_loss'], key='randomforest')
        max_depth = st.text_input('max_depth : int, default=None', value=None)
        if max_depth != 'None':
            max_depth = int(max_depth)
        else: max_depth = None
        min_samples_split = st.number_input('min_samples_split : int or float, default=2', value=2, key='tab7_min_samples_split')
        min_samples_leaf = st.number_input('min_samples_leaf : int or float, default=1', value=1, key='tab7_min_samples_leaf')
        min_weight_fraction_leaf = st.number_input('min_weight_fraction_leaf : float, default=0.0', min_value=0.0, max_value=1.0, value=0.0, key='tab7_min_weight_fraction_leaf')
        max_features = st.radio('max_features : {“sqrt”, “log2”, None}, int or float, default=”sqrt”', ['sqrt', 'log2', None, 'int', 'float'])
        if max_features in ['int', 'float']:
            max_features = st.number_input('Enter value of max_features')
        max_leaf_nodes = st.text_input('max_leaf_nodes : int, default=None', value=None, key='tab7_max_leaf_nodes')
        if max_leaf_nodes != 'None':
            max_leaf_nodes = int(random_state)
        else: max_leaf_nodes = None
        bootstrap = st.radio('bootstrap : bool, default=True', [True, False])
        oob_score = st.radio('oob_score : bool, default=False', [True, False], index=1)
        n_jobs = st.text_input('n_jobs : int, default=None', value=None, key='tab7_n_jobs')
        if n_jobs != 'None':
            n_jobs = int(n_jobs)
        else: n_jobs = None
        random_state = st.text_input('random_state : int, RandomState instance or None, default=None', value=0, key='tab7_random_state')
        if random_state != 'None':
            random_state = int(random_state)
        else: random_state = None
        max_samples = st.radio('max_samples : int or float, default=None', ['int', 'float', None], index=2)
        if max_samples in ['int', 'float']:
            max_samples = st.number_input('Enter value of max_samples')
        
        button = st.button('Run Randomforest model')
    if button:
        with hc.HyLoader('Wait for it...😅',hc.Loaders.standard_loaders,index=[3,0,5]):
            if [n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
                max_features, max_leaf_nodes, bootstrap, oob_score, n_jobs, random_state, max_samples] == [100, 'gini', None, 2, 1, 0.0, 'sqrt', None, True, False, None, 0, None]:
                y_pred_cv = RF_model_cv.predict(X_test_cv)
                y_pred_tfidf = RF_model_tfidf.predict(X_test_tfidf)
            else:
                model_cv = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, max_samples=max_samples)
                model_tfidf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, max_samples=max_samples)
            
                y_pred_cv = Train_model(model_cv, 'RF_model_cv')
                y_pred_tfidf = Train_model(model_tfidf, 'RF_model_tfidf')

            measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
            
            col1, col2 = st.columns(2)
            with col1:
                Plot_confusion_matrix(RF_model_cv, RF_model_tfidf, model_name='RandomForest')
            with col2:
                Plot_bar_chart(measure_cv, measure_tfidf)
        
# Votingclassifier model
if menu_id == 'VotingClassifier':
    _, center, _ = st.columns(3)
    with center:
        st.subheader('Select 3 best model for VotingClassifier')
        name_models = ['Naive Bayes (Recommend)', 'Logistic Regression (Recommend)', 'Support Vector Machine (Recommend)', 'K-Nearest Neighbors', 'Decision Tree', 'RandomForest']
        select_model = []
        for name_id in range(len(name_models)):
            if name_id in [0,1,2]:
                select_model.append(st.checkbox(name_models[name_id], True))
            else: select_model.append(st.checkbox(name_models[name_id]))
        button = st.button('Run VotingClassifier')
    if button:
        columns = ['Naive Bayes', 'Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbors', 'Decision Tree', 'RandomForest']
        col = [columns[idx] for idx, name in enumerate(select_model) if name==True]
        score_cv, score_tfidf, estimate, ii = [], [], [], []
        with hc.HyLoader('Wait for it...😅',hc.Loaders.standard_loaders,index=[3,0,5]):
                
            name_model_cv = [NB_model_cv, LR_model_cv, SVM_model_cv, KNN_model_cv, DT_model_cv, RF_model_cv]
            name_model_tfidf = [NB_model_tfidf, LR_model_tfidf, SVM_model_tfidf, KNN_model_tfidf, DT_model_tfidf, RF_model_tfidf]
            
            SVM_model_cv.probability=True
            SVM_model_tfidf.probability=True
            
            for i in range(len(select_model)):
                if select_model[i]:
                    score_cv.append(round(f1_score(y_test, name_model_cv[i].predict(X_test_cv), average='macro'),4))
                    score_tfidf.append(round(f1_score(y_test, name_model_tfidf[i].predict(X_test_tfidf), average='macro'),4))
                    
            ii = [i for i, val in enumerate(select_model) if val==True]
            weight_cv = [1+ index for index in np.argsort(score_cv)]    #Bộ trọng số weight cho Voting
            weight_tfidf = [1+ index for index in np.argsort(score_tfidf)]
            
            if select_model == [True, True, True, False, False, False]:
                voting_clf_cv = Voting_clf_cv
                voting_clf_tfidf = Voting_clf_tfidf
            else:
                voting_clf_cv = VotingClassifier(estimators=[('model1',name_model_cv[ii[0]]),('model2',name_model_cv[ii[1]]),('model3',name_model_cv[ii[2]])], voting='soft',weights=weight_cv)
                voting_clf_tfidf = VotingClassifier(estimators=[('model1',name_model_tfidf[ii[0]]),('model2',name_model_tfidf[ii[1]]),('model3',name_model_tfidf[ii[2]])], voting='soft',weights=weight_tfidf)
                voting_clf_cv.fit(X_train_cv, y_train)
                voting_clf_tfidf.fit(X_train_tfidf, y_train)
            
            y_pred_cv = voting_clf_cv.predict(X_test_cv)
            y_pred_tfidf = voting_clf_tfidf.predict(X_test_tfidf)
            col.append('VotingClassifier')
            score_cv.append(round(f1_score(y_test, y_pred_cv, average='macro'),4))
            score_tfidf.append(round(f1_score(y_test, y_pred_tfidf, average='macro'),4))
            score = ['F1-score (CountVector)', 'F1-score (TfidfVector)']
            st.table(pd.DataFrame([score_cv, score_tfidf], columns=col, index=score))
            
            acc_cv, pre_cv, recall_cv, f1_cv_avg = Measure_model(y_test, y_pred_cv)
            acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg = Measure_model(y_test, y_pred_tfidf)
            col1, col2 = st.columns(2)
            with col1:
                Plot_confusion_matrix(voting_clf_cv, voting_clf_tfidf, model_name='VotingClassifier')
            with col2:
                Plot_bar_chart([acc_cv, pre_cv, recall_cv, f1_cv_avg], [acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg])
                
if menu_id == 'Enter Your Name':
    with center:
        st.markdown("<h1 style='text-align: center; color: grey;'>Enter Your Name</h1>", unsafe_allow_html=True)
        name = st.text_input('', label_visibility="collapsed")
        if st.button('Predict'):
            with st.spinner('Wait for it...'):
                name = Preprocessing(name)
                vector = st.session_state.encode_cv.transform(pd.Series(name))
                y_pred = Voting_clf_cv.predict(vector)
                if y_pred==0:
                    st.markdown("<h1 style='text-align: center; color: #F318DC;'>Gender is Female👩</h1>", unsafe_allow_html=True)
                else: st.markdown("<h1 style='text-align: center; color: #18B7F3;'>Gender is Male👨</h1>", unsafe_allow_html=True)

if menu_id == 'Enter Your File (Excel)':
    st.subheader('Upload File Fullname (xlsx):')
    file_upload = st.file_uploader('', type='XLSX')
    if file_upload is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Display Input Table')
            df = pd.read_excel(file_upload, header=None, names=['Full_Name'])
            st.dataframe(df)
        with col2:
            X_test = list(df.iloc[:,0])
            for i in range(len(X_test)):
                X_test[i] = Preprocessing(X_test[i])
            X_test = st.session_state.encode_cv.transform(X_test)
            y_pred = Voting_clf_cv.predict(X_test)
            st.subheader('Display Gender Predict:')
            table = pd.concat([df, pd.DataFrame(y_pred, columns=['Gender_Predict'])], axis=1)
            st.dataframe(table)
            
            output = BytesIO()
            workbook = xlsxwriter.Workbook(output, {'in_memory': True})
            worksheet = workbook.add_worksheet()
            for i in range(table.shape[0]):
                worksheet.write('A'+str(i+1), table.iloc[i,0])
                worksheet.write('B'+str(i+1), table.iloc[i,1])
            workbook.close()
            st.download_button(
                label="Download Result⬇️",
                data=output.getvalue(),
                file_name="Gender_Predict.xlsx",
                mime="application/vnd.ms-excel"
            )
                
        
        


    