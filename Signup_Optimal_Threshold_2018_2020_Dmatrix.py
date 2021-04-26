#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
Signupdata = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/Data/signup_2020_all_Final_Before_Encoding_new_After_Encode.csv', error_bad_lines=False)


# In[90]:


Signupdata_15042021 = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/Data/signup_2021_till15April_all_Final_After_Encoding.csv', error_bad_lines=False)


# In[91]:


Signupdata_15042021.shape


# In[32]:


Signupdata.columns


# In[92]:


#Taking only those columns as requested by business
Signupdata_15042021_required = Signupdata_15042021.filter(['Fraud_Acc_Flag','Ip_Latitude',
       'Ip_Longitude', 'Ip_Carrier', 'Ip_Connection_type', 'Ip_Line_Speed',
       'Ip_Routing_type', 'IP_Anonymizer_status', 'Fullcontact_Matched',
       'Social_Profiles_Count','ageRange_Fullcontact',
       'location_Country_Fullcontact', 'browser_online', 'brwsr_lang',
       'brwsr_type', 'brwsr_version', 'device_manufacturer', 'device_name',
       'device_type', 'device_os_type', 'screen_resolution', 'country_of_residence', 'email_domain','residential_status','ad_campaign', 'affiliate_name','channel', 'keywords','referral_text',
       'reg_mode', 'search_engine', 'source', 'sub_source', 'turnover',
       'txn_value'], axis=1)


# In[ ]:


#Taking only those columns as requested by business
Signup_required = Signupdata.filter(['Fraud_Acc_Flag','Ip_Latitude',
       'Ip_Longitude', 'Ip_Carrier', 'Ip_Connection_type', 'Ip_Line_Speed',
       'Ip_Routing_type', 'IP_Anonymizer_status', 'Fullcontact_Matched',
       'Social_Profiles_Count','ageRange_Fullcontact',
       'location_Country_Fullcontact', 'browser_online', 'brwsr_lang',
       'brwsr_type', 'brwsr_version', 'device_manufacturer', 'device_name',
       'device_type', 'device_os_type', 'screen_resolution', 'country_of_residence', 'email_domain','residential_status','ad_campaign', 'affiliate_name','channel', 'keywords','referral_text',
       'reg_mode', 'search_engine', 'source', 'sub_source', 'turnover',
       'txn_value'], axis=1)


# In[34]:


Signup_required.shape


# In[35]:


Signup_required.head()


# In[94]:


Signupdata_15042021_required.isnull().sum()


# In[37]:


Signup_required_Org = Signup_required.copy()
#Imputing the NULL value with most frequent occuring for the Ip_Latitude
Signup_required['Ip_Latitude'].fillna(Signup_required['Ip_Latitude'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the turnover
Signup_required['turnover'].fillna(Signup_required['turnover'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the Ip_Longitude
Signup_required['Ip_Longitude'].fillna(Signup_required['Ip_Longitude'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the Social_Profiles_Count
Signup_required['Social_Profiles_Count'].fillna(Signup_required['Social_Profiles_Count'].mode()[0],inplace=True)


# In[95]:


Signupdata_15042021_required_Org = Signupdata_15042021_required.copy()
#Imputing the NULL value with most frequent occuring for the Ip_Latitude
Signupdata_15042021_required['Ip_Latitude'].fillna(Signupdata_15042021_required['Ip_Latitude'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the turnover
Signupdata_15042021_required['turnover'].fillna(Signupdata_15042021_required['turnover'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the Ip_Longitude
Signupdata_15042021_required['Ip_Longitude'].fillna(Signupdata_15042021_required['Ip_Longitude'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the Social_Profiles_Count
Signupdata_15042021_required['Social_Profiles_Count'].fillna(Signupdata_15042021_required['Social_Profiles_Count'].mode()[0],inplace=True)


# In[99]:


Signup_required.isnull().sum()


# In[124]:


#merging data from May 2020 till 15th April 2021
signup_052020_15042021 = pd.concat([Signupdata_15042021_required.reset_index(drop=True)                            ,Signup_required_ML.reset_index(drop=True)], axis=0)


# In[148]:


#merging data from May 2020 till 15th April 2021 with all columns
signup_052020_15042021_all_columns = pd.concat([Signupdata_15042021.reset_index(drop=True)                            ,Signupdata.reset_index(drop=True)], axis=0)


# In[151]:


signup_052020_15042021_all_columns.isnull().sum()


# In[152]:


signup_052020_15042021_all_columns_Org = signup_052020_15042021_all_columns.copy()
#Imputing the NULL value with most frequent occuring for the Ip_Latitude
signup_052020_15042021_all_columns['Ip_Latitude'].fillna(signup_052020_15042021_all_columns['Ip_Latitude'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the turnover
signup_052020_15042021_all_columns['turnover'].fillna(signup_052020_15042021_all_columns['turnover'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the Ip_Longitude
signup_052020_15042021_all_columns['Ip_Longitude'].fillna(signup_052020_15042021_all_columns['Ip_Longitude'].mode()[0],inplace=True)

#Imputing the NULL value with most frequent occuring for the Social_Profiles_Count
signup_052020_15042021_all_columns['Social_Profiles_Count'].fillna(signup_052020_15042021_all_columns['Social_Profiles_Count'].mode()[0],inplace=True)


# In[125]:


signup_052020_15042021_ML.Fraud_Acc_Flag.value_counts()


# In[138]:


print(signup_052020_15042021_ML.shape)
print(Signupdata_15042021_required.shape)
print(Signup_required_ML.shape)


# In[139]:


print(Signup_required_ML.Fraud_Acc_Flag.value_counts())
print(signup_052020_15042021_ML.Fraud_Acc_Flag.value_counts())


# In[39]:


#Seperating Dependent and independent variables 
#data_final_Org= data_final.copy()
Signup_required_ML = Signup_required.copy()
X_Signupdata = Signup_required.drop(['Fraud_Acc_Flag'], axis=1, inplace=True)
Y_Signupdata = Signup_required_ML.Fraud_Acc_Flag


# In[126]:


#Seperating Dependent and independent variables 
#data_final_Org= data_final.copy()
signup_052020_15042021_ML = signup_052020_15042021.copy()
X_Signupdata2020_21 = signup_052020_15042021.drop(['Fraud_Acc_Flag'], axis=1, inplace=True)
Y_Signupdata2020_21 = signup_052020_15042021_ML.Fraud_Acc_Flag


# In[153]:


#Seperating Dependent and independent variables  with all columns
#data_final_Org= data_final.copy()
signup_052020_15042021_all_columns_ML = signup_052020_15042021_all_columns.copy()
X_Signupdata2020_21_all_columns = signup_052020_15042021_all_columns.drop(['Fraud_Acc_Flag'], axis=1, inplace=True)
Y_Signupdata2020_21_all_columns = signup_052020_15042021_all_columns_ML.Fraud_Acc_Flag


# In[132]:


pd.set_option('display.max_rows',None)
y_train2020_21


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Signup_required, Y_Signupdata, test_size=0.30, random_state=42)


# In[128]:


X_train2020_21, X_test2020_21, y_train2020_21, y_test2020_21 = train_test_split(signup_052020_15042021, Y_Signupdata2020_21, test_size=0.30, random_state=42)


# In[154]:


#for all 44 columns
X_train2020_21_all_columns, X_test2020_21_all_columns, y_train2020_21_all_columns, y_test2020_21_all_columns = train_test_split(signup_052020_15042021_all_columns, Y_Signupdata2020_21_all_columns, test_size=0.30, random_state=42)


# In[41]:


#Applying XGBoost Classification
import xgboost as xgb
XGBClassifier = xgb.XGBClassifier()
XGBClassifier.fit(X_train,y_train)


# In[133]:


#Applying XGBoost Classification
import xgboost as xgb
XGBClassifier2020_21 = xgb.XGBClassifier()
XGBClassifier2020_21.fit(X_train2020_21,y_train2020_21)


# In[155]:


#Applying XGBoost Classification for all columns
import xgboost as xgb
XGBClassifier2020_21_all_columns = xgb.XGBClassifier()
XGBClassifier2020_21_all_columns.fit(X_train2020_21_all_columns,y_train2020_21_all_columns)


# In[42]:


from sklearn import model_selection, metrics
hardpredtst=XGBClassifier.predict(X_test)
def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(Legit)', 'True 1(Fraud)'], 
            columns=['Pred 0(Approve as Legit)', 
                            'Pred 1(Deny as Fraud)'])
conf_matrix(y_test,hardpredtst)


# In[142]:


from sklearn import model_selection, metrics
hardpredtst2020_21=XGBClassifier2020_21.predict(X_test2020_21)
def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(Legit)', 'True 1(Fraud)'], 
            columns=['Pred 0(Approve as Legit)', 
                            'Pred 1(Deny as Fraud)'])
conf_matrix(y_test2020_21,hardpredtst2020_21)


# In[156]:


# for all columns 
from sklearn import model_selection, metrics
hardpredtst2020_21_all_columns=XGBClassifier2020_21_all_columns.predict(X_test2020_21_all_columns)
def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(Legit)', 'True 1(Fraud)'], 
            columns=['Pred 0(Approve as Legit)', 
                            'Pred 1(Deny as Fraud)'])
conf_matrix(y_test2020_21_all_columns,hardpredtst2020_21_all_columns)


# In[246]:


hardpredtst2020_21_all_columns


# In[43]:


predtstXGBC=XGBClassifier.predict_proba(X_test)[:,1] #rerun again


#for XGBClassifer
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(y_test, predtstXGBC)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[45]:


import numpy as np
from sklearn.metrics import roc_curve
Optimal_Cutoff,roc = Find_Optimal_Cutoff(y_test.values, predtstXGBC)
print (Optimal_Cutoff)


# In[47]:


hardpredtst_tuned_threshXGBCT = np.where(predtstXGBC >= 0.002305733971297741, 1, 0)
conf_matrix(y_test, hardpredtst_tuned_threshXGBCT)


# In[60]:


import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import plot_roc_curve
plot_roc_curve(XGBClassifier, X_test, y_test) 
plt.show()


# In[145]:


predtstXGBC2020_21=XGBClassifier2020_21.predict_proba(X_test2020_21)[:,1] #rerun again


#for XGBClassifer
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(y_test2020_21, predtstXGBC2020_21)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[146]:


Optimal_Cutoff2020_21,roc = Find_Optimal_Cutoff(y_test2020_21.values, predtstXGBC2020_21)
print (Optimal_Cutoff2020_21)


# In[147]:


hardpredtst_tuned_threshXGBC2020_21 = np.where(predtstXGBC2020_21 >= 0.003365118522197008, 1, 0)
conf_matrix(y_test2020_21, hardpredtst_tuned_threshXGBC2020_21)


# In[157]:


# for all columns
predtstXGBC2020_21_all_columns=XGBClassifier2020_21_all_columns.predict_proba(X_test2020_21_all_columns)[:,1] #rerun again


#for XGBClassifer
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(y_test2020_21_all_columns, predtstXGBC2020_21_all_columns)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[182]:


predtstXGBC2020_21_all_columns=XGBClassifier2020_21_all_columns.predict_proba(X_test2020_21_all_columns)[:,1]
print(predtstXGBC2020_21_all_columns)


# In[159]:


optimal_Cutoff2020_21_all_columns,roc = Find_Optimal_Cutoff(y_test2020_21_all_columns.values, predtstXGBC2020_21_all_columns)
print (optimal_Cutoff2020_21_all_columns)


# In[180]:


hardpredtst_tuned_threshXGBC2020_21_all_columns = np.where(predtstXGBC2020_21_all_columns >= 0.0030371882021427155, 1, 0)
conf_matrix(y_test2020_21_all_columns, hardpredtst_tuned_threshXGBC2020_21_all_columns)


# In[161]:


#Saving the Model
import joblib
XGBClassifier2020_21_all_columns.save_model('SignupallcolXGB.model')


# In[190]:


#import joblib
filename = 'SignupallcolXGB.model'
joblib.dump(XGBClassifier2020_21_all_columns, filename)
#XGBClassifier2020_21_all_columns.save_model('SignupallcolXGB.model')


# In[212]:


print(xgb.__version__)


# In[213]:


#marco_model = joblib.load('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/xgboost_Signup_Marco_Java_Encoder_joblib_V1.model')
import joblib
filename_marco = 'xgboost_Signup_Marco_Java_Encoder_joblib_V1.model'
loaded_model_marco = joblib.load(filename_marco)


# In[214]:


# load the model from disk
loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, Y_test)
#print(result)


# In[215]:


Testdata = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/New_TestSignupFraud.csv', error_bad_lines=False)

result=loaded_model.predict_proba(Testdata)[:,1]
print(result)


# In[216]:


testoutput = np.where(result >= 0.0030371882021427155, 1, 0)
print(testoutput)


# In[162]:


pwd


# ## Creating DMatrix for wrapper compatibility

# In[ ]:


#X_train2020_21_all_columns, X_test2020_21_all_columns, y_train2020_21_all_columns, y_test2020_21_all_columns = train_test_split(signup_052020_15042021_all_columns, Y_Signupdata2020_21_all_columns, test_size=0.30, random_state=42)


# In[285]:


#data_train, data_valid = train_test_split(signup_052020_15042021_all_columns_ML,test_size=0.05,random_state=123)
dtrain = xgb.DMatrix(X_train2020_21_all_columns, label=y_train2020_21_all_columns)
dtrain1 = xgb.DMatrix(X_train2020_21_all_columns)
dtest = xgb.DMatrix(X_test2020_21_all_columns, label=y_test2020_21_all_columns)
dtest1 = xgb.DMatrix(X_test2020_21_all_columns)


# In[227]:


dtrain


# In[228]:


dtest


# In[206]:


predictors =   [ 
                  'EIDStatus', 'SanctionStatus', 'Ip_Latitude',
                   'Ip_Longitude', 'Ip_Carrier', 'Ip_Connection_type', 'Ip_Line_Speed',
                   'Ip_Routing_type', 'IP_Anonymizer_status', 'Fullcontact_Matched',
                   'Social_Profiles_Count', 'gender_Fullcontact', 'ageRange_Fullcontact',
                   'location_Country_Fullcontact', 'browser_online', 'brwsr_lang',
                   'brwsr_type', 'brwsr_version', 'device_manufacturer', 'device_name',
                   'device_type', 'device_os_type', 'screen_resolution', 'address_type',
                   'aza', 'country_of_residence', 'email_domain', 
                   #'onQueue',
                   'region_suburb', 'residential_status', 'title', 'ad_campaign',
                   'affiliate_name', 'branch', 'channel', 'keywords', 'op_country',
                   'referral_text', 'reg_mode', 'search_engine', 'source', 'sub_source',
                   'turnover', 'txn_value'
                   ]


# In[207]:


target = 'Fraud_Acc_Flag'


# In[224]:


xgtrain = xgb.DMatrix(data_train[predictors].values, label=data_train[target].values, feature_names=[str(c) for c in predictors])


# In[223]:


xgtrain


# In[236]:


param = {
    'base_score': 0.5, 
    'booster': 'gbtree', 
    'colsample_bylevel': 1,
    'colsample_bynode': 1, 
    'colsample_bytree': 1, 
    'gamma': 0, 
    'gpu_id': -1,
    'importance_type': 'gain', 
    'interaction_constraints': '',
    'learning_rate': 0.300000012, 
    'max_delta_step': 0, 
    'max_depth': 6,
    'min_child_weight': 1, 
    'monotone_constraints': '()',
    'n_estimators': 100, 
    'n_jobs': 16, 
    'num_parallel_tree': 1, 
    'random_state': 0,
    'reg_alpha': 0, 
    'reg_lambda': 1, 
    'scale_pos_weight': 1, 
    'subsample': 1,
    'tree_method': 'exact', 
    'validate_parameters': 1}


# In[ ]:


#dM_X_train2020_21_all_columns = xgb.DMatrix(X_train2020_21_all_columns[predictors], label = data_valid[target], feature_names=[str(c) for c in predictors])


# In[287]:


import xgboost as xgb
#XGBClassifier2020_21_all_columns_DM = xgb.XGBClassifier()
bst = xgb.train(param, dtrain)

#X_test
#y_test


# In[281]:


preds = bst.predict(dtest1)


# In[282]:


preds


# In[283]:


# for all columns
#predtstXGBC2020_21_all_columns=XGBClassifier2020_21_all_columns.predict_proba(X_test2020_21_all_columns)[:,1] #rerun again

preds_DM = bst.predict(dtest1)

#for XGBClassifer
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(y_test2020_21_all_columns, preds_DM)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[284]:


optimal_Cutoff2020_21_all_columns,roc = Find_Optimal_Cutoff(y_test2020_21_all_columns.values, preds_DM)
print (optimal_Cutoff2020_21_all_columns)


# In[247]:


import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])


# In[268]:


best_preds_DM = np.where(preds >= 0.0030371882021427155, 1, 0)
#conf_matrix(y_test2020_21_all_columns, hardpredtst_tuned_threshXGBC2020_21_all_columns)


# In[272]:


best_preds_DM1 = np.where(preds >= 0.01892554759979248, 1, 0)


# In[276]:


best_preds_DM1


# In[275]:


# for all columns 
from sklearn import model_selection, metrics
hardpredtst2020_21_all_columns_DM=best_preds_DM1
def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(Legit)', 'True 1(Fraud)'], 
            columns=['Pred 0(Approve as Legit)', 
                            'Pred 1(Deny as Fraud)'])
conf_matrix(y_test2020_21_all_columns,hardpredtst2020_21_all_columns_DM)


# In[256]:


#Saving the Model
import joblib
bst.save_model('SignupallcolXGB_DM.model')


# In[257]:


#import joblib
filename = 'SignupallcolXGB_DM.model'
joblib.dump(bst, filename)
#XGBClassifier2020_21_all_columns.save_model('SignupallcolXGB.model')


# In[258]:


loaded_model_DM = joblib.load(filename)


# In[260]:


TestdataDM = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/New_TestSignupFraudDM.csv', error_bad_lines=False)

#result=loaded_model.predict_proba(Testdata)[:,1]
#print(result)


# In[261]:


TestdataDM1 = xgb.DMatrix(TestdataDM[predictors], label = TestdataDM[target], feature_names=[str(c) for c in predictors])


# In[277]:


TestdataDM2 = xgb.DMatrix(TestdataDM[predictors])


# In[266]:


TestdataDM[target]


# In[278]:


predictions  = loaded_model_DM.predict(TestdataDM2)
print(predictions)


# In[279]:


new_predictions = np.where(predictions >= 0.01892554759979248, 1, 0)
print(new_predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[194]:


#https://xgboost.readthedocs.io/en/latest/python/python_intro.html
dnewtest = xgb.DMatrix('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/New_TestSignupFraud.csv?format=csv&label_column=0')


# In[197]:


dnewtest


# In[244]:


dnewresult=bst.predict(dnewtest)


# In[204]:


dnewtestoutput = np.where(dnewresult >= 0.0030371882021427155, 1, 0)
print(dnewtestoutput)


# ## MLwatcher Implementation

# In[163]:


from MLWatcher.agent import MonitoringAgent


# In[178]:


agent = MonitoringAgent(frequency=5, max_buffer_size=500, n_classes=2, agent_id='1', server_IP='127.0.0.1', server_port=8000)


# In[165]:


agent.run_local_server()


# In[179]:


agent.collect_data(
predict_proba_matrix = 2 ##mandatory
#input_matrix = <your feature matrix>,  ##optional
#label_matrix = <your label matrix>   ##optional
)


# In[171]:


hardpredtst2020_21_all_columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


#Oversampling the data to handle imbalance
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler(0.95) #80
X_train_os, y_train_os = os.fit_resample(X_train,y_train)
print("The number of class before fit {}".format(Counter(y_train)))
print("The number of class after fit {}".format(Counter(y_train_os)))


# In[52]:


#Applying XGBoost Classification
import xgboost as xgb
XGBClassifier_os = xgb.XGBClassifier()
XGBClassifier_os.fit(X_train_os,y_train_os)


# In[54]:


from sklearn import model_selection, metrics
hardpredtst_os=XGBClassifier_os.predict(X_test)
def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(Legit)', 'True 1(Fraud)'], 
            columns=['Pred 0(Approve as Legit)', 
                            'Pred 1(Deny as Fraud)'])
conf_matrix(y_test,hardpredtst)


# In[55]:


predtstXGBC_os=XGBClassifier_os.predict_proba(X_test)[:,1] #rerun again


#for XGBClassifer
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(y_test, predtstXGBC_os)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[56]:


import numpy as np
from sklearn.metrics import roc_curve
Optimal_Cutoff,roc = Find_Optimal_Cutoff(y_test.values, predtstXGBC_os)
print (Optimal_Cutoff)


# In[57]:


hardpredtst_tuned_threshXGBCT_os = np.where(predtstXGBC_os >= 0.01437988318502903, 1, 0)
conf_matrix(y_test, hardpredtst_tuned_threshXGBCT_os)


# In[89]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(XGBClassifier_os, X_test, y_test) 
plt.show()


# # XGBoost on unsampled data gives better result compared to XGBoost on oversampled data

# In[61]:


## Hyper Parameter Optimization for XGBoost Classifier

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
"scale_pos_weight"  : [1, 10, 25, 50, 75, 99, 100, 200]
    
}


# In[62]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


# In[63]:


#cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[65]:


#https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
XGBClassifier = xgb.XGBClassifier()
random_search=RandomizedSearchCV(XGBClassifier,param_distributions=params,n_iter=6,scoring='accuracy',n_jobs=-1,#cv=4,
                                 refit=True,cv=cv,
                                 verbose=3)


# In[66]:


random_search.fit(Signup_required,Y_Signupdata)


# In[67]:


random_search.best_estimator_


# In[69]:


XGBClassifier_Tuned = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=15,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=100, n_jobs=16, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=25, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


# In[70]:


XGBClassifier_Tuned.fit(X_train, y_train)


# In[71]:


from sklearn import model_selection, metrics
hardpredtst_tuned=XGBClassifier_Tuned.predict(X_test)
def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(Legit)', 'True 1(Fraud)'], 
            columns=['Pred 0(Approve as Legit)', 
                            'Pred 1(Deny as Fraud)'])
conf_matrix(y_test,hardpredtst_tuned)


# In[72]:


predtstXGBC_tuned=XGBClassifier_Tuned.predict_proba(X_test)[:,1] #rerun again


#for XGBClassifer
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(y_test, predtstXGBC_tuned)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[73]:



Optimal_Cutoff,roc = Find_Optimal_Cutoff(y_test.values, predtstXGBC_tuned)
print (Optimal_Cutoff)


# In[74]:


hardpredtst_tuned_threshXGBCT = np.where(predtstXGBC_tuned >= 0.0006503670010715723, 1, 0)
conf_matrix(y_test, hardpredtst_tuned_threshXGBCT)


# In[79]:


#import matplotlib.pylab as plt
#%matplotlib inline
from sklearn.metrics import plot_roc_curve
plot_roc_curve(XGBClassifier_Tuned, X_test, y_test) 
plt.show()


# # Optimal threshold on untuned XGboost gives slight better result compared to Random search tuned XGBoost model

# In[75]:


#Hyperparametertunning on Oversample data
os_fulldata=RandomOverSampler(0.95) #80
X_os, y_os = os_fulldata.fit_resample(Signup_required,Y_Signupdata)
print("The number of class before fit {}".format(Counter(Y_Signupdata)))
print("The number of class after fit {}".format(Counter(y_os)))


# In[76]:


# Hyper Parameter Optimization for XGBoost Classifier

params_os={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
#"scale_pos_weight"  : [1, 10, 25, 50, 75, 99, 100, 200]
    
}


# In[77]:


XGBClassifier = xgb.XGBClassifier()
random_search_os=RandomizedSearchCV(XGBClassifier,param_distributions=params_os,n_iter=6,scoring='accuracy',n_jobs=-1,#cv=4,
                                 refit=True,cv=cv,
                                 verbose=3)


# In[80]:


random_search_os.fit(X_os,y_os)


# In[81]:


random_search_os.best_estimator_


# In[82]:


XGBClassifier_Tuned_os = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=15,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=100, n_jobs=16, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


# In[83]:


XGBClassifier_Tuned_os.fit(X_train_os, y_train_os)


# In[84]:


hardpredtst_tuned_os=XGBClassifier_Tuned_os.predict(X_test)
def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(Legit)', 'True 1(Fraud)'], 
            columns=['Pred 0(Approve as Legit)', 
                            'Pred 1(Deny as Fraud)'])
conf_matrix(y_test,hardpredtst_tuned_os)


# In[85]:


predtstXGBC_tuned_os=XGBClassifier_Tuned_os.predict_proba(X_test)[:,1] #rerun again


#for XGBClassifer
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(y_test, predtstXGBC_tuned_os)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[86]:



Optimal_Cutoff,roc = Find_Optimal_Cutoff(y_test.values, predtstXGBC_tuned_os)
print (Optimal_Cutoff)


# In[87]:


hardpredtst_tuned_threshXGBCT_os = np.where(predtstXGBC_tuned_os >= 0.0004091068112757057, 1, 0)
conf_matrix(y_test, hardpredtst_tuned_threshXGBCT_os)


# In[88]:


from sklearn.metrics import plot_roc_curve
plot_roc_curve(XGBClassifier_Tuned_os, X_test, y_test) 
plt.show()


# # Tuned XGboost model on oversample gives better result than Tuned XGboost model on unsampled data. The result is similar to untuned XGboost model on unsampled data

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




