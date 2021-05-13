#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org xgboost==0.80


# In[1]:


import xgboost as xgb
print(xgb.__version__)


# In[21]:


import pandas as pd
Signupdata = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/signup_052020_15042021_all_columns_ML.csv', error_bad_lines=False, index_col=0)


# In[22]:


Signupdata.head()


# In[23]:


#Seperating Dependent and independent variables  with all columns
#data_final_Org= data_final.copy()
Signupdata_ML = Signupdata.copy()
X_Signupdata = Signupdata.drop(['Fraud_Acc_Flag'], axis=1, inplace=True)
Y_Signupdata = Signupdata_ML.Fraud_Acc_Flag


# In[25]:


#for all 44 columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Signupdata, Y_Signupdata, test_size=0.30, random_state=42)


# In[162]:


X_test.to_csv('testdataSIT.csv')


# In[302]:


X_train.to_csv('traindataSIT.csv', index=False)


# In[303]:


X_test.head()


# In[307]:


X_train.shape


# In[26]:


#data_train, data_valid = train_test_split(signup_052020_15042021_all_columns_ML,test_size=0.05,random_state=123)
dtrain = xgb.DMatrix(X_train, label=y_train)
#dtrain1 = xgb.DMatrix(X_train2020_21_all_columns)
dtest = xgb.DMatrix(X_test, label=y_test)
#dtest1 = xgb.DMatrix(X_test2020_21_all_columns)


# In[111]:


param = {
    'base_score': 0.5, 
    'booster': 'gbtree', 
    'colsample_bylevel': 1,
    'colsample_bynode': 1, 
    'colsample_bytree': 1, 
    'gamma': 0, 
    'gpu_id': 0,
    'objective' : 'binary:logistic',
    'importance_type': 'gain', 
    'interaction_constraints': '',
    'learning_rate': 0.300000012, 
    'max_delta_step': 0, 
    'max_depth': 6,
    'min_child_weight': 1, 
    #'monotone_constraints': '()',
    'n_estimators': 100, 
    'n_jobs': 16, 
    'num_parallel_tree': 1, 
    'random_state': 0,
    'reg_alpha': 0, 
    'reg_lambda': 1, 
    'scale_pos_weight': 1, 
    'subsample': 1,
    'tree_method': 'exact',
    #'num_class': 2,
    'eta': 0.6,
    'validate_parameters': 1}
num_round = 1000


# In[259]:


import xgboost as xgb
#XGBClassifier2020_21_all_columns_DM = xgb.XGBClassifier()
bst = xgb.train(param, dtrain, num_round)


# In[260]:


preds = bst.predict(dtest)


# In[261]:


preds_DM = bst.predict(dtest)

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
    fpr, tpr, threshold = roc_curve(y_test, preds_DM)
    i = np.arange(len(tpr)) 
    # tpr -(1-fpr) is zero near the optimal point
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']),roc


# In[262]:


import numpy as np
from sklearn.metrics import roc_curve
optimal_Cutoff2020_21_all_columns,roc = Find_Optimal_Cutoff(y_test.values, preds_DM)
print (optimal_Cutoff2020_21_all_columns)


# In[263]:


best_preds_DM1 = np.where(preds_DM >= 4.1071583837037906e-05, 1, 0)
#0.00380784273147583
#4.1071583837037906e-05


# In[308]:


# Comparing with previous model currently present in Production
best_preds_DMold= np.where(preds_DM >= 0.007600000000000007, 1, 0)
#0.007600000000000007


# In[353]:


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
conf_matrix(y_test,hardpredtst2020_21_all_columns_DM)


# In[323]:


#target = 'Fraud_Acc_Flag'
# compute fpr, tpr, thresholds and roc_auc
from sklearn.metrics import roc_curve, auc
#fpr, tpr, thresholds = roc_curve(train[target].values, y_pred_train)
fpr_test_old, tpr_test_old, thresholds_test_old = roc_curve(y_test, best_preds_DM1)
roc_auc_test_old = auc(fpr_test_old, tpr_test_old) # compute area under the curve
roc_auc_test_old


# In[327]:


def plot_ROC(fpr_r, tpr_r,thresholds,roc_auc_r,cut_off_value_r=0):
    plt.clf()
    # Plot ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_r, tpr_r,label='ROC curve (area = %0.2f)' % (roc_auc_r))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate',fontsize=15)
    plt.title('Receiver operating characteristic',fontsize=15)
    plt.legend(loc="lower right")
    if cut_off_value_r != 0:
        plt.axvline(x=cut_off_value_r,color='r')


    # create the axis of thresholds (scores)
    #ax2 =lt.gca().twinx()
    #ax2.plot(fpr_r, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    #ax2.set_ylabel('Th preshold',color='r')
    #ax2.set_ylim([thresholds[-1],thresholds[0]])
    #ax2.set_xlim([fpr_r[0],fpr_r[-1]])
    

plot_ROC(fpr_test_old, tpr_test_old,thresholds_test_old,roc_auc_test_old)


# In[352]:


print(confusion_matrix(y_test, best_preds_DM1))
print(accuracy_score(y_test, best_preds_DM1))
print(classification_report(y_test, best_preds_DM1))


# In[365]:


def PlotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud):

    cfn_matrix = confusion_matrix(y_test,pred)
    #cfn_norm_matrix = np.array([[1.0 / y_test_legit,1.0/y_test_legit],[1.0/y_test_fraud,1.0/y_test_fraud]])
    #norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(25,5))
    ax = fig.add_subplot(2,2,2)
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.9,annot=True,ax=ax,fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    #ax = fig.add_subplot(1,2,2)
    #sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)

    #plt.title('Normalized Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()
    
    #print('---Classification Report---')
    #print(classification_report(y_test,pred))


# In[366]:


#y_test = data_valid[target].values
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve,accuracy_score
import seaborn as sns
pred = best_preds_DM1
y_test_legit = y_test.value_counts()[0]
y_test_fraud = y_test.value_counts()[1]
sns.set(font_scale=1.4)
#sns.set(fmt='g')   
PlotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud) 


# In[372]:


y_test.value_counts()


# In[321]:


# feature importance
bst.get_score(fmap='', importance_type='weight')


# ## Checking other technique with XGBClassifer(old)
# 

# In[120]:


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


# In[121]:


target = 'Fraud_Acc_Flag'


# In[155]:


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
                
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval = True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], verbose=True)
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Predict test set:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    
    #Predict on original set:
    #data_valid_predictions = alg.predict(data_valid[predictors])
    #data_valid_predictions = alg.predict(dtest)

        
    #Print model report:
    print ("\nModel Report")
    print ("------------------------------------------------------------------------------------")
    print ("\nPredictions on training data")
    print ("Accuracy (Train) : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print ("Logloss (Train): %f" % metrics.log_loss(dtrain[target], dtrain_predprob))
    print (metrics.confusion_matrix(dtrain[target].values, dtrain_predictions))
    print ("------------------------------------------------------------------------------------")
    print ("\nPredictions on testing data")
    print ("Accuracy (Test): %.4g" % metrics.accuracy_score(dtest[target].values, dtest_predictions))
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(dtest[target], dtest_predprob))
    print ("Logloss (Test): %f" % metrics.log_loss(dtest[target], dtest_predprob))
    print (metrics.confusion_matrix(dtest[target].values, dtest_predictions))
    #print ("------------------------------------------------------------------------------------")
    #print ("\nPredictions on original data")
    #print (metrics.confusion_matrix(data_encode_valid[target].values, data_encode_predictions))
    #Y_test = y_test.values
    #pred = data_valid_predictions
    #y_test_legit = Y_test.values[0]
    #y_test_fraud = Y_test.values[1]
    
    #PlotConfusionMatrix(Y_test,pred,y_test_legit,y_test_fraud)    
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print (feat_imp)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    


# In[127]:


from xgboost.sklearn import XGBClassifier
xgb1 = XGBClassifier(
    base_score = 0.5, 
    booster= 'gbtree', 
    colsample_bylevel= 1,
    colsample_bynode=  1, 
    colsample_bytree=  1, 
    gamma=  0, 
    gpu_id=  0,
    objective =  'binary:logistic',
    importance_type=  'gain', 
    interaction_constraints=  '',
    learning_rate = 0.300000012, 
    max_delta_step = 0, 
    max_depth=  6,
    min_child_weight = 1, 
    #'monotone_constraints': '()',
    n_estimators =  100, 
    n_jobs=  16, 
    num_parallel_tree=  1, 
    random_state = 0,
    reg_alpha=  0, 
    reg_lambda=  1, 
    scale_pos_weight=  1, 
    subsample=  1,
    tree_method = 'exact',
    #'num_class': 2,
    eta =  0.6,
    validate_parameters =  1
                      )


# In[156]:


modelfit(xgb1,Signupdata_ML, predictors)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[114]:


xgb.plot_importance(bst, max_num_features=5)


# In[117]:


bst.get_fscore()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[140]:


#Saving the Model
import joblib
#bst.save_model('SignupallcolXGBBST.model')


# In[157]:


pwd


# In[141]:


filename = 'SignupallcolXGBBST.model'
joblib.dump(bst, filename)


# In[201]:


filename1 = 'SignupallcolXGBBST_1.model'
joblib.dump(bst, filename1)


# In[159]:


bst.save_model('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/SignupallcolXGBBST_1.model')


# In[254]:


bst.load_model('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/SignupallcolXGBBST_1.model')


# In[142]:


loaded_model_DM = joblib.load(filename)


# In[241]:


loaded_model_DM1 = joblib.load('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/SignupallcolXGBBST_1.model')


# In[246]:


loaded_model_DM1


# In[256]:


bst = xgb.Booster({'nthread': 4})  # init model
#bst1 = xgb.Booster()
#bst1.load_model('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/SignupallcolXGBBST_1.model')
bst.load_model('model.bin')


# In[ ]:





# In[143]:


TestdataDM = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/New_TestSignupFraudDM.csv', error_bad_lines=False)

#result=loaded_model.predict_proba(Testdata)[:,1]
#print(result)


# In[218]:


Testdata400 = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/fp_test_encoded_400.csv', error_bad_lines=False)


# In[144]:


TestdataDM2 = xgb.DMatrix(TestdataDM[predictors])


# In[219]:


Testdata400 = xgb.DMatrix(Testdata400[predictors])


# In[220]:


predictions400  = loaded_model_DM1.predict(Testdata400)
print(predictions400)


# In[221]:


new_predictions400 = np.where(predictions400 >= 4.1071583837037906e-05, 1, 0)
print(new_predictions400)


# In[291]:


Testdata_new = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/fs_fraud_test.csv', error_bad_lines=False)


# In[292]:


pd.set_option('display.max_columns',None)
Testdata_new.head()


# In[293]:


Testdata_new_DM = xgb.DMatrix(Testdata_new)


# In[266]:


print(Testdata_new_DM)


# In[294]:


predictions_1  = loaded_model_DM1.predict(Testdata_new_DM)
#print(predictions_1)


# In[295]:


Fraud_Pred_Jup = np.where(predictions_1 >= 4.1071583837037906e-05, 1, 0)
print(Fraud_Pred_Jup)


# In[ ]:


# testing with OLD Marco model


# In[249]:


loaded_model_Marco = joblib.load('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/xgboost_Signup_Marco_Java_Encoder_joblib_V1.model')


# In[252]:


import pickle
loaded_model_Marco = pickle.load(open('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/xgboost_Signup_Marco_Java_Encoder_joblib_V1.model', 'rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[203]:


predictions_2  = loaded_model_DM1.predict(Testdata_new_DM)
print(predictions_2)


# In[204]:


Fraud_Pred_Jup1 = np.where(predictions_2 >= 4.1071583837037906e-05, 1, 0)
print(Fraud_Pred_Jup1)


# In[296]:


predictions_1_df = pd.DataFrame(predictions_1, columns =['jup_output_score'])


# In[297]:


Fraud_Pred_Jup_df = pd.DataFrame(Fraud_Pred_Jup, columns = ['Fraud_Pred_Jup'])


# In[298]:


Fraud_Pred_Jup_df.Fraud_Pred_Jup.value_counts()


# In[177]:


Fraud_Pred_Jup_df.to_csv('Fraud_Pred_Jup.csv')


# In[205]:


predictions_2_df = pd.DataFrame(predictions_2, columns =['jup_output_score'])


# In[206]:


Fraud_Pred_Jup_df1 = pd.DataFrame(Fraud_Pred_Jup1, columns = ['Fraud_Pred_Jup'])


# In[207]:


Fraud_Pred_Jup_df1.Fraud_Pred_Jup.value_counts()


# In[179]:


Fraud_Pred_Jup_df.shape


# In[300]:


Jup_output_score_predcition = pd.concat([Testdata_new.reset_index(drop='True'), predictions_1_df.reset_index(drop='True'),Fraud_Pred_Jup_df.reset_index(drop='True'),],axis=1)


# In[210]:


Jup_output_score_predcition1 = pd.concat([Testdata_new.reset_index(drop='True'),predictions_2_df.reset_index(drop='True'),Fraud_Pred_Jup_df1.reset_index(drop='True'),],axis=1)


# In[301]:


Jup_output_score_predcition.to_csv('Jup_output_score_predcition.csv', index=False)


# In[211]:


Jup_output_score_predcition1.to_csv('Jup_output_score_predcition1.csv', index=False)


# In[186]:


Jup_Pred_Test = pd.concat([Testdata_new.reset_index(drop='True'),Fraud_Pred_Jup_df.reset_index(drop='True')],axis=1)


# In[196]:


Jup_Pred_outputscore_Test =pd.concat([Testdata_new.reset_index(drop='True'),Jup_output_score_predcition.reset_index(drop='True')],axis=1)


# In[197]:


Jup_Pred_outputscore_Test.to_csv('Jup_Pred_outputscore_Test.csv', index =False)


# In[188]:


Jup_Pred_Test.head()


# In[184]:


#importing Predicted values from Java wrapper
java_predicted = pd.read_csv('/home/mvisi/Project/DLP/Core/FraudPredict/Notebook/Arpan/fp_test_data_encoded.csv')


# In[185]:


java_predicted.head()


# In[189]:


java_predicted['Fraud_Predict_Java'] = np.where(java_predicted.model_output_score >= 4.1071583837037906e-05, 1, 0)


# In[191]:


java_predicted.Fraud_Predict_Java.value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




