#!/usr/bin/env python
# coding: utf-8

# In[6]:


from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd


# In[7]:


ap = PlainTextAuthProvider(username='fpadmin', password='5p@dm!n1')
cass_contact_points=['10.0.12.75']
cluster = Cluster(cass_contact_points,auth_provider=ap)
#session = cluster.connect('fraudpredict')
session = cluster.connect('cd_ml_api')


# ### Live Signup data before encoding from 0727 to Now

# In[10]:


signup = pd.DataFrame()
rows = session.execute("Select * from payload_response where payload_type = 'signup' and year = 2018")

for row in rows:
    #print (row[0], row[1], row[2])
    response = pd.DataFrame([row[0], row[1], row[2],row[3],row[4], row[5], row[6],row[7],row[8],row[9]]).T
    response.columns = ['payload_type','year','request_time','prediction_request_aggregate','customer_no','model_input_encoded','prediction_request_atlas','model_output_score','prediction_response','trans_id']
    signup = signup.append(response)


# In[11]:


signup.shape 


# In[12]:


signup_update = pd.DataFrame()
rows = session.execute("Select * from payload_response where payload_type = 'update' and year = 2018")
for row in rows:
    #print (row[0], row[1], row[2])
    response = pd.DataFrame([row[0], row[1], row[2],row[3],row[4], row[5], row[6],row[7],row[8],row[9]]).T
    response.columns = ['payload_type','year','request_time','prediction_request_aggregate','customer_no','model_input_encoded','prediction_request_atlas','model_output_score','prediction_response','trans_id']
    signup_update = signup_update.append(response)


# In[13]:


signup_update.shape


# In[14]:


signup_all = pd.concat([signup.reset_index(drop=True)                       ,signup_update.reset_index(drop=True)], axis=0)


# In[15]:


signup_all.shape


# In[16]:


signup_all


# In[17]:


signup_all['model_output_score'] = signup_all['model_output_score'].convert_objects(convert_numeric=True)
signup_all['model_output_score'] = pd.to_numeric(signup_all['model_output_score'])


# In[18]:


signup_all.info()


# In[19]:


signup_all.sort_values('model_output_score',ascending=False)


# ### Only keep the latest record for each customer

# In[20]:


signup_all_Rank = signup_all.reset_index()


# In[21]:


signup_all_Rank['Ranking_By_Time'] = signup_all_Rank.groupby('customer_no')['request_time'].rank(ascending=False)


# In[22]:


signup_all_Rank.sort_values(['Ranking_By_Time','customer_no'],ascending=True)


# In[23]:


signup_all_Rank_input = signup_all_Rank.loc[
                                          (signup_all_Rank['Ranking_By_Time'] == 1)
                                          &                                        
                                          (signup_all_Rank['customer_no'] != '100000') ]


# In[24]:


signup_all_Rank_input.shape


# In[25]:


signup_all_Rank_input


# ### Prepare all the features needed for training

# In[26]:


### prediction_request_aggregate
import json
#data['Device_Information__c'] = data['Device_Information__c'].fillna('')    
from pandas.io.json import json_normalize
prediction_request_aggregate = [json.loads(row) for row in signup_all_Rank_input['prediction_request_aggregate']]
prediction_request_aggregate_Final = json_normalize(prediction_request_aggregate)


# In[27]:


prediction_request_aggregate_Final.head()


# In[ ]:


eflexcomputers.


# In[28]:


Other_Fields = signup_all_Rank_input['request_time']


# In[29]:


signup_all_Rank_input_Final = pd.concat([Other_Fields.reset_index(drop=True),                                         prediction_request_aggregate_Final.reset_index(drop=True)], axis=1)


# In[30]:


signup_all_Rank_input_Final.columns


# In[31]:


predictors_live =   [ 'customer_no', 'request_time',
                  'eid_status', 'sanction_status', 'ip_latitude',
                   'ip_longitude', 'ip_carrier', 'ip_connection_type', 'ip_line_speed',
                   'ip_routing_type', 'ip_anonymizer_status', 'fullcontact_matched',
                   'social_profiles_count', 'gender_fullcontact', 'age_range_fullcontact',
                   'location_country_fullcontact', 'browser_online', 'browser_lang',
                   'browser_type', 'browser_version', 'device_manufacturer', 'device_name',
                   'device_type', 'device_os_type', 'screen_resolution', 'address_type',
                   'aza', 'country_of_residence', 'email_domain', 
                   #'onQueue',
                   'region_suburb', 'residential_status', 'title', 'ad_campaign',
                   'affiliate_name', 'branch', 'channel', 'keywords', 'op_country',
                   'referral_text', 'reg_mode', 'search_engine', 'source', 'sub_source',
                   'turnover', 'txn_value'
                   ]


# In[32]:


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


# In[33]:


signup_all_Rank_input_Final[predictors_live]


# In[34]:


numeric_columns = [ #'Fraud_Acc_Flag',
                    'turnover',
                    'ip_latitude',
                   'ip_longitude',
                    'social_profiles_count',
                   #'Social_Media_Total_followers',
                   #'Social_Media_Total_following',
                  ]


# In[35]:


cat_predictors = list(set(predictors_live) - set(numeric_columns))


# In[36]:


data_final_model = signup_all_Rank_input_Final.copy()


# In[37]:


for feature in cat_predictors:
    data_final_model[feature] = data_final_model[feature].astype(str)
    data_final_model[feature] = data_final_model[feature].fillna('None') 
    data_final_model[feature] = data_final_model[feature].str.replace('nan', '')
    data_final_model[feature] = data_final_model[feature].apply(lambda x: x.upper())


# In[38]:


for feature in numeric_columns:
    data_final_model[feature] = data_final_model[feature].convert_objects(convert_numeric=True)
    data_final_model[feature] = pd.to_numeric(data_final_model[feature])
    #data_final_model[feature] = data_final_model[feature].fillna(0) 


# In[39]:


data_final_model = data_final_model[predictors_live].drop_duplicates()


# In[40]:


from IPython.display import display, HTML
HTML(data_final_model.head(70).to_html())


# ### Export Live Signup data before encoding from 0727 to Now

# In[41]:


#data_final_model.to_pickle('/home/ML/FraudPredict/Data/Signup_LIVE_fileds_Final_Flag_Before_Encode_upto_0726.pkl',compression='gzip')

data_final_model.to_pickle('../Data/Signup_LIVE_fileds_Final_Flag_Before_Encode_upto_now.pkl',compression='gzip')


# In[42]:


data_final_model


# In[ ]:




