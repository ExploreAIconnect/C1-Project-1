data=data.drop(data.columns[[0,2]],axis=1)

data.nunique()


data = pd.get_dummies(data=data, columns=['xyz_campaign_id','age','gender','interest'])
data.columns


## Scaling the features with MinMaxScaler


cols_to_scale = ['Impressions','Clicks','Spent']

scaler = MinMaxScaler()

data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

data.to_csv("check.csv")


X = data.drop(columns=['Total_Conversion'],axis=1)
list(X) 

Y = data['Total_Conversion']

data.info()

#### End PART 3 Kalpana ####
