
#sns.boxplot(x=data['gender'],y=data['interest'],hue=data['age'])
#plt.show()
#
#sns.catplot(data=data, x="age", y="Impressions",hue="gender")

########################################################
################### function to detect outliers
def detect_outliers_iqr(data1):
    outliers = []
    data1 = sorted(data1)
    q1 = np.percentile(data1, 25)
    q3 = np.percentile(data1, 75)
    print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    print(lwr_bound, upr_bound)
    for i in data1: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code

Impressions_outliers = detect_outliers_iqr(data['Impressions'])
print("Impressions Outliers from IQR method: ", Impressions_outliers)

data['Impressions']=np.where(data['Impressions']>544667.25,544667.25,data['Impressions'])

Clicks_outliers = detect_outliers_iqr(data['Clicks'])
print("Clicks Outliers from IQR method: ", Clicks_outliers)

data['Clicks']=np.where(data['Clicks']>92.25,92.25,data['Clicks'])

Spent_outliers = detect_outliers_iqr(data['Spent'])
print("Spent Outliers from IQR method: ", Spent_outliers)

data['Spent']=np.where(data['Spent']>147.842499759,147.842499759,data['Spent'])


## Total_Conversion_outliers = detect_outliers_iqr(data['Total_Conversion'])
## print("Total_Conversion Outliers from IQR method: ", Total_Conversion_outliers)
## 
## data['Total_Conversion']=np.where(data['Total_Conversion']>6,6,data['Total_Conversion'])

#Approved_Conversion_outliers = detect_outliers_iqr(data['Approved_Conversion'])
#print("Approved_Conversion Outliers from IQR method: ", Approved_Conversion_outliers)

#data['Approved_Conversion']=np.where(data['Approved_Conversion']>2.5,2.5,data['Approved_Conversion'])



data['Total_Conversion']=np.where(data['Total_Conversion']>=1,1,data['Total_Conversion'])
data['Approved_Conversion']=np.where(data['Approved_Conversion']>=1,1,data['Approved_Conversion'])


#### End PART 2 Indra ####
