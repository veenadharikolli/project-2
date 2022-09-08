import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel(r"C:/Users/USER/Desktop/DS Project 73/MLR Code and Deployment/Project_73_Data-MLR.xlsx")
df.info()
df.columns
df.head()

#check for missing values
df.isna()
df.isna().sum()

df.columns

# let's find outliers in Given_Mileage_km_litre
sns.boxplot(df.Given_Mileage_km_litre)

# No outliers in age column

# Detection of outliers (find limits for salary based on IQR)
IQR = df['Given_Mileage_km_litre'].quantile(0.75) - df['Given_Mileage_km_litre'].quantile(0.25)
lower_limit = df['Given_Mileage_km_litre'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Given_Mileage_km_litre'].quantile(0.75) + (IQR * 1.5)


from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Given_Mileage_km_litre'])

df['Given_Mileage_km_litre'] = winsor.fit_transform(df[['Given_Mileage_km_litre']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df.Given_Mileage_km_litre)

# let's find outliers in Fuel_Consumption_Overall_in_per_100km
sns.boxplot(df.Fuel_Consumption_Overall_in_per_100km)
# no outliers were found

# lets see boxplot
sns.boxplot(df.Total_Emission_of_pollutants_gms_per_ltr)
# no outliers were found


## Univariate Visualization
# Heat Map
corr = df.corr()
sns.heatmap(corr, annot = True , cmap = 'coolwarm')
plt.figure(figsize = (20,20))

## Creating Count plot
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 6)
sns.countplot(x=df["vehicleType"])
plt.xticks(rotation = 90)


# Creating Pie Chart
pollutants = ['N2_Emission','CO2_Emission', 'O2_Emission', 'H2O_Emission','Other_Particulate_Matters']
class1_pollutants = [67, 12, 11, 9, 1]

plt.pie(class1_pollutants, labels = pollutants)
plt.show()

##############################
import seaborn as sns
sns.pairplot(df.iloc[:,:])

# Label Encoder
from sklearn.preprocessing import LabelEncoder
# Creating instance of labelencoder
labelencoder = LabelEncoder()

###################
# Label encoding
df.columns

# Data Split into Input and Output variables
X = df[['vehicleType', 'Type', 'Dist_To_Travel', 'region/non_region']]
X

X['vehicleType'] = labelencoder.fit_transform(X['vehicleType'])
X['Type'] = labelencoder.fit_transform(X['Type'])
X['Dist_To_Travel'] = labelencoder.fit_transform(X['Dist_To_Travel'])
X['region/non_region'] = labelencoder.fit_transform(X['region/non_region'])

y = df[['Payload', 'Given_Mileage_km_litre', 'Fuel_Consumption_Overall_in_per_100km', 'Total_Emission_of_pollutants_gms_per_ltr']]

### We have to convert y to data frame so that we can use concatenate function
# concatenate X and y
df_new = pd.concat([X, y], axis = 1)

## rename column name
df_new.columns

import scipy.stats as stats
import pylab
#4probplot
stats.probplot(df_new.Given_Mileage_km_litre, dist="norm", plot=pylab)
stats.probplot(df_new.Fuel_Consumption_Overall_in_per_100km, dist="norm", plot=pylab)
stats.probplot(df_new.Payload, dist="norm", plot=pylab)
stats.probplot(df_new.Total_Emission_of_pollutants_gms_per_ltr, dist="norm", plot=pylab)

x = df_new[['vehicleType','Fuel_Consumption_Overall_in_per_100km']]
x

Y = df_new.iloc[:, -1:]
Y

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x,Y)

import statsmodels.formula.api as smf # for regression model       
ml1 = smf.ols('Total_Emission_of_pollutants_gms_per_ltr ~ vehicleType + Payload + Fuel_Consumption_Overall_in_per_100km',  data = df_new).fit() # regression model
ml1.summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=42)

regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)


from sklearn.metrics import r2_score
y_pred = regr.predict(x)
r2_score1 = r2_score(y_pred,Y)
r2_score1

##  Linear Regression
from sklearn.model_selection import cross_val_score
mse = cross_val_score(regr,x,Y,scoring = "neg_mean_squared_error", cv = 5)
mean_mse = np.mean(mse)
mean_mse


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(df_new, test_size = 0.2) # 20% test data

##############################################
# prediction on test data set 
test_pred = ml1.predict(x_test)
# test residual values 
test_resid = test_pred - x_test.Total_Emission_of_pollutants_gms_per_ltr
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

##############################################
# train_data prediction
train_pred = ml1.predict(x_train)
# train residual values 
train_resid  = train_pred - x_train.Total_Emission_of_pollutants_gms_per_ltr
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

import pickle
# Saving model to disk
pickle.dump(regr, open('model_Prediction_73_MLR.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model_Prediction_73_MLR.pkl','rb'))
print(model.predict([[14,12]]))


