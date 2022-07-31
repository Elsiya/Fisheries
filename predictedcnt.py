import pandas
from sklearn import linear_model
df = pandas.read_csv("elsiya.csv")
X = df[['temp', 'hum', 'windspeed']]
y = df['cnt']
regr = linear_model.LinearRegression()
regr.fit(X, y)
#predict the cnt where the temp is 7, humidity is 55 and the windspeed is 12:
predictedcnt = regr.predict([[7, 55, 12]])

print(predictedcnt)
