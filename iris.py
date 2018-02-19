from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)

#Descriptive and Visual statistics to get more information on the data and identify any possible relations between columns.
print(dataset.shape)
print(dataset.head(50))
print(dataset.describe())
# From the above commands, we can see that all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 centimeters.
# So we do not need to standardize or normalize the data any further.
print(dataset.groupby('class').size())
# From the above command, we can see that each class has the same number of instances (50 or 33% of the dataset).
# Univariate plots to better understand each attribute and Multivariate plots help us understand the relationship between them.
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
dataset.hist()
pyplot.show()
# From the above histogram, It looks like perhaps 'sepal-length', 'sepal-width' have a Gaussian distribution i.e in layman term a bell curve. This is useful to note as we can use algorithms that can exploit this assumption.
scatter_matrix(dataset)
pyplot.show()
# The plots above show a high correlation between 'sepal-length', 'petal-length', 'petal-width'

# Prepare Data Split-out validation dataset into Train and Test
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 5
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot-Check Various Algorithms
models = []
# Linear
models.append(('LR', LogisticRegression()))
# Non-Linear
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
# We use 10 fold cross validation to evaluate each Algorithm. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. We will use
# the metric of accuracy as the scoring variable to evaluate models.
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# We can see that it looks like SVC (or Support Vector Classifiers) has the largest estimated accuracy score. This incidentally also has a lower Standard deviation. We can also see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.
# Make predictions on validation dataset
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#The Confusion Matrix above clearly shows that the prediction is very close to 100% accurate.
print(classification_report(Y_validation, predictions))