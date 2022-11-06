
# -----------------------------------------------------------------------------------------
# GIVEN: For use in all testing for the purpose of grading
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.spatial import distance
import timeit
from sklearn import model_selection
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):

    def __init__ (self):
        pass

    def fit (self,inputDf, outputSeries):
        return self

    def predict (self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return 1
        else:
            howMany = testInput.shape[0]
            return pd.Series(np.ones(howMany), index=testInput.index, dtype="int64")

        
#---------------------------------------------------------------------------------------#
    
   
def testAlwaysOneClassifier():
    wineDF, inputCols, outputCols = readData()
    testInputDF = wineDF.loc [0:9,inputCols]
    testOutputSeries= wineDF.loc [0:9,outputCols]
    trainInputDF = wineDF.loc [10:,inputCols]
    trainOutputSeries = wineDF.loc [10:,outputCols]

    print("testInputDF:", testInputDF, sep='\n', end='\n\n')
    print("testOutputSeries:", testOutputSeries, sep='\n', end='\n\n')
    print("trainInputDF:", trainInputDF, sep='\n', end='\n\n')
    print("trainOutputSeries:", trainOutputSeries, sep='\n', end='\n\n')
              
    AlwaysOneClassifier1 = AlwaysOneClassifier ()
    AlwaysOneClassifier1.fit (trainInputDF, trainOutputSeries)
    
    CorrectfirstRow = testOutputSeries.iloc[0]
    
    print ("Correct Answer: " + str (CorrectfirstRow))
    
    manyFirstRow = testInputDF.iloc[0,:]
    
    predictedAnswer= AlwaysOneClassifier1.predict(manyFirstRow)
    
    print ("Predicted Answer:" + str (predictedAnswer))
    
    correctTestSet = testOutputSeries.loc [0:9]
    
    print ("Correct Answers:" )
    
    print (correctTestSet)
    
    predictedTestSet = AlwaysOneClassifier1.predict(testInputDF)
    
    print ("Predicted Answers:" )
    
    print (predictedTestSet)
    
    print ("Accuracy:" + str (accuracyOfActualVsPredicted(correctTestSet, predictedTestSet)))
    
def findNearestLoop (df, testRow):

    minID = 0
    minDist = distance.euclidean (df.iloc[0,:], testRow)
    
    for rowID in range (df.shape[0]):
        if distance.euclidean (df.iloc [rowID,:], testRow) < minDist:
            minDist = distance.euclidean (df.iloc [rowID,:], testRow) 
            minID = rowID
    return df.index [minID]


def findNearestHOF (df, testRow):
    
    distances = df.apply (lambda row : distance.euclidean(row, testRow) , axis =1 )
    return distances.idxmin()


def testFindNearest():
    
    df, inputCols, outputCols = readData()
    startTime = timeit.default_timer()
    
    for i in range(100):
        findNearestLoop(df.iloc[100:107, :], df.iloc[90, :])
        
    elapsedTime = timeit.default_timer() - startTime
    
    print (findNearestLoop(df.iloc[100:107, :], df.iloc[90, :]))
    
    print("findNearestLoop timing:")
    print(elapsedTime,"seconds \n")
    
    startTime = timeit.default_timer()
    for i in range(100):
        findNearestHOF(df.iloc[100:107, :], df.iloc[90, :])
        
    elapsedTime = timeit.default_timer() - startTime
    
    print (findNearestHOF(df.iloc[100:107, :], df.iloc[90, :]))    
    print("findNearestHOF timing:")
    print(elapsedTime,"seconds \n")

#we see that when we use higher order function it takes less time as 
#compared to while using a loop
#


class OneNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__ (self):
        self.inputDf = None 
        self.outputSeries = None

    def fit (self,inputDf, outputSeries):
        self.inputDf = inputDf
        self.outputSeries = outputSeries 

    def predict (self, testInput):
        
        #label = findNearestHOF(testInput, self.inputDf)
        if isinstance(testInput, pd.core.series.Series):
            return self.__predictOne (testInput)
            
            #return self.outputSeries.loc [label]
        else:
            #howMany = testInput.shape[0]
            #return pd.Series(np.ones(howMany), index=testInput.index, dtype="int64")
            
            outputSeriesForDF = testInput.apply (lambda redRow: self.__predictOne (redRow) ,axis=1)
            return outputSeriesForDF
         
    def __predictOne (self, testInput):
        label = findNearestHOF(self.inputDf, testInput)
        return self.outputSeries.loc [label]
    
        
def testOneNNClassifier():
    wineDF, inputCols, outputCol = readData()
    testInputDF = wineDF.loc [0:9,inputCols]
    testOutputSeries= wineDF.loc [0:9,outputCol]
    trainInputDF = wineDF.loc [10:,inputCols]
    trainOutputSeries = wineDF.loc [10:,outputCol]
    
    testCase= OneNNClassifier()
    testCase.fit (trainInputDF, trainOutputSeries)
    
    CorrectAnswer = testOutputSeries.loc [2]
    predictRow = wineDF.loc [2,inputCols]
    predictedAnswer = testCase.predict(predictRow)

    print ("Correct Answer: " + str (CorrectAnswer))
    print ("Predicted Answer" + str (predictedAnswer))
    
    correctTestSet = testOutputSeries.loc [0:9]
    predictedTestSet = testCase.predict(testInputDF)
    
    print ("Correct Answer:")
    print (correctTestSet)
    print ("Predicted Answer:")
    print (predictedTestSet)
    print ("Accuracy:" + str (accuracyOfActualVsPredicted(correctTestSet, predictedTestSet)))
    

def cross_val_score_manual (model,inputDF, outputSeries, k, verbose):
    
    numberOfElements = inputDF.shape[0]
    foldSize = numberOfElements/k
    
    results = []
    
    for i in range (k):
        start = int(i*foldSize)
        uptoNotIncluding = int((i+1)*foldSize)
        testInputDF = inputDF.iloc [start:uptoNotIncluding, :]
        testOutputSeries = outputSeries.iloc [start:uptoNotIncluding]
        before = inputDF.iloc [:start, :]
        after = inputDF.iloc [uptoNotIncluding:, :]
        trainInputDF = pd.concat ([before, after])
        before1 = outputSeries.iloc [:start]
        after1 = outputSeries.iloc [uptoNotIncluding:]
        trainOutputSeries = pd.concat ([before1, after1])
        
        if (verbose): # print data structure info
            print("================================")
            print("Iteration:", i)
            print("Train input:\n", list(trainInputDF.index))
            print("Train output:\n", list(trainOutputSeries.index))
            print("Test input:\n", testInputDF.index)
            print("Test output:\n", testOutputSeries.index)
            print("================================")
            
        model= OneNNClassifier()
        model.fit (trainInputDF, trainOutputSeries)
            
        correctTestSet = testOutputSeries
        predictedTestSet = model.predict(testInputDF)
        
        results.append (accuracyOfActualVsPredicted (correctTestSet, predictedTestSet))
        
    return results 
        
def testCVManual (model, k):
    
    wineDF, inputCols, outputCol = readData()
    inputDF = wineDF.loc [:,inputCols]
    outputSeries = wineDF.loc [:, outputCol]
    
    accuracies = cross_val_score_manual (model, inputDF, outputSeries, k, verbose= True)
    print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
    

def testCVBuiltIn (model,k):
    
    wineDF, inputCols, outputCol = readData()
    inputDF = wineDF.loc [:,inputCols]
    outputSeries = wineDF.loc [:, outputCol]
    
    model= OneNNClassifier()
    k =10
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies = model_selection.cross_val_score(model, inputDF, outputSeries, cv=k, scoring=scorer)

    print("Accuracies:", accuracies)
    print("Average:", np.mean(accuracies))
    

def compareFolds():
    
    wineDF, inputCols, outputCol = readData()
    inputDF = wineDF.loc [:,inputCols]
    outputSeries = wineDF.loc [:, outputCol]
    model = OneNNClassifier ()
    
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies1 = model_selection.cross_val_score(model, inputDF, outputSeries, cv=10, scoring=scorer)
    accuracies2 = model_selection.cross_val_score(model, inputDF, outputSeries, cv=3, scoring=scorer)

    print("Mean accuracy for k = 10 : ", np.mean (accuracies1))
    print("Mean accuracy for k = 3 : ", np.mean (accuracies2))
    
    
def standardize (df, ls):
    
    #df = df.loc [:, ls]
    mean = df.loc [:, ls].mean()
    std = df.loc [:, ls].std()
    
    newDf = df.loc [:,ls] = (df.loc[:,ls]- mean) /std
    return newDf

    #newDf = df.loc [:,ls] = (df.loc [:,ls] - df.loc [:,ls].mean )/ df.loc [:,ls].std()
    #the mean and std are the mean and std of the specified colums 


def normalize (df,ls):
    
    dfMin = df.loc [:, ls].min()
    dfMax = df.loc [:, ls].max()

    newDf = df.loc [:,ls] = (df.loc [:,ls] - dfMin)/(dfMax -dfMin)
    return newDf

def comparePreprocessing() :
    wineDF, inputCols, outputCol = readData()
    model = OneNNClassifier ()
    inputDF = wineDF.loc [:,inputCols]
    outputSeries = wineDF.loc [:, outputCol]
    
    k = 10
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies = model_selection.cross_val_score(model, inputDF, outputSeries, cv=k, scoring=scorer)
    print("Mean accuracy for original set: ", np.mean (accuracies))
    
    dfCopyStd = wineDF.copy()
    standardize (dfCopyStd, inputCols)
    stdInputDF = dfCopyStd.loc [:,inputCols]
    
    accuraciesStd = model_selection.cross_val_score(model, stdInputDF, outputSeries, cv=k, scoring=scorer)
    print("Mean accuracy for standardized set: ", np.mean (accuraciesStd))
    
    dfCopyNrml = wineDF.copy()
    normalize (dfCopyNrml, inputCols)
    nrmlInputDF = dfCopyNrml.loc [:,inputCols]
    
    AccuraciesNrml = model_selection.cross_val_score(model, nrmlInputDF, outputSeries, cv=k, scoring=scorer)
    print("Mean accuracy for normalized set : ", np.mean (AccuraciesNrml))
    
#Comments
''' We see that the mean accuracies for the standardized set is the highest 
followed by that for normalized test and finally for the original set because 
Standardized and normalized data are essential for accurate data analysis; 
it's easier to draw clear conclusions about your current data when you have other
data to measure it against.
Standardization and normalization comes into the picture when features of the input data set 
have large differences between their ranges.  
For example, for the models that are based on distance computation, 
if one of the features has a broad range of values, the distance will be
governed by this particular feature.

KEEPS MAGNITUDE IN RANGE 

The result obtained for 1-NN is 96.1% (z-transformed data) 
z-transformed data is a data in which all the values are transformed 
into z-scores (standardizing the distribution).
The z-transform is also called standardization or auto-scaling. 
z-Scores become comparable by measuring the observations in multiples 
of the standard deviation of that sample. 
Leave-one-out cross-validation is a special case of cross-validation 
in which the number of folds equals the number of instances in the data set.
the “1NN” results reported in wine.names are slightly higher than the
results we obtained here because with leave one out we have more folds than with
k cross validation--which gives us more training set and makes the result we 
obtained more accurate. 
''' 

def visualization():
    fullDF, inputCols, outputCol = readData()
    standardize(fullDF, inputCols)
    # other code here, discussed below
    sns.displot(fullDF.loc[:, 'Malic Acid']) #positively skewed
    sns.displot(fullDF.loc[:, 'Alcohol']) 
    print(fullDF.loc[:, 'Malic Acid'].skew())
    print(fullDF.loc[:, 'Alcohol'].skew()) #negatively skewed
    sns.jointplot(x='Malic Acid', y='Alcohol', data=fullDF.loc[:, ['Malic Acid', 'Alcohol']], kind='kde')
    sns.jointplot (x='Ash', y='Magnesium', data=fullDF.loc[:, ['Ash', 'Magnesium']], kind='kde')
    sns.pairplot(fullDF, hue=outputCol)
    plt.show() #code to make everything we have done show up
    
#Comments
''' The malic acid is positively skewed and the skew measure is 1.0396511925814444
the alcohol is negatively skewed and the skew measure is -0.051482331077132064
The combination of values that are the most likely for Ash is -0.2 and for 
Magnesium is -0.4.

Question c:
    If Proline has a positive value, which classification is most likely?
    If Proline has a positive value, classification 1 is most likely. 

Question d:
    Suppose you dropped most input columns from your dataset, keeping only 
    Diluted and Proline. Clearly 1-NN accuracy would suffer. Would you expect 
    accuracy to drop a lot, or only some? Why?
    
    We would expect the accuracy would be descent and would drop by only some
    because each class is clustered with its own type as we see in the figure. 
    
Question e:
    Answer question (d) again, but for Nonflavanoid Phenols and Ash.
    
    We would expect the accuracy would drop a lot because different classes are  
    clustered together as we see in the figure.

'''

def testSubsets ():
    fullDF, inputCols, outputCol = readData()
    standardize(fullDF, inputCols)
    

    model = OneNNClassifier ()

    outputSeries = fullDF.loc [:,outputCol]
    inputDF1 = fullDF.loc [:, ['Diluted', 'Proline']]
    k = 3
    
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies1 = model_selection.cross_val_score(model, inputDF1, outputSeries, cv=k, scoring=scorer)
    
    print ('Accuracy for Diluted and Proline attributes: ', np.mean(accuracies1) )
    

    inputDF2 = fullDF.loc [:, ['Nonflavanoid Phenols', 'Ash']]
    
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies2 = model_selection.cross_val_score(model, inputDF2, outputSeries, cv=k, scoring=scorer)
    print ('Accuracy for Nonflavanoid Phenols and Ash attributes: ', np.mean (accuracies2 ))
    
    
#Comments 
'''
Accuracy for Diluted and Proline attributes: 0.8426553672316385
Accuracy for Nonflavanoid Phenols and Ash attributes: 0.5447269303201506
    
Question f:
    Do your experimental results match your hypotheses in (d) and (e) based 
    on the pair plot? If not, revise your answers to d and e and explain what’s
    happening.

    Yes, our experimental results match our hypotheses in (d) and (e) based 
    on the pair plot. 
'''

class kNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
        self.k=k
        self.inputDF = None
        self.outputSeries = None

        
    def fit(self, inputDF, outputSeries):
        self.inputDF = inputDF
        self.outputSeries = outputSeries
        return self
    
    def predict(self, testInput):
         if isinstance(testInput, pd.core.series.Series):
             return self.__predOfKNearest(testInput)
         else:
             result = testInput.apply(lambda row: self.__predOfKNearest(row), axis = 1)
             return result

    #Step1: find all the distances, 
    #step2: figure out which k distances are smallest and get the corr rows
    #step3: do a mode among the predictions from those k smallest
        
    #findnearest
    #another method nsmallest (k) and call it on the structure of distances
    #mode just get the first one in the series 
      
    #classification: we have 3 classes in wine dataset, whichever class has
    #the highest frequency among the neighbors will be our predicted answer
    
    def __predOfKNearest(self, testInput):
        distances = self.inputDF.apply(lambda row: distance.euclidean(row, testInput), axis=1)
        nearest = distances.nsmallest(self.k).index
        return self.outputSeries.loc[nearest].mode().loc[0]
        
            
def testkNN():
     df, inputCols, outputCol = readData(None)
     model1 = OneNNClassifier()
     model8 = kNNClassifier(8)
     
     scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
     accuracies1 = model_selection.cross_val_score(model1, df.loc[:,inputCols], df.loc[:,outputCol], cv=10, scoring=scorer)
     print("Unaltered dataset, 1NN, accuracy: ",np.mean(accuracies1))
     
     dfCopy = df.copy()
     standardize(dfCopy, inputCols)
     accuraciesStd1 = model_selection.cross_val_score(model1, dfCopy.loc[:, inputCols], df.loc[:, outputCol], cv=10, scoring=scorer)
     print("Standardized dataset, 1NN, accuracy: ",np.mean(accuraciesStd1))
     
     dfCopy = df.copy()
     standardize(dfCopy, inputCols)
     accuraciesStd8NN = model_selection.cross_val_score(model8, dfCopy.loc[:, inputCols], df.loc[:, outputCol], cv=10, scoring=scorer)
     print("Standardized dataset, 8NN, accuracy:", np.mean(accuraciesStd8NN))
     
#Comments
# the 8-NN model has higher accuracy than the 1-NN model and is much smoother 
#and is able to generalize well on previously unseen problems with unknown 
#solution because of the fact that a higher value
#of k reduces the overall complexity and flexibility of the model by reducing
#the edginess by taking more data into consideration. Generally, a large value of 
#K is more accurate as it tends to reduce the overall noise but is not always true. 
#K-NN Algorithm uses the entire dataset in its training phase. 
#Every time a prediction is needed for an unseen data instance, 
#the whole training dataset is searched for the k-most similar instances,
#and the data with the closest instance is ultimately returned as the prediction. 
#Let's use a straightforward example: If an apple resembles fruits like pear,
#cherry, or peach more than animals—such as monkeys, cats, 
#or dogs—it is most likely a fruit.

     
def paramSearchPlot():
     neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
     accuracies = neighborList.apply(calMeanAccuracy)
     print(accuracies)
     plt.plot(neighborList, accuracies)
     plt.xlabel('Neighbors')
     plt.ylabel('Accuracy')
     plt.show()
     print(neighborList.loc[accuracies.idxmax()])
     
def calMeanAccuracy(k):
     model = kNNClassifier(k)
     df, inputCols, outputCol = readData(None)
     standardize(df, inputCols)
     scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
     standardizedAccu = model_selection.cross_val_score(model, df.loc[:,inputCols], df.loc[:,outputCol], cv=10, scoring=scorer)
     meanAccu = np.mean(standardizedAccu)
     return meanAccu

def paramSearchPlotBuiltIn():
     df, inputCols, outputCol = readData(None)
     standardize(df, inputCols)
     stdInputDF = df.loc[:,inputCols]
     outputSeries = df.loc[:,outputCol]
     neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
     
     alg = KNeighborsClassifier(n_neighbors = 8)
     cvScores = model_selection.cross_val_score(alg, stdInputDF, outputSeries, cv=10, scoring='accuracy')
     print("Standardized dataset, 8NN, accuracy:", np.mean(cvScores))
     
     accuracies = neighborList.apply(lambda row: model_selection.cross_val_score(kNNClassifier(row), stdInputDF, outputSeries, cv=10, scoring='accuracy').mean())
     plt.plot(neighborList, accuracies)
     plt.xlabel('Neighbors')
     plt.ylabel('Accuracy')
     plt.show()
     print(accuracies)


def testMain():
    '''
    This function runs all the tests we'll use for grading. Please don't change it!
    When certain parts need to be graded, uncomment those parts only.
    Please keep all the other parts commented out for grading.
    '''
    pass

    print("========== testAlwaysOneClassifier ==========")
    testAlwaysOneClassifier()

    print("========== testFindNearest() ==========")
    testFindNearest()

    print("========== testOneNNClassifier() ==========")
    testOneNNClassifier()

    print("========== testCVManual(OneNNClassifier(), 5) ==========")
    testCVManual(OneNNClassifier(), 5)

    print("========== testCVBuiltIn(OneNNClassifier(), 5) ==========")
    testCVBuiltIn(OneNNClassifier(), 5)

    print("========== compareFolds() ==========")
    compareFolds()

    print("========== testStandardize() ==========")
    testStandardize()

    print("========== testNormalize() ==========")
    testNormalize()

    print("========== comparePreprocessing() ==========")
    comparePreprocessing()

    print("========== visualization() ==========")
    visualization()

    print("========== testKNN() ==========")
    testkNN()

    print("========== paramSearchPlot() ==========")
    paramSearchPlot()

    print("========== paramSearchPlotBuiltIn() ==========")
    paramSearchPlotBuiltIn()
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Reading in the data" step
def readData(numRows=None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids",
                 "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows=numRows)

    # Need to mix this up before doing CV
    wineDF = wineDF.sample(frac=1, random_state=50).reset_index(drop=True)

    return wineDF, inputCols, outputCol
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Testing AlwaysOneClassifier" step
def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0

    return accuracy
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def operationsOnDataFrames():
    d = {'x': pd.Series([1, 2], index=['a', 'b']),
         'y': pd.Series([10, 11], index=['a', 'b']),
         'z': pd.Series([30, 25], index=['a', 'b'])}
    df = pd.DataFrame(d)
    print("Original df:", df, type(df), sep='\n', end='\n\n')

    cols = ['x', 'z']

    df.loc[:, cols] = df.loc[:, cols] / 2
    print("Certain columns / 2:", df, type(df), sep='\n', end='\n\n')

    maxResults = df.loc[:, cols].max()
    print("Max results:", maxResults, type(maxResults), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def testStandardize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    standardize(df, colsToStandardize)
    print("After standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of standardization:
    print("Means are approx 0:", df.loc[:, colsToStandardize].mean(), sep='\n', end='\n\n')
    print("Stds are approx 1:", df.loc[:, colsToStandardize].std(), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Normalization on a DataFrame" step
def testNormalize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    normalize(df, colsToStandardize)
    print("After normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of normalization:
    print("Maxes are 1:", df.loc[:, colsToStandardize].max(), sep='\n', end='\n\n')
    print("Mins are 0:", df.loc[:, colsToStandardize].min(), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------


