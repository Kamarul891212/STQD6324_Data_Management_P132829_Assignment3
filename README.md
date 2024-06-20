# STQD6324 Data Management

# Assignment 3

# Iris Dataset Classification using Spark Machine Learning Library

## Introduction

This assignment will use Spark MLlib to perform classification on the Iris dataset. The Iris dataset is accessed from the R environment or through url: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data.

The classification process will involve the following steps:

+ Loading the Iris dataset into Spark DataFrame
+ Splitting the dataset into training and testing sets.
+ Selecting a classification algorithms such as Decision Trees, Random Forest and Logistic Regression from Spark MLlib.
+ Employing a techniques such as cross-validation and grid search to fine tune the hyperparameters of the chosen algorithms.
+ Evaluating the performance of the tuned model using relevant evaluation metrics such as accuracy, precision, recll and F1-score.
+ Using the tuned model to generate predictions on the testing data.
+ Conducting a comparative analysis between the predicted labels and actual labels to assess the model's performance.

## Dataset

The dataset used in this analysis is **Iris dataset** that can be assessed from R environment. This dataset is widely-used in the field of machine learning and statistics. It was introduced by the British biologist and statistician Ronald A. Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems." The dataset contains **150 observations** of iris flowers. There are four features or attributes for each observation consists of sepal length, sepal width, petal length and petal width. The class of this dataset is the species of the iris flower consist of Iris setosa, Iris versicolor and Iris virginica.

The Iris dataset is an excellent starting point for beginners to learn about data analysis and machine learning. Its relatively small size allows for quick experimentation and learning. The clear separation between the species in the feature space makes it suitable for learning classification algorithms.

## Spark Machine Learning Library

**Apache Spark MLlib** is a powerful machine learning library integrated with Apache Spark, designed for scalable and easy to use machine learning. It provides a wide range of machine learning algorithms, including classification, regression, clustering and collaborative filtering. MLlib also offers tools for feature engineering, pipelines and evaluation metrics, making it a comprehensive solution for building and deploying machine learning models on large scale datasets.

MLlib supports multiple programming languages such as Python, Java, Scala and R and integrates seamlessly with Spark' core APIs for data processing, SQL, streaming and graph processing. This integration allows users to create complex data processing pipelines and scale their machine learning tasks across distributed computing environments, making it suitable for big data applications.

## Data Exploratory

### Writing the Python Script using PuTTY 

+ The Iris dataset need to be uploaded into the HDFS ecosystem before loading the dataset into the Spark DataFrame. This analysis will use a terminal which is PuTTY to access a remote server with Spark installed. In PuTTY, we use 'vi' to create and edit a Python script named 'iris_classification.py'. The Python script will consist of all the steps from loading the dataset until comparing the predicted and actual labels to assess the performance of the tuned model.

```
vi iris_classification.py
```

+ Press 'i' to enter insert mode in 'vi' and write the complete Python script as below:

```
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

schema = StructType([
    StructField("sepal_length", DoubleType(), True),
    StructField("sepal_width", DoubleType(), True),
    StructField("petal_length", DoubleType(), True),
    StructField("petal_width", DoubleType(), True),
    StructField("species", StringType(), True)
])

iris_df = spark.read.csv("/user/maria_dev/kamarul/iris.data", header = False, schema = schema)
iris_df.show()

indexer = StringIndexer(inputCol="species", outputCol="label").fit(iris_df)

assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")

training_data, test_data = iris_df.randomSplit([0.8, 0.2], seed=1234)

rf = RandomForestClassifier(labelCol="label", featuresCol="features")

pipeline = Pipeline(stages=[indexer, assembler, rf])

param_grid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

cv_rf = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid_rf, numFolds=5)

cv_model_rf = cv_rf.fit(training_data)

predictions_rf = cv_model_rf.transform(test_data)

def evaluate(predictions):
    accuracy = evaluator.evaluate(predictions)
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)

    return accuracy, precision, recall, f1_score

accuracy_rf, precision_rf, recall_rf, f1_score_rf = evaluate(predictions_rf)

print("Random Forest - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_rf, precision_rf, recall_rf, f1_score_rf))
print("Test Error: %g" % (1.0 - accuracy_rf))


predictions_rf.select("prediction", "label").show()

spark.stop()
```

## Classification steps

### Step 1: Load the dataset into Spark MLlib

```
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

schema = StructType([
    StructField("sepal_length", DoubleType(), True),
    StructField("sepal_width", DoubleType(), True),
    StructField("petal_length", DoubleType(), True),
    StructField("petal_width", DoubleType(), True),
    StructField("species", StringType(), True)
])

iris_df = spark.read.csv("/user/maria_dev/kamarul/iris.data", header = False, schema = schema)
iris_df.show()
```

+ The initial part of the script above consist of the process to import the required libraries for the analysis and creating a Spark session with the application name "IrisClassification". The script also define the schema for the dataset and load the Iris dataset into a DataFrame. It also displays the first few rows of the DataFrame.


![Alt text](https://github.com/Kamarul891212/STQD6324_Data_Management_P132829_Assignment3/blob/master/images/image01.jpg)

### Step 2: Splitting the dataset into training and testing sets

```
indexer = StringIndexer(inputCol="species", outputCol="label").fit(iris_df)

assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")

training_data, test_data = iris_df.randomSplit([0.8, 0.2], seed=1234)
```

+ The DataFrame is splits into training and testing sets with 80% of the data is used for training and 20% is used for testing. The seed is set to "1234" for a random number generation to ensure reproducibility.

### Step 3: Selecting the Random Forest for the classification algorithm

```
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

pipeline = Pipeline(stages=[indexer, assembler, rf])

param_grid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
```

+ The Random Forest algorithm is used for this analysis. 

### Step 5: Evaluate the performance of the model

```
predictions_rf = cv_model_rf.transform(test_data)

def evaluate(predictions):
    accuracy = evaluator.evaluate(predictions)
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)

    return accuracy, precision, recall, f1_score

accuracy_rf, precision_rf, recall_rf, f1_score_rf = evaluate(predictions_rf)
```

+ The performance of the model is evaluated by relevant metrices consist of accuracy, precision, recall and F1 score.

### Step 6: Comparative analysis

```
print("Random Forest - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_rf, precision_rf, recall_rf, f1_score_rf))
print("Test Error: %g" % (1.0 - accuracy_rf))


predictions_rf.select("prediction", "label").show()

spark.stop()
```
![Alt text](https://github.com/Kamarul891212/STQD6324_Data_Management_P132829_Assignment3/blob/master/images/image02.jpg)

+ The Random Forest classifier achieved an impressive accuracy of 95.83%, precision of 96.35%, recall of 95.83%, and F1 score of 95.81%, with a test error of only 4.17%. These metrics indicate that the model is highly effective at correctly classifying the Iris species with minimal false positives and false negatives, demonstrating a good balance between precision and recall.

## Conclusion

The high performance metrics indicate that the Random Forest classifier is a robust model for classifying Iris species, showing excellent generalization to unseen data. The low test error further confirms the model's reliability in practical applications, making it suitable for deployment in real-world scenarios where accurate species identification is critical.

Despite the strong performance, further validation with different datasets or enhanced cross-validation techniques is recommended to ensure the model's robustness. Additionally, comparing the Random Forest model with other classification algorithms and exploring more advanced hyperparameter tuning methods could potentially yield even better results, ensuring the best possible model for Iris species classification.
