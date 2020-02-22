// Databricks notebook source
// MAGIC %python
// MAGIC print("Hello World! Term Project Credit Card Fraud Detection using Random Forest Classification")

// COMMAND ----------

val dataPath = "/FileStore/tables/creditcard.csv"
val dataset = sqlContext.read.format("com.databricks.spark.csv")
  .option("header","true")
  .option("inferSchema", "true")
  .load(dataPath)

display(dataset)

// COMMAND ----------

val df = spark.read.format("csv").option("header", "true").load(dataPath)

df.printSchema();

// COMMAND ----------

val ccdataframe = df.na.fill(0)
ccdataframe.show

// COMMAND ----------

val nonFeatureCols = Array("V28")
val featureCols = ccdataframe.columns.diff(nonFeatureCols)

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

val df3 = assembler.transform(dataset)

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("Class").setOutputCol("label")
val df4 = labelIndexer.fit(df3).transform(df3)

// COMMAND ----------

val Array(trainingData, testData) = df4.randomSplit(Array(0.7, 0.3))

// COMMAND ----------

import org.apache.spark.ml.classification.RandomForestClassifier

val classifier = new RandomForestClassifier()
  .setImpurity("gini")
  .setMaxDepth(3)
  .setNumTrees(20)
  .setFeatureSubsetStrategy("auto")
  .setSeed(5043)

val model = classifier.fit(trainingData)

// COMMAND ----------

val predictions = model.transform(testData)

// COMMAND ----------

// Displays a column of prediction
display(predictions)

// COMMAND ----------

//Insert an index column
import org.apache.spark.sql.functions._
val predictions_WithID = predictions.withColumn("id", monotonically_increasing_id)
display(predictions_WithID)

// COMMAND ----------

val fraudDF = predictions_WithID.filter("Class = '1'")
display(fraudDF)

// COMMAND ----------

//###############################################################################################################
//[1] True Negative 
// Class = 1; Label = 1; Prediction = 0
//Comparison of the Expected of the Prediction, what is labeled as fraudulent is predicted to be non-fraudulent
//###############################################################################################################

val filter_FraudDF = fraudDF
  .filter($"label" === "1.0" && $"prediction" === "0.0")

// COMMAND ----------

// [1] True Negative 
// Class = 1; Label = 1; Prediction = 0 
filter_FraudDF.select("id","Class", "label", "prediction").count()

// COMMAND ----------

// [1] True Negative 
// Class = 1; Label = 1; Prediction = 0 
filter_FraudDF.select("id","Class", "label", "prediction").show(1000)

// COMMAND ----------

//###############################################################################################################
//[2] True Positive 
// Class = 1; Label = 1; Prediction = 1
//Comparison of the Expected of the Prediction, what is labeled as fraudulent is predicted to be non-fraudulent
//###############################################################################################################

val filter_FraudDF_Prediction = fraudDF
  .filter($"label" === "1.0" && $"prediction" === "1.0")

// COMMAND ----------

//[2] True Positive 
// Class = 1; Label = 1; Prediction = 1
// Count
filter_FraudDF_Prediction.select("id","Class", "label", "prediction").count()

// COMMAND ----------

filter_FraudDF_Prediction.select("id","Class", "label", "prediction").show(1000)

// COMMAND ----------

val nonFraudDF = predictions_WithID.filter("Class = '0'")

// COMMAND ----------

//###############################################################################################################
//[3] False Positive 
// Class = 0; Label = 0; Prediction = 1 , no errors
//Comparison of the Expected and the Prediction, what is labeled as non-fraudulent is predicted to be fraudulent
//###############################################################################################################
val filter_NonFraudDF = nonFraudDF
  .filter($"label" === "0.0" && $"prediction" === "1.0")

// COMMAND ----------

//[3] False Positive 
// Class = 0; Label = 0; Prediction = 1
// Count
filter_NonFraudDF.select("id","Class", "label", "prediction").count()

// COMMAND ----------

//Comparison of the the Expected and Prediction, does not exist, no errors
filter_NonFraudDF.select("id","Class", "label", "prediction").show(100)

// COMMAND ----------

//###############################################################################################################
//[4] False Negative 
// Class = 0; Label = 0; Prediction = 0
//Comparison of the Expected and the Prediction, what is labeled as non-fraudulent is predicted to be fraudulent
//###############################################################################################################

val filter_NonFraudDF_Prediction = nonFraudDF
  .filter($"label" === "0.0" && $"prediction" === "0.0")

// COMMAND ----------

//[4] False Positive 
// Class = 0; Label = 0; Prediction = 0
// Count
filter_NonFraudDF_Prediction.select("id","Class", "label", "prediction").count()

// COMMAND ----------

filter_NonFraudDF_Prediction.select("id","Class", "label", "prediction").show()

// COMMAND ----------

//Total number of records
predictions_WithID.select("id","Class", "label", "prediction").count()

// COMMAND ----------


