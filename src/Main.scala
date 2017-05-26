import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Dataset, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.supercsv.util.CsvContext

/**
  * Created by Developer on 26.05.2017.
  */
object Main {
  def main(args: Array[String]) = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("MLTutorial")
    val sparkContext = new SparkContext(sparkConf)

    val sqlContext = new SQLContext(sparkContext)
    import sqlContext.implicits._

    val data = sparkContext.textFile("C:\\Datasets\\breast-cancer-wisconsin.data.txt")

    val rdd = data.map(_.split(",")).filter(_ (6) != "?").map(_.drop(1))
      .map(_.map(_.toDouble))

    val labeledPoints = rdd.map(x => LabeledPoint(if (x.last == 4) 1 else 0, Vectors.dense(x.init))).toDS()

    val splits = labeledPoints.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    classsifyAndCrossValidateWithLR(training,test)

    classifyWithNN(training,test)
  }


  def classsifyAndCrossValidateWithLR(trainData:Dataset[LabeledPoint],testData:Dataset[LabeledPoint])=
  {
    val lr = new LogisticRegression()
      .setMaxIter(10)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(lr))

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val cvModel = cv.fit(trainData)

    val result = cvModel.transform(testData)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("LR Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
  }

  def classifyWithNN(trainData:Dataset[LabeledPoint],testData:Dataset[LabeledPoint]) =
  {
    val layers = Array[Int](9, 5, 4, 2)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setBlockSize(128)
      .setLayers(layers)
      .setSeed(1234L)
      .setMaxIter(100)

    val model =  trainer.fit(trainData)

    val result = model.transform(testData)

    val predictionAndLabels = result.select("prediction","label")

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    println("NN Test set accuracy = " + evaluator.evaluate(predictionAndLabels))

  }
}
