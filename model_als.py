import sys

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.mllib.evaluation import RankingMetrics


def main(spark, data_file_train, data_file_val):

    # reading training and validation files
    df_train = spark.read.parquet(data_file_train)
    df_val = spark.read.parquet(data_file_val)

    window_user_ordered = Window.partitionBy('user_id').orderBy('rating')
    window_user = Window.partitionBy('user_id')

    actual_df_val = df_val.withColumn('actual_books', F.collect_list('book_id').over(window_user_ordered)).groupBy('user_id').agg(F.max('actual_books').alias('actual_books'))

    print("datasets loaded")

    als = ALS(maxIter=10, regParam=0.01, userCol="user_id", itemCol="book_id", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(df_train)

    print("Done with model fitting")

    # predictions = model.transform(df_val)
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    # rmse = evaluator.evaluate(predictions)

    # print("RMSE: {}".format(rmse))

    recommendations = model.recommendForUserSubset(df_val,500)
    userPredictions=recommendations.select('user_id',F.explode('recommendations.book_id')).withColumn('pred_books', F.collect_list('col').over(window_user)).groupBy('user_id').agg(F.max('pred_books').alias('pred_books'))
    predAndLabels=userPredictions.join(actual_df_val,on='user_id').select('pred_books','actual_books')
    metrics=RankingMetrics(predAndLabels.rdd)
    print('MAP: {}'.format(metrics.meanAveragePrecision)) 


if __name__ == "__main__":
    spark = SparkSession.builder.appName('model').getOrCreate()
    data_file_train = "recom/train_zero.parquet"
    data_file_val = "recom/val_zero.parquet"
    main(spark, data_file_train, data_file_val)
