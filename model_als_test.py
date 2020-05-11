import sys
import time 

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.mllib.evaluation import RankingMetrics


def main(spark, data_file_train, data_file_val):

    start = time.time()

    # reading training and validation files
    df_train = spark.read.parquet(data_file_train)
    df_val = spark.read.parquet(data_file_val)

    window_user_ordered = Window.partitionBy('user_id').orderBy('rating')
    window_user = Window.partitionBy('user_id')

    actual_df_val = df_val.withColumn('actual_books', F.collect_list('book_id').over(window_user_ordered)).groupBy('user_id').agg(F.max('actual_books').alias('actual_books'))

    print("Datasets loaded | Time taken: {}".format(time.time() - start))

    start = time.time()

    als = ALS(maxIter=10, regParam=0.001, userCol="user_id", itemCol="book_id", ratingCol="rating", rank=100)
    model = als.fit(df_train)

    print("Done with model fitting | Time taken: {}".format(time.time()-start))
    start = time.time()

    # predictions = model.transform(df_val)
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    # rmse = evaluator.evaluate(predictions)

    # print("RMSE: {}".format(rmse))

    recommendations = model.recommendForUserSubset(df_val,500)
    userPredictions=recommendations.select('user_id',F.explode('recommendations.book_id')).withColumn('pred_books', F.collect_list('col').over(window_user)).groupBy('user_id').agg(F.max('pred_books').alias('pred_books'))
    predAndLabels=userPredictions.join(actual_df_val,on='user_id').select('pred_books','actual_books')
    metrics=RankingMetrics(predAndLabels.rdd)
    score = metrics.meanAveragePrecision
    print('MAP for test data: {}'.format(score))
    print('Time taken: {}'.format(time.time() - start))



if __name__ == "__main__":
    spark = SparkSession.builder.appName('model').getOrCreate()
    data_file_train = "recom/train_zero_1per.parquet"
    data_file_val = "recom/test_zero_1per.parquet"
    main(spark, data_file_train, data_file_val)
