#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import itertools
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import explode
from pyspark.sql import Window
from pyspark.sql import functions as F

def AlStraininghyper(spark,training, model_loc):
 

    # Check progress
    print("Loading Training and Validation Data")
    
    trainingdata = spark.read.parquet("hdfs:/user/hlp276/recom/train_zero.parquet")
    valdata = spark.read.parquet("hdfs:/user/hlp276/recom/val_zero.parquet")
   
    trainingdata.createOrReplaceTempView("trainingdata")
   
    # Get a list format for all track_ids corresponding to each user_id
    valdata.createOrReplaceTempView("valdata")
    
    window_user_ordered = Window.partitionBy('user_id').orderBy('rating')
    window_user = Window.partitionBy('user_id')

    actual_df_val = valdata.withColumn('actual_books', F.collect_list('book_id').over(window_user_ordered)).groupBy('user_id').agg(F.max('actual_books').alias('actual_books'))

    print("datasets loaded")
    
    maxscore= 0.0
    hyperconfigmodel = None
    #Hyperparameters used for best scores
    rank = [10,20,50,100, 200]
    reg = [0.01, 0.1, 0.5, 1]
       
    
    #trying Various configurations for Hyperparameter Tuning
    for hyperparam in itertools.product(*[rank, reg]):
        rankpara = hyperparam[0] # Rank
        regularization = hyperparam[1] #regularization
        
        # Create a new als model based on the combination of parameters
        als = ALS(rank = rankpara, regParam = regularization,
                  userCol="user_id",
                  itemCol="book_id",
                  ratingCol="rating")
        
        # Fit the model on Training data
        alsmodel = als.fit(trainingdata)
        print("Training Model")

        #Will  Try to get Recommendation on Validation Data
        recommendations = alsmodel.recommendForUserSubset(valdata,500)
        userPredictions=recommendations.select('user_id',F.explode('recommendations.book_id')).withColumn('pred_books', F.collect_list('col').over(window_user)).groupBy('user_id').agg(F.max('pred_books').alias('pred_books'))
        predAndLabels=userPredictions.join(actual_df_val,on='user_id').select('pred_books','actual_books')
        metrics=RankingMetrics(predAndLabels.rdd)
        
        score = metrics.meanAveragePrecision
        print("Rank is {}".format(rankpara))
        print("Regularization term is {}".format(regularization))
        print("MAP {}".format(score))
        if score> maxscore:
            maxscore = score
            hyperconfigmodel = alsmodel
            print("New best MAP {}".format(maxscore))
            print("Rank is {}".format( rankpara))
            print("Regularization term is {}".format(regularization))
            
        

       

    # Save the moddel on the location with the best paramateres
    hyperconfigmodel.save(model_loc)



if __name__ == "__main__":
    # Creating a Spark Session
    spark = (SparkSession.builder.appName("AlS with hyperparam").config("spark.executor.cores", "4").config("spark.executor.memory", "20g").config("spark.driver.cores", "6").config("spark.driver.memory", "32g").getOrCreate())

      
    
    training= sys.argv[0]

    model_loc = sys.argv[1]

    AlStraininghyper(spark, training, model_loc)