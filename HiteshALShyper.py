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


def AlStraininghyper(spark,training, model_loc):
 

    # Check progress
    print("Loading Training and Validation Data")
    
    trainingdata = spark.read.parquet("hdfs:/user/hlp276/train.parquet")
    valdata = spark.read.parquet("hdfs:/user/hlp276/val.parquet")
   
    trainingdata.createOrReplaceTempView("trainingdata")
   
    # Get a list format for all track_ids corresponding to each user_id
    valdata.createOrReplaceTempView("valdata")
    
    labels = spark.sql('SELECT user_id, collect_list(book_id) AS label FROM valdata GROUP BY user_id')
    labels.createOrReplaceTempView("labels")

    # Extract users column
    valuser = labels.select("user_id").distinct()
    
    maxscore= 0.0
    hyperconfigmodel = None
    #Hyperparameters used for best scores
    rank = [10, 20, 50, 100]
    reg = [0.01, 0.1, 0.5, 1]
    alpha = [0.1, 1, 2, 5, 10]
   
    
    #trying Various configurations for Hyperparameter Tuning
    for hyperparam in itertools.product(*[rank, reg, alpha]):
        rankpara = hyperparam[0] # Rank
        regularization = hyperparam[1]
        alphapara = hyperparam[2] # alpha
        # Create a new als model based on the combination of parameters
        als = ALS(rank = rankpara, regParam = regularization,
                  alpha = alphapara,
                  implicitPrefs=True,
                  userCol="user_id",
                  itemCol="book_id",
                  ratingCol="rating")
        print(als)
        # Fit the model on Training data
        alsmodel = als.fit(trainingdata)
        print("Training Model")

        #Will  Try to get Recommendation on Validation Data
        alsrec= alsmodel.recommendForUserSubset(valuser, 500)
        alsrec.createOrReplaceTempView("alsrec")

      

       #performing comaparison between actual and recommended
        testingrecommend = (alsrec.select("user_id", explode("recommendations").alias("predvalue")).\
                           select("user_id", "predvalue.*"))
        testingrecommend .createOrReplaceTempView("testingrecommend")
        predicted = spark.sql('SELECT user_id, collect_list(book_id) AS prediction FROM testingrecommend GROUP BY user_id')
        predicted.createOrReplaceTempView("predicted")
        

        
        comparison= spark.sql('SELECT predicted.prediction AS predictions, labels.label AS label FROM predicted INNER JOIN labels ON predicted.user_id = labels.user_id')

       
        
        actualandpred= comparison.select("predictions","label")
        
        metrics = RankingMetrics(actualandpred.rdd)
        score = metrics.meanAveragePrecision
        if score> maxscore:
            maxscore = score
            hyperconfigmodel = alsmodel
            print("New best MAP {}".format(maxscore))
            print("Rank is {}".format( rankpara))
            print("Regularization term is {}".format(regularization))
            print("Alpha is {}".format(alphapara))
        

       

    # Save the moddel on the location with the best paramateres
    hyperconfigmodel.save(model_loc)



if __name__ == "__main__":

    # Creating a Spark Session
    spark = (SparkSession.builder.appName("AlS with hyperparam").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").getOrCreate())

    
    training= sys.argv[0]

    model_loc = sys.argv[1]

    AlStraininghyper(spark, training, model_loc)