# ssh sr5387@dumbo.hpc.nyu.edu # load the cluster
# module load python/gnu/3.6.5
# module load spark/2.4.0

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit


def read_data(spark):
    interactions = spark.read.csv("hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv", 
    schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
    interactions = interactions.na.drop()
    interactions.createOrReplaceTempView('interactions')

    interactions = spark.sql('SELECT * FROM interactions WHERE rating!=0')

    # using 1% of total data
    interactions = interactions.sample(False, 0.01, seed=1)
    interactions.createOrReplaceTempView('interactions')
    
    print("1% subsample of 'interactions' dataset taken")


    # dropping all users with less than 10 interactions
    g10 = spark.sql('SELECT user_id, COUNT(book_id) as c FROM interactions GROUP BY user_id HAVING c>10')
    g10.createOrReplaceTempView('g10')

    # creating a df of the remaining users
    ss_df = spark.sql('SELECT interactions.* FROM interactions INNER JOIN g10 on interactions.user_id=g10.user_id')
    ss_df.createOrReplaceTempView('ss_df')
    # ss_df.count() # for 1% data: 171,023

    print("Created subset of users with more than 10 interactions")

    spark.sql('SELECT Count(Distinct(user_id)) FROM ss_df').show() # for 1% data: 10,482; for 5% data: 138,040; for 10% data:266,086

    # Make a folder in dumbo hdfs -recom and save this file in parquet format
    ss_df.write.mode('overwrite').parquet("recom/ss_zero_1per.parquet")
    print("Subset saved")

    # read the file 
    # ss_df = spark.read.parquet("recom/ss_zero_1per.parquet")
    # ss_df.createOrReplaceTempView('ss_df') 

    # making validation and test sets using 40% data on the new set - 106,436

    subset_users = spark.sql("SELECT user_id FROM (SELECT Distinct(ss_df.user_id) FROM ss_df) ORDER BY RAND() LIMIT 5242") # made a limit for 40% of users
    subset_users.createOrReplaceTempView('subset_users')
    subset_df = spark.sql('SELECT ss_df.* FROM ss_df INNER JOIN subset_users on ss_df.user_id=subset_users.user_id')
    # subset_df = temp_df[interactions.user_id.isin(val_users.user_id)]

    subset_df.createOrReplaceTempView('subset_df')

    print("40% of all unique users extracted for validation and test sets")

    #taking half of users for both val and test sets
    fractions = subset_df.select("user_id").distinct().withColumn("fraction", lit(0.5)).rdd.collectAsMap()

    # holdout right now has 40% of the total users
    holdout_df = subset_df.sampleBy("user_id", fractions, 12)
    holdout_df.createOrReplaceTempView('holdout_df')
    print("Half interactions of all 40% users captured")

    # creating training dataset first
    train_df = ss_df.subtract(holdout_df)
    print("Train set created")
    train_df.createOrReplaceTempView('train_df')

    #removing books from val and test which are not present in the training set
    m = spark.sql('''
    SELECT DISTINCT(holdout_df.book_id) as book_id 
    FROM holdout_df 
    INNER JOIN train_df 
    ON holdout_df.book_id=train_df.book_id
    ''')

    m.createOrReplaceTempView('m')

    holdout_df = spark.sql('''
    SELECT holdout_df.* 
    FROM holdout_df
    INNER JOIN m
    ON holdout_df.book_id=m.book_id
    ''')

    # getting a list of half of the unique users in holdout
    half_users = spark.sql('SELECT DISTINCT(holdout_df.user_id) FROM holdout_df ORDER BY RAND() LIMIT 2621')
    half_users.createOrReplaceTempView('half_users')

    val_df = spark.sql('SELECT holdout_df.* FROM holdout_df INNER JOIN half_users on holdout_df.user_id=half_users.user_id')
    print("Validation set created")
    val_df.createOrReplaceTempView('val_df')
    test_df = holdout_df.subtract(val_df)
    print("Test set created")
    test_df.createOrReplaceTempView('test_df')
    

    val_df.write.mode('overwrite').parquet('recom/val_zero_1per.parquet')
    print("Validation set saved in 'recom' folder")
    test_df.write.mode('overwrite').parquet('recom/test_zero_1per.parquet')
    print("Test set saved in 'recom' folder")
    train_df.write.mode('overwrite').parquet('recom/train_zero_1per.parquet')
    print("Train set saved in 'recom' folder")

if __name__ == "__main__":
    spark = SparkSession.builder.appName('read_data_final').getOrCreate()
    read_data(spark)