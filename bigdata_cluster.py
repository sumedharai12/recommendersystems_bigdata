ssh sr5387@dumbo.hpc.nyu.edu # load the cluster
module load python/gnu/3.6.5
module load spark/2.4.0

# will try to run this on my local first

pyspark # load Spark
interactions = spark.read.csv("hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv", 
schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
interactions = interactions.na.drop()

interactions_master = interactions
# create a subsample of 5%
# interactions = interactions.sample(False,0.05,seed=1)

userid = spark.read.csv("hdfs:/user/bm106/pub/goodreads/user_id_map.csv", 
schema='user_id_csv INT, user_id INT')
userid = userid.na.drop()

bookid = spark.read.csv("hdfs:/user/bm106/pub/goodreads/book_id_map.csv", 
schema='user_id_csv INT, user_id INT')

# shape of a dataframe
# print((interactions.count(), len(interactions.columns)))
# number of rows 
# interactions.count()

interactions.createOrReplaceTempView('interactions')

g10 = spark.sql('SELECT user_id, COUNT(book_id) as c FROM interactions GROUP BY user_id HAVING c>10')
g10.createOrReplaceTempView('g10')

# temp_df - RENAME THIS - ACTUAL INTERACTIONS DATA THAT I WILL BE USING TO CREATE TRAIN/VAL/TEST

temp_df = spark.sql('SELECT interactions.* FROM interactions INNER JOIN g10 on interactions.user_id=g10.user_id')
temp_df.createOrReplaceTempView('temp_df') 
n = spark.sql('SELECT Count(Distinct(user_id)) FROM temp_df')
n.show() #766717

# Make a folder in dumbo hdfs -recom and save this file in parquet format
temp_df.write.parquet("temp_df.parquet")

# read the file 
temp_df = spark.read.parquet("recom/temp_df.parquet")
temp_df.createOrReplaceTempView('temp_df') 
#counting unique users
# m = spark.sql('SELECT Count(Distinct(user_id)) FROM interactions')
# m.show()  #342415
# emp_df.coalesce(1).write.parquet("recom/temp_df.parquet")


# making a df of less than 10 users
# less10 = spark.sql("""
# SELECT user_id, COUNT(book_id) as c
# FROM interactions
# GROUP BY user_id
# HAVING c<=10
# """)

# less10.count() #246910 
# less10.createOrReplaceTempView('less10')
# create a copy for backup
#interactions_master = interactions

#interactions = interactions[~interactions.user_id.isin(less10.user_id)]

#interactions = interactions.where('user_id NOT IN (SELECT user_id FROM less10)')

# temp_df = spark.sql("SELECT * FROM interactions WHERE user_id NOT IN (SELECT user_id FROM less10)")
# temp_df.createOrReplaceTempView('temp_df')
# temp_df.coalesce(1).write.parquet("temp_df.parquet")

# n = spark.sql('SELECT Count(Distinct(user_id)) FROM temp_df')

# making validation and test sets using 40% data  - 306688 (approx)

subset_users = spark.sql("SELECT user_id FROM (SELECT Distinct(temp_df.user_id) FROM temp_df) ORDER BY RAND() LIMIT 306688") # made a limit for 40% of users
subset_users.createOrReplaceTempView('subset_users')
subset_df = spark.sql('SELECT temp_df.* FROM temp_df INNER JOIN subset_users on temp_df.user_id=subset_users.user_id')
# subset_df = temp_df[interactions.user_id.isin(val_users.user_id)]

subset_df.createOrReplaceTempView('subset_df')

from pyspark.sql.functions import lit

#taking half of users for both val and test sets
fractions = subset_df.select("user_id").distinct().withColumn("fraction", lit(0.5)).rdd.collectAsMap()
fractions.createOrReplaceTempView('fractions')

holdout_df = subset_df.sampleBy("user_id", fractions, 12)
val_df = holdout_df.sample(False,0.5,seed=1)
test_df = holdout_df.subtract(val_df)

train_df = temp_df.subtract(holdout_df)

val_df.write.mode('overwrite').parquet('recom/val_df.parquet') # 22,800,716 - new 
test_df.write.mode('overwrite').parquet('recom/test_df.parquet') # 22,804,855 - new
train_df.write.mode('overwrite').parquet('recom/train_df.parquet') # 182,582,423 - new

# train_df.count()
# test_df.count()
# val_df.count()

temp_df.count() # 228,187,994

columns_to_drop = ['is_read', 'is_reviewed']
val_df = val_df.drop(*columns_to_drop)
test_df = test_df.drop(*columns_to_drop)
train_df = train_df.drop(*columns_to_drop)

val_df.write.mode('overwrite').parquet('recom/val.parquet')
test_df.write.mode('overwrite').parquet('recom/test.parquet')
train_df.write.mode('overwrite').parquet('recom/train.parquet')


#---------------------------------------
# SPLITTING DONE

# Working on ALS now

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
als = ALS(maxIter=10, regParam=0.01, userCol="user_id”, itemCol=“book_id”, ratingCol="rating")
model = als.fit(train)














#------------------------------------------
# OLD CODE

# check = spark.sql("""
# SELECT user_id, COUNT(book_id) as c
# FROM temp_df
# GROUP BY user_id
# HAVING c<=10
# """)

# from pyspark.sql.functions import col, countDistinct
# u = interactions.agg(countDistinct(col("user_id")).alias("count")).show() # unique users - 876145


# val_users = spark.sql("""
# SELECT user_id 
# FROM (SELECT user_id,COUNT(book_id) AS c FROM interactions GROUP BY user_id) TEMP 
# WHERE TEMP.c>=10 ORDER BY RAND() LIMIT 108429
# """)

# val_super_df = interactions[interactions.user_id.isin(val_users.user_id)]

# from pyspark.sql.functions import lit
# fractions = val_super_df.select("user_id").distinct().withColumn("fraction", lit(0.5)).rdd.collectAsMap()
# val_df = val_super_df.sampleBy("user_id", fractions, 12)
# train_df = interactions.subtract(val_df)

