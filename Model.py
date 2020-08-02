# -*- coding: utf-8 -*-
"""
Erickson, Holly

1. Movie Recommendation Engine

a. Prepare Data

Load the data from the ratings.csv and movies.csv files and combine them on movieId. 
The resultant data set should contain all of the user ratings and include movie titles.

"""

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
#%%
spark = SparkSession.builder.appName("Week11").getOrCreate()

ratings_path = 'C://Master/Semester_5/movielens/ratings.csv'
movies_path = 'C://Master/Semester_5/movielens/movies.csv'
         
df_ratings = spark.read.load(
  ratings_path,
  format="csv",
  sep=",",
  inferSchema=True,
  header=True
)

df_movies = spark.read.load(
  movies_path,
  format="csv",
  sep=",",
  inferSchema=True,
  header=True
)

# combine them on movieId
joinType = "outer"
df_join = df_ratings.join(df_movies, "movieId")
df_join.printSchema()

#%%
"""
b. Train Recommender

Create a movie recommendation model using collaborative filtering. 

"""

#split the data into train / test
train, test = df_join.randomSplit([0.8, 0.2], seed = 2018)

#%%
# Use the RegressionEvaluator to calculate the root-mean-square error of the model.

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(train)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

"""
Output:
Root-mean-square error = 1.089806933470268
"""

#%%
"""
#c. Generate top 10 movie recommendations
Generate the top ten recommendations for each user. 
Using the show method, print the recommendations for the user IDs, 127, 151, and 300. 
You should not truncate the results and so should call the show method like this recommendations_127.show(truncate=False).
"""
userRecs = model.recommendForAllUsers(10)
#%%
userRecs.filter(userRecs.userId == 127).show(truncate=False)
userRecs.filter(userRecs.userId == 151).show(truncate=False)
userRecs.filter(userRecs.userId == 300).show(truncate=False)

"""
Output:
+------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|userId|recommendations                                                                                                                                                                      |
+------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|127   |[[48877, 11.117], [4678, 9.030599], [8228, 8.817692], [3358, 8.782554], [6380, 8.649328], [25850, 8.646203], [1293, 8.566614], [42723, 8.558058], [3477, 8.508026], [3030, 8.415057]]|
+------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

+------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|userId|recommendations                                                                                                                                                                              |
+------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|151   |[[158783, 8.376521], [1147, 7.0066714], [1464, 6.8068166], [37384, 6.7451925], [928, 6.697016], [3270, 6.5985575], [56145, 6.576035], [2398, 6.5551996], [3925, 6.4736543], [3089, 6.398668]]|
+------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

+------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|userId|recommendations                                                                                                                                                                      |
+------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|300   |[[232, 7.9599967], [2491, 7.919761], [3503, 7.032155], [417, 7.0151343], [7234, 6.847977], [6385, 6.846513], [945, 6.843505], [1292, 6.8395333], [6993, 6.8005753], [7323, 6.709377]]|
+------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
"""
