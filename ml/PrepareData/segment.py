from pyspark import SparkContext
from pyspark.sql import SQLContext 
import pandas as pd


sc = SparkContext() 
sqlContext=SQLContext(sc) 
df=pd.read_csv(r'game-clicks.csv')
sdf=sqlContext.createDataFrame(df)
