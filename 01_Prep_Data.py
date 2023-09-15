# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the data required for the Zingg Product Matching with Image Data solution accelerator.  This notebook ###.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC To demonstrate how we might use Zingg to product comparisons leveraging image inputs, we first need access to a dataset containing product metadata and images.  The [Amazon-Berkeley Objects dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (ABO) is one such dataset.
# MAGIC
# MAGIC Assembled and maintained by Matthieu Guillaumin (Amazon.com), Thomas Dideriksen (Amazon.com), Kenan Deng (Amazon.com), Himanshu Arora (Amazon.com), Jasmine Collins (UC Berkeley) and Jitendra Malik (UC Berkeley) and made available under a Creative Commons Attribution 4.0 International Public License (CC BY 4.0), this dataset provides access to a wide variety of information about products available on the Amazon website. More details about the dataset and its preparation can be found in [this paper](https://amazon-berkeley-objects.s3.amazonaws.com/static_html/ABO_CVPR2022.pdf)
# MAGIC
# MAGIC
# MAGIC For our purposes, we will make use of the *downscaled (small) product images and metadata* dataset which provides access to a considerable amount of product metadata and 256-pixel images for nearly 150k products. The ABO dataset is quite well curated.  In order to demonstrate matching, we'll need to introduce some duplicates into the dataset.  After we've loaded to original ABO dataset, we will insert duplicates, some with metadata and image modifications.  We will then divide the dataset into a set of records on which we might initialize our fuzzy matching solution and another set representing new, incremental additions to the dataset.  Setting up the data this way will allow us to demonstrate a broader array of Zingg's functionality that others can then leverage to design their own solutions. 

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -U sentence-transformers 

# COMMAND ----------

# DBTITLE 1,Get Configuration Values
# MAGIC %run "./00_Intro_&_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn
from pyspark.sql.types import *

from sentence_transformers import SentenceTransformer, util
import torch
import pickle

from PIL import Image, ImageEnhance
import random
import os
from pathlib import Path

import pandas as pd

# COMMAND ----------

# MAGIC %md ##Step 1: Download the Data
# MAGIC
# MAGIC To access the data, we first need to download it to our environment. The images and metadata (listings) are provided as separate compressed *tar* files.  It is recommended that you download and extract these files to a [cloud storage account](https://docs.databricks.com/en/storage/index.html#how-do-you-configure-cloud-object-storage-for-databricks) associated with your environment.  To simplify things, the initial configuration of this notebook employs an internal DBFS file location - not an external cloud storage account -  within which the extraction phase of this work can be very time consuming %pytbecause of the very large number of small files within the images tarball.  So while we have used such a location for this demonstration, we do not recommend this in production.
# MAGIC
# MAGIC **NOTE** The path to this storage location is configured in the *00* notebook.
# MAGIC
# MAGIC Once you have determined where you want to download and extract your data to, you can download and extract it using steps as follows:
# MAGIC
# MAGIC ```
# MAGIC %python 
# MAGIC
# MAGIC # configure an environmental variable for the download path
# MAGIC os.environ['DOWNLOADS_FOLDER'] = '/dbfs' + config['dir']['downloads']
# MAGIC ```
# MAGIC
# MAGIC ```
# MAGIC %sh 
# MAGIC
# MAGIC rm -rf $DOWNLOADS_FOLDER 2> /dev/null
# MAGIC mkdir -p $DOWNLOADS_FOLDER
# MAGIC cd $DOWNLOADS_FOLDER
# MAGIC
# MAGIC # download the images files
# MAGIC echo "downloading images ..."
# MAGIC wget -q https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
# MAGIC
# MAGIC # decompress the images files
# MAGIC # (untars to a folder called images)
# MAGIC echo "unzipping images ..."
# MAGIC tar -xf ./abo-images-small.tar --no-same-owner
# MAGIC gzip -df ./images/metadata/images.csv.gz
# MAGIC rm ./abo-images-small.tar
# MAGIC
# MAGIC # download the listings files
# MAGIC echo "downloading listings ..."
# MAGIC wget -q https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
# MAGIC
# MAGIC # decompress the listings files
# MAGIC echo "unzipping listings ..."
# MAGIC tar -xf ./abo-listings.tar --no-same-owner
# MAGIC gunzip ./listings/metadata/*.gz
# MAGIC rm ./abo-listings.tar
# MAGIC
# MAGIC echo "complete"
# MAGIC ```

# COMMAND ----------

# MAGIC %md Once downloaded, you can confirm the presence of the extracted files as follows.  You should have just under 400K files:

# COMMAND ----------

# DBTITLE 1,Count Extracted Files
path = Path('/dbfs' + config['dir']['downloads'])

# loop through folder structure
total = 0
for root, dirs, files in os.walk(path):
    total += len(files) # count files

# print total count
print(total)

# COMMAND ----------

# MAGIC %md ##Step 2: Access the Data
# MAGIC
# MAGIC The downloaded data consists of product metadata and images.  We can access these data as follows:

# COMMAND ----------

# DBTITLE 1,Reset the Database
# drop the database
_ = spark.sql(f"DROP DATABASE IF EXISTS {config['database name']} CASCADE")

# create database to house mappings
_ = spark.sql(f"CREATE DATABASE IF NOT EXISTS {config['database name']}")

# set database as default for queries
_ = spark.catalog.setCurrentDatabase(config['database name'] )

# COMMAND ----------

# DBTITLE 1,Access Product Metadata
# read metadata
products = (
  spark
    .read
    .json( config['dir']['downloads']+'/listings/metadata' )
  ) 

# display data
display(products)

# COMMAND ----------

# DBTITLE 1,Access Image Metadata
images = (
  spark
      .read
      .csv(
        path=config['dir']['downloads']+'/images/metadata',
        sep=',',
        header=True
        )   
      .withColumn('path', fn.expr(f"concat('/dbfs', '{config['dir']['downloads']}', '/images/small/', path)"))
    )

display(images)

# COMMAND ----------

# MAGIC %md Images are referenced in the product metadata as a single *main image* and an array of *other images*.  Those images are identified using a string identifier which does not correspond with the item id assigned to a given product.  This means we need to join up a few datasets to connect product metadata with specific image paths:

# COMMAND ----------

# DBTITLE 1,Convert Image Identifiers to File Paths
# get path for other image id values (from within an array)
other_image_path = (
  products
    .select('item_id','other_image_id')
    .withColumn('image_id', fn.explode('other_image_id'))
    .join(images, on='image_id')
    .groupBy('item_id')
      .agg(fn.collect_list('path').alias('other_image_path'))
  )

# combine products and image metadata
products_and_images = (
  products
    .join(images, on=fn.expr('main_image_id=image_id'), how='left') # lookup main image path
    .drop('image_id', 'height', 'width')
    .withColumnRenamed('path', 'main_image_path')
    .join(other_image_path, on='item_id', how='left') # join in other image pathes (array)
  )

# display results
display(products_and_images)

# COMMAND ----------

# MAGIC %md With the product and image data reconciled, we can persist this data to enable further processing:

# COMMAND ----------

# DBTITLE 1,Persist Product and Image Data
_ = (
  products_and_images
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('original')
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Extract Elements of Interest
# MAGIC
# MAGIC Examining the schema of the original dataset, we can see that there are many complex data elements.  Many of these are used to enable language-specific variations of product information:

# COMMAND ----------

# DBTITLE 1,Retrieve Original Dataset
original = spark.table('original')

display(original)

# COMMAND ----------

# MAGIC %md To simplify things, we might extract a single string from many of these elements with a preference for US English:

# COMMAND ----------

# DBTITLE 1,Define Function to Retrieve English-Language Elements from Arrays
@fn.udf(ArrayType(StringType()))
def get_english_values_from_array(array=None):

   # prioritized list of english language codes (biased towards us english)
  english = ['en_US','en_CA','en_GB','en_AU','en_IN','en_SG','en_AE']

  # initialize search 
  values = []
  if array is None: array=[]

  # for each potential english code
  for e in english:

    # for each item in array
    for a in array:
      # if we found the english variant we want
      if a['language_tag']==e and len(a['value'].strip())>0: 
        # add the value to th eset of values for this variant
        values += [a['value']] 

    # if value(s) has been found, then break
    if len(values) > 0: break
    
  return values


# COMMAND ----------

# DBTITLE 1,Extract Required Elements from Product Data
cleansed = (
  original
    .filter("country='US'") # just get the US listings for our purposes
    .select( 
      'item_id',
      'domain_name',
      'marketplace',
      fn.expr("element_at(product_type,1).value").alias('product_type'),
      get_english_values_from_array('brand')[0].alias('brand'),
      get_english_values_from_array('item_name')[0].alias('item_name'),
      get_english_values_from_array('product_description')[0].alias('product_description'),
      get_english_values_from_array('bullet_point').alias('bulletpoint'),
      get_english_values_from_array('item_keywords').alias('item_keywords'),
      fn.split( fn.col('node')[0]['node_name'], '/').alias('hierarchy'),
      fn.expr('main_image_path as image_path')
      )
  )
   
display(cleansed)

# COMMAND ----------

# MAGIC %md From a quick review of the data, we can see that many of our items are missing a product description, making that field less than useful for our work.  The *bulletpoint* and *item_keywords* fields provide far more robust opportunities for item matching but the fact that they are organized in arrays of strings which aren't usable by Zingg. (Arrays in Zingg must be numeric.)  If we were to convert these to continuous strings, Zingg would employ them and learn how best to compare items using the information within these fields:

# COMMAND ----------

# DBTITLE 1,Flatten String Arrays to Delimited Strings
flattened = (
  cleansed
    .withColumn('bulletpoint', fn.expr("array_join(bulletpoint, ', ')"))
    .withColumn('item_keywords', fn.expr("array_join(array_distinct(item_keywords), ', ')")) # apply distinct because of large num of duplicates
)

display(flattened)

# COMMAND ----------

# MAGIC %md Before wrapping up the clean up of the various fields we intend to use, we have found that Zingg has trouble handling NULL strings written to Parquet or Delta formats.  (In many of the steps we wish to perform, Zingg is calling the *read_parquet* method and it's complaining about unknown data types).  To simplify things and to avoid these issues, we might convert any NULL values in string fields to empty strings:

# COMMAND ----------

# DBTITLE 1,Convert NULL Strings to Empty Strings
# get list of string columns in current dataset
string_cols = [col for col, dtype in flattened.dtypes if dtype == 'string']

# convert any nulls in these columns to empty strings
flattened = (
  flattened
    .fillna(
      value='',
      subset=string_cols
    )
)

# COMMAND ----------

# MAGIC %md We can now persist our data for use in subsequent data preparation steps:

# COMMAND ----------

# DBTITLE 1,Persist Cleansed Data
# write the cleansed data to storage as base table
_ = (
  flattened
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('inputs')
  )

# COMMAND ----------

# MAGIC %md ##Step 4: Create Duplicates for Entity Resolution
# MAGIC
# MAGIC In order to perform product matching, we need to have a dataset that contains some duplicate records. Our initial step will be to access our cleansed dataset and insert a number of records back into it with no modifications, creating pure duplicates:

# COMMAND ----------

# DBTITLE 1,Access Cleansed Data
# reference data in base table
inputs = spark.table('inputs')

# count records for on-going tracking
inputs.count()

# COMMAND ----------

# DBTITLE 1,Add Duplicates with No Modifications
_ = (
  inputs
    .sample(fraction=0.10)
    .write
      .format('delta')
      .mode('append')
      .saveAsTable('inputs')
  )

inputs.count()

# COMMAND ----------

# MAGIC %md Next, we will insert a small number of records with some character substitutions:

# COMMAND ----------

# DBTITLE 1,Duplicates with Character Substitutions
# define function to perform some character-level substitutions
@fn.udf('string')
def make_char_subs(input):
  char_find = 'zoBeA' # find these chars
  char_repl = '50834' # replace with these chars
  subs = str.maketrans(char_find, char_repl) # compile translation
  return str(input).translate(subs) # apply translation

# register function for use in SQL
_ = spark.udf.register('make_char_subs', make_char_subs)

# alter name and description for a random sample of records
_ = (
  inputs
    .sample(fraction=0.10) # grap some random records
    .withColumn('item_name', fn.expr("case when rand() > 0.5 then make_char_subs(item_name) else item_name end")) # alter item name
    .withColumn( # alter item description
      'product_description', 
      fn.expr("case when rand() > 0.5 then make_char_subs(product_description) else product_description end")
      ) 
    .write
      .format('delta')
      .mode('append')
      .saveAsTable('inputs')
  )

inputs.count()

# COMMAND ----------

# MAGIC %md To create duplicates with altered images, we will create a full set of modified images and then attach these to a subset of randomly selected duplicate records:

# COMMAND ----------

# DBTITLE 1,Create Modified Images
# define function to modify images
def modify_image(path, overwrite=False):

  # determine name and path of modified image file
  fname = path.replace('/small','/small/modified') # modify the path for the file
  fpath = '/'.join(fname.split('/')[:-1])

  # delete modified file if necessary
  if os.path.exists(fname) and overwrite: 
    os.remove(fname)

  # if modified file does not exist:
  if not os.path.exists(fname):

    # open image
    try:
      im = Image.open(path).convert('RGB')

      # crop image
      w, h = im.size
      w_margin = int(w * 0.25 * random.random())
      h_margin = int(h * 0.25 * random.random())
      im = im.crop( (w_margin, h_margin, w - w_margin,  h - h_margin) )

      # adjust contrast
      factor = random.uniform(0.5, 1.5)
      enhancer = ImageEnhance.Contrast(im)
      im = enhancer.enhance(factor)

      # persist modified image
      
      if not os.path.exists(fpath): os.makedirs(fpath) # create dir if not exists
      
      im.save( fname ) # write modified image

    except:
      pass

  return fname

# get list of images to modify
images = (
 inputs
    .selectExpr('image_path') # primary images
    .distinct() # get unique image paths
    .filter('image_path is not null')
    .toPandas()['image_path'] # extract to a series
    .tolist() # to list
  )

# iterate through list 
for i, image in enumerate(images):
  modify_image(image)
  if (i+1)%1000==0:
    print(f"{i+1} of {len(images)} modified", end='\r')

print(f"{i+1} of {len(images)} modified", end='\r')  

# COMMAND ----------

# DBTITLE 1,Duplicates with Modified Images
_ = (
 inputs
    .sample(fraction=0.10) # randomly select records to associate with modified images
    .withColumn('image_path', fn.expr("replace(image_path, '/small', '/small/modified')"))
    .write
      .format('delta')
      .mode('append')
      .saveAsTable('inputs')
  )

inputs.count()

# COMMAND ----------

# MAGIC %md ##Step 5: Convert Image Data to Embeddings
# MAGIC
# MAGIC To support product matching on image data, we will need to convert our images into embeddings.  An embedding is an array of floating point values generated by a model that's typically trained to understand how a wide range of images typically differ from one another.  The process of converting an image to an array of floating point values is complex, but if we think of these arrays as coordinates, the distance between the arrays associated with two images tells us something about the degree of similiarity between those images.  More similiar images will have coordinates that are closer together whild more dissimiliar images will be further apart.
# MAGIC
# MAGIC To support this work, we need a model that's been trained on a set of images that will produce effective arrays for us.  We like [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32) because its reasonably lightweight and trained on a combination of images and descriptive text that align images around both their physical composition and how a human might describe them.  Feel free to experiment with other models to see if they give you better results in subsequent item matching exercises:

# COMMAND ----------

# DBTITLE 1,Install Model to Generate Embeddings
model = SentenceTransformer('clip-ViT-B-32')

# COMMAND ----------

# MAGIC %md We can now use this model in a function to convert an image to an embedding.  Notice that we are receiving the local absolute path to our file (as defined in an earlier step), reading the image and then passing that image to our model for *encoding*..  The encoded values is converted to a list to align the value what what is supported by Zingg:

# COMMAND ----------

# DBTITLE 1,Define Function to Convert Images to Embeddings
@fn.udf(ArrayType(DoubleType()))
def get_image_embedding(path):

  # define default return value
  ret = []

  # if we have a path to a file
  if path is not None:
  
    # open image and convert to embedding
    try:
      image = Image.open(path).convert('RGB')
      embedding = model.encode(image, batch_size=128, convert_to_tensor=False, show_progress_bar=False)
      ret = embedding.tolist()
    except:
      pass
    
  # return embedding value
  return ret

# COMMAND ----------

# MAGIC %md Now we can apply our function to the images to generate the embeddings:

# COMMAND ----------

# DBTITLE 1,Convert Images to Embeddings
inputs = spark.table('inputs')

inputs = (
  inputs
    .withColumn('image_embedding', get_image_embedding('image_path'))
  )

display(inputs)

# COMMAND ----------

# MAGIC %md At this point, we will save our data before attemping further modifications. We will overwrite our original input dataset as updating each record in the set with an embedding is more time-consuming.  We will take advantage of this opportunity to create a row identifier that will simplify our last data preparateion step:

# COMMAND ----------

# DBTITLE 1,Save Inputs with Embeddings
_ = (
  inputs
    .withColumn('rowid', fn.expr("row_number() over(order by item_id)"))
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('inputs')
    )

# COMMAND ----------

# MAGIC %md ##Step 6: Create Initial & Incremental Sets
# MAGIC
# MAGIC In order to demonstrate how we might perform product matching in both initial and incremental phases, we will seperate our data into *initial* and *incremental* subsets using a simple random sampling of the ABO dataset.  We will limit these data to just items in one particular category so that we can define a more consistent set of criteria for identifying product matches:

# COMMAND ----------

# DBTITLE 1,Read Prepared Data
inputs = (
  spark
    .table('inputs')
  )

# COMMAND ----------

# DBTITLE 1,Examine Info by Product Category
display(
  inputs
    .groupBy('product_type')
      .agg(fn.count('*').alias('items'))
    .orderBy('items',ascending=False)
  )

# COMMAND ----------

# DBTITLE 1,Filter Data to a Single Category
   inputs = (
     inputs
      .filter("product_type='SHOES'")
      .drop('product_type')
    )

# COMMAND ----------

# MAGIC %md Now we can split the data into our *initial* and *incremental* sets: 

# COMMAND ----------

# DBTITLE 1,Separate Data into Initial & Incremental Subsets
incremental = inputs.sample(fraction=0.10)
initial = inputs.join(incremental, on='rowid', how='leftanti')

# COMMAND ----------

# MAGIC %md To make this data available for product matching, we can save it to a Databricks table.  Notice that we are specifying a path for our table which will make Spark persist this as an [external table](https://docs.databricks.com/sql/language-manual/sql-ref-external-tables.html).  External tables can be accessed either as database tables or as folders within the file system.  Zingg does not yet support database table access within Spark so that we will need file system access in order to read the data in subsequent notebooks:

# COMMAND ----------

# DBTITLE 1,Persist the Data
_ = (
  initial
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .option('path',f"{config['dir']['tables']}/initial")
    .saveAsTable('initial')
    )


_ = (
  incremental
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .option('path',f"{config['dir']['tables']}/incremental")
    .saveAsTable('incremental')
    )

# COMMAND ----------

# DBTITLE 1,Verify Dataset Counts
print(f"Initial:     {spark.table('initial').count()}")
print(f"Incremental: {spark.table('incremental').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Zingg                      | entity resolution library with support for user-defined arrays|  AGPL-3.0 license    | https://github.com/zinggAI/zingg                      |
# MAGIC | sentence-transformers | multilingual text embeddings | Apache 2.0 | https://www.zingg.ai/company/zingg-enterprise-databricks
# MAGIC | Amazon-Berkeley Objects | a CC BY 4.0-licensed dataset of Amazon products with metadata, catalog images, and 3D models | Creative Commons Attribution 4.0 International Public License (CC BY 4.0) | https://amazon-berkeley-objects.s3.amazonaws.com/index.html |
