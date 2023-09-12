# Databricks notebook source
# MAGIC %md The purpose of this notebook is to perform incremental processing of *incoming* data to existing records using a previously trained model as part of the Zingg Product Matching with Image Data solution accelerator. This notebook was developed using a **Databricks 12.2 LTS** cluster.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC The incremental Zingg workflow consists of two tasks, each of which is intended to examine incoming data for the inclusion of duplicate records.  These tasks are:</p>
# MAGIC
# MAGIC 1. **link** - identify duplicates between incoming and previously observed records
# MAGIC 2. **match** - identify duplicates within the incoming dataset
# MAGIC
# MAGIC These tasks are performed on a portion of the data withheld from the initial workflow, referred to earilier as our *incremental* dataset.
# MAGIC
# MAGIC **NOTE** In the steps below, we will focus on displaying *link* and *match* results without generating product image thumbnails.  If you'd like to see how thumbnail images can be displayed in the notebook, please check out the different techniques employed in the prior notebook.

# COMMAND ----------

# DBTITLE 1,Initialize Config
# MAGIC %run "./00_Intro_&_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from zingg.client import Arguments, ClientOptions, ZinggWithSpark
from zingg.pipes import Pipe, FieldDefinition, MatchType

import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ##Step 1: Perform Record Linking
# MAGIC
# MAGIC The *[link](https://docs.zingg.ai/zingg/stepbystep/link)* task performs a comparison of two different datasets to determine which members of one are likely matches for the other.  If we are thinking of this as an incremental workflow, our incoming, incremental dataset will serve as one of the datasets and the previously processed data against which we wish to compare these for matches will serve as the other.  
# MAGIC
# MAGIC Both datasets will serve as inputs into our *link* task.  Because we need both datasets to have the same data structure, we won't simply point to the previously processed data in the *cluster_members* table but instead will extract the relevant fields as follows:

# COMMAND ----------

# DBTITLE 1,Record Prior Members
# define path where to save the prior data
prior_data_dir = config['dir']['input'] + '/prior'
  
# save the data to a file location
_ = (
  spark
    .table('cluster_members')
    .selectExpr(
      'item_id','domain_name','marketplace','brand','item_name',
      'product_description','bulletpoint','item_keywords',
      'hierarchy','image_path','image_embedding',
      '-1 as rowid'
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .save(prior_data_dir)
  )


# display data
display(
  spark
    .read
    .format('delta')
    .load(prior_data_dir)
  )

# COMMAND ----------

# MAGIC %md With our priors saved as such, we might then define the *link* task.  Notice that we are defining two input pipes, the first of which is the priors and the second of which is the incoming/incremental.  The order in which they are added to our task configuration does not matter:

# COMMAND ----------

# DBTITLE 1,Initialize Zingg Arguments
args = Arguments()

# COMMAND ----------

# DBTITLE 1,Assign Model Arguments
# this is where zingg models, labels, and other data will be stored
args.setZinggDir(config['dir']['zingg'] )

# this uniquely identifies the model you are training
args.setModelId(config['model name'])

# COMMAND ----------

# DBTITLE 1,Config Model Inputs
# configure priors Zingg input pipe
priors_inputPipe = Pipe(name='priors', format='delta')
priors_inputPipe.addProperty('path', prior_data_dir)
args.setData(priors_inputPipe)

# configure incoming Zingg input pipe
incoming_input_path = spark.sql("DESCRIBE DETAIL incremental").select('location').collect()[0]['location']
incoming_inputPipe = Pipe(name='incoming', format='delta')
incoming_inputPipe.addProperty('path', incoming_input_path )

# set input data pipelines
args.setData(priors_inputPipe, incoming_inputPipe)

# COMMAND ----------

# DBTITLE 1,Config Model Output
linked_output_dir = config['dir']['output'] + '/incremental/linked'

# configure Zingg output pipe
outputPipe = Pipe(name='linked', format='delta')
outputPipe.addProperty('path', linked_output_dir)

# add output pipe to arguments collection
args.setOutput(outputPipe)

# COMMAND ----------

# DBTITLE 1,Configure Field Definitions
# define logic for each field in incoming dataset
item_id = FieldDefinition('item_id', 'string', MatchType.DONT_USE)
domain_name = FieldDefinition('domain_name', 'string', MatchType.DONT_USE)
marketplace = FieldDefinition('marketplace', 'string', MatchType.DONT_USE)
brand = FieldDefinition('brand','string', MatchType.FUZZY)
item_name = FieldDefinition('item_name', 'string', MatchType.TEXT)
product_description = FieldDefinition('product_description', 'string', MatchType.DONT_USE)
bulletpoint = FieldDefinition('bulletpoint', 'string', MatchType.TEXT)
item_keywords = FieldDefinition('item_keywords', 'string', MatchType.TEXT)
hierarchy = FieldDefinition('hierarchy', 'array<string>', MatchType.DONT_USE)
image_path = FieldDefinition('image_path', 'string', MatchType.DONT_USE)
image_embedding = FieldDefinition('image_embedding', 'array<double>', MatchType.FUZZY)

# define sequence of fields to receive
field_defs = [item_id, domain_name, marketplace, brand, item_name, product_description, bulletpoint, item_keywords, hierarchy, image_path, image_embedding]

# add field definitions to arguments collection
args.setFieldDefinition(field_defs)

# COMMAND ----------

# DBTITLE 1,Config Performance Settings
# define number of partitions to distribute data across
args.setNumPartitions( sc.defaultParallelism * 20 ) # default parallelism reflects databricks's cluster capacity

# define sample size
args.setLabelDataSampleSize(0.1)  

# COMMAND ----------

# DBTITLE 1,Define Link Task
# define task
link_options = ClientOptions([ClientOptions.PHASE, 'link'])

# configure findTrainingData task
link = ZinggWithSpark(args, link_options)

# initialize task
link.init()

# COMMAND ----------

# MAGIC %md We can now run the *link* task as follows:

# COMMAND ----------

# DBTITLE 1,Link Incoming & Prior Records
link.execute()

# COMMAND ----------

# MAGIC %md The output of the link task can be viewed by retreiving data from the linked output directory:

# COMMAND ----------

# DBTITLE 1,Review Linked Record Output
linked = (
  spark
    .read
      .format('delta')
      .load(linked_output_dir)
    
  )

display(linked.orderBy(['z_cluster', 'z_zsource'], ascending=[True, False]))

# COMMAND ----------

# MAGIC %md The link job output assigns a *z_cluster* value to records in the incoming dataset likely to match a record in the prior dataset.  A *z_score* helps us understand the probability of that match. The *z_source* field differentiates between records coming from the prior and the incoming datasets.  Please note that if a prior record is in a cluster with multiple incoming records, it's *z_score* reflects the highest scored incoming match. 
# MAGIC
# MAGIC It's important to note that an incoming record may be linked to more than one prior records. Also, incoming records that do not have likely matches in the prior dataset (as determined by the blocking portion of the Zingg logic), will not appear in the linking output.  This knowledge needs to be taken into the data processing steps that follow.
# MAGIC
# MAGIC To help us work with the linked data, we might separate those records from the prior dataset from those in the incoming dataset.  For the prior dataset, we can lookup the *cluster_id* in our *cluster_members* table to make the appending of new data to that table easier in later steps:

# COMMAND ----------

# DBTITLE 1,Get Linked Priors
linked_prior = (
  linked
    .alias('a')
    .filter(fn.expr("a.z_zsource = 'priors'"))
    .join( 
      spark.table('cluster_members').alias('b'), # join on fields relevant for matching
      on=fn.expr("""
      a.item_id=COALESCE(b.item_id,'') AND
      a.domain_name=COALESCE(b.domain_name,'') AND 
      a.marketplace=COALESCE(b.marketplace,'') AND 
      a.brand=COALESCE(b.brand,'') AND 
      a.item_name=COALESCE(b.item_name,'') AND 
      a.bulletpoint=COALESCE(b.bulletpoint,'') AND 
      a.item_keywords=COALESCE(b.item_keywords,'') AND 
      a.image_path=COALESCE(b.image_path,'')
      """)      
      )
    .selectExpr(
      'b.cluster_id',
      'a.z_cluster',
      "COALESCE(a.item_id,'') as item_id",
      "COALESCE(a.domain_name,'') as domain_name",
      "COALESCE(a.marketplace,'') as marketplace",
      "COALESCE(a.brand,'') as brand",
      "COALESCE(a.item_name,'') as item_name",
      "COALESCE(a.product_description,'') as product_description",
      "COALESCE(a.bulletpoint,'') as bulletpoint",
      "COALESCE(a.item_keywords,'') as item_keywords",
      "a.hierarchy", 
      "COALESCE(a.image_path,'') as image_path", 
      "a.image_embedding"    
      )
  )

display(
  linked_prior
  )

# COMMAND ----------

# DBTITLE 1,Get Linked Incoming
# get priors
linked_incoming = (
  linked
    .filter(fn.expr("z_zsource = 'incoming'"))
    .selectExpr(
      'z_cluster',
      "COALESCE(item_id,'') as item_id",
      "COALESCE(domain_name,'') as domain_name",
      "COALESCE(marketplace,'') as marketplace",
      "COALESCE(brand,'') as brand",
      "COALESCE(item_name,'') as item_name",
      "COALESCE(product_description,'') as product_description",
      "COALESCE(bulletpoint,'') as bulletpoint",
      "COALESCE(item_keywords,'') as item_keywords",
      "hierarchy", 
      "COALESCE(image_path,'') as image_path", 
      "image_embedding",
      "z_score"
      )
  )

# get highest scored priors if multiple entries
max_linked_incoming = (
  linked_incoming
    .groupBy(
      'item_id','domain_name','marketplace','brand','item_name','bulletpoint','item_keywords','image_path'
      )
      .agg(fn.max('z_score').alias('z_score'))
  )

# restrict to just the highest scored
linked_incoming = (
  linked_incoming
    .join(max_linked_incoming, on=['item_id','domain_name','marketplace','brand','item_name','bulletpoint','item_keywords','image_path','z_score'])
    .selectExpr(
      'z_cluster',
      "COALESCE(item_id,'') as item_id",
      "COALESCE(domain_name,'') as domain_name",
      "COALESCE(marketplace,'') as marketplace",
      "COALESCE(brand,'') as brand",
      "COALESCE(item_name,'') as item_name",
      "COALESCE(product_description,'') as product_description",
      "COALESCE(bulletpoint,'') as bulletpoint",
      "COALESCE(item_keywords,'') as item_keywords",
      "hierarchy", 
      "COALESCE(image_path,'') as image_path", 
      "image_embedding"    
      )
  )

display(
  linked_incoming
  )

# COMMAND ----------

# MAGIC %md We now have our set of prior records to which one or more incoming records have a linkage.  We also have the highest scored version of an incoming record and its link-cluster assignment.  It is possible that an incoming record could be linked to two different prior records with identical scores so we'll need to take that into consideration as we design our persistance logic.  That logic should then be something like the following:</P>
# MAGIC
# MAGIC 1. If there is only one link for an incoming record and the score for that record is above a given upper threshold, assign that record to its linked prior's cluster.
# MAGIC 2. If there are multiple links for an incoming record and the score for those records is above a given upper threshold, send those records to manual review.
# MAGIC 3. If a record is below a given lower threshold, reject the record as a possible match.
# MAGIC 4. If a record is between a given lower and upper threshold, send that record to manual review.

# COMMAND ----------

# MAGIC %md ##Step 2: Perform Record Matching
# MAGIC
# MAGIC The *[match](https://docs.zingg.ai/zingg/stepbystep/match)* task is now used to examine potential matches within the *incoming* dataset. The configuration for this task is more straightforward than with *link* as we are only dealing with one input dataset and closely mirrors the configuration used in the last notebook, though our input is the *incoming* dataset: 

# COMMAND ----------

# DBTITLE 1,Initialize Zingg Arguments
args = Arguments()

# COMMAND ----------

# DBTITLE 1,Assign Model Arguments
# this is where zingg models, labels, and other data will be stored
args.setZinggDir(config['dir']['zingg'] )

# this uniquely identifies the model you are training
args.setModelId(config['model name'])

# COMMAND ----------

# DBTITLE 1,Config Model Inputs
# configure incoming Zingg input pipe
incoming_input_path = spark.sql("DESCRIBE DETAIL incremental").select('location').collect()[0]['location']
incoming_inputPipe = Pipe(name='incoming', format='delta')
incoming_inputPipe.addProperty('path', incoming_input_path )

# set input data pipelines
args.setData(incoming_inputPipe)

# COMMAND ----------

# DBTITLE 1,Config Model Outputs
matched_output_dir = config['dir']['output'] + '/incremental/matched'

# configure Zingg output pipe
outputPipe = Pipe(name='matched', format='delta')
outputPipe.addProperty('path', matched_output_dir)

# add output pipe to arguments collection
args.setOutput(outputPipe)

# COMMAND ----------

# DBTITLE 1,Configure Field Definitions
# define logic for each field in incoming dataset
item_id = FieldDefinition('item_id', 'string', MatchType.DONT_USE)
domain_name = FieldDefinition('domain_name', 'string', MatchType.DONT_USE)
marketplace = FieldDefinition('marketplace', 'string', MatchType.DONT_USE)
brand = FieldDefinition('brand','string', MatchType.FUZZY)
item_name = FieldDefinition('item_name', 'string', MatchType.TEXT)
product_description = FieldDefinition('product_description', 'string', MatchType.DONT_USE)
bulletpoint = FieldDefinition('bulletpoint', 'string', MatchType.TEXT)
item_keywords = FieldDefinition('item_keywords', 'string', MatchType.TEXT)
hierarchy = FieldDefinition('hierarchy', 'array<string>', MatchType.DONT_USE)
image_path = FieldDefinition('image_path', 'string', MatchType.DONT_USE)
image_embedding = FieldDefinition('image_embedding', 'array<double>', MatchType.FUZZY)

# define sequence of fields to receive
field_defs = [item_id, domain_name, marketplace, brand, item_name, product_description, bulletpoint, item_keywords, hierarchy, image_path, image_embedding]

# add field definitions to arguments collection
args.setFieldDefinition(field_defs)

# COMMAND ----------

# DBTITLE 1,Define Match Task
# define task
match_options = ClientOptions([ClientOptions.PHASE, 'match'])

# configure findTrainingData task
match = ZinggWithSpark(args, match_options)

# initialize task
match.init()

# COMMAND ----------

# MAGIC %md We can now run the *match* task to look for matches within the incoming dataset:

# COMMAND ----------

# DBTITLE 1,Identify Matches in Incoming Dataset
match.execute()

# COMMAND ----------

# DBTITLE 1,Review Matched Records
matches = (
  spark
    .read
      .format('delta')
      .load(matched_output_dir)
      )

dupes = (
  matches
    .groupBy('z_cluster')
      .agg(fn.count('*').alias('rows'))
    .filter('rows > 1')
    .select('z_cluster')
  )

# retrieve matches
dupe_matches = (
  matches
    .join(dupes, on='z_cluster')  
    .orderBy('z_cluster')
  )

display(dupe_matches)

# COMMAND ----------

# MAGIC %md The output of the incremental match is the same as the output of the initial match (though the records involved are the same). As before, you'll want to carefully consider which matches to accept and which to reject.  And once you've got that sorted, you'll want to insert records in the matched output not already in the the *cluster_members* table into it under new *cluster_id* values.

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
