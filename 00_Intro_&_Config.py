# Databricks notebook source
# MAGIC %md The purpose of this notebook is to introduce the Zingg Product Matching with Image Data solution accelerator and provide access to the configuration settings used across the various notebooks in the accelerator.  This notebook was developed using a **Databricks 12.2 LTS** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC A common need in many retail and consumer goods companies is to match products between internal and external datasets.  This may take place as part of the process of onboarding new products into an online marketplace or when performing product comparisons between websites.
# MAGIC
# MAGIC The problem sounds simple enough to tackle: just look at the products and see if they are the same thing.  But companies are often dealing with vast product catalogs and some items may have a number of variations that depending on the scenario surrounding the comparison may or may not be relevant to determine if the products are essentially the same thing.
# MAGIC
# MAGIC Using fuzzy matching techniques, product metadata can be used to perform this comparison.  In this approach, user feedback on likely matches are used to determine how various characteristics such as product names, descriptions, manufacturer, price, *etc.* should factor into the estimation that two products are (or are not) the same item.  But what about using product images?
# MAGIC
# MAGIC If we look at how users might compare products across two websites, they might read names, descriptions, *etc.*, but quite often they will start by looking at the images associated with the product and use that as a key factor in determining whether two items are the *same*.  How might we incorporate image information into a fuzzy matching exercise?
# MAGIC
# MAGIC Advances in computer vision and neural networks have given us the ability to condense a complex image into an embedding. An embedding is an array of numbers that tells us something about the structure of an image relative to the other images on which a model has been trained.  When two images are passed through a pre-trained model, we can examine the differences in their embeddings to arrive at an estimation of how similar or dissimilar the two images are to one another.  This can then be used as another input into a machine learning-based fuzzy matching exercise.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ##Understanding Zingg
# MAGIC
# MAGIC Zingg is a library supporting the development of a wide range of *entity-resolution*, *i.e.* fuzzy matching, applications.  We've used Zingg in the past to tackle customer (person) entity-resolution challenges, but with Zingg's recent introduction of support for user-defined arrays, we now have the opportunity to use it for product entity-resolution and take advantage of image embeddings as part of this process.
# MAGIC
# MAGIC To build a Zingg-enabled application, it's easiest to think of Zingg as being deployed in two phases.  In the first phase that we will refer to as the *initial workflow*, candidate pairs of potential duplicates are extracted from an initial dataset and labeled by expert users.  These labeled pairs are then used to train a model capable of scoring likely matches.
# MAGIC
# MAGIC In the second phase that we will refer to as the *incremental workflow*, the trained model is applied to newly arrived data.  Those data are compared to data processed in prior runs to identify likely matches between in incoming and previously processed dataset. The application engineer is responsible for how matched and unmatched data will be handled, but typically information about groups of matching records are updated with each incremental run to identify all the record variations believed to represent the same entity.
# MAGIC
# MAGIC The initial workflow must be run before we can proceed with the incremental workflow.  The incremental workflow is run whenever new data arrive that require linking and deduplication. If we feel that the model could be improved through additional training, we can perform additional cycles of record labeling by rerunning the initial workflow.  The retrained model will then be picked up in the next incremental run.
# MAGIC

# COMMAND ----------

# MAGIC %md ##Installation & Verification
# MAGIC
# MAGIC Zingg with support for array-comparisons (which is the basis for our image comparisons) will be supported in the 0.4 version of the product.  At the time of development, this version was available as a preview which can be accessed [here](https://github.com/zinggAI/zingg/releases/tag/0.4.0_ARRAY).
# MAGIC
# MAGIC To enable Zingg support, you will need to install the JAR file within your Databricks cluster as a [workspace library](https://docs.databricks.com/libraries/workspace-libraries.html).  In addition, you will need to install the corresponding WHL file, as either a workspace or a notebook-scoped library.  For simplicity, it's recommended you install the WHL, also available [here](https://github.com/zinggAI/zingg/releases/tag/0.4.0_ARRAY), as a workspace library.
# MAGIC
# MAGIC Alternatively you can run the ./RUNME notebook and use the Workflow and Cluster created in that notebook to run this accelerator. The RUNME notebook automated the download, extraction and installation of the Zingg jar.

# COMMAND ----------

# DBTITLE 1,Verify JAR Installed
# set default zingg path
zingg_jar_path = None

# for each jar in the jars folder
for j in dbutils.fs.ls('/FileStore/jars'):
  # locate the zingg jar
  if '-zingg_' in j.path:
    zingg_jar_path = j.path
    print(f'Zingg JAR found at {zingg_jar_path}')
    break
    
if zingg_jar_path is None: 
  raise Exception('The Zingg JAR was NOT found.  Please install the JAR file before proceeding.')

# COMMAND ----------

# DBTITLE 1,Verify WHL Installed
try:
  import zingg # attempt to import the zingg python library
except:
   raise Exception('The Zingg python library was NOT found.  Please install the WHL file before proceeding.')

# COMMAND ----------

# MAGIC %md ## Configuration
# MAGIC
# MAGIC To enable consistent settings across the notebooks in this accelerator, we establish the following configuration settings:

# COMMAND ----------

# DBTITLE 1,Initialize Configuration Variable
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
# set database name
config['database name'] = 'zingg_abo'

# create database to house mappings
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database name']))

# set database as default for queries
_ = spark.catalog.setCurrentDatabase(config['database name'] )

# COMMAND ----------

# DBTITLE 1,Zingg Model
config['model name'] = 'zingg_abo'

# COMMAND ----------

# MAGIC %md The Zingg workflow depends on access to various folder locations where the trained model and intermediary assets can be placed between various steps.  The purpose of these locations will be explained in the subsequent notebooks:

# COMMAND ----------

# DBTITLE 1,Directories
# path where files are stored
mount_path = '/tmp/abo'

config['dir'] = {}

# folder locations where you place your data
config['dir']['downloads'] = f'{mount_path}/downloads' # original unzipped data files that you will upload to the environment
config['dir']['input'] = f'{mount_path}/inputs' # folder where downloaded files will be seperated into initial and incremental data assets
config['dir']['tables'] = f'{mount_path}/tables' # location where external tables based on the data files will house data 

# folder locations Zingg writes data
config['dir']['zingg'] = f'{mount_path}/zingg' # zingg models and temp data
config['dir']['output'] = f'{mount_path}/output'

# make sure directories exist
for dir in config['dir'].values():
  dbutils.fs.mkdirs(dir)

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
