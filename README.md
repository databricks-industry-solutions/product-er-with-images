![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)


##Introduction

A common need in many retail and consumer goods companies is to match products between internal and external datasets.  This may take place as part of the process of onboarding new products into an online marketplace or when performing product comparisons between websites.

The problem sounds simple enough to tackle: just look at the products and see if they are the same thing.  But companies are often dealing with vast product catalogs and some items may have a number of variations that depending on the scenario surrounding the comparison may or may not be relevant to determine if the products are essentially the same thing.

Using fuzzy matching techniques, product metadata can be used to perform this comparison.  In this approach, user feedback on likely matches are used to determine how various characteristics such as product names, descriptions, manufacturer, price, *etc.* should factor into the estimation that two products are (or are not) the same item.  But what about using product images?

If we look at how users might compare products across two websites, they might read names, descriptions, *etc.*, but quite often they will start by looking at the images associated with the product and use that as a key factor in determining whether two items are the *same*.  How might we incorporate image information into a fuzzy matching exercise?

Advances in computer vision and neural networks have given us the ability to condense a complex image into an embedding. An embedding is an array of numbers that tells us something about the structure of an image relative to the other images on which a model has been trained.  When two images are passed through a pre-trained model, we can examine the differences in their embeddings to arrive at an estimation of how similar or dissimilar the two images are to one another.  This can then be used as another input into a machine learning-based fuzzy matching exercise.

##Understanding Zingg

Zingg is a library supporting the development of a wide range of *entity-resolution*, *i.e.* fuzzy matching, applications.  We've used Zingg in the past to tackle customer (person) entity-resolution challenges, but with Zingg's recent introduction of support for user-defined arrays, we now have the opportunity to use it for product entity-resolution and take advantage of image embeddings as part of this process.

To build a Zingg-enabled application, it's easiest to think of Zingg as being deployed in two phases.  In the first phase that we will refer to as the *initial workflow*, candidate pairs of potential duplicates are extracted from an initial dataset and labeled by expert users.  These labeled pairs are then used to train a model capable of scoring likely matches.

In the second phase that we will refer to as the *incremental workflow*, the trained model is applied to newly arrived data.  Those data are compared to data processed in prior runs to identify likely matches between in incoming and previously processed dataset. The application engineer is responsible for how matched and unmatched data will be handled, but typically information about groups of matching records are updated with each incremental run to identify all the record variations believed to represent the same entity.

The initial workflow must be run before we can proceed with the incremental workflow.  The incremental workflow is run whenever new data arrive that require linking and deduplication. If we feel that the model could be improved through additional training, we can perform additional cycles of record labeling by rerunning the initial workflow.  The retrained model will then be picked up in the next incremental run.

##Installation & Verification

Zingg with support for array-comparisons (which is the basis for our image comparisons) will be supported in the 0.4 version of the product.  At the time of development, this version was available as a preview which can be accessed [here](https://github.com/zinggAI/zingg/releases/tag/0.4.0_ARRAY).

To enable Zingg support, you will need to install the JAR file within your Databricks cluster as a [workspace library](https://docs.databricks.com/libraries/workspace-libraries.html).  In addition, you will need to install the corresponding WHL file, as either a workspace or a notebook-scoped library.  For simplicity, it's recommended you install the WHL, also available [here](https://github.com/zinggAI/zingg/releases/tag/0.4.0_ARRAY), as a workspace library.

Alternatively you can run the ./RUNME notebook and use the Workflow and Cluster created in that notebook to run this accelerator. The RUNME notebook automated the download, extraction and installation of the Zingg jar.


-----------------------------------------

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |
| Zingg                      | entity resolution library with support for user-defined arrays|  AGPL-3.0 license    | https://github.com/zinggAI/zingg                      |
| sentence-transformers | multilingual text embeddings | Apache 2.0 | https://www.zingg.ai/company/zingg-enterprise-databricks
| Amazon-Berkeley Objects | a CC BY 4.0-licensed dataset of Amazon products with metadata, catalog images, and 3D models | Creative Commons Attribution 4.0 International Public License (CC BY 4.0) | https://amazon-berkeley-objects.s3.amazonaws.com/index.html |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
