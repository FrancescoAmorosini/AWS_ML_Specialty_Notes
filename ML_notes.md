# AWS Machine Learning Specialty Notes
## Table of Contents

- [AWS Machine Learning Specialty Notes](#aws-machine-learning-specialty-notes)
- [Amazon Web Services](#amazon-web-services)
  - [Amazon SageMaker](#amazon-sagemaker)
    - [Elastic Network Interface](#elastic-network-interface)
    - [Pipe Input mode](#pipe-input-mode)
    - [Inter-Container Traffic Encryption](#inter-container-traffic-encryption)
    - [Deploy a dev-model in a test-environment](#deploy-a-dev-model-in-a-test-environment)
    - [Amazon SageMaker Data Wrangler](#amazon-sagemaker-data-wrangler)
    - [Amazon SageMaker Feature Store](#amazon-sagemaker-feature-store)
    - [Available Amazon SageMaker Images](#available-amazon-sagemaker-images)
    - [PrivateLink](#privatelink)
    - [Autopilot](#autopilot)
    - [SageMaker Available Algorithms](#sagemaker-available-algorithms)
  - [Amazon Forecast](#amazon-forecast)
    - [CNN-QR](#cnn-qr)
    - [DeepAR+](#deepar)
    - [Prophet](#prophet)
    - [NPTS](#npts)
    - [ARIMA](#arima)
    - [ETS](#ets)
  - [AWS Lake Formation](#aws-lake-formation)
  - [Amazon Kinesis](#amazon-kinesis)
  - [AWS Glue](#aws-glue)
  - [Amazon QuickSight vs Kibana](#amazon-quicksight-vs-kibana)
    - [Kibana](#kibana)
    - [Amazon QuickSight](#amazon-quicksight)
  - [Amazon Redshift](#amazon-redshift)
  - [Amazon Kinesis](#amazon-kinesis)
  - [Amazon Athena](#amazon-athena)
  - [Amazon Kendra](#amazon-kendra)
  - [Amazon CodeGuru Reviewer](#amazon-codeguru-reviewer)
  - [Amazon Fraud Detector](#amazon-fraud-detector)
- [Machine Learning Preprocessing](#machine-learning-preprocessing)
  - [Scaling & Normalization](#scaling--normalization)
    - [Box-CoX Transformation](#box-cox-transformation)
    - [Yeo-Johnson Transformation](#yeo-johnson-transformation)
  - [Imputation Methods for Missing Values](#imputation-methods-for-missing-values)
  - [Variable Enconding](#variable-enconding)
    - [One Hot Encoding](#one-hot-encoding)
    - [Label Encoding](#label-encoding)
    - [Ordinal Encoding](#ordinal-encoding)
    - [Helmert Encoding](#helmert-encoding)
    - [Binary Encoding](#binary-encoding)
    - [Frequency Encoding](#frequency-encoding)
    - [Target Encoding](#target-encoding)
    - [Weight of Evidence Encoding](#weight-of-evidence-encoding)
    - [Hashing](#hashing)
    - [James-Stein Encoding](#james-stein-encoding)
  - [Word Embeddings](#word-embeddings)
  - [Class Imbalance](#class-imbalance)
    - [Sampling Techniques](#sampling-techniques)
  - [Correlation Analysis](#correlation-analysis)
    - [Covariance](#covariance)
    - [Pearson's correlation](#pearsons-correlation)
    - [Spearman's correlation](#spearmans-correlation)
    - [Polychoric correlation](#polychoric-correlation)
    - [Mutual Information](#mutual-information)
  - [Variance Methods](#variance-methods)
    - [Principal Component Analysis](#principal-component-analysis)
    - [T-Distributed Stochastic Neighbourhood Embedding (t-SNE)](#t-distributed-stochastic-neighbourhood-embedding-t-sne)
    - [Linear Discriminant Analysis](#linear-discriminant-analysis)
    - [R-Squared](#r-squared)
  - [Spark](#spark)
  - [Securing Sensitive Information in AWS Data Stores](#securing-sensitive-information-in-aws-data-stores)
- [Model Selection](#model-selection)
  - [Feature Combination](#feature-combination)
  - [Naive Bayes](#naive-bayes)
  - [Support Vector Machines](#support-vector-machines)
  - [Factorization Machines](#factorization-machines)
- [Model Evaluation](#model-evaluation)
  - [Confusion Matrix](#confusion-matrix)
    - [Statistics on Confusion Matrix](#statistics-on-confusion-matrix)
    - [Receiver Operating Characteristic Curve](#receiver-operating-characteristic-curve)
  - [Satisficing Metrics](#satisficing-metrics)


# Amazon Web Services

## Amazon SageMaker

### [Elastic Network Interface](https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html)
Amazon SageMaker Elastic Inference (EI) enables users to accelerate throughput and decrease latency. It also reduces costs by allowing users to attach the desired amount of GPU-powered inference acceleration to an instance without code changes. It supports TensorFlow, PyTorch, and MXNet. Amazon Elastic Inference accelerators are network attached devices that work along with SageMaker instances in your endpoint to accelerate your inference calls. Elastic Inference accelerates inference by allowing you to attach fractional GPUs to any SageMaker instance. You can select the client instance to run your application and attach an Elastic Inference accelerator to use the right amount of GPU acceleration for your inference needs. Elastic Inference helps you lower your cost when not fully utilizing your GPU instance for inference.

### Pipe Input mode
SageMaker Pipe Mode is an input mechanism for SageMaker training containers based on Linux named pipes. SageMaker makes the data available to the training container using named pipes, which allows data to be downloaded from S3 to the container while training is running. For larger datasets, this dramatically improves the time to start training, as the data does not need to be first downloaded to the container. 

### Inter-Container Traffic Encryption
SageMaker automatically encrypts machine learning data and related artifacts in transit and at rest. However, SageMaker does not encrypt all intra-network data in transit such as inter-node communications in distributed processing and training jobs. Enabling inter-container traffic encryption via console or API meets this requirement. Distributed ML frameworks and algorithms usually transmit information that is directly related to the model such as weights, and enabling inter-container traffic encryption can increase training time, especially if you are using distributed deep learning algorithms.

### [Deploy a dev-model in a test-environment](https://aws.amazon.com/it/premiumsupport/knowledge-center/sagemaker-cross-account-model/)
To deploy the model to the test account, the engineer must first create an AWS KMS customer master key (CMK) for the SageMaker training job in the development account and link it to the test account. Then, an IAM role also needs to be created in the test account. This role requires SageMaker access and access to the training job output S3 bucket and CMK which are both in the development account. The output S3 bucket policy also needs to be updated to allow access from the test account. Finally, create the Amazon SageMaker deployment model, endpoint configuration, and endpoint from the test account.

### Amazon SageMaker Data Wrangler
Data Wrangler is a feature of Amazon SageMaker Studio that provides an end-to-end solution to import, prepare, transform, featurize, and analyze data. It allows you to run your own python code.

### Amazon SageMaker Feature Store
Serves as the single source of truth to store, retrieve, remove, track, share, discover, and control access to features.

### [Available Amazon SageMaker Images](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html)
A SageMaker image is a file that identifies the kernels, language packages, and other dependencies required to run a Jupyter notebook in Amazon SageMaker Studio.

* Python
* PySpark
* Datascience (Conda, Numpy, Scikit-Learn)
* MXNet
* Pytorch
* Tensorflow

If you need different functionality, you can bring your own custom images to Studio. In order to do this, you need to build your Dockerfile locally (satisfy the [requirements](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-byoi-specs.html) to be used in Amazon SageMaker Studio), push it to ECR, and then use it as source for a SageMaker image. After you have created your custom SageMaker image, you must attach it to your domain to use it with Studio.

### [PrivateLink](https://aws.amazon.com/it/blogs/machine-learning/secure-prediction-calls-in-amazon-sagemaker-with-aws-privatelink/)
Amazon SageMaker now supports Amazon Virtual Private Cloud (VPC) Endpoints via AWS PrivateLink so you can initiate prediction calls to your machine learning models hosted on Amazon SageMaker inside your VPC, without going over the internet. With AWS PrivateLink support, the SageMaker Runtime API can be called through an interface endpoint within the VPC instead of connecting over the internet. Since the communication between the client application and the SageMaker Runtime API is inside the VPC, there is no need for an Internet Gateway, a NAT device, a VPN connection, or AWS Direct Connect.

### [Autopilot](https://github.com/aws/amazon-sagemaker-examples/blob/main/autopilot/sagemaker_autopilot_direct_marketing.ipynb)
Amazon SageMaker Autopilot is an automated machine learning (commonly referred to as AutoML) solution for tabular datasets. It explores your data, selects the algorithms relevant to your problem type, and prepares the data to facilitate model training and tuning. Autopilot applies a cross-validation resampling procedure automatically to all candidate algorithms when appropriate to test their ability to predict data they have not been trained on. It also produces metrics to assess the predictive quality of its machine learning model candidates. It ranks all of the optimized models tested by their performance. It finds the best performing model that you can deploy at a fraction of the time normally required.

Autopilot currently supports regression and binary and multiclass classification problem types. It supports tabular data formatted as **CSV or Parquet** files in which each column contains a feature with a specific data type and each row contains an observation. The column data types accepted include numerical, categorical, text, and time series that consists of strings of comma-separate numbers.

Available algorithms are **Linear Regression, XGBoost, MLP**.

### [SageMaker Available Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

|  Use Case |  ML Problem  | ML Domain  | Algorithms  |
|---|---|---|---|
| Spam filter  | Classification  | Supervised  |  AutoGluon-Tabular, CatBoost, FMA, KNN, LightGBM, Linear Learner, TabTransformer, XGBoost |
| Estimate house value  | Regression  | Supervised  | AutoGluon-Tabular, CatBoost, FMA, KNN, LightGBM, Linear Learner, TabTransformer, XGBoost |
| Predict sales on historical data | Time series forecasting  | Supervised | DeepAR Forecasting Algorithm |
|Identify duplicate support tickets or find the correct routing | Embeddings | Supervised | Object2Vec |
|Drop those columns from a dataset that have a weak relation with the label | Feature engineering | Unsupervised | PCA |
|Spot when an IoT sensor is sending abnormal readings | Anomaly detection | Unsupervised | Random Cut Forest|
|Detect if an IP address accessing a service might be from a bad actor | IP anomaly detection | Unsupervised | IP Insights |
|Group similar objects/data together | Clustering | Unsupervised | K-means |
|Organize a set of documents into topics | Topic Modeling | Unsupervised | Latent Dirichlet Analysis, Neural-Topic Model|
|Assign pre-defined categories to documents | Text classification | Textual Analysis | Blazing Text |
|Convert text from one language to other | Machine translation | Textual Analysis | Seq2Seq |
|Summarize a long text corpus | Machine translation | Textual Analysis | Seq2Seq |
|Convert audio files to text | Speech2Text | Textual Analysis | Seq2Seq |
|Tag an image based on its content | Image Classification | Image Processing |Image Classification |
|Detect people and objects in an image | Object Detection | Image Processing |Object Detection |
|Tag every pixel of an image individually with a category | Computer Vision | Image Processing | Semantic Segmentation |

---

## Amazon Forecast
Amazon Forecast is a fully managed service that uses statistical and machine learning algorithms to deliver highly accurate time-series forecasts. Available algorithms are:

### [CNN-QR](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-algo-cnnqr.html)
Amazon Forecast CNN-QR, Convolutional Neural Network - Quantile Regression, is a proprietary machine learning algorithm for forecasting time series using causal convolutional neural networks (CNNs). CNN-QR works best with large datasets containing hundreds of time series. It accepts item metadata, and is the only Forecast algorithm that accepts related time series data without future values. CNN-QR supports both historical and forward looking related time series datasets.

### [DeepAR+](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-deeparplus.html)
Amazon Forecast DeepAR+ is a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNNs). DeepAR+ works best with large datasets containing hundreds of feature time series. The algorithm accepts forward-looking related time series and item metadata.

### [Prophet](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-prophet.html) 
Prophet is a time series forecasting algorithm based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality. It works best with time series with strong seasonal effects and several seasons of historical data.

### [NPTS](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-npts.html)
The Amazon Forecast Non-Parametric Time Series (NPTS) proprietary algorithm is a scalable, probabilistic baseline forecaster. NPTS is especially useful when working with sparse or intermittent time series. Forecast provides four algorithm variants: Standard NPTS, Seasonal NPTS, Climatological Forecaster, and Seasonal Climatological Forecaster.

### [ARIMA](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-arima.html)
Autoregressive Integrated Moving Average (ARIMA) is a commonly used statistical algorithm for time-series forecasting. The algorithm is especially useful for simple datasets with under 100 time series.

### [ETS](https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-ets.html)
Exponential Smoothing (ETS) is a commonly used statistical algorithm for time-series forecasting. The algorithm is especially useful for simple datasets with under 100 time series, and datasets with seasonality patterns. ETS computes a weighted average over all observations in the time series dataset as its prediction, with exponentially decreasing weights over time.

---

## AWS Lake Formation

**Personas**: One of the key features of AWS Lake formation is its ability to secure access to data in your data lake. Lake Formation provides its own permissions model that augments the AWS Identity and Access Management (IAM) permissions model. Configuring both IAM groups and Lake Formation personas are required to provide access permissions.

---

## Amazon Kinesis
You cannot stream data directly to Kinesis Data Analytics, you have to use Kinesis Data Stream of Kinesis Firehose first. Kinesis Firehose can produce parquet formatted data (from JSON!), but cannot apply transformations like Kinesis Data Analytics.

---

## AWS Glue

**Pushdown Predicates (Pre-Filtering)**: In many cases, you can use a pushdown predicate to filter on partitions without having to list and read all the files in your dataset. Instead of reading the entire dataset and then filtering in a DynamicFrame, you can apply the filter directly on the partition metadata in the Data Catalog. Then you only list and read what you actually need into a DynamicFrame. The predicate expression can be any Boolean expression supported by Spark SQL. Anything you could put in a WHERE clause in a Spark SQL query will work.

**Partition Predicates (Server-Side Filtering)**:The *push_down_predicate* option is applied after listing all the partitions from the catalog and before listing files from Amazon S3 for those partitions. If you have a lot of partitions for a table, catalog partition listing can still incur additional time overhead. To address this overhead, you can use server-side partition pruning with the *catalogPartitionPredicate* option that uses partition indexes in the AWS Glue Data Catalog. This makes partition filtering much faster when you have millions of partitions in one table. You can use both *push_down_predicate* and *catalogPartitionPredicate* in *additional_options* together.

> The *push_down_predicate* and *catalogPartitionPredicate* use different syntaxes. The former one uses Spark SQL standard syntax and the later one uses JSQL parser.

**SelectField**: You can create a subset of data property keys from the dataset using the SelectFields transform. You indicate which data property keys you want to keep and the rest are removed from the dataset.

Glue's **Relationalize** transformation can be used to convert data in a DynamicFrame into a relational data format. Once relationaized, the data can be written to an Amazon Redshift cluster from the Glue job using a JDBC connection.

> GlueTransform is the parent class for all of the AWS Glue Transform classes such as ApplyMapping, Unbox, and Relationize.

**Data Brew**: similar to Sagemaker Data Wrangles, you can run profile job (fully managed) that will give you basic statistics on your data (correlation, statistics, etc). 

---

## [Amazon QuickSight vs Kibana](https://wisdomplexus.com/blogs/kibana-vs-quicksight/)

### Kibana
Kibana is a visualization tool.
Volumes of data are initially stored in the formats of index or indices.Mostly these indexes are supported by **Elasticsearch**.
Indices convert the data into a structured form for Elasticsearch with Logstash or beats that collect the data from log files.
Results for the data are presented in visualized format through Lens, canvas, and maps.

* Within Kibana, users can set up their dashboards and categories, which means only those dashboards will be visible to users that are selected for a particular category.
* Kibana Lens helps users in reaching out to the data insights through a simple drag and drop action. No prior knowledge or experience is required for using the Kibana Lens.
* Through dashboard-only mode in Kibana, restrictions can be applied based on the roles of users.
* *Amazon Elasticsearch* Service can integrate with *Kinesis* to visualize data in real-time.
* Kibana has no data matching built-in function.

### Amazon QuickSight
Amazon QuickSight is a cloud-based business intelligence service that provides insights into the data through visualization. With the help of QuickSight, users can access the dashboard from any device and can be further integrated into any of the applications or websites to derive insight into the data received through them.It can collect data from any connected source. These sources can be any 3rd party applications, cloud, or on-premise sources. Amazon QuickSight derives the insights from these collected sources through ML techniques and provides user information in interactive dashboards, email-reports, or embedded analytics.

* Users can share particular dashboards with any staff member in the organization simply through browsers or mobile devices.
* QuickSight calculation engine is supported by SPICE (super-fast, parallel, in-memory, calculation engine). This SPICE engine replicates data quickly, thus assisting multiple users to perform various analysis tasks.
* You can discover hidden trends and insights into your data through the AWS proven machine learning capabilities.
* Quicksight has an anomaly insights feature, removing the need to write custom code.
* If you want to use SPICE and you don't have enough space, choose Edit/Preview data. In data preparation, you can remove fields from the data set to decrease its size. You can also apply a filter or write a SQL query that reduces the number of rows or columns returned.

> Athena fits the use case for the basic analysis of S3 data. Quicksight, which integrates with Athena, is a fully managed service that can be used to create rich visualizations for customers. The data is better suited for a relational data solution than a graph DB. Kibana cannot visualize data stored in S3 directly, as it would first have to be read into Elasticsearch. Athena is more appropriate for basic analysis than creating models in Sagemaker.

---

## Amazon Redshift
[**Resizing Clusters**](https://docs.aws.amazon.com/redshift/latest/mgmt/managing-cluster-operations.html): you can use elastic resize to scale your cluster by changing the node type and number of nodes. We recommend using elastic resize, which typically completes in minutes. If your new node configuration isn't available through elastic resize, you can use classic resize.
* *Elastic resize* – Use it to change the node type, number of nodes, or both. Elastic resize works quickly by changing or adding nodes to your existing cluster. If you change only the number of nodes, queries are temporarily paused and connections are held open, if possible. Typically, elastic resize takes 10–15 minutes. During the resize operation, the cluster is read-only.
* *Classic resize* – Use it to change the node type, number of nodes, or both. Classic resize provisions a new cluster and copies the data from the source cluster to the new cluster. Choose this option only when you are resizing to a configuration that isn't available through elastic resize, because it takes considerably more time to complete. An example of when to use it is when resizing to or from a single-node cluster. During the resize operation, the cluster is read-only. Classic resize can take several hours to several days, or longer, depending on the amount of data to transfer and the difference in cluster size and computing resources.

---

## Amazon Kinesis

Kinesis Firehose should be used instead of Kinesis Data Streams since there are multiple data sources for the shipment records.

 [**Use Elasticsearch to read a Firehose delivery stream un another account**](https://docs.aws.amazon.com/firehose/latest/dev/controlling-access.html#cross-account-delivery-es): Create an IAM role under account A using the steps described in Grant Kinesis Data Firehose Access to a Public OpenSearch Service Destination. To allow access from the IAM role that you created in the previous step, create an OpenSearch Service policy under account B. Create a Kinesis Data Firehose delivery stream under account A using the IAM role that you created in such account. When you create the delivery stream, use the AWS CLI or the Kinesis Data Firehose APIs and specify the ClusterEndpoint field instead of DomainARN for OpenSearch Service.
 >To create a delivery stream in one AWS account with an OpenSearch Service destination in a different account, you must use the AWS CLI or the Kinesis Data Firehose APIs.

---

## Amazon Athena

**Federated query** is a new Amazon Athena feature that enables data analysts, engineers, and data scientists to execute SQL queries across data stored in relational, non-relational, object, and custom data sources. With Athena federated query, customers can submit a single SQL query and analyze data from multiple sources running on-premises or hosted on the cloud.

---

## [Amazon Kendra](https://docs.aws.amazon.com/kendra/latest/dg/adjusting-capacity.html)

Amazon Kendra is a highly accurate and intelligent search service that enables your users to search unstructured and structured data using natural language processing and advanced search algorithms.

You can use Amazon Kendra to create an updatable index of documents of a variety of types, including *plain text, HTML files, Microsoft Word documents, Microsoft PowerPoint presentations, and PDF files*.

---

## Amazon CodeGuru Reviewer
Amazon CodeGuru Reviewer is a service that uses program analysis and machine learning to detect potential defects that are difficult for developers to find and offers suggestions for improving your Java and Python code.

You can associate CodeGuru Reviewer with a repository to allow CodeGuru Reviewer to provide recommendations. After you associate a repository with CodeGuru Reviewer, CodeGuru Reviewer automatically analyzes pull requests that you make, and you can choose to run repository analyses on the code in your branch to analyze all the code at any time.

---

## Amazon Fraud Detector
To generate fraud predictions, Amazon Fraud Detector uses machine learning models that are trained with historical fraud data that you provide. There are three available models: ONLINE_FRAUD_INSIGHTS, TRANSACTION_FRAUD_INSIGHTS, ACCOUNT_TAKEOVER_INSIGHTS.

---

# Machine Learning Preprocessing

## Scaling & Normalization

**Scaling** means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1. You want to scale data when you're using methods based on measures of how far apart data points are, like support vector machines (SVM) or k-nearest neighbors (KNN). By scaling your variables, you can help compare different variables on equal footing.

**Normalization** is a more radical transformation. The point of normalization is to change your observations so that they can be described as a normal distribution $^{1}$. In general, you'll normalize your data if you're going to be using a machine learning or statistics technique that assumes your data is normally distributed. Some examples of these include linear discriminant analysis (LDA) and Gaussian naive Bayes.

> In scaling, you're changing *the range* of your data, while in normalization you're changing *the shape of the distribution* of your data.

The Box-Cox and Yeo-Johnson transformations are two different ways to transform a continuous (numeric) variable so that the resulting variable looks more normally distributed. They are often used in feature engineering to reduce skew in the raw variables.

### Box-CoX Transformation 
The transformation is really a family of transformations indexed by a parameter $\lambda$:
![](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D+%5Cpsi%28y%2C+%5Clambda%29+%3D+%5Cbegin%7Bcases%7D+%5Cdfrac%7By%5E%5Clambda+-+1%7D%7B%5Clambda%7D+%26%5Clambda+%5Cneq+0%2C+%5C%5C+%5Clog+y+%26%5Clambda+%3D+0.+%5Cend%7Bcases%7D+%5Cend%7Baligned%7D&bg=ffffff&fg=333333&s=0&c=20201002)

The parameter $\lambda$ is calculated usin Maximum Likelihood Estimation.

### Yeo-Johnson Transformation
Yeo and Johnson note that the tweak above only works when $y$ is bounded from below, and also that standard asymptotic results of maximum likelihood theory may not apply. 

![](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D+%5Cpsi%28y%2C+%5Clambda%29+%3D+%5Cbegin%7Bcases%7D++%5Cdfrac%7B%28y%2B1%29%5E%5Clambda+-+1%7D%7B%5Clambda%7D+%26y+%5Cgeq+0+%5Ctext%7B+and+%7D%5Clambda+%5Cneq+0%2C+%5C%5C++%5Clog+%28y%2B1%29+%26y+%5Cgeq+0+%5Ctext%7B+and+%7D+%5Clambda+%3D+0%2C+%5C%5C++-%5Cdfrac%7B%28-y+%2B+1%29%5E%7B2+-+%5Clambda%7D+-+1%7D%7B2+-+%5Clambda%7D+%26y+%3C+0+%5Ctext%7B+and+%7D+%5Clambda+%5Cneq+2%2C+%5C%5C++-+%5Clog%28-y+%2B+1%29+%26y+%3C+0%2C+%5Clambda+%3D+2.++%5Cend%7Bcases%7D+%5Cend%7Baligned%7D&bg=ffffff&fg=333333&s=0&c=20201002)

The motivation for this transformation is rooted in the concept of relative skewness introduced by [van Zwet (1964)](https://onlinelibrary.wiley.com/doi/abs/10.1002/bimj.19680100134). This transformation has the following properties:

1. $\psi$ is concave in $y$ for $\lambda < 1$ and convex for $\lambda > 1$.
2. The constant shift of $+1$ makes is such that the transformed value will always have the same sign as the original value.
3. The new transformations on the positive line are equivalent to the Box-Cox transformation for $y > -1$ (after accounting for the constant shift), so the Yeo-Johnson transformation can be viewed as a **generalization of the Box-Cox transformation**.
 
---

## Imputation Methods for Missing Values

![](https://editor.analyticsvidhya.com/uploads/30381Imputation%20Techniques%20types.JPG)

1. **Complete Case Analisys**: This is a quite straightforward method of handling the Missing Data, which directly removes the rows that have missing data i.e we consider only those rows where we have complete data i.e data is not missing. This method is also popularly known as "Listwise deletion".
    * **Assumptions**:
        * Data is Missing At Random(MAR)
        * Missing data is completely removed from the table.
    * **Advantages**:
        * Easy to implement.
        * No Data manipulation required.
    * **Limitations**:
        * Deleted data can be informative.
        * Can lead to the deletion of a large part of the data.
        * Can create a bias in the dataset, if a large amount of a particular type of variable is deleted from it.
        * The production model will not know what to do with Missing data.
    * **When to Use**:
        * Data is MAR(Missing At Random).
        * Good for Mixed, Numerical, and Categorical data.
        * Missing data is not more than 5% - 6% of the dataset.
        * Data doesn't contain much information and will not bias the dataset.

2. **Arbitrary Value Imputation**: It can handle both the Numerical and Categorical variables. This technique states that we group the missing values in a column and assign them to a new value that is far away from the range of that column. Mostly we use values like 99999999 or -9999999 or "Missing" or "Not defined" for numerical & categorical variables.

    * **Assumptions**:
        * Data is **NOT** Missing At Random(MAR)
        * The missing data is imputed with an arbitrary value that is not part of the dataset or Mean/Median/Mode of data.
    * **Advantages**:
        * Easy to implement.
        * We can use it in production.
        * It retains the importance of missing values, if it exists
    * **Limitations**:
        * Can distort original variable distribution.
        * Arbitrary values can create outliers.
        * Extra caution required in selecting the Arbitrary value.
    * **When to Use**:
        * Data is not MAR(Missing At Random).
        * Suitable for All.

3. **Frequent Category Imputation**: This technique says to replace the missing value with the variable with the highest frequency or in simple words replacing the values with the Mode of that column. This technique is also referred to as **mode**.

    * **Assumptions**:
        * Data is missing at random.
        * There is a high probability that the missing data looks like the majority of the data.
    * **Advantages**:
        * Implementation is easy.
        * We can obtain a complete dataset in very little time.
        * We can use this technique in the production model.
    * **Limitations**:
        * The higher the percentage of missing values, the higher will be the distortion.
        * May lead to over-representation of a particular category.
        * Can distort original variable distribution.
    * **When to Use**:
        * Data is Missing at Random(MAR)
        * Missing data is not more than 5% - 6% of the dataset.

4. **Hot Deck Imputation**: A randomly chosen value from an individual in the sample who has similar values on other variables. In other words, find all the sample subjects who are similar on other variables, then randomly choose one of their values on the missing variable. One advantage is you are constrained to only possible values. In other words, if Age in your study is restricted to being between 5 and 10, you will always get a value between 5 and 10 this way. Another is the random component, which adds in some variability. This is important for accurate standard errors.

5. **Cold Deck Imputation**: A systematically chosen value from an individual who has similar values on other variables. This is similar to Hot Deck in most ways, but removes the random variation. So for example, you may always choose the third individual in the same experimental condition and block.

6. **Regression Imputation**: The predicted value obtained by regressing the missing variable on other variables. So instead of just taking the mean, you're taking the predicted value, based on other variables. This *preserves relationships* among variables involved in the imputation model, *but not variability* around predicted values.

7. **Stochastic Regression Imputation**: The predicted value from a regression plus a random residual value. This has all the advantages of regression imputation but adds in the advantages of the random component. Most multiple imputation is based off of some form of stochastic regression imputation.

8. **Interpolation and extrapolation**: An estimated value from other observations from the same individual. It usually only works in longitudinal data. Use caution, though. Interpolation, for example, might make more sense for a variable like height in children–one that can't go back down over time. Extrapolation means you're estimating beyond the actual range of the data and that requires making more assumptions that you should.

9. **Single and Multiple Imputation**: Single refers to the fact that you come up with a single estimate of the missing value, using one of the methods listed above.It's popular because it is conceptually simple and because the resulting sample has the same number of observations as the full data set. **Careful**: some imputation methods result in biased parameter estimates, such as means, correlations, and regression coefficients, unless the data are Missing Completely at Random (MCAR). The bias is often worse than with listwise deletion. The extent of the bias depends on many factors, including the imputation method, the missing data mechanism, the proportion of the data that is missing, and the information available in the data set. Moreover, all single imputation methods underestimate standard errors.

    Since the imputed observations are themselves estimates, their values have corresponding random error. But when you put in that estimate as a data point, your software doesn't know that. So it overlooks the extra source of error, resulting in too-small standard errors and too-small p-values. And although imputation is conceptually simple, it is difficult to do well in practice. So it's not ideal but might suffice in certain situations. So [Multiple Imputation](https://www.statisticshowto.com/multiple-imputation/) comes up with multiple estimates. Two of the methods listed above work as the imputation method in multiple imputation–hot deck and stochastic regression. This re-introduces some variation that your software can incorporate in order to give your model accurate estimates of standard error.

10. **Iterative Imputation**: A sophisticated approach involves defining a model to predict each missing feature as a function of all other features and to repeat this process of estimating feature values multiple times. The repetition allows the refined estimated values for other features to be used as input in subsequent iterations of predicting missing values. This is generally referred to as iterative imputation. 

Iterative imputation refers to a process where each feature is modeled as a function of the other features, e.g. a regression problem where missing values are predicted. Each feature is imputed sequentially, one after the other, allowing prior imputed values to be used as part of a model in predicting subsequent features. This approach may be generally referred to as fully conditional specification (FCS) or multivariate imputation by chained equations (MICE).

Different regression algorithms can be used to estimate the missing values for each feature, although linear methods are often used for simplicity. The number of iterations of the procedure is often kept small, such as 10. Finally, the order that features are processed sequentially can be considered, such as from the feature with the least missing values to the feature with the most missing values.

---

## Variable Enconding

Most of the Machine learning algorithms can not handle **categorical variables** unless we convert them to numerical values. Many algorithm's performances vary based on how Categorical variables are encoded.

### One Hot Encoding
In this method, we map each category to a vector that contains 1 and 0, denoting the presence or absence of the feature. The number of vectors depends on the number of categories for features.

One hot encoding with N-1 binary variables should be used in linear Regression to ensure the correct number of degrees of freedom (N-1). This means that N-1 binary variables give complete information about (represent completely) the original categorical variable to the linear Regression. This approach can be adopted for any machine learning algorithm that looks at ALL the features simultaneously during training (for example, *support vector machines* and *neural networks* as well as *clustering algorithms*).


### Label Encoding
In this encoding, each category is assigned a value from 1 through N (where N is the number of categories for the feature. One major issue with this approach is *there is no relation or order between these classes*, but the algorithm might consider them as some order or some relationship.

### Ordinal Encoding
We do Ordinal encoding to ensure the encoding of variables retains the ordinal nature of the variable. This is reasonable only for ordinal variables, unlike the previous case.

### Helmert Encoding
In this encoding, the mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels.

### Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are $n$ unique categories, then binary encoding results in the only $\log_2(n)$ features. *Compared to One Hot Encoding, this will require fewer feature columns* (for 100 categories, One Hot Encoding will have 100 features, while for Binary encoding, we will need just seven features).

### Frequency Encoding
It is a way to utilize the frequency of the categories as labels. In the cases where the frequency is related somewhat to the target variable, it helps the model understand and assign the weight in direct and inverse proportion, depending on the nature of the data. If two different categories appear the same amount of times in the dataset, that is, they appear in the same number of observations, they will be replaced by the same number,hence, *may lose valuable information*.

### [Target Encoding](https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69)
Target encoding is similar to label encoding, except here labels are correlated directly with the target. For example, in mean target encoding for each category in the feature label is decided with the mean value of the target variable on training data. This encoding method brings out the relation between similar categories, *but the connections are bounded within the categories and target itself*. The advantages of the mean target encoding are that *it does not affect the volume of the data* and helps in faster learning. 

The problem of target encoding has a name: **over-fitting**. Indeed, relying on an average value isn't always a good idea when the number of values used in the average is low. To overcome such problem, **additive smoothing** is often employed, which means that the (weighted) global mean is taken into consideration when calculating the encoding.

Mean encoding can embody the target in the label, whereas label encoding does not correlate with the target. In the case of many features, mean encoding could prove to be a much simpler alternative.

### Weight of Evidence Encoding
Weight of evidence (WOE) measures how much the evidence supports or undermines a hypothesis.

$$WoE = [\ln\frac{P(0)}{P(1)}]*100$$

WoE is well suited for *Logistic Regression* because the Logit transformation is simply the log of the odds. Therefore, by using WoE-coded predictors in Logistic Regression, the predictors are prepared and coded to the same scale.

The WoE transformation has (at least) three advantage:
1. It can transform an independent variable to establish a monotonic relationship to the dependent variable on a "logistic" scale.
2. For variables with too many (sparsely populated) discrete values, these can be grouped into categories (densely populated), and the WoE can be used to express information for the whole category.
3. The (univariate) effect of each category on the dependent variable can be compared across categories and variables because WoE is a standardized value (for example, you can compare WoE of married people to WoE of manual workers).

It also has (at least) three drawbacks:
1. Loss of information (variation) due to binning to a few categories.
2. It is a "univariate" measure, so it does not take into account the correlation between independent variables.
3. It is easy to manipulate (over-fit) the effect of variables according to how categories are created.

### Hashing
Hashing converts categorical variables to a higher dimensional space of integers, where the distance between two vectors of categorical variables is approximately maintained by the transformed numerical dimensional space.

### James-Stein Encoding
For feature value, the James-Stein estimator returns a weighted average of:
1. The mean target value for the observed feature value.
2. The mean target value (regardless of the feature value).

The James-Stein encoder shrinks the average toward the overall average. It is a target based encoder. James-Stein estimator has, however, one practical limitation - it was defined only for normal distributions.

![](https://miro.medium.com/max/875/0*NBVi7M3sGyiUSyd5.png)

---

## [Word Embeddings](https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d)

There are three main step in text preprocessing:
1. Tokenization ([n-grams](https://cran.r-project.org/web/packages/textrecipes/vignettes/Working-with-n-grams.html))
2. Normalization
3. Denoising

In a nutshell, **tokenization** is about splitting strings of text into smaller pieces, or "tokens". Paragraphs can be tokenized into sentences and sentences can be tokenized into words. **Normalization** aims to put all text on a level playing field, e.g., converting all characters to lowercase. **Noise removal** cleans up the text, e.g., remove extra whitespaces.

---

## Class Imbalance
 Ideally, it would be great if you could come up with more examples representing fraudulent behavior.  However, for some problems, the minority class does not happen in real life very often, so these examples might be hard to find.  An alternative is to use a data preparation approach called downsampling and upweighting.  You would apply this technique to the majority class;
 * *Downsampling*: Use a smaller set of the examples labeled with 'not fraud' (majority class) so that there is less of an imbalance between the two classes.
* *Upweighting*: After downsampling, you need to add weight to the data that makes up the downsampling set. This weight should be proportional to the factor by which you downsampled the majority class.

### Sampling Techniques
Resampling methods are designed to change the composition of a training dataset for an imbalanced classification task. 

* [**Undersampling**](https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/): Undersampling refers to a group of techniques designed to balance the class distribution for a classification dataset that has a skewed class distribution. Undersampling techniques remove examples from the training dataset that belong to the majority class in order to better balance the class distribution,
    * **Near Miss**: Near Miss refers to a collection of undersampling methods that select examples based on the distance of majority class examples to minority class examples. There are three versions of the technique.
        * *NearMiss-1* selects examples from the majority class that have the smallest average distance to the $n$ closest examples from the minority class.
        * *NearMiss-2* selects examples from the majority class that have the smallest average distance to the $n$ furthest examples from the minority class.
        * *NearMiss-3* involves selecting a given number of majority class examples for each example in the minority class that are closest.

    * **Condensed Nearest Neighbor Rule**: Condensed Nearest Neighbors, or CNN for short, is an undersampling technique that seeks a subset of a collection of samples that results in no loss in model performance, referred to as a minimal consistent set. It is achieved by enumerating the examples in the dataset and adding them to the "store" only if they cannot be classified correctly by the current contents of the store. When used for imbalanced classification, the store is comprised of all examples in the minority set and only examples from the majority set that cannot be classified correctly are added incrementally to the store.
        > For KNN, It's a relatively slow procedure, so small datasets and small k values are preferred.
        >> A criticism of the Condensed Nearest Neighbor Rule is that examples are selected randomly, especially initially. This has the effect of allowing redundant examples into the store and in allowing examples that are internal to the mass of the distribution, rather than on the class boundary, into the store.
    * **Tomek Links**: it's a rule that finds pairs of examples, one from each class; they together have the smallest Euclidean distance to each other in feature space. These cross-class pairs are now generally referred to as *Tomek Links* and are valuable as they define the class boundary. If the examples in the minority class are held constant, the procedure can be used to find all of those examples in the majority class that are closest to the minority class, then removed. These would be the ambiguous examples.
        > Because the procedure only removes so-named "Tomek Links", we would not expect the resulting transformed dataset to be balanced, only less ambiguous along the class boundary.
    * **Edited Nearest Neighbors Rule**: This rule involves using k=3 nearest neighbors to locate those examples in a dataset that are misclassified and that are then removed before a k=1 classification rule is applied. For each instance $a$ in the dataset, its three nearest neighbors are computed. If $a$ is a majority class instance and is misclassified by its three nearest neighbors, then $a$ is removed from the dataset. Alternatively, if $a$ is a minority class instance and is misclassified by its three nearest neighbors, then the majority class instances among a's neighbors are removed.
    * **One-Sided Selection**: One-Sided Selection is an undersampling technique that combines Tomek Links and the Condensed Nearest Neighbor (CNN) Rule. Specifically, Tomek Links are ambiguous points on the class boundary and are identified and removed in the majority class. The CNN method is then used to remove redundant examples from the majority class that are far from the decision boundary.
    ![](https://machinelearningmastery.com/wp-content/uploads/2019/10/Overview-of-the-One-Sided-Selection-for-Undersampling-Procedure2.png)
    * **Neighborhood Cleaning Rule**: The Neighborhood Cleaning Rule is an undersampling technique that combines both the Condensed Nearest Neighbor (CNN) Rule to remove redundant examples and the Edited Nearest Neighbors (ENN) Rule to remove noisy or ambiguous examples. Like OSS, the CSS method is applied in a one-step manner, then the examples that are misclassified according to a KNN classifier are removed, as per the ENN rule. Unlike OSS, less of the redundant examples are removed and more attention is placed on "cleaning" those examples that are retained. 
    ![](https://machinelearningmastery.com/wp-content/uploads/2019/10/Summary-of-the-Neighborhood-Cleaning-Rule-Algorithm.png)
* [**Oversampling**](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/): One approach to addressing imbalanced datasets is to oversample the minority class. The simplest approach involves duplicating examples in the minority class, although these examples don't add any new information to the model. Instead, new examples can be synthesized from the existing examples. This is a type of data augmentation for the minority class and is referred to as the **Synthetic Minority Oversampling Technique** (SMOTE). SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
    > A general downside of the approach is that synthetic examples are created without considering the majority class, possibly resulting in ambiguous examples if there is a strong overlap for the classes.
    >>We can be selective about the examples in the minority class that are oversampled using SMOTE. Hence we will see some extension to the algorithm.

    * **Borderline-SMOTE**: A popular extension to SMOTE involves selecting those instances of the minority class that are misclassified, such as with a k-nearest neighbor classification model. We can then oversample just those difficult instances, providing more resolution only where it may be required.
    * **Borderline-SMOTE SVM**: Instead of KNN, an SVM is used to locate the decision boundary defined by the support vectors and examples in the minority class that close to the support vectors become the focus for generating synthetic examples. New instances will be randomly created along the lines joining each minority class support vector with a number of its nearest neighbors using the interpolation. If majority class instances count for less than a half of its nearest neighbors, new instances will be created with extrapolation to expand minority class area toward the majority class.
    * **Adaptive Synthetic Sampling**: Another approach that involves generating synthetic samples inversely proportional to the density of the examples in the minority class. With online Borderline-SMOTE, a discriminative model is not created. Instead, examples in the minority class are weighted according to their density, then those examples with the lowest density are the focus for the SMOTE synthetic example generation process.

---

## [Correlation Analysis](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)
There may be complex and unknown relationships between the variables in your dataset. It is important to discover and quantify the degree to which variables in your dataset are dependent upon each other, which can help you better prepare your data to meet the expectations of machine learning algorithms.

> **Correlation** is a statistical summary of the relationship between variables. A correlation could be positive, meaning both variables move in the same direction, or negative, meaning that when one variable's value increases, the other variables' values decrease.
>> The performance of some algorithms can deteriorate if two or more variables are tightly related, called **multicollinearity**. An example is linear regression, where one of the offending correlated variables should be removed in order to improve the skill of the model. We may also be interested in the correlation between input variables with the output variable in order provide insight into which variables may or may not be relevant as input for developing a model.

### Covariance
Variables can be related by a **linear relationship**. This is a relationship that is consistently additive across the two data samples. This relationship can be summarized between two variables using the covariance.

$$cov(X,Y) = \frac{\sum((X - mean(X))*(Y-mean(Y)))}{n-1} $$
> The use of the mean in the calculation suggests the need for each data sample to have a **Gaussian or Gaussian-like distribution**.

The sign of the covariance can be interpreted as whether the two variables change in the same direction (positive) or change in different directions (negative). The magnitude of the covariance is **not easily interpreted**. A covariance value of zero indicates that both variables are **completely independent**.

### Pearson's correlation
The Pearson correlation coefficient can be used to summarize the strength of the **linear relationship** between two data samples. The Pearson's correlation coefficient is calculated as the covariance of the two variables divided by the product of the standard deviation of each data sample.

$$ \rho(X,Y) = \frac{cov(X,Y)}{std(X)*std(Y)}$$

> Pearson's correlation coefficient ranges from -1 to 1, and two variables are considered strongly correlated when $|\rho(X,Y)| > 0.5$. Each variable is always completely correlated with itself ($\rho(X,X) = 1$).

### Spearman's correlation
Two variables may be related by a **nonlinear relationship**, such that the relationship is stronger or weaker across the distribution of the variables. In this case, the Spearman's correlation coefficient can be used to summarize the strength between the two data samples. Instead of calculating the coefficient using covariance and standard deviations on the samples themselves, these statistics are calculated from the relative rank of values on each sample. This is a common approach used in non-parametric statistics, e.g. statistical methods where **we do not assume a distribution of the data** such as Gaussian.

$$\rho(X,Y) = \frac{cov(Rank(X),Rank(Y))}{std(Rank(X))*std(Rank(Y))}$$
>A linear relationship between the variables is not assumed, although a **monotonic relationship is assumed**.

---

### [Polychoric correlation](https://www.r-bloggers.com/2021/02/how-does-polychoric-correlation-work-aka-ordinal-to-ordinal-correlation/)

In statistics, polychoric correlation is a technique for estimating the correlation between two hypothesised normally distributed continuous latent variables, from two **observed ordinal variables**.

Polychoric correlation assumes that each ordinal data point represents a binned continuous value from a **normal distribution**, and then tries to estimate the correlation coefficient on that assumption. The distribution of the ordinal responses, along with the assumption that the latent values follow a normal distribution, allows to estimate the correlation between latent variables as if you actually knew what those values were.

> *For example*: "How was your last movie: poor, fair, good, very good, or excellent?" We treat the responses like they're binned values for some abstract variable we didn't ask for, like a quality scale from 0-100, so "very good" could be a 73 or a 64. The actual number of movies watched, or the actual 0-100 quality score that someone would give their last movie aren't recorded in the data, so we would call them **latent variables**.
>>  Ideally, polychoric correlation on the (realistic) binned / ordinal data will closely match the Pearson correlation on the latent data.

### [Mutual Information](https://quantdare.com/what-is-mutual-information/)
Mutual Information measures the **entropy drops** under the condition of the target value, which indicates how much information can be obtained from a random variable by observing another random variable.

The main difference with correlation (and by extent, [Pearson's Correlation Coefficient](#pearsons-correlation)) is that the latter is a measure of linear dependence, whereas mutual information measures general dependence (including non-linear relations). Therefore, mutual information detects dependencies that do not only depend on the covariance. Thus, mutual information is zero when the two random variables are strictly independent.

---

## Variance Methods

### [Principal Component Analysis](https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/)

PCA is an unsupervised linear dimensionality reduction and data visualization technique for very high dimensional data. As having high dimensional data is very hard to gain insights from adding to that, it is very computationally intensive. The main idea behind this technique is to reduce the dimensionality of data that is highly correlated by transforming the original set of vectors to a new set which is known as Principal component.

PCA tries to preserve the Global Structure of data i.e when converting d-dimensional data to d-dimensional data then it tries to map all the clusters as a whole due to which local structures might get lost. 

### [T-Distributed Stochastic Neighbourhood Embedding (t-SNE)](https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/)

T-SNE is also a unsupervised non-linear dimensionality reduction and data visualization technique. The math behind t-SNE is quite complex but the idea is simple. It embeds the points from a higher dimension to a lower dimension trying to preserve the neighborhood of that point.

Unlike PCA it tries to preserve the Local structure of data by minimizing the Kullback-Leibler divergence (KL divergence) between the two distributions with respect to the locations of the points in the map.


### [Linear Discriminant Analysis](https://towardsai.net/p/data-science/lda-vs-pca)

LDA is very similar to PCA , they both look for linear combinations of the features which best explain the data. The main difference is that the Linear discriminant analysis is a supervised dimensionality reduction technique that also achieves classification of the data simultaneously.Linear Discriminant Analysis projects the data points onto new axes such that these new components maximize the separability among categories while keeping the variation within each of the categories at a minimum value. LDA focuses on finding a feature subspace that maximizes the separability between the groups.

![](https://cdn-images-1.medium.com/max/434/1*BfrSQg2wZoCHxs89zjcW4w.png)

>LDA assumes that the independent variables are normally distributed for each of the categories.
>>LDA assumes the independent variables have equal variances and covariances across all the categories. This can be tested with Box's M statistic $^{1}$. When this assumption fails, another variant of Discriminant analysis is used which is the **Quadratic Discriminant Analysis (QDA)**.
>>>The performance of prediction can decrease with the increased correlation between the independent variables.

$\small{1}.$ Box's M test is a multivariate statistical test used to check the equality of multiple variance-covariance matrices.


### [R-Squared](https://www.investopedia.com/terms/r/r-squared.asp)

R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. Whereas correlation explains the strength of the relationship between an independent and dependent variable, R-squared explains to what extent the variance of one variable explains the variance of the second variable. So, if the R2 of a model is 0.50, then approximately half of the observed variation can be explained by the model's inputs.

$$R^2 = 1- \frac{Unexplained Variation}{Total Variation} $$

The actual calculation of R-squared requires several steps. This includes taking the data points (observations) of dependent and independent variables and finding the line of best fit, often from a regression model. From there you would calculate predicted values, subtract actual values and square the results. This yields a list of errors squared, which is then summed and equals the unexplained variance. To calculate the total variance, you would subtract the average actual value from each of the actual values, square the results and sum them. 

R-Squared only works as intended in a simple linear regression model with one explanatory variable. With a multiple regression made up of several independent variables, the R-Squared must be **adjusted**. The adjusted R-squared compares the descriptive power of regression models that include diverse numbers of predictors. 

> R-squared will give you an estimate of the relationship between movements of a dependent variable based on an independent variable's movements. It doesn't tell you whether your chosen model is good or bad, nor will it tell you whether the data and predictions are biased.

---

## Spark

When a task is **parallelized** in Spark, it means that concurrent tasks may be running on the driver node or worker nodes. How the task is split across these different nodes in the cluster depends on the types of data structures and libraries that you're using. When a task is **distributed** in Spark, it means that the data being operated on is split across different nodes in the cluster, and that the tasks are being performed concurrently. Ideally, you want to author tasks that are both parallelized and distributed.

---

## [Securing Sensitive Information in AWS Data Stores](https://aws.amazon.com/it/blogs/database/best-practices-for-securing-sensitive-data-in-aws-data-stores/)
An effective strategy for securing sensitive data in the cloud requires a good understanding of general data security patterns and a clear mapping of these patterns to cloud security controls. You then can apply these controls to implementation-level details specific to data stores such as Amazon Relational Database Service (Amazon RDS) and Amazon DynamoDB. 

1. **Classify data based on their confidentiality**.
![](https://d2908q01vomqb2.cloudfront.net/887309d048beef83ad3eabf2a79a64a389ab1c9f/2019/04/12/BestPracticesSensitiveData1_2.png)

2. **Consider how and from where data can be accessed**.  security zone provides a well-defined network perimeter that implements controls to help protect all assets within it. A security zone also enables clarity and ease of reasoning for defining and enforcing network flow control into and out of the security zone based on its characteristics. You can define a network flow control policy through AWS network access control lists (ACLs) in combination with a complementary IAM policy. With these, you enforce access to the secured zone only from the restricted zone and never from the internet-facing external zone. This approach places your sensitive data two security layers beneath internet accessibility.

3. **Preventative and Detective Controls**. There are three main categories of preventative controls: *IAM, Infrastructure Security and Data Protection*. The sequence in which you layer the controls together can depend on your use case. The effective application of detective controls allows you to get the information you need to respond to changes and incidents. The effective application of detective controls allows you to get the information you need to respond to changes and incidents. A robust detection mechanism with integration into a security information and event monitoring (SIEM) system enables you to respond quickly to security vulnerabilities and the continuous improvement of the security posture of your data.


4. **Swim Lane Isolation**. Swim-lane isolation can be best explained in the context of domain-driven design. If you think about grouping microservices into domains that resemble your business model, you can also think about the security of the data stores attached to each of those microservices from the context of a business domain. This enables you to achieve two things: Enforce a pure microservices data-access pattern to ensure that microservice data store access is available only through the owning microservice APIs, and ensure that sensitive data from one microservice domain does not leak out through another microservice domain.


---

# Model Selection
Model selection is the task of selecting a statistical model from a set of candidate models, given data. In the simplest cases, a pre-existing set of data is considered.


## [Feature Combination](https://datascience.stackexchange.com/questions/28883/why-adding-combinations-of-features-would-increase-performance-of-linear-svm)

Combination of existing features can give new features and help for classification. Polynomial features ($x^2$, $x^3$, $y^2$, etc.) do not give additional data points, but instead increase the complexity the objective function (which sometimes leads to overfitting). 

---

## [Naive Bayes](https://sebastianraschka.com/Articles/2014_naive_bayes_1.html#3_3_multivariate)
The Naive Bayes classifier is a simple probabilistic classifier which is based on Bayes theorem with strong and naive independence assumptions. Despite the naive design and oversimplified assumptions that this technique uses, Naive Bayes performs well in many complex real-world problems. You can use Naive Bayes when you have limited resources in terms of CPU and Memory. Moreover when the training time is a crucial factor, Naive Bayes comes handy since it can be trained very quickly. It comes in three flavours:

* **Bernoulli Naive Bayes**: It assumes that all our features are binary such that they take only two values. Means 0s can represent "word does not occur in the document" and 1s as "word occurs in the document".
* **Multinomial Naive Bayes**: It is used when we have discrete data (e.g. movie ratings ranging 1 and 5 as each rating will have certain frequency to represent). In text learning we have the count of each word to predict the class or label.
* **Gaussian Naive Bayes** : Because of the assumption of the normal distribution, Gaussian Naive Bayes is used in cases when all our features are continuous. For example in Iris dataset features are sepal width, petal width, sepal length, petal length. So its features can have different values in data set as width and length can vary. We can't represent features in terms of their occurrences.

---

## [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

---

## [Factorization Machines](https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db)

---

# Model Evaluation

## Confusion Matrix
A confusion matrix  is a performance measurement for machine learning classification problem where output can be two or more classes. It is extremely useful for measuring Recall, Precision, Specificity, Accuracy, and most importantly AUC-ROC curves.

![](https://miro.medium.com/max/445/1*Z54JgbS4DUwWSknhDCvNTQ.png)

### Statistics on Confusion Matrix

$$Precision = \frac{TP}{TP+FP} $$
$$Recall = \frac{TP}{TP+FN} $$
$$FPR = \frac{FP}{FP+TN} $$
$$Accuracy = \frac{TP+TN}{Total} $$
$$F_{score} = \frac{2*Recall*Precision}{Recall+Precision} $$



### [Receiver Operating Characteristic Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
An ROC curve is a graph showing the performance of a classification model **at all classification thresholds**. This curve plots two parameters: **Recall(TPR) and FPR**

![](https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg)

An ROC curve plots TPR vs. FPR at different classification thresholds. To compute the points in an ROC curve, we could evaluate a model many times with different classification thresholds, but this would be inefficient. Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called **Area Under (ROC) Curve**.

AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.

AUC is desirable for the following two reasons:

* AUC is **scale-invariant**. It measures how well predictions are ranked, rather than their absolute values.
* AUC is **classification-threshold-invariant**. It measures the quality of the model's predictions regardless of what classification threshold is chosen.

However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

* Scale invariance is not always desirable. For example, sometimes we really do need well calibrated probability outputs.
* Classification-threshold invariance is not always desirable. In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. 
AUC isn't a useful metric for anomaly detection.

> In general, raising the classification threshold reduces false positives, **thus raising precision**.
>> Raising our classification threshold will cause the number of true positives to decrease or stay the same and will cause the number of false negatives to increase or stay the same. **Thus, recall will either stay constant or decrease**.

---

## Satisficing Metrics
You may want to select a classifier that maximizes accuracy, but subject to running time, that is the time it takes to classify an image that must be less than 100 milliseconds or equal to it. That is, it just needs to be nice enough, it just needs to be less than 100 milliseconds and beyond that you just don't care about, or at least you don't care that much. That will therefore be a fairly reasonable way to trade off or put off Both accuracy and running time together.