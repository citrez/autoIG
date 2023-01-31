## Automated trading using IG's API

The aim of this repository is to allow for as-easy-as-possible exploration of various trading stategies and techniques. To this end, I want to make it as easy as possible to
1) Train new models with new stagties/ hyperparamters.
2) Deploy these models into a shadoow environemnt
3) Monitor the performance of said models

Below I set out the specifics of how I address these 3 points.


I try to keep documentation as close to code as possible. 
There is documentation in the DBML. Comments on the code. 
## Getting started
start grafana with `./bin/grafana-server web ` in the grafana repo. Listens on localhost post 3000

Start a local server on port 8000 using `python3 -m http.server` in the autoIG folder. 
grafana username and password is admin
TODO: Make this in root dir 
Start mlflow using `mlflow ui` command
In grafana, connect to mysql with root user 


### Development Reference
- [IG API Glossary](https://labs.ig.com/glossary)
- [IG REST Reference](https://labs.ig.com/rest-trading-api-reference)
- [IG Stream Reference](https://labs.ig.com/streaming-api-reference)
- [Database diagram](https://dbdocs.io/citrez/autoIG)
  - run `dbml verify` then `dbml build` to build the entity relationship diagram
- [Grafana](https://citrez.grafana.net/a/grafana-easystart-app/?src=hg_notification_trial) can be used the monitor databases and produce analytics.
- [IG-markets-api-python-library](https://github.com/ig-python/ig-markets-api-python-library)
- [IG Dashboard](https://www.ig.com/uk/myig/dashboard)

MLflow used for tracking trained models. Storing the model itself, performance metrics and parameters needed to deploy. 

graphna used for live monitoring on modelsdeployment

mermaid used for flowchart live documenation 

dbml used for database schema and producing . 

## Features
Ability to list a model (from mlflow) in config and then deploy (with monitoring) seamlessly.
Abilty to see where the training data came from. 

### Choices
Do not use Oanda. Use IG API for making trades. 
Use yahoo finance for historical data

Arrange into proper docs:
Set LIMIT and STOP losses on all trades. Each trade lasts 3 mins, as set out in the training 
However often the stream of data comes through, we could resample to 10 second intervals, so a) the price is sufficiently different. b) we are not predicting all the time. If we need 3 signals, we only predict every 30 seconds. 


### RULES
Don't save data with indexes, if you want read them in with indexes. 

Don’t need to focus on perfecting a model for a stock. Get shape working shape then can search predictable socks

Think about data model. Every prediction has a model ticket associated. Anything else?

Stack of different length tails. Feeding into linear model

Think about data collected. Deal I’d . Bought for. Sold at. Model used

Set up multiple models in shadow, promote after good performance

Make frequency of incoming data minute by resampling and not taking the most recent

Make a training script, which inputs data and outputs a model. 

I like the idea of using an instance based model. Like KNN regression. We can also use a metric for 'how near' the 5 nearest neighbors are to the instance we wish the predict. Only predict when it is near enough. 

### GLOSSARY
ASK: What I buy for
OFFER: I sell for

### Explanation in simple terms

### Hints
Use `watch -n 1 tail /Users/ezracitron/my_projects/auto_IG/autoIG/resources/tmp/raw_stream.csv` to set up looking at raw stream

## Mermaid diagram

``` mermaid
---
title: The flow of my decisioning
---
flowchart TB
A[MARKET stream] --1min resample--> E[MARKET stream resampled]
A --> F[(raw_stream.csv)]
F --> C

C --> D[sklearn pipeline]
E --> C[(stream.csv)]
D --> P[prediction]
```


Raw stream comes in and get minute resampled.

Everytime we buy. OPU gives us the time we bought. We wants to have a record of the prediction and model used. 






