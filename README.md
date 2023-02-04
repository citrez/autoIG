## Automated trading using IG's API

The aim of this repository is to allow for as-easy-as-possible exploration of various trading strategies and techniques. To this end, I want to make it as easy as possible to
1) Train new models with new strategies/ hyper-paramters.
2) Deploy these models into a shadow environemnt
3) Monitor the performance shadow models
4) Promote well performing shadow models to live

## Open source tech used:
- mlflow - track trained model metrics, store the model, make deployment easier by saving
- parameters needed in deployments with the model.
- grafana - Monitor deployment, monitor shadow performance
- dbml/mermaid - Documentation diagrams
- sklearn - ML pipelines
- pandas/numpy

## Getting started
start grafana with `./bin/grafana-server web ` in the grafana repo. Listens on localhost post 3000

Start a local server on port 8000 using `python3 -m http.server` in the autoIG folder. 
grafana username and password is admin
TODO: Make this in root dir 
Start mlflow using `mlflow ui` command
In grafana, connect to mysql with root user 


### Reference
- [IG API Glossary](https://labs.ig.com/glossary)
- [IG REST Reference](https://labs.ig.com/rest-trading-api-reference)
- [IG Stream Reference](https://labs.ig.com/streaming-api-reference)
- [Database diagram](https://dbdocs.io/citrez/autoIG)
  - run `dbml verify` then `dbml build` to build the entity relationship diagram
- [Grafana](https://citrez.grafana.net/a/grafana-easystart-app/?src=hg_notification_trial) can be used the monitor databases and produce analytics.
- [IG-markets-api-python-library](https://github.com/ig-python/ig-markets-api-python-library)
- [IG Dashboard](https://www.ig.com/uk/myig/dashboard)
 

## Features
Ability to list a model (from mlflow) in config and then deploy (with monitoring) seamlessly.
Abilty to see where the training data came from. 

### Choices
Do not use Oanda. Use IG API for making trades. 
Use yahoo finance for historical data.

Arrange into proper docs:
Set LIMIT and STOP losses on all trades. Each trade lasts 3 mins, as set out in the training 
However often the stream of data comes through, we could resample to 10 second intervals, so a) the price is sufficiently different. b) we are not predicting all the time. If we need 3 signals, we only predict every 30 seconds. 


### RULES

Keep documentation as close to code as possible. 
Use code as documentation solutions such as mermaid/dbml. 
Don't save data with indexes, if you want read them in with indexes. 
Don’t need to focus on perfecting a model for a stock. Get shape working shape then can search predictable socks

Think about data model. Every prediction has a model ticket associated. Anything else?

Stack of different length tails. Feeding into linear model

Think about data collected. Dealid . Bought for. Sold at. Model used

Set up multiple models in shadow, promote after good performance

Make frequency of incoming data minute by resampling and not taking the most recent

Make a training script, which inputs data and outputs a model. 

I like the idea of using an instance based model. Like KNN regression. We can also use a metric for 'how near' the 5 nearest neighbors are to the instance we wish the predict. Only predict when it is near enough. 

### GLOSSARY
ASK: What I buy for
OFFER: I sell for

### Explanation in simple terms

``` mermaid
---
title: on_update
---
flowchart TB
s --> scsv[(stream.csv)]
rs[raw_stream] --if market open\nif stream larger--> s[stream]
mn[model name]-->mp
rs --> rscsv[(raw_stream.csv)]
s --if steam larger than past_periods_needed-->p
e[epic]-->rs
conf[deploy config]-->thr[buy threshold]
thr-->op
pm-->pmm[(position_metrics_merged.csv)]
sold-->pmm

mv[model version]-->mp
mlf[MlflowClient] -->e
mlf -->mn
mlf -->mv
mlf -->ppn[past_periods_needed]
ppn-->p


mp[Model pipeline]-->p[Make prediction]
p--if latest prediction>threshold-->op[Open a Position]
op-->pm[(position_metrics.csv)]
pm-->cp[Close a position]
cp-->pm
cp-->sold[(sold.csv)]
```

``` mermaid
---
title: fetch_data
---
flowchart TB
intr[instrument]-->c[config]
source[Source]-->c
res[Resolution]-->c
f[fetch_all_training_data]-->cd[Create directory]

c-->cd
fh-->d3[(training/\nsource/\ninstrument/\nresolution/\nstart_to_end)]
cd-->fh[fetch_history]
subgraph training_buckets
d1[(training/\nsource/\ninstrument/\nresolution/\nstart_to_end)]
d2[(training/\nsource/\ninstrument/\nresolution/\nstart_to_end)]
d3
end



```


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

Everytime we buy. OPU gives us the time we bought. We wants to have a record of the prediction and model used. 






