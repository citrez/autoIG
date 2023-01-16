## Automated trading using IG's API

I try to keep documentation as close to code as possible. 
There is documentation in the DBML. Comments on the code. 
## Getting started
start grafana with `./bin/grafana-server web ` in the grafana repo
Start a local server on port 8000 using `python3 -m http.server` in the autoIG folder. 
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

### Choices
Do not use Oanda. Use IG API for making trades. 

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










