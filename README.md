## Automated trading using IG's API

I try to keep documentation as close to code as possible. 
There is documentation in the DBML. Comments on the code. 

### Data documentation: 
https://dbdiagram.io/d  
IG API Docs:  
IG_trading (lightweight python wrapper for IG markets API): https://github.com/ig-python/ig-markets-api-python-library/blob/master/sample/rest_ig.py  
IG trading dashboard: https://www.ig.com/uk/myig/dashboard

### Glossary
DFB = daily funded bet. Remains open util you close it

[dbdiagram](https://dbdiagram.io/d/62949e0cf040f104c1bff2c0) helps me draw the entity relationship database model seamlessly. 

[grafana](https://citrez.grafana.net/a/grafana-easystart-app/?src=hg_notification_trial) can be used the monitor databases and produce analytics. 

Choices:
Do not use Oanda. Use IG API for making trades. 

Arrange into proper docs:
Set LIMIT and STOP losses on all trades.
However often the stream of data comes through, we could resample to 10 second intervals, so a) the price is sufficiently different. b) we are not predicting all the time. If we need 3 signals, we only predict every 30 seconds. 

TODO:
Set up tracking and analytics based on IGs API.
Set up backtesting functionality.

Resample 1 minute intervals (or whatever was used during training), and then take head(1), rather than mean() or something else.

RULES:
Don't save data with indexes, if you want read them in with indexes. 


Don’t need to focus on perfecting a model for a stock. Get shape workingshape working then can search predictable socks

Think about data model. Eve prediction has a model ticket associated. Anything else?

Stack of different length tails. Feeding into linear model

Think about data collected. Deal I’d . Bought for. Sold at. Model used

Create models with script with mlflow giving I’d

Set up multiple models in shadow, promote after good performance

Make frequency of incoming data minute by resampling and not taking the most recent

Make a training script, which inputs data and outputs a model. 

I like the idea of using an instance based model. Like KNN regression. We can also use a metric for 'how near' the 5 nearest neighbors are to the instance we wish the predict. Only predict when it is near enough. 

GLOSSARY:
ASK: What I buy for
OFFER: I sell for










