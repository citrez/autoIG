This is for automated trading using IG's API

Data documentation: https://dbdiagram.io/d  
IG API Docs:  
ig_trading (lightweight python wrapper for IG markets API): https://github.com/ig-python/ig-markets-api-python-library/blob/master/sample/rest_ig.py  
IG trading dashboard: https://www.ig.com/uk/myig/dashboard

Glossary:
DFB = daily funded bet. Remains open util you close it

[dbdiagram](https://dbdiagram.io/d/62949e0cf040f104c1bff2c0) helps me draw the entity relationship database model seamlessly. 

[grafana](https://citrez.grafana.net/a/grafana-easystart-app/?src=hg_notification_trial) can be used the moniotr dataabases and produce analytics. 

Choices:
Do not use Oanda. Use IG API for making trades. 

Arrange into proper docs:
Set LIMIT and STOP losses on all trades.
However often the stream of data comes through, we could resample to 10 second intervals, so a) the price is sufficiently different. b) we are not predicting all the time. If we need 3 signals, we only predict every 30 seconds. 

TODO:
Set up tracking and analytics based on IGs API.
Set up backtesting functionality.

Resample 1 minute intervals (or whatever was used during training), and then take head(1), rather than mean() or something else.






