CREATE SCHEMA `IG`;

CREATE SCHEMA `tmp`;

CREATE SCHEMA `pub`;

CREATE SCHEMA `stream`;

CREATE TABLE `IG`.`transactions` (
  `closelevel` float COMMENT 'Price deal closed at',
  `currency` varchar(255),
  `date` date COMMENT 'This is not a datetime, better to join to activity to get datetime of closing/opening position',
  `dateutc` date,
  `openlevel` float COMMENT 'Price deal opened at',
  `opendateutc` date,
  `period` varchar(255) COMMENT 'e.g daily funded bet (DFB)',
  `profitandloss` float COMMENT '(closelevel-openlevel)*size',
  `reference` varchar(255) PRIMARY KEY COMMENT 'The dealId of the closing deal',
  `size` float,
  `transationtype` varchar(255)
);

CREATE TABLE `IG`.`activity` (
  `channel` varchar(255),
  `date` datetime,
  `dealId` varchar(255) PRIMARY KEY,
  `description` varchar(255),
  `details` json COMMENT 'Think this is a JSON of stream.OPU of the affected positions.
An action opens or closed a position?
These are all the dealIds affetion by the actions of an activity
actiontype e.g POSITION_CLOSED, POSITION_DELETED, POSITION_OPENED,
POSITION_PARTIALLY_CLOSED, POSITION_ROLLED',
  `dealreference` varchar(255),
  `direction` ENUM ('BUY', 'SELL'),
  `epic` varchar(255)
);

CREATE TABLE `IG`.`open_positions` (
  `contractSize` float,
  `continent_name` varchar(255),
  `createdDate` datetime,
  `createdDateUTC` datetime,
  `dealId` varchar(255) PRIMARY KEY,
  `dealRefernce` varchar(255),
  `direction` varchar(255),
  `size` float,
  `name` varchar(255)
);

CREATE TABLE `IG`.`instrument` (
  `epic` varchar(255) PRIMARY KEY,
  `expiry` varchar(255),
  `name` varchar(255),
  `type` varchar(255),
  `streamingPricesAvailable` boolean,
  `marketId` varchar(255),
  `openingHours` date
);

CREATE TABLE `tmp`.`raw_stream` (
  `epic` varchar(255),
  `updated_at` datetime,
  `bid_open` float,
  `ask_open` float,
  `market_state` varchar(255),
  PRIMARY KEY (`epic`, `updated_at`)
);

CREATE TABLE `tmp`.`to_sell` (
  `dealId` varchar(255),
  `to_sell_date` datetime COMMENT 'The datetime I want to sell',
  `sold` boolean
);

CREATE TABLE `pub`.`stream` (
  `epic` varchar(255),
  `updated_at` datetime,
  `bid_open` float,
  `ask_open` float,
  `market_state` varchar(255),
  PRIMARY KEY (`epic`, `updated_at`)
);

CREATE TABLE `pub`.`position_metrics` (
  `dealId` varchar(255) PRIMARY KEY,
  `model_used` varchar(255) COMMENT 'The name+version is taken from the model registery',
  `prediction` float,
  `buy_date` datetime,
  `buy_level_resp` float
);

CREATE TABLE `pub`.`transactions_joined` (
  `y_true` float,
  `buy_date` datetime,
  `closing_dealId` varchar(255),
  `closeLevel` float,
  `currency` varchar(255),
  `instrumentName` varchar(255),
  `model_used` varchar(255),
  `opening_dealId` varchar(255),
  `openLevel` float,
  `y_pred` float,
  `profitAndLoss` varchar(255),
  `profitAndLoss_numeric` float,
  `sell_date` date,
  `size` float
);

CREATE TABLE `stream`.`price` (
  `bid` float COMMENT 'I sell for',
  `change` float,
  `change_pct` float,
  `epic` varchar(255),
  `high` float,
  `low` float,
  `market_delay` boolean,
  `market_state` ENUM ('CLOSED', 'OFFLINE', 'TRADEABLE', 'EDIT', 'AUCTION', 'AUCTION_NO_EDIT', 'SUSPENDED'),
  `mid_open` float,
  `offer` float COMMENT 'I buy for',
  `update_time` datetime
);

CREATE TABLE `stream`.`trade` (
  `affecteddeals` json,
  `direction` ENUM ('BUY', 'SELL'),
  `dealReference` varchar(255),
  `dealstatus` ENUM ('ACCEPTED', 'DECLINED'),
  `dealId` varchar(255),
  `epic` varchar(255),
  `level` float COMMENT 'instrument price',
  `size` float,
  `status` ENUM ('AMENDED', 'DELETED', 'FULLY_CLOSED', 'UPDATED', 'OPENED', 'PARTIALLY_CLOSED')
);

CREATE TABLE `stream`.`OPU` (
  `channel` ENUM ('DEALER', 'MOBILE', 'pub_FIX_API', 'pub_WEB_API', 'SYSTEM', 'WEB'),
  `currency` varchar(255),
  `date` datetime,
  `dealId` varchar(255),
  `dealstatus` ENUM ('ACCEPTED', 'DECLINED'),
  `dealReference` varchar(255),
  `direction` ENUM ('BUY', 'SELL'),
  `epic` varchar(255),
  `level` float,
  `status` ENUM ('AMENDED', 'DELETED', 'FULLY_CLOSED', 'UPDATED', 'OPENED', 'PARTIALLY_CLOSED'),
  `size` float
);

ALTER TABLE `transactions` COMMENT = 'Transactions are a possition that has been both opened and closed.
So we can calculate profit';

ALTER TABLE `activity` COMMENT = 'Activities are made up from actions.
Activities can be on position or limit orders.
Actions can be ammending, creating, deleting a position';

ALTER TABLE `open_positions` COMMENT = 'The currently open poistions.
This can be calculated by looking at the dealID in create_open_position_responce 
that are not in close_position_responce';

ALTER TABLE `instrument` COMMENT = 'This is information about the instrument itself';

ALTER TABLE `raw_stream` COMMENT = 'A selection from stream.price';

ALTER TABLE `to_sell` COMMENT = 'For each opening dealId when should it be sold.';

ALTER TABLE `stream` COMMENT = 'A resampled (1 min) version of raw_stream';

ALTER TABLE `position_metrics` COMMENT = 'We want to keep track of many things about the position we open, when we open them. ';

ALTER TABLE `transactions_joined` COMMENT = 'IG.transaction, joined with other useful things to get some metrics.';

ALTER TABLE `price` COMMENT = 'This is a stream of prices for a particular instrument';

ALTER TABLE `trade` COMMENT = 'These are the distinct confirms
i.e everytime we get a confirm, this returns an item
It does not necessariliy need to be open a successfull position.
A rejection will show here, but not in OPU table';

ALTER TABLE `OPU` COMMENT = 'Open position updates (OPU) for an account. This is the stream for activity?';

ALTER TABLE `tmp`.`raw_stream` ADD FOREIGN KEY (`epic`) REFERENCES `IG`.`instrument` (`epic`);

ALTER TABLE `tmp`.`to_sell` ADD FOREIGN KEY (`dealId`) REFERENCES `pub`.`position_metrics` (`dealId`);

