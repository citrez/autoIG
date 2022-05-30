# TODO:
# - add logging capabilities everytime it is run
# - We want buy outperform overweight.
# - Look at recently covered stock and see if theyve gone up
# - make exception handling
# - small leaway for stock to go down. If it goes down by even a small amout. Sell.
# - Get bazinga API working
# - only add bezinga lines to the table, if they are not already there
# - make csv of initiations used almost as a log
# - make csv of initiations, ratings and number of analysts
# - Make get_yf_estimate_table function more robust
# - only get bezinga rows if they are not already in the db. Or only add them if they arent in.

import logging
import sqlite3
from datetime import datetime
# import numpy as np
import pandas as pd
# import requests
# import yfinance as yf
import time  # for sleeping selenium
from bs4 import BeautifulSoup
from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class Bezinga:
    """This class scrapes information from the bezinga website"""
    def __init__(self, minimise=True):
        self.minimise = minimise

        self.url = "https://www.benzinga.com/analyst-ratings"
        
        self.css_selector = ".ag-header-cell-text , .ag-cell-value"
        # self.css_selector =  "bz-ag-table"
        
    def _extract_bezinga_table(self, html_doc):
        """
        Feed in the raw html bezinga table html and extract the relevant html with table data
        """
        soup = BeautifulSoup(html_doc, "html.parser")
        raw_table = soup.select(self.css_selector)
        return raw_table

    def _parse_bezinga_table(self, table, number_of_cols=10):
        """
        Parse the raw table html elements into a pandas dateframe.
        """
        raw_data = [row.text for row in table]
        table_headers = raw_data[0:number_of_cols]
        table_body = raw_data[10:]
        df = pd.DataFrame(
            {table_headers[i]: table_body[i::10] for i in range(number_of_cols)}
        )
        df.columns = self._clean_bezinga_cols(df.columns)

        # df = df.rename(columns={'current_price':'current_price ($)'})

        return df

    @staticmethod
    def _change_dtypes(df):
        df = (
            df.
                assign(
                current_price = lambda df: df.current_price.str.replace('$','',regex = False),
                date = lambda df: pd.to_datetime(df.date)
            )
        )

    @staticmethod
    def _rename_bezinga_cols(df):
        return df.rename(columns={'current_price': 'current_price ($)'})


    @staticmethod
    def _clean_bezinga_cols(cols):
        """Clean the bezinga column names"""

        updated_cols = (
            cols.str.replace(" ", "_", regex=True)
            .str.replace("\\.|\\(|\\)|\\/", "", regex=True)  # remove whitespace
            .str.replace("/", "or", regex=True)
        )
        updated_cols = updated_cols.str.lower()
        return updated_cols

    def block_popup(self):
        pass

    def get_bezinga_table(self):
        """
        This retrieves the Bezinga Table which gives latest table initialisation coverage.
        Also writes to a bezinga_log database
        """
        s = Service(ChromeDriverManager().install())

        with webdriver.Chrome(service=s) as driver:
            driver.minimize_window() if not self.minimise else driver.maximize_window() 

            try: # get the bezinga website
                driver.get(self.url)
            except Exception as e:
                logging.error(f"Error retrieving URL: {e}")

            try: #click on the initiations tab
                initiations_tab = driver.find_elements_by_class_name("top-tabs__tab")[3]
                initiations_tab.click()
            except Exception as e:
                logging.error(f"Initiations tab not working: {e}")
            try: # get rid of pop-up
                # TODO: find a way to only block up pop-up if it appears
                time.sleep(15)
                pop_up = driver.find_element_by_xpath("""//*[@id="om-vwxzgy4xhurijhsolzhf-optin"]/div/button""")
                pop_up.click()
            except Exception as e:
                logging.error(e)
            # try: # retrieve the table from the page
            #     bz_table = driver.find_element_by_class_name("bz-ag-table")
            #     ag_row_even = bz_table.find_elements_by_class_name("ag-row-even")
            #     ag_row_odd = bz_table.find_elements_by_class_name("ag-row-odd")
            #     ag_rows = ag_row_even + ag_row_odd

            #     for row  in ag_rows:
            #         time.sleep(2)
            #         try: # get
            #             row_index = row.get_attribute('row-index')
            #             cells = row.find_elements_by_class_name( "ag-cell" )
            #             print( row_index, [i.text for i in cells] )
                    # except Exception as e:
                        # print(e)
            # except Exception as e:
                # print(e)
            # time.sleep(8)
            # html_doc = driver.page_source

        # raw_table = self._extract_bezinga_table(html_doc)
        # table_1 = [i for i in enumerate(raw_table.split("\n")) ]
        # table = self._parse_bezinga_table(raw_table)
        # logging.info(f"Successfully retrieved bezinga initilisations")
        # table = table.assign(created_at = datetime.now() )

        # try:
        #     con = sqlite3.connect('bezinga_log.db')
        #     table.to_sql(name='bezinga', con=con, if_exists='append')
        #     logging.info(f"Successfully written {table.shape[0]} records to bezinga_log.db")
        # except Exception as e:
        #     logging.info(f"Failed to write to bezinga_log.db")
        # return table

class Coverage:

    def __init__(self, ticker_search, minimise=True,url = None):
        self.ticker_search = ticker_search
        self.minimise = minimise
        self.yf_estimate_table = None
        if url is None:
            self.url  = f"https://uk.finance.yahoo.com/quote/{self.ticker_search}/analysis?p={self.ticker_search}"

    def _extract_df_estimate_table(self, html_doc):
        soup = BeautifulSoup(html_doc, "html.parser")
        table = soup.find("table")
        return table

    def _parse_yf_estimate_table(self, table):
        estimate_df = pd.read_html(str(table))
        estimate_df = estimate_df[0]
        return estimate_df

    def get_yf_estimate_table(self):
        """
        Get yahoo-finance (https://uk.finance.yahoo.com/quote/CCSI/analysis?p=CCSI) table
        in order to extract number of analysts CY
        """
        s = Service(ChromeDriverManager().install())
        with webdriver.Chrome(service=s) as driver:
            driver.minimize_window() if self.minimise else driver.maximize_window()
            driver.get(self.url)
            try:  # accept cookies
                accept_cookies = driver.find_element(By.NAME, "agree")
                accept_cookies.click()
            except Exception as e:
                print("Could not except cookies", e)
                logging.error("Could not except cookies")
            html_doc = driver.page_source

        table = self._extract_df_estimate_table(html_doc=html_doc)
        table = self._parse_yf_estimate_table(table)
        self.yf_estimate_table = table
        return table

    def no_of_analysts(self):
        return self.yf_estimate_table.query(" `Earnings estimate` == 'No. of analysts' ")


if __name__ == "__main__":
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.ERROR)

    bezinga = Bezinga()
    new_initiates_df = bezinga.get_bezinga_table()
    # new_initiates_df[new_initiates_df.rating_change == 'Initiates' ]

    # new_initiates_df = new_initiates_df.assign(date_added=datetime.now())
    # add logic that it also has to be rating chage = initiates?

    # log_message = 'number of initiations: ' + str( new_initiates_df.shape[0] )
    # logging.warning(log_message)

    # try:
    #     old_initiates_df = pd.read_csv("data/initiates.csv")
    #     initiates_df = pd.concat([old_initiates_df, new_initiates_df], axis=0)
    # except:
    #     initiates_df = new_initiates_df
    # else:
    #     pass
    # finally:
    #     initiates_df.to_csv("data/initiates.csv")

    # print("\n\n\n")
    # for ticker in list(initiates_df.ticker):
    #     no_of_current_analysts = get_yf_estimate_table(ticker).iloc[0,1]
    #     if  no_of_current_analysts > 0:
    #         print(f"There are {no_of_current_analysts} analysts covering {ticker}")
    #     else:
    #         print("**INVEST**")
    #         print(f"There are {no_of_current_analysts} analysts covering {ticker}")
