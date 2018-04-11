import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import *
from copy import copy
from pprint import pprint as pp
import re
from pathlib import PurePath, Path 
# ================================================================== #
# datetime management

d = pd.datetime.today()
# ---------- Days ----------
l10 = d - 10 * BDay()
l21 = d - 21 * BDay()
l63 = d - 63 * BDay()
l252 = d - 252 * BDay()
# ---------- Years ----------
l252_x2 = d - 252 * 2 * BDay()
l252_x3 = d - 252 * 3 * BDay()
l252_x5 = d - 252 * 5 * BDay()
l252_x7 = d - 252 * 7 * BDay()
l252_x10 = d - 252 * 10 * BDay()
l252_x20 = d - 252 * 20 * BDay()
l252_x25 = d - 252 * 25 * BDay()
# ================================================================== #

class utilities():

    def __init__(self):
        self.today = d
    # -------------------------------------------------------- #
    def _create_date_format(self, start_date=None, end_date=None, timeseries=None):
        """Function that formats timeseries index begin and end for blackarbs format

        Parameters:
        ===========
        start_date, end_date = pd.Timestamp/pd.Datetime
        timeseries = pd.DataFrame, pd.Series

        Returns:
        ========
        date_text_format = formatted string containing start/end dates"""

        fmt = '%A, %m.%d.%Y'
        if timeseries:
            start_date = timeseries.index[0].strftime(fmt)
            end_date = timeseries.index[-1].strftime(fmt)
            date_text_format = '[{} - {}]'.format(start_date, end_date)
            return date_text_format
        elif start_date:
            start = start_date.strftime(fmt)
            end = end_date.strftime(fmt)
            date_text_format = '[{} - {}]'.format(start, end)
            return date_text_format
    # -------------------------------------------------------- #
    def _remove_pct_symbol(self, val):
        '''function to remove `%` symbol from scraped percentages'''
        if '%' in val:
            val = val.replace('%', '')
        val = float(val)
        return val
    # -------------------------------------------------------- #
    def _remove_commas(self, val):
        '''function to remove `,` symbol from values'''
        if ',' in val:
            val = val.replace(',', '')
        val = float(val)
        return val
    # -------------------------------------------------------- #
    def _remove_non_alphanumeric(self, val):
        """Function to remove All non-alphanumeric symbols from value"""
        val = re.sub(r"[^\d\.]","", val)
        return val
    # -------------------------------------------------------- #
    # symbol management / clean holdings symbols for price download

    def _clean_symbols_strings(self, symbols):
        if isinstance(symbols, pd.DataFrame):
            sf = symbols#.replace('\.', '-', regex=True)
            df = pd.DataFrame()
            for i in sf:
                df[i] = sf[i].str.upper()
            df = df[df.notnull()]
            return df
        elif isinstance(symbols, pd.Series):
            new_series = symbols#.replace('\.','-', regex=True)
            new_series = new_series.map(str.upper)
            return new_series
    # -------------------------------------------------------- #
    def _clean_millions_billions(self, abbrev_value_string):
        '''convert abbreviated numbers from string to float'''
        val = abbrev_value_string
        if '$' in val:
            val = val.replace('$', '')
        try:
            if 'B' in val:
                val = float(val.rstrip('B')) * 1e9
            elif 'M' in val:
                val = float(val.rstrip('M')) * 1e6
            elif 'K' in val:
                val = float(val.rstrip('K')) * 1e3
        except Exception as e:
            print(e)

        return val
    # -------------------------------------------------------- #
    def _etf_count(self, symbol_dict):
        sd = symbol_dict
        if isinstance(sd, dict):
            etf_val_count = 0
            for i in sd.values():
                etf_val_count += len(i)
            return etf_val_count
        else:
            etf_val_count = sd.count().sum()
            return etf_val_count

    # -------------------------------------------------------- #
    def _N_days_ytd(self):
        '''Calculate number of days since the beginning of the year'''
        year_begin = (self.today - BYearBegin())
        lbd = (self.today - BDay())
        idx_diff = pd.bdate_range(year_begin, lbd) # date range between periods in biz days
        N_days_diff = len(idx_diff)
        return N_days_diff
    # -------------------------------------------------------- #
    def _create_output_path(self, PATH):
        '''Create timestamped output directory for code output'''
        dtStr = str(pd.datetime.today().strftime('%m-%d-%y'))
        detailed_output_path = PurePath(PATH/dtStr)
        print(f'checking {detailed_output_path}...')
        try:
            if not Path(detailed_output_path).is_dir():
                Path(detailed_output_path).mkdir(parents=True,exist_ok=True)
                print('creating directory...')
                print('directory created [complete]')
        except Exception as e:
            print(e)
        return detailed_output_path
    # -------------------------------------------------------- #
    def _create_path(self, PATH):
        '''Create timestamped output directory for code output'''
        detailed_output_path = PATH
        try:
            if not Path(detailed_output_path).is_dir():
                Path(detailed_output_path).mkdir(parents=True,exist_ok=True)
                #pp('creating directory...')
                #pp('directory created [complete]')
            else:
                pass
        except Exception as e:
            print( e )
        return detailed_output_path
    # -------------------------------------------------------- #
    def _code_timer(self, start, end):
        secs      = np.round( ( end  - start ), 4 )
        time_secs = "{timeSecs} seconds to run".format(timeSecs = secs)
        mins      = np.round( ( end -  start )  / 60, 4 )
        time_mins = "| {timeMins} minutes to run".format(timeMins = mins)
        hours     = np.round( (  end  -  start )  / 60 / 60, 4 )
        time_hrs  = "| {timeHrs} hours to run".format(timeHrs = hours)
        print( time_secs, time_mins, time_hrs )
        return time_secs, time_mins, time_hrs
    # -------------------------------------------------------- #
    def _create_holdings(self, holdings_dir):
        holdings_loc = PurePath(holdings_dir / '_SPDR_holdings')
        ETF = ['xli', 'xlk', 'xlv', 'xlp', 'xlu', 'xlb', 'xlf', 'xly', 'xle']
        holdings = pd.DataFrame()
        for sector in ETF:
            symbols = pd.read_excel(PurePath(holdings_loc/'holdings-{}.xls'.format(sector)), skiprows=1)
            holdings[sector.upper()] = symbols['Symbol']
        holdings = self._clean_symbols_strings(holdings)
        return holdings
# =========================================================================== #