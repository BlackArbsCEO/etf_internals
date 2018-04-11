import time
import os
import sys
from pathlib import PurePath
from pathlib import Path  
pdir = list(PurePath(Path.cwd()).parents)[1] # .../etf_internals/.
price_dir = PurePath(pdir/'data'/'external'/'prices')
print(price_dir)
# ------------------- ^ --------- ^ ------------------- #
import _blk_utilities_
utils = _blk_utilities_.utilities()
# =========================================================================== #
import pandas as pd
from pandas.tseries.offsets import *
import pandas_datareader.data as web
import numpy as np
from decimal import Decimal
from collections import OrderedDict as od

dplaces = Decimal('0.0001')

sz = 11
mlt = 1.66
size=(mlt * sz, sz)
import seaborn as sns
sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
sns.set_palette(sns.color_palette(flatui,7))

# =========================================================================== #
class formatSymbols:
    def __init__(self, holdings_path):
        self.holdings_dir = holdings_path
        
    def _remove_cash_holdings(self, etf):
        return pd.Series([sym for sym in etf if 'CASH' not in sym])
    # ============================================================================================ #
    # EUROPE
    def _format_FTSE_symbols(self):
        ftse100 = pd.read_csv(PurePath(self.holdings_dir / '_Foreign' / 'ISF_holdings.csv'), skiprows=2)
        ftse100 = ftse100[ ftse100['Sector'] != r'Cash and/or Derivatives' ].dropna()
        ftse100 = (ftse100['Issuer Ticker'].replace(['/', '' , '.'], '', regex=True)).sort_values().reset_index(drop=True)    
        idx_loc = None
        for i in range(len(ftse100)):
            if ftse100.ix[i] == 'BTA':
                idx_loc = i
        ftse100[idx_loc] = 'BT-A'
        ftse100 =   pd.Series( [sec + '.L' for sec in ftse100 ] ).map(str.strip)
        return ftse100

    def _format_DAX_symbols(self):
        dax = pd.read_csv(PurePath(self.holdings_dir / '_Foreign' / 'EXS1_holdings.csv'), skiprows=2)
        dax = dax[ dax['Sector'] != r'Cash and/or Derivatives' ].dropna()    
        dax = pd.Series( [sec + '.DE' for sec in dax['Issuer Ticker'] ] ).map(str.strip)
        return dax
    # ============================================================================================ #
    # CANADA

    def _format_TSX_symbols(self):
        tsx = pd.read_csv(PurePath(self.holdings_dir / '_Foreign' / 'XIU_holdings.csv'), skiprows=10)
        tsx = tsx[ tsx['Sector'] != r'Cash and/or Derivatives' ].dropna()
        tsx = utils._clean_symbols_strings(tsx['Ticker'])
        tsx = pd.Series( [sec + '.TO' for sec in tsx] ).map(str.strip)
        return tsx
    # ============================================================================================ #
    # AUSTRALIA

    def _format_ASX_symbols(self):
        asx = pd.read_excel(PurePath(self.holdings_dir / '_Foreign' / 'fund_allholdings_STW.xls'), skiprows=4)
        asx = asx[asx['Sector Classification'] != 'Unassigned']  
        asx = asx['Ticker'].map(str.strip)
        asx = asx.replace('-AU', '.AX', regex=True)
        return asx
    # ============================================================================================ #
    # US MAJOR AVERAGES

    def _format_SPY_symbols(self):
        spy = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-spy.xls'))
        spy = spy[spy['Shares Held']>0]
        spy = spy['Identifier'].map(str.strip)
        spy = utils._clean_symbols_strings(spy)
        spy = self._remove_cash_holdings(spy)
        return spy

    def _format_DIA_symbols(self):
        dia = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-dia.xls'), skiprows=3)
        dia = dia[(dia['Shares Held']>0) & (dia['Identifier'] != 'CASH_USD')]    
        dia = dia['Identifier'].map(str.strip)
        return dia

    def _format_QQQ_symbols(self):
        qqq = pd.read_csv(PurePath(self.holdings_dir / '_QQQ_holdings' / 'holdings-QQQ.csv'))
        qqq = qqq['HoldingsTicker'].map(str.strip)
        qqq = self._remove_cash_holdings(qqq)
        return qqq
    # ============================================================================================ #
    # SECTOR SPDRS

    def _format_XLB_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xlb.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms) 
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLE_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xle.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms)
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLF_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xlf.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms)
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLI_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xli.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLK_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xlk.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms)
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLP_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xlp.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms)
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLU_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xlu.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms)
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLV_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xlv.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms)
        syms = self._remove_cash_holdings(syms)
        return syms

    def _format_XLY_symbols(self):
        syms = pd.read_excel(PurePath(self.holdings_dir / '_SPDR_holdings' / 'holdings-xly.xls'), skiprows=1)
        syms = syms['Symbol'].map(str.strip)
        syms = utils._clean_symbols_strings(syms)
        syms = self._remove_cash_holdings(syms)
        return syms
    
    '''
    def _format_STOXX_symbols(self):
        stoxx50 = pd.read_csv(PurePath(self.holdings_dir / '_Foreign' / 'EUE_holdings.csv'), skiprows=2)
        stoxx50= stoxx50[ stoxx50['Sector'] != r'Cash and/or Derivatives' ].dropna()    
        stoxx50 = pd.Series( [sec + '.PA' for sec in stoxx50['Issuer Ticker'] ] ).map(str.strip)
        return stoxx50

    '''    
    def format_all_symbols(self):
        # ================================================================== #
        # EUROPE
        FTSE = self._format_FTSE_symbols()
        DAX = self._format_DAX_symbols()
        # ================================================================== #
        # CANADA
        TSX = self._format_TSX_symbols()
        # ================================================================== #
        # AUSTRALIA
        ASX = self._format_ASX_symbols()
        # ================================================================== #
        # US MAJOR AVERAGES

        SPY = self._format_SPY_symbols()
        #DIA = self._format_DIA_symbols()
        QQQ = self._format_QQQ_symbols()
        # ================================================================== #
        # SECTOR SPDRS

        XLB = self._format_XLB_symbols()
        XLE = self._format_XLE_symbols()
        XLF = self._format_XLF_symbols()
        XLI = self._format_XLI_symbols()
        XLK = self._format_XLK_symbols()
        XLP = self._format_XLP_symbols()
        XLU = self._format_XLU_symbols()
        XLV = self._format_XLV_symbols()
        XLY = self._format_XLY_symbols()

        all_etf_symbols = od([
            ('SPY',SPY), #('DIA',DIA), 
            ('QQQ',QQQ), 
            ('XLB',XLB), ('XLE',XLE), ('XLF',XLF), 
            ('XLI',XLI), ('XLK',XLK), ('XLP',XLP),
            ('XLU',XLU), ('XLV',XLV), ('XLY',XLY),
            ('FTSE',FTSE), ('DAX',DAX), ('TSX',TSX),
            ('ASX',ASX)])
        
        return all_etf_symbols
# =========================================================================== #

class etfInternals():
    def __init__(self, price_dir):
        self.today = pd.datetime.today().date().strftime('%m-%d-%y')
        self.price_dir = utils._create_output_path(price_dir)
        #self.PATH = utils._create_output_path(pdir)
        #pos_thresh = [0., .01, .02, .03, .04, .05, .06, .07, .1, .15, .2]
        #neg_thresh = [ flt * -1. for flt in pos_thresh ]

    def _get_query_dates(self, n_weeks=104):
        '''
        Function to generate start and end dates for price downloads

        Parameters:
        ===========
        n_weeks = int() in case query is generated mid week without full week requires 2 to anchor to previous full week

        Returns:
        ========
        start_date, end_date = pd.Timestamp()
        '''
        today = pd.datetime.today().date()
        query_start_date = today - Week(weekday=0) * n_weeks # anchored to previous Monday
        query_end_date = query_start_date + Week(weekday=4) * n_weeks # anchored to start date + one week ending Friday
        return query_start_date, query_end_date

    def _get_px(self, symbol, source, start, end):
        """
        Helper Function: Get stock price using pandas-datareader

        Note Yahoo API is deprecated, Google is inconsistent

        """
        return web.DataReader(symbol, source, start, end)

    """
    def _create_hdf_store(self, ETF_symbol):
        '''
        Function to create ETF HDF Store dynamically
        '''
        today = pd.datetime.today().date()
        PATH = utils._create_output_path(price_path)
        store = pd.HDFStore(PATH + r'{} ETF Market Internals Price Data {}.h5'.format(ETF_symbol, today.strftime('%m-%d-%y')))
        location = PATH + r'{} ETF Market Internals Price Data {}.h5'.format(ETF_symbol, today.strftime('%m-%d-%y'))
        return store, location
    """

    def _calc_log_returns(self, ts):
        """Function that takes a dataframe\series and calculates the log returns (assumes no 0 prices)"""
        lrets = np.log(ts / ts.shift(1)) #.dropna(axis=0)
        return lrets

    def _calc_cuml_returns(self, log_ts):
        """Function that takes log returns dataframe\series and calculates cuml returns"""
        crets = log_ts.cumsum(axis=1)
        return crets

    def _calc_avg_log_returns_vol(self, log_returns, n_periods):
        """Function to calculate average ETF cumulative returns and volatility"""
        avg_cuml_ret = log_returns.ix[-n_periods:].cumsum(axis=0).mean(axis=1).ix[-1]
        avg_std = log_returns.ix[-n_periods:].std(axis=1).ix[-1]
        return avg_cuml_ret, avg_std

    def _convert2decimal(self, value):
        "Convert `value` to Decimal and percent"
        val = Decimal(value).quantize(dplaces)
        val = "{:.2%}".format(val)
        return val

    def make_outfp(self, symbol, date=None):
        """make outfp for saving price data to parquet"""
        if date is None: date = self.today 
        return PurePath(self.price_dir / f'{symbol}_{date}.parquet')

    def _get_symbol_px_data(self, ETF_symbol, symbols, source='iex'):
        """Function to create hdf5 px storage for each ETF's components

        Parameters:
        ===========
        ETF_symbol = str() the ETF symbol
        symbols = pd.Series() of ETF symbol components

        Returns:
        ========
        location_h5 = str() filepath location of datastore
        """
        #datastore, location_h5 = self._create_hdf_store(ETF_symbol)
        start, end = self._get_query_dates()

        sym_errors = []
        symbol_count = len(symbols)
        N = len(symbols)
        for sym in symbols:
          N -= 1
          if 'CASH' not in sym:
            print(f"downloading symbol: {sym} | {(N/symbol_count):.2%}")
            try:
                px_dataframe = self._get_px(sym, source, start=start, end=end)
                outfp = self.make_outfp(sym)
                px_dataframe.to_parquet(outfp)
            except Exception as e:
                print(e, sym)
                sym_errors.append(sym)
        print("{} sym_errors:\n{}".format(ETF_symbol, sym_errors))
        pd.Series(sym_errors).to_csv(PurePath(self.price_dir/'{} missing symbols {}.txt'
                               .format(ETF_symbol, self.today)).as_posix())
        return 

    def _extract_col_price(self, symbol, date, col='close'):
        infp = self.make_outfp(symbol, date)
        try: 
            df = pd.read_parquet(infp).loc[:,col].rename(symbol)
            return df
        except Exception as e:
            print(f'{symbol} error: {e}')
        return 

    def extract_close_prices(self, symbols, col='close', date_str=None):
        """extract close prices from downloaded price data"""
        if date_str is None: date_str = self.today 
        lst_dfs = []
        for sym in symbols:
            try:
                df = self._extract_col_price(sym,date=date_str,col=col)
                lst_dfs.append(df)
            except Exception as e:
                print(f'{sym} error: {e}')
                continue
        try:
            return (pd.concat(lst_dfs,axis=1)
                    .assign(date=lambda df:pd.to_datetime(df.index))
                    .set_index('date')
                    .drop_duplicates())
        except Exception as e:
            print(f'error: {e}')
            return 

    def _typecheck(self, obj): return hasattr(obj, '__iter__') and not isinstance(obj, str)

    def _extract_adjclose_px(self, location_h5, symbols):
        """Function to collect all the adjusted close price series from each Dataframe, returns PX df"""
        adjclose = pd.DataFrame()

        if self._typecheck(symbols) == True:
            with pd.HDFStore(location_h5) as dat:
                for sym in symbols:
                    adjclose[sym] = dat[sym]['Adj Close']
            return adjclose
        else:
            with pd.HDFStore(location_h5) as dat:
                adjclose[symbols] = dat[symbols]['Adj Close']
            return adjclose

    def _52_week_highs(self, ts, symbols):
        """function to calculate Number of stocks at 52 week highs

        Parameters:
        ===========
        ts = pd.DataFrame() all adj. close pxx
        symbols = pd.Series() of ETF symbols

        Returns:
        ========
        high_syms_list = list containing the symbols (str) of stocks that fit criteria
        N_52_highs = int() number of symbols at 52 week highs
        highs_df = pd.DataFrame() simply df containing all ETF symbols' highs and current prices
        """
        last_52 = ts.index[-1] - Week(weekday=0) * 52
        ts = ts.ix[last_52:]
        today_px = ts.ix[-1]
        max_px = ts.max()
        high_syms_list = []
        highs_df = pd.DataFrame(columns=['52 week high', 'last'], index=symbols.values)
        for sym in symbols:
            highs_df.loc[sym] = max_px[sym], today_px[sym]
            if today_px[sym] >= max_px[sym]:
                high_syms_list.append(sym)
        N_52_highs = len(high_syms_list)
        return high_syms_list, N_52_highs, highs_df

    def _52_week_lows(self, ts, symbols):
        """function to calculate Number of stocks at 52 week lows

        Parameters:
        ===========
        ts = pd.DataFrame() all adj. close pxx
        symbols = pd.Series() of ETF symbols

        Returns:
        ========
        low_syms_list = list containing the symbols (str) of stocks that fit criteria
        N_52_lows = int() number of symbols at 52 week lows
        lows_df = pd.DataFrame() simply df containing all ETF symbols' lows and current prices
        """
        last_52 = ts.index[-1] - Week(weekday=0) * 52
        ts = ts.ix[last_52:]
        today_px = ts.ix[-1]
        low_px = ts.min()
        low_syms_list = []
        lows_df = pd.DataFrame(columns=['52 week low', 'last'], index=symbols.values)
        for sym in symbols:
            lows_df.loc[sym] = low_px[sym], today_px[sym]
            if today_px[sym] <= low_px[sym]:
                low_syms_list.append(sym)
        N_52_lows = len(low_syms_list)
        return low_syms_list, N_52_lows, lows_df

    def _calc_high_low_metrics(self, highs_df, lows_df, N_stocks):
        """Function to calculate high/low data points

        Parameters:
        ==========
        highs_df = dataframe containing all ETF symbol 52 week highs, Current prices
        lows_df = dataframe containing all ETF symbol 52 week lows, current prices

        Returns:
        =======
        average/median_below_highs = float()
        average/median_above_lows = float()

        N_stocks_off_10 = float()
        N_stocks_off_20 = float()

        pct_stocks_off_10 = float()
        pct_stocks_off_20 = float()

        N_stocks_up_10 = float()
        N_stocks_up_20 = float()

        pct_stocks_up_10 = float()
        pct_stocks_up_20 = float()
        """
        highs_df['pct_below_highs'] = (highs_df['last'] / highs_df['52 week high'] - 1)
        lows_df['pct_above_lows'] = (lows_df['last'] / lows_df['52 week low'] - 1)

        avg_pct_below_highs = highs_df['pct_below_highs'].mean()
        median_pct_below_highs = highs_df['pct_below_highs'].median()

        avg_pct_above_lows = lows_df['pct_above_lows'].mean()
        median_pct_above_lows = lows_df['pct_above_lows'].median()

        # stocks off 10% or more from 52 week highs
        stocks_off_10 = highs_df[['pct_below_highs']][highs_df['pct_below_highs'] < -0.10]
        N_stocks_off_10 = len(stocks_off_10)
        pct_stocks_off_10 = N_stocks_off_10 / N_stocks

        # stocks off 20% or more from 52 week highs
        stocks_off_20 = highs_df[['pct_below_highs']][highs_df['pct_below_highs'] < -0.20]
        N_stocks_off_20 = len(stocks_off_20)
        pct_stocks_off_20 = N_stocks_off_20 / N_stocks

        # stocks 10% or more from 52 week lows
        stocks_up_10 = lows_df[['pct_above_lows']][lows_df['pct_above_lows'] > 0.10]
        N_stocks_up_10 = len(stocks_up_10)
        pct_stocks_up_10 = N_stocks_up_10 / N_stocks

        # stock up 20% or more from 52 week lows
        stocks_up_20 = lows_df[['pct_above_lows']][lows_df['pct_above_lows'] > 0.20]
        N_stocks_up_20 = len(stocks_up_20)
        pct_stocks_up_20 = N_stocks_up_20 / N_stocks

        return avg_pct_below_highs, median_pct_below_highs, avg_pct_above_lows, median_pct_above_lows, \
        N_stocks_off_10, N_stocks_off_20, pct_stocks_off_10, pct_stocks_off_20, \
        N_stocks_up_10, N_stocks_up_20, pct_stocks_up_10, pct_stocks_up_20

    def _up_down_last_N_periods(self, n_periods, log_returns):
        """
        function to calculate the number of stocks up/down over user defined period of time

        Parameters:
        ===========
        n_periods = int() in days from end date of ts
        log_returns = pd.DataFrame(); log returns timeseries

        Returns:
        ========
        up = pd.Series()
        N_up = int()
        down = pd.Series()
        N_down = int()
        nup_pct = float()
        ndown_pct = float()
        """
        N_stocks = len(log_returns.columns)
        lrets_trunc = log_returns.ix[-n_periods:]
        crets = lrets_trunc.cumsum(axis=0).ix[-1]
        up = crets[crets > 0.]
        down = crets[crets < 0.]

        N_up = len(up)
        N_down = len(down)

        nup_pct = N_up / N_stocks
        ndown_pct = N_down / N_stocks
        return up, N_up, nup_pct, down, N_down, ndown_pct

    def _calc_EMA(self, ts, span):
        '''
        Parameters:
        ==========
        ts = pd.Series(), individual equity timeseries
        span = int(); moving average window

        Returns: pd.Series()
        '''
        return pd.ewma(ts, span=span, min_periods=span)

    def _get_EMA(self, ts, list_of_spans, symbols):
        """Function to create dataframes containing current prices and current EMA

        Parameters:
        ===========
        ts = timeseries (DataFrame) of stock Prices
        list_of_spans = periods to calculate ema (by default this is of length 3)
        symbols = pd.Series() of ETF component symbols

        Returns:
        ========
        ema_comp = pd.DataFrame()
        """
        cols = ['EMA_{}'.format(list_of_spans[0]),
                'EMA_{}'.format(list_of_spans[1]),
                'EMA_{}'.format(list_of_spans[2]),
                'Current Price'
               ]
        last_px = ts.ix[-1]
        ema_comp = pd.DataFrame(columns=cols, index=ts.columns)

        ema_zero = ( self._calc_EMA(ts, list_of_spans[0]) ).ix[-1]
        ema_one = ( self._calc_EMA(ts, list_of_spans[1]) ).ix[-1]
        ema_two = ( self._calc_EMA(ts, list_of_spans[2]) ).ix[-1]

        for sym in symbols:
            ema_comp.loc[sym] = ema_zero[sym], ema_one[sym], ema_two[sym], last_px[sym]
        return ema_comp

    def _calc_ema_stats(self, spans, emas, N_stocks):
        """Function to calc above ema statistics

        Parameters:
        ===========
        emas = pd.DataFrame() containing all the symbols' prices, emas

        Returns:
        ========
        n_abv_252 = int()
        n_abv_63 = int()
        n_abv_21 = int()

        pct_abv_252 = float()
        pct_abv_63 = float()
        pct_abv_21 = float()
        """
        above_252 = emas[['EMA_{}'.format(spans[0]), 'Current Price']][emas['Current Price'] > emas['EMA_{}'.format(spans[0])]]
        above_63 = emas[['EMA_{}'.format(spans[1]), 'Current Price']][emas['Current Price'] > emas['EMA_{}'.format(spans[1])]]
        above_21 = emas[['EMA_{}'.format(spans[2]), 'Current Price']][emas['Current Price'] > emas['EMA_{}'.format(spans[2])]]
        n_abv_252 = len(above_252)
        n_abv_63 = len(above_63)
        n_abv_21 = len(above_21)
        pct_abv_252 = n_abv_252 / N_stocks
        pct_abv_63 = n_abv_63 / N_stocks
        pct_abv_21 = n_abv_21 / N_stocks

        return n_abv_252, n_abv_63, n_abv_21, pct_abv_252, pct_abv_63, pct_abv_21


    def _convert_thresh_index(self, value):
        "Helper Function: Convert `value` to Decimal and percent"
        if value > 0:
            val = Decimal(value).quantize(dplaces)
            val = "> {:.2%}".format(val)
            return val
        elif value < 0:
            val = Decimal(value).quantize(dplaces)
            val = "< {:.2%}".format(val)
            return val
        else:
            val = Decimal(value).quantize(dplaces)
            val = "{:.2%}".format(val)
            return val

    def _calc_return_thresholds(self, list_of_thresholds, log_returns, N_stocks, pos=None):
        """Function to calculate the quantity of stocks meeting or exceeding a return threshold

        Parameters:
        ===========
        list_of_thresholds: a list of each return level
        log_returns: dataframe of equity returns
        pos = boolean, if True calculates positive thresholds else negative

        Returns:
        ========
        N_thresh = pd.Series() with stock counts for each return threshold
        """
        if pos:
            N_thresh = pd.DataFrame(columns=['Count', 'Percent of Total Stocks'], index=list_of_thresholds)
            for thresh in list_of_thresholds:
                l = []
                for i in log_returns.ix[-1]:
                    if i > thresh:
                        l.append(1)
                N_thresh.loc[thresh]['Count'] = sum(l)
            N_thresh['Percent of Total Stocks'] = (N_thresh['Count'] / N_stocks) #.apply(_convert2decimal)
            N_thresh.index = N_thresh.index.map(self._convert_thresh_index)
            return N_thresh
        else:
            N_thresh = pd.DataFrame(columns=['Count', 'Percent of Total Stocks'], index=list_of_thresholds)
            for thresh in list_of_thresholds:
                l = []
                for i in log_returns.ix[-1]:
                    if i <= thresh:
                        l.append(1)
                N_thresh.loc[thresh]['Count'] = sum(l)
            N_thresh['Percent of Total Stocks'] = (N_thresh['Count'] / N_stocks) #.apply(_convert2decimal)
            N_thresh.index = N_thresh.index.map(self._convert_thresh_index)
            return N_thresh


    def _format_odf_percent(self, output_series):
        """Helper function: convert data series into proper number format"""
        for stat in output_series.index:
            if 'Percent' in stat:
                output_series[stat] = self._convert2decimal(output_series.loc[stat])
        return output_series

    def _create_output_dataframe(self, n_periods,
                                start_timestamp, end_timestamp, N_stocks, \
                                N_52_highs, N_52_lows, \
                                avg_pct_below_highs, median_pct_below_highs, avg_pct_above_lows, median_pct_above_lows, \
                                N_stocks_off_10, N_stocks_off_20, pct_stocks_off_10, pct_stocks_off_20, \
                                N_stocks_up_10, N_stocks_up_20, pct_stocks_up_10, pct_stocks_up_20,
                                N_up, nup_pct, N_down, ndown_pct, \
                                n_abv_252, n_abv_63, n_abv_21, pct_abv_252, pct_abv_63, pct_abv_21, \
                                avg_cuml_ret, avg_std):
        """Construct output dataframe for excel"""

        odf = pd.Series()
        odf['Total Number of ETF Components'] = N_stocks
        odf['Quantity of Stocks UP over period'] = N_up
        odf['Quantity of Stocks DOWN over period'] = N_down
        odf['Percent Stocks UP over period'] = nup_pct
        odf['Percent Stocks DOWN over period'] = ndown_pct
        odf['Stocks at 52 Week Highs'] = N_52_highs
        odf['Stocks at 52 Week Lows'] = N_52_lows
        odf['Percent Stocks at 52 Week Highs'] = N_52_highs / N_stocks
        odf['Percent Stocks at 52 Week Lows'] = N_52_lows / N_stocks
        odf['Quantity of Stocks above 252 Day EMA'] = n_abv_252
        odf['Quantity of Stocks above 63 Day EMA'] = n_abv_63
        odf['Quantity of Stocks above 21 Day EMA'] = n_abv_21
        odf['Percent Stocks above 252 Day EMA'] = pct_abv_252
        odf['Percent Stocks above 63 Day EMA'] = pct_abv_63
        odf['Percent Stocks above 21 Day EMA'] = pct_abv_21
        odf['Average Percent DOWN from 52 Week Highs'] = avg_pct_below_highs
        odf['Median Percent DOWN from 52 Week Highs'] = median_pct_below_highs
        odf['Average Percent UP from 52 Week Lows'] = avg_pct_above_lows
        odf['Median Percent UP from 52 Week Lows'] = median_pct_above_lows
        odf['Quantity of Stocks DOWN 10%+ from 52 Week Highs'] = N_stocks_off_10
        odf['Quantity of Stocks DOWN 20%+ from 52 Week Highs'] = N_stocks_off_20
        odf['Quantity of Stocks UP 10%+ from 52 Week Lows'] = N_stocks_up_10
        odf['Quantity of Stocks UP 20%+ from 52 Week Lows'] = N_stocks_up_20
        odf['Percent Stocks DOWN 10%+ from 52 Week Highs'] = pct_stocks_off_10
        odf['Percent Stocks DOWN 20%+ from 52 Week Highs'] = pct_stocks_off_20
        odf['Percent Stocks UP 10%+ from 52 Week Lows'] = pct_stocks_up_10
        odf['Percent Stocks UP 20%+ from 52 Week Lows'] = pct_stocks_up_20
        odf['Average Cumulative Return'] = self._convert2decimal(avg_cuml_ret)
        odf['Average Daily Volatility'] = self._convert2decimal(avg_std)
        # ================================================================== #
        new_odf = self._format_odf_percent(odf)
        new_odf = new_odf.to_frame(name='ETF Data')
        new_odf.index.name = utils._create_date_format(start_date=start_timestamp, end_date=end_timestamp)
        return new_odf

    def _create_excelwriter_format(self, savefp, ETF_symbol, new_odf, rev_N_pos, N_neg):
        """Function to create xlsxwriter format and output final report"""
        outfp=PurePath(savefp/f'{ETF_symbol}_ETF-Internals_{self.today}.xlsx')
        writer = pd.ExcelWriter(outfp, engine='xlsxwriter')

        new_odf.to_excel(writer, sheet_name='{} ETF Internals'.format(ETF_symbol))

        rev_N_pos.to_excel(writer, sheet_name='{} ETF Internals'.format(ETF_symbol), startcol=3)
        N_neg.to_excel(writer, sheet_name='{} ETF Internals'.format(ETF_symbol), \
                                            header=False, startrow=len(rev_N_pos)+1, startcol=3)

        # access the worksheet
        workbook = writer.book
        worksheet = writer.sheets['{} ETF Internals'.format(ETF_symbol)]
        worksheet.set_zoom(110)

        index_name_fmt = workbook.add_format({'align':'center', 'bold':True, 'valign':'vcenter'})
        worksheet.write('D1', '1 Week\nCumulative\nReturn Bins', index_name_fmt)
        index_name_fmt.set_text_wrap()
        total_percent_fmt = workbook.add_format({'align':'center', 'num_format': '0.00%','bold':True})
        reg_num_fmt = workbook.add_format({'align':'center', 'bold':True, 'valign':'vcenter'})
        reg_num_fmt_f = workbook.add_format({'valign':'vjustify', 'align':'center', 'bold':False})
        index_fmt = workbook.add_format({'align':'left', 'bold':True,'valign':'vcenter'})

        ## TODO: must change color scale
        worksheet.conditional_format('E2:E23', {'type':'3_color_scale',
                                               'min_color':'#fee6ce'.upper(),
                                               'mid_color':'#fdae6b'.upper(),
                                               'max_color':'#e6550d'.upper()})
        worksheet.conditional_format('F2:F23', {'type':'3_color_scale',
                                               'min_color':'#fee6ce'.upper(),
                                               'mid_color':'#fdae6b'.upper(),
                                               'max_color':'#e6550d'.upper()})

        worksheet.set_column('A:A', 55, index_fmt)
        worksheet.set_column('B:B', 10, reg_num_fmt_f)
        worksheet.set_column('C:C', 5)
        worksheet.set_column('D:D', 15)
        worksheet.set_column('E:E', 10, reg_num_fmt)
        worksheet.set_column('F:F', 25, total_percent_fmt)
        writer.save()
        return
