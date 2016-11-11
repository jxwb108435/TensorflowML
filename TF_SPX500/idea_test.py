import numpy as np
import pandas as pd
import pickle
# from pandas.tools.plotting import scatter_matrix

# US
# ---------------------------------------------------------------------
snp = pd.read_csv('snp.csv', index_col=['Date'])
snp.index = pd.DatetimeIndex(snp.index)
djia = pd.read_csv('djia.csv', index_col=['Date'])
djia.index = pd.DatetimeIndex(djia.index)

# Asia
# ---------------------------------------------------------------------
aord = pd.read_csv('aord.csv', index_col=['Date'])
aord.index = pd.DatetimeIndex(aord.index)
nikkei = pd.read_csv('nikkei.csv', index_col=['Date'])
nikkei.index = pd.DatetimeIndex(nikkei.index)
hangseng = pd.read_csv('hangseng.csv', index_col=['Date'])
hangseng.index = pd.DatetimeIndex(hangseng.index)

# Europe
# ---------------------------------------------------------------------
dax = pd.read_csv('dax.csv', index_col=['Date'])
dax.index = pd.DatetimeIndex(dax.index)
ftse = pd.read_csv('ftse.csv', index_col=['Date'])
ftse.index = pd.DatetimeIndex(ftse.index)


# ori_df
# ______________________________________________________________________________________________________________________
ori_df = pd.DataFrame()

ori_df['snp_o'] = snp['Open']
ori_df['snp_h'] = snp['High']
ori_df['snp_l'] = snp['Low']
ori_df['snp_c'] = snp['Close']

ori_df['djia_o'] = djia['Open']
ori_df['djia_h'] = djia['High']
ori_df['djia_l'] = djia['Low']
ori_df['djia_c'] = djia['Close']

ori_df['aord_o'] = aord['Open']
ori_df['aord_h'] = aord['High']
ori_df['aord_l'] = aord['Low']
ori_df['aord_c'] = aord['Close']

ori_df['nikkei_o'] = nikkei['Open']
ori_df['nikkei_h'] = nikkei['High']
ori_df['nikkei_l'] = nikkei['Low']
ori_df['nikkei_c'] = nikkei['Close']

ori_df['hangseng_o'] = hangseng['Open']
ori_df['hangseng_h'] = hangseng['High']
ori_df['hangseng_l'] = hangseng['Low']
ori_df['hangseng_c'] = hangseng['Close']

ori_df['dax_o'] = dax['Open']
ori_df['dax_h'] = dax['High']
ori_df['dax_l'] = dax['Low']
ori_df['dax_c'] = dax['Close']

ori_df['ftse_o'] = ftse['Open']
ori_df['ftse_h'] = ftse['High']
ori_df['ftse_l'] = ftse['Low']
ori_df['ftse_c'] = ftse['Close']

ori_df = ori_df.fillna(method='bfill')  # fill gaps
ori_df = ori_df.reindex(index=ori_df.index[::-1])

# corr_select = feature_select.corr().loc[::, 'snp_c0o0_0']
# scatter_matrix(feature_select, figsize=(20, 20), diagonal='kde')


# no effect
def test_cc(_df):
    """ 根据前一天的变化预测今天的close变化,
    今天的close-昨天的close趋势 ＝ 昨天的close-前天的close趋势,
    验证结果 2010-01-04:2016-10-14,
    snp:48.39%, djia:47.69%, aord:51.28%, nikkei:52%, hangseng:51.41%, dax:50.14%, ftse:52.6%"""
    t = pd.DataFrame(index=_df.index)
    t = t.reindex(index=t.index[::-1])

    t['c'] = _df['Close']

    t['c1'] = t['c'].shift()
    t['c_c1'] = t['c'] - t['c1']
    t['diff_c'] = 0
    t.ix[t.c_c1 >= 0, 'diff_c'] = 1
    t['diff_c1'] = t['diff_c'].shift()

    r = t.ix[t.diff_c == t.diff_c1]
    return float(len(r)) / float(len(_df))


# no effect
def test_csma40(_df):
    """如果收盘价大于40天均值的收盘价, 在第二天开盘价买入, 收盘价出场.
       如果收盘价小于40天均值的收盘价, 在第二天开盘价做空, 收盘价买入回补.
       概念 市场在上涨后继续上涨概率更大"""
    t = pd.DataFrame(index=_df.index)
    t = t.reindex(index=t.index[::-1])

    t['c'] = _df['Close']
    t['o'] = _df['Open']

    t['csma40'] = t['c'].rolling(window=40).mean()
    t['d_c_csma40'] = t['c'] - t['csma40']
    t = t.dropna()
    t['bd_c_csma40'] = 0
    t.ix[t.d_c_csma40 >= 0, 'bd_c_csma40'] = 1

    t['signal'] = t['bd_c_csma40'].shift()

    t['d_c_o'] = t['c'] - t['o']
    t['bd_co'] = 0
    t.ix[t.d_c_o >= 0, 'bd_co'] = 1

    r = t.ix[t.signal == t.bd_co]

    return float(len(r)) / float(len(t))


# no effect
def test_sma2_5(_df):
    """ 如果2天均值收盘价小于5天均值收盘价, 第二天在开盘价买入;
        如果2天均值收盘价大于5天均值收盘价, 第二天在开盘价做空, 在收盘价平仓
        概念 2天均值和5天均值说明了市场有一定倾向, 一般认为参数小的均值相对于参数大的均值表明了方向
        大部分人凭直觉知道连续两天上涨后市场可能要下跌, 反之亦然"""
    t = pd.DataFrame(index=_df.index)
    t = t.reindex(index=t.index[::-1])

    t['c'] = _df['Close']
    t['o'] = _df['Open']

    t['csma2'] = t['c'].rolling(window=2).mean()
    t['csma5'] = t['c'].rolling(window=5).mean()

    t['d_csma2_5'] = t['csma2'] - t['csma5']

    t = t.dropna()
    t['bd_csma2_5'] = 0
    t.ix[t.d_csma2_5 <= 0, 'bd_csma2_5'] = 1

    t['signal'] = t['bd_csma2_5'].shift()

    t['d_c_o'] = t['c'] - t['o']
    t['bd_co'] = 0
    t.ix[t.d_c_o >= 0, 'bd_co'] = 1
    t = t.dropna()

    r = t.ix[t.signal == t.bd_co]

    return float(len(r)) / float(len(t))

