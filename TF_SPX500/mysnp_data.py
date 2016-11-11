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

# feature_select
# ______________________________________________________________________________________________________________________
feature_select = pd.DataFrame()
feature_select['snp_c0o0_0'] = np.log(ori_df['snp_c'] / ori_df['snp_o'])

# ln(today's open / today's close)  0, 1, 2, day before  np.log() = ln()
feature_select['aord_c0o0_0'] = np.log(ori_df['aord_c'] / ori_df['aord_o'])
feature_select['aord_c0o0_1'] = feature_select['aord_c0o0_0'].shift()
feature_select['aord_c0o0_2'] = feature_select['aord_c0o0_0'].shift(2)

# ln(today's open / today's close)  0, 1, 2, day before
feature_select['nikkei_c0o0_0'] = np.log(ori_df['nikkei_c'] / ori_df['nikkei_o'])
feature_select['nikkei_c0o0_1'] = feature_select['nikkei_c0o0_0'].shift()
feature_select['nikkei_c0o0_2'] = feature_select['nikkei_c0o0_0'].shift(2)

# ln(today's open / today's close)  0, 1, 2, day before
feature_select['hangseng_c0o0_0'] = np.log(ori_df['hangseng_c'] / ori_df['hangseng_o'])
feature_select['hangseng_c0o0_1'] = feature_select['hangseng_c0o0_0'].shift()
feature_select['hangseng_c0o0_2'] = feature_select['hangseng_c0o0_0'].shift(2)

# ln(today's open / today's close)  0, 1, 2, day before
feature_select['dax_c0o0_0'] = np.log(ori_df['dax_c'] / ori_df['dax_o'])
feature_select['dax_c0o0_1'] = feature_select['dax_c0o0_0'].shift()
feature_select['dax_c0o0_2'] = feature_select['dax_c0o0_0'].shift(2)

# ln(today's open / today's close)  0, 1, 2, day before
feature_select['ftse_c0o0_0'] = np.log(ori_df['ftse_c'] / ori_df['ftse_o'])
feature_select['ftse_c0o0_1'] = feature_select['ftse_c0o0_0'].shift()
feature_select['ftse_c0o0_2'] = feature_select['ftse_c0o0_0'].shift(2)

# corr_select = feature_select.corr().loc[::, 'snp_c0o0_0']
# scatter_matrix(feature_select, figsize=(20, 20), diagonal='kde')


# one-hot, up -> [1, 0], down -> [0, 1]
feature_select['snp_co_up'] = 0
feature_select.loc[feature_select['snp_c0o0_0'] >= 0, 'snp_co_up'] = 1
feature_select['snp_co_down'] = 0
feature_select.loc[feature_select['snp_c0o0_0'] < 0, 'snp_co_down'] = 1

feature_select['date'] = feature_select.index
feature_select['week'] = feature_select['date'].apply(lambda _var: _var.weekday() + 1)

# re_arrange columns
feature_select = feature_select[['date', 'week', 'snp_c0o0_0', 'snp_co_up', 'snp_co_down',

                                 'aord_c0o0_0', 'aord_c0o0_1', 'aord_c0o0_2',
                                 'nikkei_c0o0_0', 'nikkei_c0o0_1', 'nikkei_c0o0_2',
                                 'hangseng_c0o0_0', 'hangseng_c0o0_1', 'hangseng_c0o0_2',

                                 'dax_c0o0_0', 'dax_c0o0_1', 'dax_c0o0_2',
                                 'ftse_c0o0_0', 'ftse_c0o0_1', 'ftse_c0o0_2']]

feature_select = feature_select.dropna()

# save data as .pkl by using pickle
# ______________________________________________________________________________________________________________________
output = open('data_co.pkl', 'wb')
pickle.dump(feature_select, output)
output.close()
