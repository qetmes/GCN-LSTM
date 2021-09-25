import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from sklearn.linear_model import LinearRegression
import csv    #加载csv包便于读取csv文件


df = pd.read_pickle('./data.pkl')
# print(df)

flt_col = ['AIR_CODE', 'FLIGHT_NO', 'DEP_DATE', 'UP_LOCATION', 'DIS_LOCATION']   # 确认具体航班的五个字段
cls_col = ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL','CM',
           'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ']   # 26个舱位字段


# 利用LKZK（旅客折扣）和CA到CZ的26个舱位字段来计算平均折扣
def get_discount(d):
    d = d[d.LKZK > 0]    # LKZK（旅客折扣）
    print(d)
    X = d[cls_col].loc[:, d[cls_col].sum(axis=0) > 10]
    Y = d.LKZK * (X.sum(axis=1))

    reg = LinearRegression(fit_intercept=False)  # 用线性回归求每家航司的系数k
    reg.fit(X, Y)

    return pd.DataFrame([reg.coef_.round(2)], columns=X.columns)


disc = df.groupby(['AIR_CODE']).apply(lambda x:get_discount(x))
disc = disc.applymap(lambda x:x if x>0 else 0)
disc.index = [x[0] for x in disc.index]

print(disc)


def get_arv_bks(flt):
    flt = flt.sort_values(by='EX_DIF')
    bkd = flt[cls_col].diff(-1).applymap(lambda x: x if x > 0 else 0)
    bkd.iloc[-1] = flt[cls_col].iloc[-1]  # 最后一天用累计值

    bks = bkd.sum(axis=1)
    arv = (bkd * disc.loc[flt.AIR_CODE.iloc[0]]).sum(axis=1) / bks

    arv_bks = pd.DataFrame([arv, bks], index=['平均折扣', '订票数']).T
    #     arv_bks.bks = arv_bks.bks.apply(int)

    return pd.concat([flt, arv_bks], axis=1)




for k, flt in df.groupby(flt_col):
    arv_bks = get_arv_bks(flt)
    csv_file = open('hb_name_3_.csv')  # 打开csv文件
    # csv_reader_lines = csv.reader(csv_file)
    # for one_line in csv_reader_lines:
    #     if (k[0] == one_line[2] and str(k[1]) == one_line[3] and k[3] == one_line[4] and k[4] == one_line[5]):
    #         arv_bks.to_csv('./new_data/data_'+one_line[2]+'_'+one_line[3]+'.csv', mode='a',header=False)
    #     print(k[0]+'=='+one_line[2] +'--'+str(k[1])+ '=='+one_line[3]+'--'+k[3]+ '=='+one_line[4]+'--'+k[4]+ '=='+one_line[5])




    # hb_name = pd.DataFrame({'a': [k[0]], 'b': [k[1]], 'c':[k[3]], 'd':[k[4]]})
    # hb_name.to_csv('hb_name_1.csv',mode='a', header=False)
    # print(k)
    #
    # if ( k[0] == '3U' and k[1] == 2543 and k[3] == 'HPG' and k[4] == 'SHE' ):
    #     arv_bks.to_csv('data_3U_2543.csv', mode='a', header=False)
    #
    # if ( k[0] == '3U' and k[1] == 2675 and k[3] == 'HPG' and k[4] == 'BRE' ):
    #     arv_bks.to_csv('data_3U_2675.csv', mode='a', header=False)
    # #
    # if ( k[0] == '3U' and k[1] == 2776 and k[3] == 'BRE' and k[4] == 'CTS' ):
    #     arv_bks.to_csv('data_3U_2776.csv', mode='a', header=False)
    # #
    # if ( k[0] == '3U' and k[1] == 2833 and k[3] == 'MTY' and k[4] == 'UYN' ):
    #     arv_bks.to_csv('data_3U_2540.csv', mode='a', header=False)
    #
    #
    #
    #
    # if ( k[0] == 'BK' and k[1] == 2355 and k[3] == 'MIA' and k[4] == 'WUT' ):
    #     arv_bks.to_csv('data_BK_2355.csv', mode='a', header=False)
    #
    # if ( k[0] == 'BK' and k[1] == 4057 and k[3] == 'HAK' and k[4] == 'UME' ):
    #     arv_bks.to_csv('data_BK_4057.csv', mode='a', header=False)
    #
    # if ( k[0] == 'BK' and k[1] == 4373 and k[3] == 'HAK' and k[4] == 'PIT' ):
    #     arv_bks.to_csv('data_BK_4373.csv', mode='a', header=False)
    #
    # if ( k[0] == 'BK' and k[1] == 4750 and k[3] == 'WDS' and k[4] == 'BPX' ):
    #     arv_bks.to_csv('data_BK_4750.csv', mode='a', header=False)
    #
    # if ( k[0] == 'BK' and k[1] == 4797 and k[3] == 'HUY' and k[4] == 'WDS'):
    #     arv_bks.to_csv('data_BK_4797.csv', mode='a', header=False)
    #
    #
    #
    #
    # if (k[0] == 'CA' and k[1] == 140 and k[3] == 'WDS' and k[4] == 'WNH'):
    #     arv_bks.to_csv('data_CA_140.csv', mode='a', header=False)
    #
    # if (k[0] == 'CA' and k[1] == 180 and k[3] == 'WDS' and k[4] == 'YLW'):
    #     arv_bks.to_csv('data_CA_180.csv', mode='a', header=False)
    #
    # if (k[0] == 'CA' and k[1] == 1993 and k[3] == 'WNH' and k[4] == 'NZL'):
    #     arv_bks.to_csv('data_CA_1993.csv', mode='a', header=False)
    #
    # if (k[0] == 'CA' and k[1] == 2245 and k[3] == 'HSN' and k[4] == 'DUD'):
    #     arv_bks.to_csv('data_CA_2245.csv', mode='a', header=False)
    #
    # if (k[0] == 'CA' and k[1] == 2465 and k[3] == 'HPG' and k[4] == 'HAK'):
    #     arv_bks.to_csv('data_CA_2465.csv', mode='a', header=False)
    #
    #
    #
    #
    #
    # if (k[0] == 'CZ' and k[1] == 802 and k[3] == 'LIM' and k[4] == 'MCO'):
    #     arv_bks.to_csv('data_CZ_802.csv', mode='a', header=False)
    #
    # if (k[0] == 'CZ' and k[1] == 2151 and k[3] == 'JXA' and k[4] == 'CDG'):
    #     arv_bks.to_csv('data_CZ_2151.csv', mode='a', header=False)
    #
    # if (k[0] == 'CZ' and k[1] == 6936 and k[3] == 'LIM' and k[4] == 'JXA'):
    #     arv_bks.to_csv('data_CZ_6936.csv', mode='a', header=False)
    #
    # if (k[0] == 'CZ' and k[1] == 9799 and k[3] == 'JXA' and k[4] == 'LIM'):
    #     arv_bks.to_csv('data_CZ_9799.csv', mode='a', header=False)
    #
    # if (k[0] == 'CZ' and k[1] == 7438 and k[3] == 'CAI' and k[4] == 'CWJ'):
    #     arv_bks.to_csv('data_CZ_7438.csv', mode='a', header=False)



    # if k == ('3U', 2534, Timestamp('2020-10-23 00:00:00'), 'LIM', 'DDG'):
    #     # arv_bks.to_csv('data_3U_2534.csv', mode='a', header=True)
    #     arv_bks.to_csv('data_3U_2534.csv', mode='a', header=False)
    #     break
    # print(k)
    # print(arv_bks)
    # 作为样例，只计算一个
    # 全部计算需要10个小时




