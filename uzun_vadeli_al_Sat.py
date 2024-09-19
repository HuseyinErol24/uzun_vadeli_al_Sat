import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import vectorbt as vbt
# -- Hüseyin Erol --
hisse_adini_girin = 'KCHOL.IS'
df = yf.download(hisse_adini_girin,start="2020-01-01")

def money_flow_index(df, length=14):
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['money_flow'] = df['typical_price'] * df['Volume']
    df['money_flow_positive'] = np.where(df['typical_price'] > df['typical_price'].shift(1), df['money_flow'], 0)
    df['money_flow_negative'] = np.where(df['typical_price'] < df['typical_price'].shift(1), df['money_flow'], 0)
    df['money_flow_positive_sum'] = df['money_flow_positive'].rolling(window=length).sum()
    df['money_flow_negative_sum'] = df['money_flow_negative'].rolling(window=length).sum()
    df['mfi'] = 100 - (100 / (1 + df['money_flow_positive_sum'] / df['money_flow_negative_sum']))
    return df['mfi']

df['mfi'] = money_flow_index(df)

def median_hesapla(df, length):
    df['median'] = df['mfi'].rolling(window=length).median()
    return df['median']

median_length = 3

df['median'] = median_hesapla(df, median_length)
df['median_ema'] = df['median'].ewm(span=median_length).mean()
df["VWMA"] = ta.vwma(df["Close"], df["Volume"], length=17)


plt.figure(figsize=(14,5))
plt.plot(df["VWMA"],label="VWMA")
plt.plot(df["Close"],label="Kapanış Degeri")
plt.grid()
plt.figure(figsize=(14,5))
plt.plot(df["mfi"])

plt.axhline(80,linestyle="--",color="red",label="üstünde sat")
plt.axhline(30,linestyle="--",color="green",label="altında al")
plt.fill_between(df.index, df["median"], df["median_ema"],
                 where=df["median_ema"] > df["median"], color='red', label="Burdan Alma")
plt.fill_between(df.index, df["median"], df["median_ema"],
                 where=df["median_ema"] <= df["median"], color='green', label='Al Bekle')
plt.legend()
plt.grid()

df = df.drop(columns=["typical_price","money_flow","money_flow_positive","money_flow_negative","money_flow_positive_sum","money_flow_negative_sum"])
df["yeşil"]  = np.where((df["median_ema"] <= df["median"]),1,0)
df["AL_sinyali"] = np.where(
    (df["Close"] > df["VWMA"]) &
    (df["yeşil"] == 1),
    1,  
    0   
)

df["AL_sinyali"] = df["AL_sinyali"].diff()

df["Sat_sinyali"] = np.where(
    (df["VWMA"].shift(1) < df["Close"].shift(1)) &   
    (df["VWMA"].shift(0) > df["Close"].shift(0)),    
    1, 
    0   
)

plt.figure(figsize=(14,8))
plt.plot(df["Close"])
plt.plot(df["VWMA"])
plt.plot(df.loc[df["AL_sinyali"]==1]["Close"],"^")
plt.plot(df.loc[df["Sat_sinyali"]==1]["Close"],"v")
plt.legend()
plt.grid()
plt.show()



al = df["AL_sinyali"] == 1
sat = df["Sat_sinyali"] == 1

portfolio = vbt.Portfolio.from_signals(
    close=df['Close'],
    entries=al,
    exits=sat,
    direction='longonly',  
    freq='1D', 
    init_cash=1000  

)

print(portfolio.stats())

portfolio.plot().show()
