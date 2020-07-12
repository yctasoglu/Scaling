import pandas as pd
import numpy as np

#! Reindex Dataset Before Normalizing
def df_reindex(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep]
    print("DataSet Reindexed...")
    df.info()
    return df
df = df_reindex(df)

#! Normalize the Data
def df_normalize(df):
    from sklearn.preprocessing import Normalizer
    normalizer = Normalizer(norm='l2')
    df = pd.DataFrame(normalizer.fit_transform(df),columns=df.columns)
    print("DataSet Normalized...")
    df.head()
    return df
df = df_normalize(df)

#! StandardScale the Data
def df_stdscale(df):
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    df = pd.DataFrame(std_scaler.fit_transform(df),columns=df.columns)
    print("DataSet StdScaled...")
    df.head()
    return df
df = df_stdscale(df)

#! MinMaxScale the Data
def df_minmaxscale(df):
    from sklearn.preprocessing import MinMaxScaler
    minmax_scaler = MinMaxScaler()
    df = pd.DataFrame(minmax_scaler.fit_transform(df),columns=df.columns)
    print("DataSet MinMaxScaled...")
    df.head()
    return df
df = df_minmaxscale(df)

#! RobustScale the Data
def df_robustscale(df):
    from sklearn.preprocessing import RobustScaler
    robust_scaler = RobustScaler().fit(df)
    df = pd.DataFrame(robust_scaler.transform(df),columns=df.columns)
    print("DataSet RobustScaled...")
    df.head()
    return df
df = df_robustscale(df)

#! MaxAbsScale the Data
def df_maxabsscale(df):
    from sklearn.preprocessing import MaxAbsScaler
    maxabs_scaler = MaxAbsScaler().fit(df)
    df = pd.DataFrame(maxabs_scaler.transform(df),columns=df.columns)
    print("DataSet MaxAbsScaled...")
    df.head()
    return df
df = df_maxabsscale(df)

#! QuantileScale the Data
def df_quantile_scale(df):
    from sklearn.preprocessing import QuantileTransformer
    quantile_scaler = QuantileTransformer().fit(df)
    df = pd.DataFrame(quantile_scaler.transform(df),columns=df.columns)
    print("DataSet QuantileScaled...")
    df.head()
    return df
df = df_quantile_scale(df)

#! Power Transform the Data
def df_power_transformer(df):
    from sklearn.preprocessing import  PowerTransformer
    power_transform_scaler =  PowerTransformer().fit(df)
    df = pd.DataFrame(power_transform_scaler.transform(df),columns=df.columns)
    print("DataSet QuantileScaled...")
    df.head()
    return df
df = df_power_transformer(df)


