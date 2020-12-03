import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression  # BayesianRidge


def linreg_residuals(x: pd.Series, y: pd.Series, summary: bool=False) -> tuple:
    x_df = pd.DataFrame(x)
    y_df = pd.DataFrame(y)
    ols = LinearRegression()  # BayesianRidge
    ols.fit(x_df, y_df)
    y_hat = ols.predict(x_df).squeeze()
    df = pd.DataFrame({
        'y': y,
        'y_hat': y_hat,
        'residual': y - y_hat,
        'intercept': ols.intercept_[0],
        'beta': ols.coef_[0][0],
        })
    if summary is True:
        import statsmodels.api as sm
        mod = sm.OLS(y, X)
        fit_mod = mod.fit()
        print(fit_mod.summary())
    return df


def colwise_linreg_residuals(df: pd.DataFrame, beta_series: pd.Series) -> pd.DataFrame:
    results = []
    for col in tqdm(df.columns):
        # 'regress out' market 'beta' and return residuals
        linreg_df = linreg_residuals(x=beta_series, y=df[col])
        results.append(linreg_df['residual'])
    # convert list of pd.Series to df
    resid_df = pd.DataFrame(results).transpose()
    resid_df.columns = df.columns
    return resid_df
