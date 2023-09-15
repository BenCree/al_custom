from sklearn.preprocessing import MinMaxScaler
import pandas as pd


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale_df(df):
    '''Scale the metrics to [0, 1] for columns containing floats'''
    scaler = MinMaxScaler()
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = scaler.fit_transform(df[float_cols])
    return df

def add_sf(file1, file2, func, scale=True):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if set(df1.columns) != set(df2.columns):
        mismatched_columns = set(df1.columns).symmetric_difference(set(df2.columns))
        raise ValueError(f"Mismatched columns: {mismatched_columns}")

    func_name = func.__name__
    if scale:
        scaled_df1 = scale_df(df1.copy())
        scaled_df2 = scale_df(df2.copy())

        df1[func_name] = func(scaled_df1)
        df2[func_name] = func(scaled_df2)
    else:
        df1[func_name] = func(df1)
        df2[func_name] = func(df2)

    df1.to_csv(f"{file1.split('.csv')[0]}_{func_name}.csv", index=False)
    df2.to_csv(f"{file2.split('.csv')[0]}_{func_name}.csv", index=False)

    return df1, df2



def plot_cycles(df, scale=False):


    # Separate 'Index' and metrics
    index_col = df['Index']
    metrics = df.drop('Index', axis=1)

    if scale:
        # Scale the metrics to [0, 1]
        scaler = MinMaxScaler()
        scaled_metrics = scaler.fit_transform(metrics)
        scaled_df = pd.DataFrame(scaled_metrics, columns=metrics.columns)
        scaled_df['Index'] = index_col
        df = scaled_df

    # Melt the DataFrame
    df_melted = df.melt(id_vars=['Index'], var_name='Metric', value_name='Value')

    # Create lineplot with standard deviation shading
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Index', y='Value', hue='Metric', data=df_melted, )

    plt.xlabel('Cycle')
    plt.ylabel('Scaled Metrics')
    plt.legend(title='Metrics')
    plt.show()

def violin_plot():
    # summary of al cycle
    pass

def qed_cnn_per_mw(df):
    '''qed * (cnn / mw)'''
    if 'QED' not in df.columns or 'cnn_per_mw' not in df.columns:
        raise KeyError("Missing required columns: 'qed' and/or 'cnn_per_mw'")
    return df['QED'] * df['cnn_per_mw']

def sa_cnn_per_mw(df):
    '''qed * (cnn / mw)'''
    if 'synthetic_accessibility' not in df.columns or 'cnn_per_mw' not in df.columns:
        raise KeyError("Missing required columns: 'qed' and/or 'cnn_per_mw'")
    return df['synthetic_accessibility'] * df['cnn_per_mw']


#probably just do a generic function for each type of combination, *, /, exp??
def plip_cnn_per_mw(df):
    '''Xstal fragment similarity * (cnn / mw)'''
    col_name = 'plip'
    if col_name not in df.columns or 'cnn_per_mw' not in df.columns:
        raise KeyError(f"Missing required columns: {col_name} and/or 'cnn_per_mw'")
    return df[col_name] * df['cnn_per_mw']


def logp_cnn_per_mw(df):
    '''Xstal fragment similarity * (cnn / mw)'''
    col_name = 'LogP'
    if col_name not in df.columns or 'cnn_per_mw' not in df.columns:
        raise KeyError(f"Missing required columns: {col_name} and/or 'cnn_per_mw'")
    return df[col_name] * df['cnn_per_mw']

# think about how to do sf's for HBA/HBD, cant be maxmin because dependent on the datasets max/min, cant just be
# multiply because values are large and not always good to maximise


#also look at correlation of features/metrics


def compute_column_similarity(df1, df2):

    total_columns = len(df1.columns)

    match_score = 0



    for col2 in df2.columns:

        parts2 = col2.split('_')



        # Full match

        if col2 in df1.columns:

            match_score += 1

        else:

            # Partial match (excluding the last part)

            partial_match_cols = [col1 for col1 in df1.columns if col1.split('_')[:-1] == parts2[:-1]]

            match_score += 0.5 * len(partial_match_cols)



    return match_score / total_columns


add_sf('gen_metrics.csv', 'initial.csv', logp_cnn_per_mw)

