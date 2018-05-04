import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import LabelEncoder

def fill_mixed_median_mode(dataframe, medians=list()):
    """ Fill missing values with median for specified column, otherwise mode
    
    Args:
        dataframe (pandas.core.frame.DataFrame): rows of observations of features
        medians (list): columns to fill missing values with median instead of mode
        
    Returns:
        dataframe with no missing values
    """
    
    
    null = dataframe.isnull().any()
    null_cols = list(null[null].index)
    
    fill = pd.Series([data[c].median() if c in medians else data[c].mode()[0]
                     for c in null_cols], index=null_cols)
    
    dataframe[null_cols] = dataframe[null_cols].fillna(fill)
    return dataframe

def ecdf_plots(data, features, plot_cols=2, quartile_markers=False):
    """Plot a grid of ECDFs for numeric data
    
    Args:
        data (pd.core.frame.DataFrame)
        features (iterable): names of numeric columns of data
        plot_cols (int): number of columns for subplot grid
        quartile_markers (bool): whether to plot the quartiles with the ecdf
        
    Returns:
        fig: matplotlib.figure.Figure object
        axs: array of Axes objects.
    
    """
    
    plot_rows = int(np.ceil(len(features) / plot_cols))
    
    fig, axs = plt.subplots(plot_rows, plot_cols)
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle('Empirical Cumulative Distribution Functions')
    
    for ix, feature in enumerate(features):
        selection = data[feature]
        
        p0 = ix // plot_cols
        p1 = ix % plot_cols
        
        ecdf = ECDF(selection)
        x = np.linspace(selection.min(), selection.max())
        y = ecdf(x)
        
        axs[p0, p1].step(x, y)
        axs[p0, p1].set(title=feature)
        
        if quartile_markers:
            quartiles = selection.describe().loc[['25%', '50%', '75%']]
            
            for q in quartiles:
                qy = ecdf(q)
                axs[p0, p1].plot(q, qy, 'kD')
        
    return fig, axs

def scatter(data, x, y):
    fig, ax = plt.subplots()

    ax.scatter(data[x], data[y], s=10, alpha=0.6)
    ax.set(xlabel=x, ylabel=y)
    
    return fig, ax

def compute_cov_corr(data, feature1, feature2):
    M = data[[feature1, feature2]].as_matrix().T
    
    cov = np.cov(M)[0, 1]
    corr = np.corrcoef(M)[0, 1]
    
    return cov, corr

def correlation_heatmap(corr_data, labels):
    """Visualize the correlation matrix of a dataframe as a heatmap
    https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    """
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(19, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio    
    sns.heatmap(corr_data, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
                ax=ax, yticklabels=labels, xticklabels=labels)
    
    return fig, ax

def encode_mixed_categoricals(data, categories):
    """Encode non-numeric categoricals in dataframe with mixed dtypes"""
    encoded = [pd.Series(LabelEncoder().fit_transform(data_f[c]))
               if data_f[c].dtype==np.dtype('O') else data_f[c] 
               for c in categories]
    
    data_enc = pd.concat(encoded, axis=1)
    data_enc.columns = categories
    
    return data_enc