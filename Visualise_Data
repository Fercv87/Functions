import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
import pandas as pd
from dateutil.relativedelta import relativedelta
import logging
import inspect
import seaborn as sns
import matplotlib.pyplot as plt

# Example usage
# df = pd.DataFrame()  # Assuming df is your DataFrame
# visualizer = DataVisualizer(df)
# visualizer.visualize_all(num_vars=['num_var1', 'num_var2'], cat_vars=['cat_var1', 'cat_var2'], hue=['hue_var1'])

class DataVisualizer:
    def __init__(self, df, style='darkgrid', palette='coolwarm', plot_width=10, plot_height=6):
        """
        Initialize the DataVisualizer with a DataFrame and default plotting styles and palettes.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the data to be visualized.
        style : str
            The style of the plots. Options include 'darkgrid', 'whitegrid', 'dark', 'white', and 'ticks'.
        palette : str or list
            The color palette to use for the plots. Can be a seaborn palette or a list of colors.
        plot_width : int or float, optional
            Default width of the plots.
        plot_height : int or float, optional
            Default height of the plots.
        """
        self.df = df
        self.style = style
        self.palette = palette
        self.plot_width = plot_width
        self.plot_height = plot_height
        sns.set_style(self.style)
        self.set_custom_palette()

    def set_custom_palette(self):
        """
        Sets the custom palette based on the initialization parameter.
        """
        if self.palette == 'custom_coolwarm':
            sns.set_palette(sns.color_palette('coolwarm', 15))
        elif self.palette == 'custom_blend':
            deep_pal = sns.color_palette('deep')
            sns.set_palette(sns.blend_palette([deep_pal[3], deep_pal[2]], 15))
        else:
            sns.set_palette(self.palette)

    def reorder_categories(self, cat_var):
        """
        Reorder categories of a categorical variable based on their frequency.
        
        Parameters:
        -----------
        cat_var : str
            The name of the categorical variable to reorder.
        """
        ordered_categories = list(self.df[cat_var].value_counts().index)
        self.df[cat_var] = pd.Categorical(self.df[cat_var], categories=ordered_categories, ordered=True)
        #self.df[cat_var] = self.df[cat_var].astype("category")
        #self.df[cat_var] = self.df[cat_var].cat.reorder_categories(new_categories=ordered_categories)
                
    def visualize_all(self, num_vars=None, cat_vars=None, hue=None):
        """
        Calls all visualization methods based on the provided parameters.

        Parameters:
        -----------
        num_vars : list of str, optional
            The names of the numerical variables to be plotted.
        cat_vars : list of str, optional
            The names of the categorical variables to be plotted.
        hue : list of str, optional
            The names of the variables to be used as hue for certain plots.
        """
        # Conditional checks to call each visualization method based on the provided arguments
        if num_vars is not None or cat_vars is not None:
            if num_vars is not None and cat_vars is not None:
                self.visualize_box_violin_plots(num_vars, cat_vars)
                self.visualize_strip_plots(num_vars, cat_vars)
                self.visualize_lm_line_plots(num_vars, cat_vars)

            if num_vars is not None:
                self.visualize_pairplots(num_vars, cat_vars, hue)
                self.visualize_kdeplots(num_vars, hue)
                self.visualize_joint_plots(num_vars)
                self.visualize_correlation_matrix(num_vars)

    def visualize_pairplots(self, num_vars=None, cat_vars=None, hue=None):
        """
        Generates pair plots for numerical variables with optional hue.

        Parameters:
        -----------
        num_vars : list of str, optional
            The names of the numerical variables to be plotted.
        cat_vars : list of str, optional
            The names of the categorical variables to reorder before plotting.
        hue : list of str, optional
            The names of the variables to be used as hue.
        """
        # Reorder categories for all categorical variables at once
        #plt.figure(figsize=(self.plot_width, self.plot_height))
        if num_vars:
            # Filter num_vars to include only numerical variables
            num_vars = [var for var in num_vars if pd.api.types.is_numeric_dtype(self.df[var])]
            
            # Check if there are numerical variables to plot
            if not num_vars:
                print("No numerical variables to plot.")
                return
            
            if cat_vars:
                for cat_var in cat_vars:
                    self.reorder_categories(cat_var)

            if hue:
                for hue_var in hue:
                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.pairplot(data=self.df, vars=num_vars, kind='scatter', hue=hue_var, palette='RdBu', diag_kws={'alpha':.5})
                    plt.suptitle(f'Pairplot of {", ".join(num_vars)} with hue on {hue_var}', y=1.02)
                    plt.show()
            else:
                plt.figure(figsize=(self.plot_width, self.plot_height))
                sns.pairplot(data=self.df, vars=num_vars)
                plt.suptitle(f'Pairplot of {", ".join(num_vars)}', y=1.02)
                plt.show()

    def visualize_kdeplots(self, num_vars=None, hue=None):
        """
        Generates KDE plots for numerical variables with optional hue.

        Parameters:
        -----------
        num_vars : list of str, optional
            The names of the numerical variables to be plotted.
        hue : list of str, optional
            The names of the variables to be used as hue.
        """
        num_vars = [var for var in num_vars if pd.api.types.is_numeric_dtype(self.df[var])]
        
        if not num_vars:
            print("No numerical variables available for KDE plots.")
            return

        if num_vars and hue:
            for num_var in num_vars:
                for hue_var in hue:
                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.kdeplot(data=self.df, x=num_var, hue=hue_var)
                    plt.title(f'KDE Plot of {num_var} with hue on {hue_var}')
                    plt.show()

                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.kdeplot(data=self.df, x=num_var, hue=hue_var, cut=0)
                    plt.title(f'KDE Plot of {num_var} with hue on {hue_var} (cut=0)')
                    plt.show()

                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.kdeplot(data=self.df, x=num_var, hue=hue_var, cut=0, cumulative=True)
                    plt.title(f'Cumulative KDE Plot of {num_var} with hue on {hue_var} (cut=0)')
                    plt.show()
    def visualize_box_violin_plots(self, num_vars, cat_vars):
        if cat_vars and num_vars:
            for cat_var in cat_vars:
                for num_var in num_vars:
                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.boxplot(data=self.df, x=cat_var, y=num_var)
                    plt.title(f'Boxplot of {num_var} by {cat_var}')
                    plt.xticks(rotation=90)
                    plt.show()

                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.violinplot(data=self.df, x=cat_var, y=num_var)
                    plt.title(f'Violinplot of {num_var} by {cat_var}')
                    plt.xticks(rotation=90)
                    plt.show()

    def visualize_strip_plots(self, num_vars, cat_vars):
        if cat_vars and num_vars:
            for num_var in num_vars:
                for cat_var_permutation in permutations(cat_vars, 2):
                    y_var, hue_var = cat_var_permutation
                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.stripplot(data=self.df, x=num_var, y=y_var, hue=hue_var, jitter=True)
                    plt.title(f'Stripplot of {num_var} by {y_var} with hue {hue_var}')
                    plt.show()

    def visualize_lm_line_plots(self, num_vars, cat_vars):
        if cat_vars and num_vars:
            for cat_var in cat_vars:
                self.reorder_categories(cat_var)  # Ensure categories are ordered
                for num_var_permutation in permutations(num_vars, 2):
                    x_var, y_var = num_var_permutation
                    if "_YYYY" in x_var or "_YYYY" in y_var:
                        x_var, y_var = (y_var, x_var) if "_YYYY" in y_var else (x_var, y_var)
                    plt.figure(figsize=(self.plot_width, self.plot_height))
                    sns.lmplot(data=self.df, x=x_var, y=y_var, hue=cat_var)
                    plt.title(f'Lmplot of {x_var} by {y_var} with hue on {cat_var}')
                    plt.show()

    def visualize_joint_plots(self, num_vars):
        """
        Generates various joint plots for pairs of numerical variables.

        Parameters:
        -----------
        num_vars : list of str
            The names of the numerical variables to be plotted in pairs.
        """
        num_vars_cleaned = [var for var in num_vars if pd.api.types.is_numeric_dtype(self.df[var])]
        
        if not num_vars_cleaned:
            print("No suitable numerical variables for joint plots.")
            return
        
        for x_var, y_var in permutations(num_vars_cleaned, 2): 
                
                # Jointplot with regression line
                plt.figure(figsize=(self.plot_width, self.plot_height))
                sns.jointplot(data=self.df, x=x_var, y=y_var, kind='reg')
                plt.suptitle(f'Regression Plot of {y_var} vs {x_var}', y=1.02)
                plt.show()

                # Jointplot with regression line (2nd order polynomial)
                plt.figure(figsize=(self.plot_width, self.plot_height))
                sns.jointplot(data=self.df, x=x_var, y=y_var, kind='reg', order=2, 
                              xlim=(self.df[x_var].min(), self.df[x_var].max()), 
                              ylim=(self.df[y_var].min(), self.df[y_var].max()))
                plt.suptitle(f'2nd Order Polynomial Regression Plot of {y_var} vs {x_var}', y=1.02)
                plt.show()

                # Jointplot showing the residuals
                plt.figure(figsize=(self.plot_width, self.plot_height))
                sns.jointplot(data=self.df, x=x_var, y=y_var, kind='resid', order=2)
                plt.suptitle(f'Residuals Plot of {y_var} vs {x_var}', y=1.02)
                plt.show()

                # Jointplot with scatter and kdeplot
                plt.figure(figsize=(self.plot_width, self.plot_height))
                g = sns.jointplot(data=self.df, x=x_var, y=y_var, kind='scatter', marginal_kws=dict(bins=10))
                g.plot_joint(sns.kdeplot)
                plt.suptitle(f'Scatter and KDE Plot of {y_var} vs {x_var}', y=1.02)
                plt.show()
    def visualize_correlation_matrix(self, num_vars, method='pearson'):
        """
        Visualizes the correlation matrix for specified numerical variables in the DataFrame.

        Parameters:
        -----------
        num_vars : list of str
            The names of the numerical variables to be plotted in pairs.
        method : {'pearson', 'kendall', 'spearman'}
            Method of correlation:
            - 'pearson' : standard correlation coefficient
            - 'kendall' : Kendall Tau correlation coefficient
            - 'spearman' : Spearman rank correlation
        save_path : str, optional
            Path to save the plot image. If None, the plot is not saved.
        """
        # Filter the DataFrame to include only the specified numerical variables
        if num_vars:
            data = self.df[num_vars]
        else:
            data = self.df.select_dtypes(include=[np.number])
        
        # Calculate the correlation matrix
        corr = data.corr(method=method)
        
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        plt.figure(figsize=(self.plot_width, self.plot_height))
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        
        plt.title(f'Correlation Matrix ({method.capitalize()} method) for Specified Numerical Variables')
        plt.show()
