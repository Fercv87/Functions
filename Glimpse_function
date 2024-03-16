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

#glimpser = DataFrameGlimpser(Attrition_data)
#glimpser.set_df_name('Attrition_data')  # Optional: Set a custom name
#glimpser.glimpse()


class DataFrameGlimpser:
    '''
    This class provides a comprehensive overview of a Pandas DataFrame.
    It is intended to be used in Jupyter Notebooks.
    '''
    def __init__(self, df):
        self.df = df
        self.df_name = "DataFrame"
        # Set Seaborn configurations here
        sns.set_palette("colorblind")
        sns.set_context('paper')

    def set_df_name(self, df_name):
        self.df_name = df_name

    def display_html(self, content):
        display(HTML(content))

    def glimpse(self):
        self.display_shape()
        self.display_info()
        self.display_head_and_tail()
        self.display_summary_statistics()
        self.calculate_additional_statistics()
        self.remove_outliers()
        self.generate_outlier_removal_snippet()
        self.display_comprehensive_statistics()
        self.calculate_correlation_matrix()
        self.display_correlation_matrix()
        self.display_custom_correlation_matrix()
        self.summarize_and_display_correlations()
        self.display_missing_values()
        self.plot_missing_values_bar_chart()
        self.suggest_columns_to_drop()
        self.display_unique_values()
        self.display_index_information()
        self.suggest_and_display_data_type_changes()
        self.identify_potential_primary_keys()
        self.calculate_memory_savings()

    def display_shape(self):
        styled_string = f'<h3>Shape of {self.df_name}: {self.df.shape[0]} rows and {self.df.shape[1]} columns</h3>'
        self.display_html(styled_string)

    def display_info(self):
        self.display_html(f'<h4>DataFrame Info ("{self.df_name}"):</h4>')
        display(self.df.info())
        
    def display_head_and_tail(self):
        self.display_html(f'<h4>First 5 Rows ("{self.df_name}"):</h4>')
        display(self.df.head())
        self.display_html(f'<h4>Last 5 Rows ("{self.df_name}"):</h4>')
        display(self.df.tail())
        
    def display_summary_statistics(self):
        self.display_html(f'<h4>Summary Statistics ("{self.df_name}"):</h4>')
        median_values = self.df.median(numeric_only=True)
        mode_values = self.df.mode().iloc[0]
        variance_values = self.df.var(numeric_only=True)
        mad_values = (self.df.select_dtypes(include=['number']) - self.df.select_dtypes(include=['number']).mean()).abs().mean()
    
    def calculate_additional_statistics(self):
        additional_stats_df = pd.DataFrame()

        num_columns = self.df.select_dtypes(include=['number']).columns
        for col in num_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = self.df[col][(self.df[col] < lower) | (self.df[col] > upper)]
            outliers_count = len(outliers)
            outliers_percentage = (outliers_count / len(self.df[col])) * 100

            additional_stats_df.loc['iqr', col] = iqr
            additional_stats_df.loc['lower', col] = lower  
            additional_stats_df.loc['upper', col] = upper 
            additional_stats_df.loc['outliers_count', col] = outliers_count
            additional_stats_df.loc['outliers_percentage', col] = outliers_percentage

        self.additional_stats_df = additional_stats_df

    def remove_outliers(self):
        df_no_outliers = self.df.copy()
        num_columns = self.df.select_dtypes(include=['number']).columns
        for col in num_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower) & (df_no_outliers[col] <= upper)]

        self.df_no_outliers = df_no_outliers
         # Rename the DataFrame
        self.df_name_no_outliers = self.df_name + "_no_outliers"

        # Print the shape of the new DataFrame
        print(f"Shape of the original DataFrame '{self.df_name}': {self.df.shape}")
        print(f"Shape of the DataFrame without outliers '{self.df_name_no_outliers}': {self.df_no_outliers.shape}")
    
    def generate_outlier_removal_snippet(self):
        code_snippet = f"""
        # Code Snippet to Remove Outliers from '{self.df_name}'
        {self.df_name}_no_outliers = {self.df_name}.copy()
        num_columns = {self.df_name}.select_dtypes(include=['number']).columns
        for col in num_columns:
            q1 = {self.df_name}[col].quantile(0.25)
            q3 = {self.df_name}[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            {self.df_name}_no_outliers = {self.df_name}_no_outliers[({self.df_name}_no_outliers[col] >= lower) & ({self.df_name}_no_outliers[col] <= upper)]
        """

        print("\nPython Code Snippet for Removing Outliers:")
        print("-------------------------------------------------")
        print(code_snippet)    
        
    def display_comprehensive_statistics(self):
        # Create a DataFrame for median, mode, variance, and mean absolute deviation
        median_values = self.df.median(numeric_only=True)
        mode_values = self.df.mode().iloc[0]
        variance_values = self.df.var(numeric_only=True)
        mad_values = (self.df.select_dtypes(include=['number']) - self.df.select_dtypes(include=['number']).mean()).abs().mean()

        other_stats_df = pd.DataFrame({
            'median': median_values,
            'mode': mode_values,
            'variance': variance_values,
            'mad': mad_values
        })

        # Calculate IQR, outliers, and standard deviation percentages for numerical columns
        num_columns = self.df.select_dtypes(include=['number']).columns
        for col in num_columns:
            mean_value = self.df[col].mean()
            std_value = self.df[col].std()

            # Calculate percentages of data within 1, 2, and 3 standard deviations
            within_one_std = len(self.df[col][(self.df[col] >= mean_value - std_value) & (self.df[col] <= mean_value + std_value)]) / len(self.df[col]) * 100
            within_two_std = len(self.df[col][(self.df[col] >= mean_value - 2 * std_value) & (self.df[col] <= mean_value + 2 * std_value)]) / len(self.df[col]) * 100
            within_three_std = len(self.df[col][(self.df[col] >= mean_value - 3 * std_value) & (self.df[col] <= mean_value + 3 * std_value)]) / len(self.df[col]) * 100

            self.additional_stats_df.loc['within_one_std', col] = within_one_std
            self.additional_stats_df.loc['within_two_std', col] = within_two_std
            self.additional_stats_df.loc['within_three_std', col] = within_three_std

        # Reset index for describe() DataFrame, other_stats_df, and additional_stats_df
        describe_df = self.df.describe().reset_index()
        other_stats_df = other_stats_df.T.reset_index()
        additional_stats_df = self.additional_stats_df.reset_index()

        # Rename the 'index' column to 'statistic' for all DataFrames
        describe_df.rename(columns={'index': 'statistic'}, inplace=True)
        other_stats_df.rename(columns={'index': 'statistic'}, inplace=True)
        additional_stats_df.rename(columns={'index': 'statistic'}, inplace=True)

        # Concatenate with describe() DataFrame and other DataFrames
        summary_stats = pd.concat([describe_df, other_stats_df, additional_stats_df])

        # Set 'statistic' as the index again
        summary_stats.set_index('statistic', inplace=True)

        # Round all numerical values to 2 decimal places
        summary_stats = summary_stats.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
        # Apply the color to the 'variance', 'mad', and 'outliers_percentage' rows
        styled_summary_stats = summary_stats.style.applymap(self.highlight_variance, subset=pd.IndexSlice['variance', :]) \
                                                  .applymap(self.highlight_mad, subset=pd.IndexSlice['mad', :]) \
                                                  .applymap(self.highlight_outliers, subset=pd.IndexSlice['outliers_percentage', :])

        display(styled_summary_stats)

    def highlight_outliers(self, val):
        if val > 10:
            color = 'red'  # red for high percentage of outliers
        elif val > 5:
            color = 'yellow'  # yellow for moderate
        else:
            color = 'green'  # green for low
        return f'background-color: {color}'

    def highlight_variance(self, val):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ''  # Return an empty string or some other default style for non-numeric or missing values
        if val > 100:
            color = 'red'  # red for high variance
        elif val > 50:
            color = 'yellow'  # yellow for moderate
        else:
            color = 'green'  # green for low
        return f'background-color: {color}'

    def highlight_mad(self, val):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ''  # Return an empty string or some other default style for non-numeric or missing values
        if val > 20:
            color = 'red'  # red for high MAD
        elif val > 10:
            color = 'yellow'  # yellow for moderate
        else:
            color = 'green'  # green for low
        return f'background-color: {color}'
    
    def calculate_correlation_matrix(self):
        num_columns = self.df.select_dtypes(include=['number']).columns
        correlation_matrix = self.df[num_columns].corr()

        # Round all numerical values to 2 decimal places
        correlation_matrix = correlation_matrix.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
        self.correlation_matrix = correlation_matrix
        
    def display_correlation_matrix(self):
        self.display_html(f'<h4>Correlation Coefficients ("{self.df_name}"):</h4>')
        styled_corr_matrix = self.correlation_matrix.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}")
        display(styled_corr_matrix)

    def color_cells(self, val):
        if pd.isna(val):
            return ''
        elif val >= 0.9 or val <= -0.9:
            return 'background-color: red'
        elif val >= 0.7 or val <= -0.7:
            return 'background-color: orange'
        elif val >= 0.5 or val <= -0.5:
            return 'background-color: yellow'
        elif val >= 0.3 or val <= -0.3:
            return 'background-color: lightgreen'
        else:
            return 'background-color: green'

    def display_custom_correlation_matrix(self):
        # Apply custom coloring
        styled_corr_matrix = self.correlation_matrix.style.applymap(self.color_cells).format("{:.2f}")
        display(styled_corr_matrix)

    def summarize_and_display_correlations(self):
        summary_text = []
        num_columns = self.df.select_dtypes(include=['number']).columns

        for i, col in enumerate(num_columns):
            for j, row in enumerate(num_columns):
                if i < j:  # Only consider lower triangle
                    corr = self.correlation_matrix.loc[row, col]  # Corrected order of indices
                    if not pd.isna(corr):  # Ignore NaN values
                        if corr >= 0.9 or corr <= -0.9:
                            interpretation = "Very high positive (negative) correlation"
                        elif corr >= 0.7 or corr <= -0.7:
                            interpretation = "High positive (negative) correlation"
                        elif corr >= 0.5 or corr <= -0.5:
                            interpretation = "Moderate positive (negative) correlation"
                        elif corr >= 0.3 or corr <= -0.3:
                            interpretation = "Low positive (negative) correlation"
                        else:
                            interpretation = "Negligible correlation"
                        summary_text.append(f'{col} and {row}: {corr:.2f} - {interpretation}')

        # Display summary
        summary_html = '<br>'.join(summary_text)
        self.display_html(f'<h4>Correlation Summary:</h4><p>{summary_html}</p>')

    def display_missing_values(self):
        missing_values = self.df.isnull().sum()
        self.display_html(f'<h4>Missing Values ("{self.df_name}"):</h4>')
        display(missing_values[missing_values > 0])

    def plot_missing_values_bar_chart(self):
        missing_values = self.df.isnull().sum()
        
        if missing_values.sum() > 0:
            plt.figure(figsize=(12, 6))
            missing_sorted = missing_values[missing_values > 0].sort_values(ascending=True)
            bars = missing_sorted.plot(kind='barh', color='salmon', edgecolor='black')
            
            # Annotate each bar with the number of records
            for bar in bars.patches:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                         f'{int(bar.get_width())}', 
                         va='center', ha='left', color='black', fontsize=10)
            
            plt.title('Volume of Missing Values per Variable')
            plt.xlabel('Number of Missing Values')
            plt.ylabel('Variables')
            plt.grid(axis='x')
            plt.show()
            
    def suggest_columns_to_drop(self):
        threshold = len(self.df) * 0.05
        cols_to_drop = self.df.columns[self.df.isna().sum() > threshold]
        self.display_html(f'<h4>Potential Columns to Drop Due to Significant Missing Values ("{self.df_name}"):</h4>')
        display(cols_to_drop)

        # Generate and display code snippet for dropping columns
        new_df_name = self.df_name + "_No_Missings"
        code_snippet = f"{new_df_name} = {self.df_name}.dropna(subset={list(cols_to_drop)}, inplace=False)"
        self.display_html(f'<h4>Python Code Snippet for Dropping Columns with Significant Missing Values:</h4>')
        print(code_snippet)

    def display_unique_values(self):
        unique_values = self.df.nunique()
        self.display_html(f'<h4>Number of Unique Values ("{self.df_name}"):</h4>')
        display(unique_values)

        # Display unique values for columns with <= 20 unique values
        self.display_html(f'<h4>Unique Values for Columns with <= 20 Unique Values:</h4>')
        for col in self.df.columns:
            if self.df[col].nunique() <= 20:
                unique_vals = ', '.join(map(str, self.df[col].unique()))
                self.display_html(f'<p><b>{col}</b>: {unique_vals}</p>')
    
    def display_index_information(self):
        self.display_html(f'<h4>Index Information ("{self.df_name}"):</h4>')
        display(self.df.index)    
    
    def suggest_and_display_data_type_changes(self):
        suggestions = {}
        code_snippets = []

        for col in self.df.columns:
            dtype = self.df[col].dtype
            if dtype == 'object':
                # Check if the column can be converted to datetime
                try:
                    pd.to_datetime(self.df[col])
                    suggestions[col] = 'datetime'
                    code_snippets.append(f" {self.df_name}['{col}'] = pd.to_datetime({self.df_name}['{col}'])")
                except:
                    # Check if the column can be converted to a category
                    if self.df[col].nunique() / len(self.df) < 0.05:  # 5% threshold for unique values
                        suggestions[col] = 'category'
                        code_snippets.append(f" {self.df_name}['{col}'] = {self.df_name}['{col}'].astype('category')")
            elif dtype == 'int64' or dtype == 'float64':
                # Check if the column can be converted to int (no decimal values)
                if self.df[col].dropna().apply(lambda x: float(x).is_integer()).all():  # Modified line
                    suggestions[col] = 'int'
                    code_snippets.append(f" {self.df_name}['{col}'] = {self.df_name}['{col}'].astype('int')")
                else:
                    suggestions[col] = 'float'
                    code_snippets.append(f" {self.df_name}['{col}'] = {self.df_name}['{col}'].astype('float')")

        # Display suggestions
        self.display_html(f'<h4>Suggestions for Column Data Types ("{self.df_name}"):</h4>')
        for col, suggestion in suggestions.items():
            if str(self.df[col].dtype) != suggestion:
                print(f"Column '{col}' of type '{self.df[col].dtype}' could be '{suggestion}'")

        # Display code snippets
        self.display_html(f'<h4>Python Code Snippets for Suggested Changes:</h4>')
        for snippet in code_snippets:
            print(snippet)  
        
    def identify_potential_primary_keys(self):
        unique_values = self.df.nunique()
        potential_keys = unique_values[unique_values == len(self.df)]
        self.display_html(f'<h4>Potential Primary Keys ("{self.df_name}"):</h4>')
        display(potential_keys)     
        
    def calculate_memory_savings(self, columns=None):
        if columns is None:
            columns = self.df.columns.tolist()

        memory_savings_data = []

        for column in columns:
            if column in self.df.columns:
                original_dtype = self.df[column].dtype
                memory_as_source = self.df[column].nbytes
                memory_as_category = self.df[column].astype('category').nbytes

                ratio_saved_memory = 1 - (memory_as_category / memory_as_source)
                ratio_saved_memory_percent = f"{ratio_saved_memory * 100:.2f}%"

                unique_count = self.df[column].nunique()
                nan_count = self.df[column].isna().sum()
                zero_count = (self.df[column] == 0).sum()
                value_counts = self.df[column].value_counts()
                most_frequent_value = value_counts.idxmax()
                least_frequent_value = value_counts.idxmin()

                memory_savings_data.append({
                    'variable_name': column,
                    'original_dtype': original_dtype,
                    'unique_count': unique_count,
                    'ratio_saved_memory': ratio_saved_memory_percent,
                    'nan_count': nan_count,
                    'zero_count': zero_count,
                    'most_frequent_value': most_frequent_value,
                    'least_frequent_value': least_frequent_value
                })

        memory_savings_df = pd.DataFrame(memory_savings_data)
        memory_savings_df['ratio_saved_memory_float'] = memory_savings_df['ratio_saved_memory'].str.rstrip('%').astype(float)
        to_convert = memory_savings_df[memory_savings_df['ratio_saved_memory_float'] > 50]['variable_name']

        conversion_code = "\n".join(f"{self.df_name}['{column}'] = {self.df_name}['{column}'].astype('category')" for column in to_convert)
        memory_savings_df = memory_savings_df.sort_values(by='ratio_saved_memory_float', ascending=False).drop(columns='ratio_saved_memory_float')

        self.display_html(f'<h4>Memory Savings Analysis for "{self.df_name}":</h4>')
        display(memory_savings_df)
        self.display_html(f'<h4>Conversion Code for Columns with Significant Memory Savings:</h4>')
        print(conversion_code)       
        
    
