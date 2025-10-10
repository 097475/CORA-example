import pandas as pd
from pathlib import Path
import os

folder = Path('.')
os.makedirs('processed_data', exist_ok=True)


for file in folder.glob('*.csv'):
    if 'proc' in file.name:
        continue
    elif 'bisim' in file.name:
        # Load the CSV file
        headers = ['Outcome', 'Running Time', 'Original Nodes', 'Bisimular Nodes', 'Bisimulation Time']
        df = pd.read_csv(file.name, header=None, names=headers)

        # Compute the cumulative sum of the second column
        cumsum = df.iloc[:, 1].cumsum()

        # Optional: add it as a new column to the DataFrame
        df['Cumulative Time'] = cumsum


        # Create a boolean mask where 'pass' is True
        is_pass = df['Outcome'].eq('Pass')

        # Compute cumulative count of 'pass'
        cumulative_pass = is_pass.cumsum()

        # Compute cumulative percentage of 'pass'
        cumulative_percentage = cumulative_pass / len(df)

        # Add it to the DataFrame
        df['Cumulative Outcome'] = cumulative_percentage

        df.to_csv('processed_data/' + file.stem + '_proc' + file.suffix, index=False)
    else:
        # Load the CSV file
        headers = ['Outcome', 'Running Time']
        df = pd.read_csv(file.name, header=None, names=headers)

        # Compute the cumulative sum of the second column
        cumsum = df.iloc[:, 1].cumsum()

        # Optional: add it as a new column to the DataFrame
        df['Cumulative Time'] = cumsum


        # Create a boolean mask where 'pass' is True
        is_pass = df['Outcome'].eq('Pass')

        # Compute cumulative count of 'pass'
        cumulative_pass = is_pass.cumsum()

        # Compute cumulative percentage of 'pass'
        cumulative_percentage = cumulative_pass / len(df)

        # Add it to the DataFrame
        df['Cumulative Outcome'] = cumulative_percentage

        df.to_csv('processed_data/' + file.stem + '_proc' + file.suffix, index=False)
