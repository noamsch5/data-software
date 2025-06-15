import pandas as pd
import glob
import os

files = glob.glob('data_software/input/*.csv')
print('--- קבצים בתיקייה ---')
for f in files:
    print(os.path.basename(f))

print('\n--- שמות עמודות בכל קובץ ---')
for f in files:
    print(f'\n{os.path.basename(f)}:')
    try:
        df = pd.read_csv(f, nrows=1)
        print(list(df.columns))
    except Exception as e:
        print(f'שגיאה: {e}') 