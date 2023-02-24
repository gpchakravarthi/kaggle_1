# from tabulate import tabulate
#
# from prettytable import PrettyTable
# import pandas as pd
#
# df = pd.read_excel(r'/Users/pradeep/Desktop/completed.xlsx')
# print(df)
# print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

for type, alias in [['SNOWFLAKE', 'SF'], ['AWS-S3', 'S3'], ['Local', 'LOCAL']]:
    print(type,alias)
#
# from werkzeug import secure_filename
#
#
# print(secure_filename('user/fixed/jam.py'))
