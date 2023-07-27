from pysychonaut import Erowid
import pandas as pd
import numpy as np

report_df = pd.DataFrame(columns=['name', 'author', 'gender', 'age', 'substance', 'year', 'date', 'url', 'exp_id','experience','dosage'])

#write your substance of instance below
reports = Erowid.search_reports("1P-LSD")
print(reports[0].keys())

#write the number of reports that you are interested in
for i, report in zip(range(50),reports[:50]):
    print(i)
    for var in ['name', 'author','date','url','exp_id']:
        report_df.loc[i,var] = report[var]

    tmp = Erowid.get_experience(report["exp_id"])
    for var in ["year", "substance",'gender','age','experience','dosage']:
        try:
            report_df.loc[i,var] = tmp[var]
        #update except for only typeonly
        except:
            report_df.loc[i,var] = np.nan

#rename your output file
report_df.to_csv('erowid_1PLSD.txt', sep='|', index=False)
