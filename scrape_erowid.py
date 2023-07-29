from pysychonaut import Erowid
import pandas as pd
import numpy as np
import time

report_df = []

#Add your list of substances here
list_of_substances = ['1P-LSD','2CB', 'Ketamine']

#Add the number of experiences you want to scrape per substance
num_of_records = 100

for sub in list_of_substances:
    print(sub)
    reports = Erowid.search_reports(sub)
    report_tmp = pd.DataFrame(columns=['name', 'author', 'gender', 'age', 'substance', 'year', 'date', 'url', 'exp_id','experience','dosage'])

    for i, report in zip(range(num_of_records),reports[:num_of_records]):
        if i==0 or i%%5==0:
            print(i)
        for var in ['name', 'author','date','url','exp_id']:
            report_tmp.loc[i,var] = report[var]
            
        # this is to prevent Erowid to prevent blocking your IP
        time.sleep(3)
        tmp = Erowid.get_experience(report["exp_id"])
        for var in ["year", "substance",'gender','age','experience','dosage']:
            try:
                report_tmp.loc[i,var] = tmp[var]
            except TypeError:
                report_tmp.loc[i,var] = np.nan
        
        report_df.append(report_tmp)

report_df = pd.concat(report_df)
filename = 'erowid_export_'+'_'.join(list_of_substances)+'.txt'
report_df.to_csv(filename, sep='|', index=False)
print('Exported to ',filename)
