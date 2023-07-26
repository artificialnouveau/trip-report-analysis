from pysychonaut import Erowid

reports = Erowid.search_reports("1P-LSD")
print(reports[0].keys())
for report in reports[:5]:
    print(report["substance"], report["url"], report["date"], report["exp_id"])


trip_report = Erowid.get_experience(1)
for key in trip_report:
    print(key, ":", trip_report[key])
