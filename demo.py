import pandas as pd

xls = pd.ExcelFile("ADS_Index_Most_Current_Vintage.xlsx")
print(xls.sheet_names)
