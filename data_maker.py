import math
import pandas as pd

ecg_path = "ppg_data/ecg_valid_24July.xlsx"
ppg_path = "ppg_data/ppg_valid_24July.xlsx"

ppg_df = pd.read_excel(ppg_path)
ecg_df = pd.read_excel(ecg_path)

ECG = []
PPG = []
BP = []
for col in ppg_df.columns:
    tmp = ppg_df[col].tolist()
    bp = tmp[0]
    BP.append([int(bp[0]), int(bp[1])])
    ppg_array = [int(ele) for ele in tmp[1:] if not math.isnan(ele)]
    ppg_array.insert(0, bp)
    PPG.append(ppg_array)

for col in ecg_df.columns:
    tmp = ecg_df[col].tolist()
    bp = tmp[0]
    ecg_array = [int(ele) for ele in tmp[1:] if not math.isnan(ele)]
    ecg_array.insert(0, bp)
    ECG.append(ecg_array)


df_csv = pd.DataFrame()
min_length = min(len(PPG), len(ECG))

for i in range(min_length):
    df_csv[f"ecg_{i}"] = ECG[i]
    df_csv[f"ppg_{i}"] = PPG[i]



print(df_csv.head(5))

df_csv.to_csv("check.csv",index = None)

