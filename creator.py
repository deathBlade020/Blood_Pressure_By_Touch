import pandas as pd
from icecream import ic

df = pd.read_excel("convertedall_data_valid_invalid_new.xlsx", index_col=None)
cnt = 0
ret_col = [i+1 for i in range(151)]
df_valid = pd.DataFrame(columns=ret_col)
for col in df.columns:
    ppg_array = df[col].tolist()
    header = ppg_array[0]
    tmp = header.split(",")
    bp = tmp[0].strip()
    is_valid = int(tmp[1].strip())
    if is_valid == 1:
        valid_index = int(tmp[2].strip())
        systolic_bp = int(bp.split("/")[0])
        if valid_index + 150 < 350:
            cnt += 1
            ic(systolic_bp, is_valid, valid_index)
            valid_ppg = ppg_array[1:][valid_index:valid_index+150]
            valid_ppg.append(systolic_bp)
            df_valid.loc[len(df_valid)] = valid_ppg

ic(cnt)
df_valid.to_csv("valid_overall_new.csv", index=None)
print("DONE")
