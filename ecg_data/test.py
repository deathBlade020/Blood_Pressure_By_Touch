import json
from icecream import ic

path = "data\\30ecgRawData.json"

with open(path,'r') as file:
    data = json.load(file)
    for item in data:
        lead = item["lead1"]
        length = len(lead)
        if length >= 2700 and length <= 3500:
            ic(len(lead))


    # ic(len(glob_dict.keys()))

    


# cnt = 0
#     store = []
#     glob_dict = {}
#     for item in data:
#         my_dict = item
#         if "lead1" in my_dict:
#             length = len(my_dict["lead1"])
#             if length >= 3000:
#                 if "creationDate" in my_dict:
#                     small_dict = my_dict["creationDate"]
#                     if "$date" in small_dict:
#                         ic(len(my_dict["lead1"]))
#                         store.append([small_dict["$date"],my_dict["lead1"]])
#                         cnt += 1
                    



#     ic(cnt)
