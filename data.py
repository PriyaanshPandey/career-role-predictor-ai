import random
import pandas as pd

roles = {
"AI_Engineer":            [5,5,2,5,2,5,5,4,4,2],
"Data_Scientist":         [4,5,2,3,3,4,5,5,3,2],
"ML_Engineer":            [5,4,2,5,2,5,4,4,4,2],
"Backend_Developer":      [4,3,2,5,2,4,3,4,3,2],
"Frontend_Developer":     [2,2,5,3,4,3,3,3,4,5],
"UI_UX_Designer":         [1,1,5,1,5,2,2,3,4,5],
"Cybersecurity_Engineer": [5,3,1,4,1,5,4,5,2,1],
"DevOps_Engineer":        [4,3,1,5,2,5,4,5,3,1],
"Game_Developer":         [3,2,5,4,3,3,4,3,4,5],
"Product_Manager":        [3,2,4,2,5,3,4,3,5,4]
}

columns = [
"logic","math","creativity","coding","communication",
"patience","curiosity","attention","risk_taking","visualization","role"
]

data = []

def vary(value):
 return max(1, min(5, value + random.choice([-1,0,1])))

for role, center in roles.items():
 for _ in range(30):  
  row = [vary(v) for v in center]
  row.append(role)
  data.append(row)

df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset.csv", index=False)

print("dataset.csv created with", len(df), "rows")
