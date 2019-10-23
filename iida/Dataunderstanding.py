import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 200)
data_definition = pd.read_csv('dataset/data_definition.txt', delimiter='\t')
print(data_definition[["項目名","データ種別","項目名（日本語）","データ型"]])