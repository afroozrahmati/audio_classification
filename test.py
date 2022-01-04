from preprocessing import *

p = preprocessing()
# p.rename_files('.\\data\\physionet\\normal')
# p.rename_files('.\\data\\physionet\\abnormal')
p.load_data('./data/physionet/normal','./data/physionet/abnormal',0,10)