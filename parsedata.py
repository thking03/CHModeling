"""
Parse data from experiments (in Excel form) into a format that's usable in the model programs.
Also classifies data as complete (PB from wk 5 and wk 12 + BM from wk 14) or incomplete.
NOTE: The default order of data will be [WT, TP53, Tet2]. Additionally, the sort_data() method called at the end of every extract_data() call automatically sorts data w/ the lowest week first.
"""

import pandas as pd
import re
from CLVModel import *

class DataProfile:
    def __init__(self):
        pass

    def fill(self, info, data):
        self.week = info[0]
        self.type = info[1] # BM or PB
        self.probs = data

class CHData:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.opt = False

    def extract_data(self, clsn, line):
        
        if "transplant" in line.keys():
            self.type = line["transplant"] # Options are 1XTet2, 1XTP53, 2X
        else:
            self.type = "2X"
        self.treatment = line["treatment"] # Options are control, cisplatin

        if clsn == "BM":
            wk = 14
            dpf = DataProfile()
            data = [(100-line["TP53"]-line["Tet2"]), line["TP53"], line["Tet2"]]
            dpf.fill([wk, clsn], data)
            self.data.append(dpf)
        
        else:
            wk = 5
            dpf = DataProfile()
            data = [(100-line["week 5 TP53"]-line["week 5 Tet2"]), line["week 5 TP53"], line["week 5 Tet2"]]            
            dpf.fill([wk, clsn], data)
            self.data.append(dpf)

            wk = 12
            dpf = DataProfile()
            data = [(100-line["week 12 TP53"]-line["week 12 Tet2"]), line["week 12 TP53"], line["week 12 Tet2"]]
            dpf.fill([wk, clsn], data)
            self.data.append(dpf)
        
        self.sort_data()

    def sort_data(self):
        self.data.sort(key = lambda x: x.week)

    def optimize(self, **kwargs):
        # The kwarg that can be passed is interaction_const, which must be spelled as such and is directly passed to do_CLVopt.
        paramopt = do_CLVopt(self, verbose=False, getloss=True, **kwargs)
        self.opt = True
        self.rates = paramopt[0]
        self.interactions = paramopt[1]
        self.optloss = paramopt[2]

def readdata(sheet, datadict, sheet_name=0):
    df = pd.read_excel(sheet, sheet_name=sheet_name)
    df = df.fillna(method="ffill")
    keys = df.keys()

    if "BM" in sheet_name:
        clsn = "BM"
    elif "PB" in sheet_name:
        clsn = "PB"
    else:
        raise Exception("Unknown measurement type.")

    for index, row in df.iterrows():
        pID = re.findall("\d{3}[a-zA-Z]", row["ID"])[0].lower()
        if pID in datadict:
            datapt = datadict[pID]
            datapt.extract_data(clsn, row)
        else:
            print("Making new CHData {}".format(pID))
            datapt = CHData(pID)
            datapt.extract_data(clsn, row)
        datadict[pID] = datapt
        
def assign_complete():
    pass
