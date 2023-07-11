"""
Parse data from experiments (in Excel form) into a format that's usable in the model programs.
Also classifies data as complete (PB from wk 5 and wk 12 + BM from wk 14) or incomplete.
NOTE: The default order of data will be [WT, TP53, Tet2]
"""

import pandas as pd
import re

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

    def check_complete(self):
        condition = 0
        if condition:
            self.complete = True
        else:
            self.complete = False

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

    def optimize(self):
        pass

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
        if row["ID"] in datadict:
            # Raise exceptions when data types don't math
            datapt = datadict[pID]
            if datapt.type != row["transplant"]:
                print("Warning: Conflicting transplant types for {name}: {t1} and {t2}!".format(name=datapt.name, t1=datapt.type, t2=row["transplant"]))
            if datapt.treatment != row["treatment"]:
                print("Warning: Conflicting treatment types for {name}: {t1} and {t2}!".format(name=datapt.name, t1=datapt.treatment, t2=row["treatment"]))
            datapt.extract_data(clsn, row)
        else:
            datapt = CHData(pID)
            datapt.extract_data(clsn, row)
        datadict[pID] = datapt
        
def assign_complete():
    pass
