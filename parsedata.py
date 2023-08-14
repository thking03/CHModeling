"""
Parse data from experiments (in Excel form) into a format that's usable in the model programs.
Also classifies data as complete (PB from wk 5 and wk 12 + BM from wk 14) or incomplete.
NOTE: The default order of data will be [WT, TP53, Tet2]. Additionally, the sort_data() method called at the end of every extract_data() call automatically sorts data w/ the lowest week first.
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

    def add_counts(self, num):
        if self.type == "BM":
            self.bmcount = num
        else:
            raise Exception("Cannot assign BM data to non BM data point.")

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

    def update_bm(self):
        self.sort_data()
        if self.data[-1].type == "BM":
            self.data[-1].add_counts()
        else:
            raise Exception("Cannot assign BM data to non BM data point.")

    def optimize(self, **kwargs):
        from CLVModel import do_CLVopt # Import here to avoid circularity
        # The kwarg that can be passed is interaction_const, which must be spelled as such and is directly passed to do_CLVopt.
        if self.treatment.lower() == "control":
            paramopt = do_CLVopt(self, verbose=False, getloss=True, **kwargs)
            self.opt = True
            self.rates = paramopt[0]
            self.interactions = paramopt[1]
            self.optloss = paramopt[2]
        else:
            raise Exception("No automatic optimize method for treatment data.")

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

def add_bm_counts(sheet, datadict, count_sheet=0, flow_sheet=1):
    df = pd.read_excel(sheet, sheet_name=count_sheet)
    df_bd = pd.read_excel(sheet, sheet_name=flow_sheet)

    # From df_bd make dict of data + HSC %
    bd_dict = {}
    for index, row in df_bd.iterrows():
        pID = re.findall("\d{3}[a-zA-Z]", row["Sample:"])[0].lower()
        bd_dict[pID] = row["HSC %"]/100

    for index, row in df.iterrows():
        pID = re.findall("\d{3}[a-zA-Z]", row["Mouse ID"])[0].lower()
        if pID not in datadict:
            print("Warning: will not assign BM counts to non-existing data for {}".format(pID))
        else:
            datapt = datadict[pID]
            if len(datapt.data) != 3:
                print("Warning: cannot assign BM counts to non-complete data for {}".format(pID))
            elif pID not in bd_dict:
                print("Warning: no HSC percent data available from flow for {}".format(pID))
            else:
                datapt.data[-1].add_counts(row["x10^6"]*10**6*bd_dict[pID])