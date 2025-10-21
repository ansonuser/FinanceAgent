import os
import json
from schema.data import RevenueData
from typing import List, Dict
from collections import defaultdict



def checkcount(src : str, company : str, verbose = True)->List[str]:
    company_path = os.path.join(src, company, "predictions")
    res = []
    filenames = os.listdir(company_path)
    if verbose:    
        print(company_path)
        print("Found number of prediction:", len(os.listdir(company_path)))
    duplicate = []
    period_filename_table = defaultdict(list)
    for filename in filenames:
        path = os.path.join(company_path, filename)
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
        period_filename_table[data["period"]].append(filename)
        if len(period_filename_table[data["period"]]) >= 2:
            duplicate.append(data["period"])
        if not data["period"]:
            res.append(filename)

    print("Found empty period:", len(res))
    print(res)

    time_cache = [v for v in period_filename_table.keys()]
    time_cache.sort()

    if verbose:
        print(time_cache)
        print("Found duplicate:", len(duplicate))
   
    for d in duplicate:
        if verbose:
            print(d)
            print(period_filename_table[d])
        res += period_filename_table[d]
    return res

def checksum(src : str, company : str, verbose = True)->List[str]: 
    res = []

    company_path = os.path.join(src, company)
 
    prediction_folder = os.path.join(company_path, "predictions")
    for file in os.listdir(prediction_folder):
        file_path = os.path.join( prediction_folder, file)
        with open(file_path, "r", encoding='utf-8') as f:
            data = RevenueData(**json.load(f))
        try:
            if abs(sum([v for v in data.product_segments.values() if v]) - data.total_revenue) / (data.total_revenue + 1e-5) < 0.01:
                continue
            else:
                res.append((file_path))
        except:
            res.append(str(data))

    if verbose:
        print("Number of mismatch:", len(res))
        print(res)
    return res 

def get_error_table(companies : List[str], verbose : bool = False):
    error_table = defaultdict(list)
    for company in companies:
        # print("=================================================")
        error_table[company] += checkcount("result", company, verbose = verbose)
        error_table[company] += checksum("result", company, verbose= verbose)
        error_table[company] = [c for c in set(error_table[company])] 
    print("error table:", error_table)
    return error_table


if __name__ == "__main__":
    companies = ["NVDA", "AAPL", "GOOGL", "AMZN", "MSFT"] + ["TSLA"]
    # companies = ["TSLA"]
    error_table = get_error_table(companies, True)
    
   