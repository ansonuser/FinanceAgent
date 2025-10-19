import os
import json
import click

def check_transformation(src : str):
    companies = os.listdir(src)
    meta = []
    print("**Security Check**")
    print("=========START=====================")
    for company in companies:
        missing_report = []
        result = {"company" : company}
        result["missing"] = []
        company_path = os.path.join(src, company)
        date_folder = os.listdir(company_path)
        result["num_folder"] = len(date_folder)
        for folder in date_folder:
            parent = os.path.join(company_path, folder)
            if not any([i for i in os.listdir(parent) if i.endswith("json")]):
                result["missing"].append(folder)
        result["missing_ratio"] = 1 - (result["num_folder"] - len(result["missing"])) / result["num_folder"]
        if result["missing_ratio"] != 0:
            missing_report.append(result)
            print(f"[Missing Alert]: -------{company}-----------\n")
            print("[INFO]:", missing_report)
        meta.append(result)
    with open("data_source_quality_report.json", "w") as f:
        json.dump(meta, f)
    if len(missing_report) == 0:
        print("[INFO] Quality Pass! No missing data.")
    print("[END] Data Source Check Done!")
    print("=========END=====================")

@click.command()
@click.option("--target", default ="data_source", help = "Check if data source has defect.")
@click.option('--src', default = "preprocessed", help = "Location or data source or target")
def main(target : str, src : str):
    if target == "data_source":
        check_transformation(src)



if __name__ == '__main__':
    main()
