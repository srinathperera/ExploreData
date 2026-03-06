import pandas as pd
import json

def check_patterns():
    pdf1 = pd.read_csv("data/patterns-data/iter_01.csv")
    pdf1_set = set(pdf1["summarized_patterns"].str.strip().explode())
    print(pdf1.head())
    print(len(pdf1_set))

    pdf2 = pd.read_csv("data/patterns-data/iter_02.csv")
    pdf2_set = set(pdf2["summarized_patterns"].str.strip().explode())
    print(len(pdf2_set))

    pdf3 = pd.read_csv("data/patterns-data/iter_03.csv")
    pdf3_set = set(pdf3["summarized_patterns"].str.strip().explode())
    print(len(pdf3_set))

    #patterns_common_to_all_iters = pdf1_set.intersection(pdf2_set).intersection(pdf3_set)
    #print(len(patterns_common_to_all_iters))

    print("p1_p2_common")
    p1_p2_common = pdf1_set & pdf2_set
    print(len(p1_p2_common))


def check_duplicate_patterns():
    pdf1 = pd.read_json("data/patterns-data/all_patterns.json")
    pdf1["pattern_id"] = pdf1["Pattern Name"].str.lower().str.replace(r'[^0-9a-z]', ' ', regex=True).str.strip()    
    print(len(pdf1))

    pdf1 = pdf1.drop_duplicates(subset=["pattern_id"])
    pdf1.sort_values(by="pattern_id", inplace=True)
    pdf1.to_csv("temp/all_patterns_unique.csv", index=False)
    print(len(pdf1))
    
#check_duplicate_patterns()


def show_clusters():
    df = pd.read_csv("~/Downloads/umap_clusters-nov-07.csv")
    print(df.head())

    print(df.groupby("cluster").apply(lambda x: x["Pattern Name"].tolist()))


#define structure of a cluster
class Cluster:
    def __init__(self, cluster_id: str, short_name: str, sub_patterns: list[str]):
        self.cluster_id = cluster_id
        self.short_name = short_name
        self.patterns = sub_patterns

class CuratedCluster:
    def __init__(self, cluster_id: str, short_name: str, l1_patterns: list[str], l2_patterns: list[str]):
        self.cluster_id = cluster_id
        self.short_name = short_name
        self.l1_patterns = l1_patterns
        self.l2_patterns = l2_patterns

    def to_json(self):
        return {
            "cluster_id": self.cluster_id,
            "short_name": self.short_name,
            "l1_patterns": self.l1_patterns,
            "l2_patterns": self.l2_patterns
        }

def list_clusters():
    file_name = "/Users/srinath/code/ai-patterns/curated_clusters_nov25.json"

    with open(file_name, 'r') as f:
        # json.load() reads the entire file object
        data_list = json.load(f)
        
    all_clusters = []
    for item in data_list:
        print(item["cluster_id"], item["short_name"])
        

def clusters_json_to_csv():
    clustersMap = {}

    file_name = "/Users/srinath/Downloads/cluster_summarizations.json"
    data_list = []


    with open(file_name, 'r') as f:
        # json.load() reads the entire file object
        data_list = json.load(f)
        
    all_clusters = []
    for item in data_list:
        #print(item["cluster_id"], item["short_name"], item["description"])
        all_clusters.append(item["cluster_id"])
        clustersMap[item["cluster_id"]] = Cluster(item["cluster_id"], item["short_name"], item["sub_patterns"])

    duplicates_file_name = "/Users/srinath/Downloads/patterns/duplicates.json"
    duplicates_data_list = []

    with open(duplicates_file_name, 'r') as f:
        duplicates_data_list = json.load(f)

    ducplicate_clusters = []
    for item in duplicates_data_list:
        list = item["patterns_cluster"]
        for cluster in list:
            #print(f"Duplicate Cluster: {cluster}")
            if cluster in ducplicate_clusters:
                print(f"Duplicate cluster in two: {cluster}")
            else:
                ducplicate_clusters.append(cluster)
    
    unduplicated_clusters = []
    for cluster in all_clusters:
        if cluster not in ducplicate_clusters:
            unduplicated_clusters.append(cluster)

    
    print("Unduplicated clusters:")
    for item in data_list:
        if item["cluster_id"] in unduplicated_clusters:
            print(item["cluster_id"], item["short_name"], item["description"])


    curated_clusters = []
    for index, item in enumerate(duplicates_data_list):
        l1_patterns_ids = item["patterns_cluster"]
        l1_patterns = []
        l2_patterns = []
        for pattern_id in l1_patterns_ids:
            cluster_obj = clustersMap[pattern_id]
            l1_patterns.append(cluster_obj.short_name)                        
            for p in cluster_obj.patterns:
                l2_patterns.append(p)

        if "short_name" in item:
            short_name = item["short_name"]
        else:
            short_name = l1_patterns[0]

        curated_clusters.append(CuratedCluster(index, short_name, l1_patterns, l2_patterns).to_json())
    
    #save curated clusters to json
    with open("/Users/srinath/Downloads/patterns/curated_clusters.json", "w") as f:
        json.dump(curated_clusters, f, indent=4)

    print("Done")


def count_l2_patterns():
    file_path = '/Users/srinath/code/ai-patterns/curated_clusters_jan21_26.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    total_count = sum(len(cluster.get("l2_patterns", [])) for cluster in data)
    print(f"Total number of L2 patterns loaded from file: {total_count}")


#clusters_json_to_csv()
#list_clusters()

count_l2_patterns()
