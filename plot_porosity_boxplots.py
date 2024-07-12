import numpy as np
from matplotlib import pyplot as plt
import scipy

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder

def load_csv(path, skip_header=True):
    with open(path, "r") as file:
        lines = file.readlines()
    if skip_header:
        lines = lines[1:]
    return [line.split(",") for line in lines]

def parse_browning_scores(metadata_path):
    with open(metadata_path) as file:
        metadata = [line.rstrip().split(",") for line in file][1:]
    
    scores = {}
    for record in metadata:    
        apple_nr = int(record[0][1:])
        label = int(record[5])-1
        scores[apple_nr] = label
        
    return scores

def select_plot_data(csv_data, required_days, included_days, excluded_days, scores, key):
    apple_days = {}
    for tup in csv_data:
        if not tup[0] in apple_days.keys():
            apple_days[tup[0]] = []
        
        apple_days[tup[0]].append(tup[1])
        
    selected_nrs = []
    for apple_nr in apple_days.keys():
        days = apple_days[apple_nr]
        included = False
        for day in days:
            if day in included_days:
                included = True
        for day in required_days:
            if not day in days:
                included = False
        for day in excluded_days:
            if day in days:
                included = False
        if included:
            selected_nrs.append(apple_nr)

    result = {}
    for nr in selected_nrs:
        result[nr] = ([], scores[int(nr)])
        
    for tup in csv_data:
        if not tup[0] in selected_nrs or not tup[1] in included_days+required_days:
            continue
        result[tup[0]][0].append(float(tup[key]))
        
    return result

if __name__ == "__main__":
    base_path = get_data_folder()
    metadata_path = base_path / "Brae_browning_score.csv"
    porosity_csv_path = get_results_folder() / "2024-03-28_calc_porosity_10" / "porosities.csv"
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    scores = parse_browning_scores(metadata_path)
    
    csv_data = load_csv(porosity_csv_path, skip_header=True)
    
    day0 = "2022-11-02 day 1" 
    
    included_days = [
        "2023-02-13 CA storage 13 weeks out day 15",
        "2023-03-13 CA storage 18 weeks out day 8",
        "2023-03-20 CA storage 18 weeks out day 15"
    ]
    

    data_whole = select_plot_data(csv_data, [day0], included_days, [], scores, 4)
    score_data_whole = {0 : [], 1 : [], 2 : [], 3 : []}
    for val, score in data_whole.values():
        score_data_whole[score].append(val[-1])
    print(len(data_whole))
    
    data_change = select_plot_data(csv_data, [day0], included_days, [], scores, 4)
    score_data_change = {0 : [], 1 : [], 2 : [], 3 : []}
    for val, score in data_change.values():
        score_data_change[score].append(val[-1]-val[0])
    print(len(data_change))
    
    data_inner = select_plot_data(csv_data, [day0], included_days, [], scores, 5)
    score_data_inner = {0 : [], 1 : [], 2 : [], 3 : []}
    for val, score in data_inner.values():
        score_data_inner[score].append(val[-1])
    print(len(data_inner))
    
    data_change_inner = select_plot_data(csv_data, [day0], included_days, [], scores, 5)
    score_data_change_inner = {0 : [], 1 : [], 2 : [], 3 : []}
    for val, score in data_change_inner.values():
        score_data_change_inner[score].append(val[-1]-val[0])
    print(len(data_change_inner))
    
    states = ["1", "2", "3", "4"]
    labels = [["a", "ab", "b", "b"], ["a", "ab", "c", "bc"], ["a", "b", "b", "b"], ["a", "b", "c", "c"]]
    
    anova_results = []

    for d in [score_data_whole, score_data_change, score_data_inner, score_data_change_inner]:
        hsd_result = scipy.stats.tukey_hsd(*(d.values()))
        print(hsd_result)
        
        anova_result = scipy.stats.f_oneway(*(d.values()))
        print(anova_result.pvalue)
        anova_results.append(anova_result.pvalue)

        
    limits = []
    for data in [score_data_whole, score_data_change, score_data_inner, score_data_change_inner]:
        values = np.concatenate([data[key] for key in data.keys()])
        vmin = np.min(values)
        vmax = np.max(values)
        vrange = vmax-vmin 
        limits.append([vmin, vmax, vrange])
    
    plt.figure(figsize=(12,3.1))
    plt.subplot(141)
    plt.boxplot(score_data_whole.values(), labels=states)
    plt.title(f"Whole apple porosity \n (ANOVA p={anova_results[0]:.5f})")
    plt.ylim([limits[0][0]-limits[0][2]*0.05, limits[0][1]+limits[0][2]*0.165])
    for i, l in enumerate(labels[0]):
        plt.text(i+1-(0.06*len(l)), limits[0][1]+limits[0][2]*0.067, l)
    plt.gca().grid(axis='y')
    plt.xlabel("Browning score")
    
    plt.subplot(142)
    plt.boxplot(score_data_change.values(), labels=states)
    plt.title(f"Change in whole apple porosity \n (ANOVA p={anova_results[1]:.5f})")
    plt.ylim([limits[1][0]-limits[1][2]*0.05, limits[1][1]+limits[1][2]*0.165])
    for i, l in enumerate(labels[1]):
        plt.text(i+1-(0.06*len(l)), limits[1][1]+limits[1][2]*0.067, l)
    plt.gca().grid(axis='y')
    plt.xlabel("Browning score")
    
    plt.subplot(143)
    plt.boxplot(score_data_inner.values(), labels=states)
    plt.title(f"Inner apple porosity \n (ANOVA p={anova_results[1]:.5f})")
    plt.ylim([limits[2][0]-limits[2][2]*0.05, limits[2][1]+limits[2][2]*0.165])
    for i, l in enumerate(labels[2]):
        plt.text(i+1-(0.06*len(l)), limits[2][1]+limits[2][2]*0.067, l)
    plt.gca().grid(axis='y')
    plt.xlabel("Browning score")
    
    plt.subplot(144)
    plt.boxplot(score_data_change_inner.values(), labels=states)
    plt.title(f"Change in inner apple porosity \n (ANOVA p={anova_results[3]:.5f})")
    plt.ylim([limits[3][0]-limits[3][2]*0.05, limits[3][1]+limits[3][2]*0.165])
    for i, l in enumerate(labels[3]):
        plt.text(i+1-(0.06*len(l)), limits[3][1]+limits[3][2]*0.067, l)
    plt.gca().grid(axis='y')
    plt.xlabel("Browning score")
    
    plt.tight_layout()
    plt.savefig(experiment_folder / "porosity_significance.png")
    plt.savefig(experiment_folder / "porosity_significance.eps")
    plt.close()
    
    
        
