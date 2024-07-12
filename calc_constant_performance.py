import numpy as np

from folder_locations import get_data_folder

def retrieve_selected_browning_scores(metadata_path, selected_days, selected_nrs):
    with open(metadata_path) as file:
        metadata = [line.rstrip().split(",") for line in file][1:]
    
    scores = []
    for record in metadata:    
        apple_nr = int(record[0][1:])
        if not (apple_nr in selected_nrs and record[7] in selected_days):
        	continue
        
        label = int(record[5])-1
        scores.append(label)
        
    return np.array(scores)

def calc_mae(scores, label):
    return np.mean(np.abs(scores-label))

def calc_best_mae(scores):
    best_mae = 100000
    for i in range(4):
        mae = calc_mae(scores, i)
        print(f"label = {i}: mae = {mae}")
        if mae < best_mae:
            best_mae = mae
            best_label = i
    return best_label
    
def calc_baseline_mae(metadata_path, selected_days, train_set_nrs, val_set_nrs, test_set_nrs):
    scores_train_detect = retrieve_selected_browning_scores(metadata_path, selected_days, train_set_nrs)
    scores_val_detect = retrieve_selected_browning_scores(metadata_path, selected_days, val_set_nrs)
    scores_test_detect = retrieve_selected_browning_scores(metadata_path, selected_days, test_set_nrs)
    
    best_label = calc_best_mae(scores_train_detect)
    val_mae = calc_mae(scores_val_detect, best_label)
    test_mae = calc_mae(scores_test_detect, best_label)
    return val_mae, test_mae, len(scores_val_detect), len(scores_test_detect)
    
if __name__ == "__main__":
    base_path = get_data_folder()
    metadata_path = base_path / "Brae_browning_score.csv"
    
    selected_days = [
        "2023-01-23 CA storage 10 weeks out 2 weeks",
        "2023-01-30 CA storage 10 weeks out 3 weeks",
        "2023-02-13 CA storage 13 weeks out day 15",
        "2023-03-13 CA storage 18 weeks out day 8",
        "2023-03-20 CA storage 18 weeks out day 15"
    ]
    train_set_nrs = [41, 42, 50, 49, 16, 19, 24, 28, 33, 45, 51, 53, 54, 55, 56,
                     57, 58, 59, 60, 61, 63, 6, 29, 64, 66, 68, 72, 74, 77, 80,
                     81, 82, 85, 22, 34, 35, 46, 67, 70, 71, 73, 75, 76, 78, 79]
    val_set_nrs = [44, 40, 20, 26, 52, 47, 62, 83, 84, 48, 69, 65, 86, 21, 39]
    test_set_nrs = [43, 13, 17, 8, 18, 4, 25, 36, 37, 32, 23, 38, 14, 3, 27]
    
    val_mae, test_mae, _, _ = calc_baseline_mae(metadata_path, selected_days, train_set_nrs, val_set_nrs, test_set_nrs)
    print(f"val_mae = {val_mae}, test_mae = {test_mae}")
    
    val_mae10, test_mae10, num_val10, num_test10 = calc_baseline_mae(metadata_path, ["2023-01-23 CA storage 10 weeks out 2 weeks", "2023-01-30 CA storage 10 weeks out 3 weeks"], train_set_nrs, val_set_nrs, test_set_nrs)
    val_mae13, test_mae13, num_val13, num_test13 = calc_baseline_mae(metadata_path, ["2023-02-13 CA storage 13 weeks out day 15"], train_set_nrs, val_set_nrs, test_set_nrs)
    val_mae18, test_mae18, num_val18, num_test18 = calc_baseline_mae(metadata_path, ["2023-03-13 CA storage 18 weeks out day 8", "2023-03-20 CA storage 18 weeks out day 15"], train_set_nrs, val_set_nrs, test_set_nrs)
    val_mae = (val_mae10*num_val10 + val_mae13*num_val13 + val_mae18*num_val18) / (num_val10+num_val13+num_val18)
    test_mae = (test_mae10*num_test10 + test_mae13*num_test13 + test_mae18*num_test18) / (num_test10+num_test13+num_test18)
    print(f"val_mae = {val_mae}, test_mae = {test_mae}")
    
    train_set_nrs.remove(50)
    
    val_mae, test_mae, _, _ = calc_baseline_mae(metadata_path, ["2023-01-23 CA storage 10 weeks out 2 weeks", "2023-02-13 CA storage 13 weeks out day 15", "2023-03-20 CA storage 18 weeks out day 15"], train_set_nrs, val_set_nrs, test_set_nrs)
    print(f"val_mae = {val_mae}, test_mae = {test_mae}")
    
    val_mae10, test_mae10, num_val10, num_test10 = calc_baseline_mae(metadata_path, ["2023-01-23 CA storage 10 weeks out 2 weeks"], train_set_nrs, val_set_nrs, test_set_nrs)
    val_mae13, test_mae13, num_val13, num_test13 = calc_baseline_mae(metadata_path, ["2023-02-13 CA storage 13 weeks out day 15"], train_set_nrs, val_set_nrs, test_set_nrs)
    val_mae18, test_mae18, num_val18, num_test18 = calc_baseline_mae(metadata_path, ["2023-03-20 CA storage 18 weeks out day 15"], train_set_nrs, val_set_nrs, test_set_nrs)
    val_mae = (val_mae10*num_val10 + val_mae13*num_val13 + val_mae18*num_val18) / (num_val10+num_val13+num_val18)
    test_mae = (test_mae10*num_test10 + test_mae13*num_test13 + test_mae18*num_test18) / (num_test10+num_test13+num_test18)
    print(f"val_mae = {val_mae}, test_mae = {test_mae}")
