from pathlib import Path
import shutil
from datetime import datetime

def find_line_containing(lines, part):
    for i, line in enumerate(lines):
        if line.find(part) != -1:
            return i
    return -1

def check_and_fix_cancelled(lines):
    cancelled_line = find_line_containing(lines, "SCRIPT CANCELLED!")
            
    if cancelled_line == -1:
        return lines, False
        
    summary_line = find_line_containing(lines, "Script summary:")
            
    if summary_line == -1:
        raise ValueError("Script does not contain a summary")
        
    return lines[:cancelled_line] + lines[summary_line:], True
    
    
def extract_position(line):
    start_char = line.find(":")
    end_char = line.find(";")
    return float(line[start_char+1:end_char].strip())
    
def extract_value(lines, value_name):
    line = lines[find_line_containing(lines, f"{value_name} :")]
    start_char = line.find(":")
    return float(line[start_char+1:].strip())
    
def replace_value(lines, value_name, new_value):
    line_i = find_line_containing(lines, f"{value_name} :")
    lines[line_i] = f"{value_name} : {new_value:.6f}\r\n"

def check_and_fix_mag_det_phys(lines):
    #Warning: The VC, HC and COR parameters are not updated, they are unused in FlexBox
    mag_det_line = find_line_containing(lines, "mag_det")
    mag_det_phys_line = find_line_containing(lines, "mag_det_phys")
    if mag_det_phys_line == -1:
        return False
    
    mag_det = extract_position(lines[mag_det_line])
    mag_det_phys = extract_position(lines[mag_det_phys_line])
    if mag_det_phys == mag_det:
        return False
    
    lines[mag_det_line] = lines[mag_det_phys_line].replace("mag_det_phys", "mag_det")
    SDD = extract_value(lines, "SDD")
    SOD = extract_value(lines, "SOD")
    binning = extract_value(lines, "Binning value")
    SDD = SDD + (mag_det_phys - mag_det)
    magnification = SDD/SOD
    voxel_size = (74.8 * binning)/magnification
    
    replace_value(lines, "SDD", SDD)
    replace_value(lines, "Magnification", magnification)
    replace_value(lines, "Voxel size", voxel_size)
    return True

def fix_scan_settings(file_path, *, fix_mag_det_phys, fix_cancelled, backup_name="scan settings backup"):
    file_path = Path(file_path)
    with open(file_path, "r") as settings_file:
        lines = settings_file.readlines()
        
    if fix_cancelled:
        lines, changed = check_and_fix_cancelled(lines)
    else:
        changed = False
    
    if fix_mag_det_phys:
        changed = check_and_fix_mag_det_phys(lines) or changed
        
    if not changed:
        return
    
    # Make absolutely sure no file by that name exists yet
    copy_number = 1
    backup_name_time = f"{backup_name} {datetime.now():%Y-%m-%d %H:%M:%S%z}"
    backup_file_name = f"{backup_name_time}.txt"
    while (file_path.parent.absolute() / backup_file_name).exists():
        copy_number += 1
        backup_file_name = f"{backup_name_time}({copy_number}).txt"
        
    shutil.copy2(file_path, file_path.parent.absolute() / backup_file_name)
    with open(file_path, "w") as settings_file:
        for line in lines:
            settings_file.write(line)
