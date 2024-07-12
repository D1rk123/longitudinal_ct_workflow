import ct_experiment_utils as ceu
import numpy as np
import skimage.transform
import skimage.filters
import skimage.measure
import SimpleITK as sitk

from folder_locations import get_results_folder, get_data_folder
from fix_scan_settings import extract_value

def load_padded(path, end_resolution):
    img = ceu.load_stack(path, prefix="", dtype=np.float32, stack_axis=0)
    np.nan_to_num(img, copy=False, nan=0.0, posinf=3, neginf=-3)
    img = np.clip(img, -3, 3)
    result = np.zeros(end_resolution, dtype=np.float32)
    offset = (np.array(result.shape)-np.array(img.shape))//2
    end = offset + np.array(img.shape)
    result[offset[0]:end[0], offset[1]:end[1], offset[2]:end[2]] = img
    return result

def np_to_sitk(np_img, voxel_size):
    sitk_img = sitk.GetImageFromArray(np_img, isVector=False)
    sitk_img.SetSpacing(np.array(sitk_img.GetSpacing())*(voxel_size/134.326786))
    return sitk_img

def make_sitk_imgs_downsampled(np_img, voxel_size):
    np_img_d2 = skimage.transform.downscale_local_mean(np_img, (2, 2, 2))
    np_img_d4 = skimage.transform.downscale_local_mean(np_img_d2, (2, 2, 2))
    
    sitk_img = np_to_sitk(np_img, voxel_size)
    sitk_img_d2 = sitk.GetImageFromArray(np_img_d2, isVector=False)
    sitk_img_d2.SetSpacing(np.array(sitk_img.GetSpacing())*2)
    sitk_img_d4 = sitk.GetImageFromArray(np_img_d4, isVector=False)
    sitk_img_d4.SetSpacing(np.array(sitk_img.GetSpacing())*4)
    
    return sitk_img, sitk_img_d2, sitk_img_d4

def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue()} : {method.GetOptimizerPosition()}")


def registerLBFGS(fixed, moving, init_transf, num_iters, fraction=0.1, accuracy=1e-5, ls_accuracy=1e-4):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingPercentage(percentage=fraction, seed=sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsLBFGS2(numberOfIterations=num_iters, solutionAccuracy=accuracy, lineSearchAccuracy=ls_accuracy)
    R.SetInitialTransform(init_transf)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    transf_out = R.Execute(fixed, moving)
    print("-------")
    print(transf_out)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    return transf_out, R.GetMetricValue()

def registerGD(fixed, moving, init_transf, num_iters, fraction=0.1, lr=1e-1, gm_tol=1e-8):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingPercentage(percentage=fraction, seed=sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=lr, minStep=1e-10, numberOfIterations=num_iters, relaxationFactor=0.5, gradientMagnitudeTolerance=gm_tol)
    R.SetInitialTransform(init_transf)
    R.SetOptimizerScalesFromJacobian()
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    transf_out = R.Execute(fixed, moving)
    print("-------")
    print(transf_out)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    return transf_out, R.GetMetricValue()

def resample(fixed, moving, transf, interpolator):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transf)

    return resampler.Execute(moving)

def calc_bounding_box(recon):
    #use otsu thresholding to segment the apple
    threshold = skimage.filters.threshold_otsu(recon)
    mask = recon > threshold
    
    # find the largest connected component
    # https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    labels = skimage.measure.label(mask)
    assert( labels.max() != 0 ) # assume at least 1 CC
    mask = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    
    coords = np.stack(np.nonzero(mask))
    min_vec = np.min(coords, axis=1)
    max_vec = np.max(coords, axis=1)
    return min_vec, max_vec
    
def find_transform(np_img1, voxel_size1, np_img2, voxel_size2):
    img1, img1_d2, img1_d4 = make_sitk_imgs_downsampled(np_img1, voxel_size1)
    img2, img2_d2, img2_d4 = make_sitk_imgs_downsampled(np_img2, voxel_size2)
    
    transforms = []
    errors = []

    for angle in np.linspace(0, 2*np.pi, 10, endpoint=False):
        init_transf = sitk.Euler3DTransform(
        sitk.CenteredTransformInitializer(
            img1_d4,
            img2_d4,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS))
        init_transf.SetRotation(0, 0, angle)
        transf, error = registerLBFGS(img1_d4, img2_d4, init_transf, 200, fraction=0.1)
        transf, error = registerLBFGS(img1_d2, img2_d2, transf, 200, fraction=0.1)
        transforms.append(transf)
        errors.append(error)
    
    print("finished initialization")
    print(f"errors = {errors}")
    transf = transforms[np.argmin(errors)]
    
    transf = sitk.Euler3DTransform(transf)
    similarity = sitk.Similarity3DTransform()
    similarity.SetTranslation(transf.GetTranslation())
    similarity.SetCenter(transf.GetCenter())
    similarity.SetMatrix(transf.GetMatrix())
    transf = similarity
    
    transf, error = registerGD(img1_d4, img2_d4, transf, 300, fraction=0.3)
    transf, error = registerGD(img1_d2, img2_d2, transf, 300, fraction=0.3)
    transf, error = registerGD(img1, img2, transf, 300, fraction=0.3)
    
    return transf


def transform_img(img, transf, sitk_img1, interpolation, voxel_size):
    if transf is not None:
        img = np_to_sitk(img, voxel_size)
        img = resample(sitk_img1, img, transf, interpolation)
        img = sitk.GetArrayFromImage(img)
    return img 

def load_and_transform_img(r_path, transf, sitk_img1, interpolation, voxel_size):
    img = load_padded(r_path, (760, 760, 760))
    return transform_img(img, transf, sitk_img1, interpolation, voxel_size)

def get_voxel_size(r_path, projections_folder, recons_folder):
    p_path = projections_folder / r_path.relative_to(recons_folder)
    p_path = p_path.parent / p_path.name.replace("_recon", "")
    settings_path = p_path / "scan settings.txt"
    with open(settings_path, "r") as settings_file:
        lines = settings_file.readlines()
    return extract_value(lines, "Voxel size")
    

if __name__ == "__main__":
    base_path = get_data_folder()
    recons_folder = base_path / "recons_bh_corr_crop"
    projections_folder = base_path / "projections"
    registrations_folder = base_path / "recons_bh_corr_registered_crop"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    transforms_folder = experiment_folder / "transforms"
    transforms_folder.mkdir()
    
    day_folders = sorted([d for d in recons_folder.iterdir() if d.is_dir()])
    
    for apple_nr in range(1,87):
        recon_paths = [d / f"{apple_nr}_recon/" for d in day_folders if (d / f"{apple_nr}_recon/").exists()]
        if not len(recon_paths)<2:
            continue
        
        voxel_sizes =  [get_voxel_size(r_path, projections_folder, recons_folder) for r_path in recon_paths]
        print(voxel_sizes)
        transforms = [None]
        
        np_img1 = load_padded(recon_paths[0], (760, 760, 760))
        sitk_img1 = np_to_sitk(np_img1, voxel_sizes[0])
        min_vec_total = np.array((760, 760, 760), dtype=int)
        max_vec_total = np.array((0,0,0), dtype=int)
        
        for r_path, voxel_size in zip(recon_paths, voxel_sizes):
            print(r_path)
            if r_path != recon_paths[0]:
                np_img2 = load_padded(r_path, (760, 760, 760))
                transforms.append(find_transform(np_img1, voxel_sizes[0], np_img2, voxel_size))
                np_img2_t = transform_img(np_img2, transforms[-1], sitk_img1, sitk.sitkLinear, voxel_size)
            else:
                np_img2_t = np_img1
            min_vec, max_vec = calc_bounding_box(np_img2_t)
            min_vec_total = np.minimum(min_vec_total, min_vec)
            max_vec_total = np.maximum(max_vec_total, max_vec)
            
        del np_img1
            
        min_vec_total = np.maximum(min_vec_total-10, 0)
        max_vec_total = np.minimum(max_vec_total+10, 760)
        print(min_vec_total)
        print(max_vec_total)
        
        transform_folder = transforms_folder / f"{apple_nr}"
        transform_folder.mkdir()
        with open(transform_folder / "extends.txt", "w") as extends_file:
            extends_file.write(f"{min_vec_total[0]},{min_vec_total[1]},{min_vec_total[2]}\n")
            extends_file.write(f"{max_vec_total[0]},{max_vec_total[1]},{max_vec_total[2]}\n")
        
        for r_path, transf, voxel_size in zip(recon_paths, transforms, voxel_sizes):
            img = load_and_transform_img(r_path, transf, sitk_img1, sitk.sitkBSpline, voxel_size)
            crop = img[min_vec_total[0]:max_vec_total[0], min_vec_total[1]:max_vec_total[1], min_vec_total[2]:max_vec_total[2]]
            ceu.save_stack(registrations_folder / f"{apple_nr}" / r_path.parent.name, crop, parents=True)
            if transf is not None:
                sitk.WriteTransform(transf, str(transform_folder / (r_path.parent.name + ".txt")))
        
    
    
