from typing import Optional, Tuple, List
from pathlib import Path

import tqdm
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data_loading import RegressionTaskData
from pretrained_resnet import ResNetRegression


def get_image_names(image_dir: Path, extensions: List[str] = ['jpg', 'png']):
    """
    This function returns a list of image names in the image_dir
    """
    glob_str = f'*.[{"|".join(extensions)}]'
    return sorted([str(x) for x in image_dir.glob(glob_str)])


def path_target_generator(image_dir: Path):
    """
    This function returns a map that is back to the image path
    """
    image_names = get_image_names(image_dir)
    return {image_name: image_name for image_name in image_names}


def run_model_on_images(model_name: Path, image_dir: Path, image_size: Tuple[int, int, int] = (1, 100, 100)):
    """
    This function runs the model on the images in the image_dir and saves the results to a csv file
    It runs the model on the GPU
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=grayscale, resize_size=resize_size)

    # Load the model, check if it is a resnet model
    if 'resnet' in model_name:
        model = ResNetRegression.load(model_name)
        model.eval()
    else:
        raise ValueError('Model name must contain resnet')
    
    def enable_dropout(mod: nn.Module):
        if isinstance(mod, nn.Dropout):
            mod.train()

    # Enable dropout for uncertainty estimation
    model.apply(enable_dropout)

    # We need to move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model.to(device)

    # Delete the existing results file if it exists
    if Path('results.csv').exists():
        Path('results.csv').unlink()

    # Run the model on the data and store the results in a csv file
    with torch.no_grad():
        evaluation_loader = regression_task.make_dataloader(image_dir, path_target_generator(image_dir / 'images'))
        for inputs, image_path_batch in tqdm.tqdm(evaluation_loader):

            n_mc_evals = 5
            output_totals = np.zeros((len(inputs), 2))
            outputs_vars = np.zeros((len(inputs), 2))
            for i in range(n_mc_evals):
                outputs = model(inputs.to(device))
                outputs_np = outputs.detach().cpu().numpy()
                output_totals += outputs_np
                outputs_vars += outputs_np**2
            outputs_mean = output_totals/(1 + n_mc_evals)
            outputs_vars = outputs_vars/(1 + n_mc_evals) - outputs_mean**2

            # Store the result of this batch in a csv file and each of the image names
            with open('results.csv', 'a') as f:
                for op_mu, op_var, image_name in zip(outputs_mean, outputs_vars, image_path_batch):
                    f.write(f'{image_name},{op_mu[0]},{op_mu[1]},{op_var[0]},{op_var[1]}\n')


def symlink_images_to_evalutation_folder(image_dir: Path):
    """
    Symlinks the images in image_dir to the evaluation folder
    The evaluation folder is the folder that the model will run on, it is example_dataset/evaluation
    """
    evaluation_dir = Path('example_dataset/evaluation/images')
    if not evaluation_dir.exists():
        evaluation_dir.mkdir(exist_ok=True, parents=True)
    for image_name in get_image_names(image_dir):
        if not (evaluation_dir / Path(image_name).name).exists():
            (evaluation_dir / Path(image_name).name).symlink_to(image_name)


def map_sun_fl_to_az_degs(sun_f: np.ndarray, sun_l: np.ndarray) -> np.ndarray:
    """
    Convert the sun vector in FL coordinates to azimuth angle in degrees
    """
    return np.rad2deg(np.arctan2(sun_l, sun_f))


def jacobian_sun_fl_to_az_rads(sun_f: np.ndarray, sun_l: np.ndarray) -> np.ndarray:
    """
    Compute the jacobian of the sun vector in FL coordinates to the azimuth angle in rads
    """
    return np.array([
        -sun_l / (sun_f**2 + sun_l**2),
        sun_f / (sun_f**2 + sun_l**2)
    ])


def transform_fl_std_to_az_degs_std(sun_f: np.ndarray, sun_l: np.ndarray, sun_f_std: np.ndarray, sun_l_std: np.ndarray) -> np.ndarray:
    """
    This function transforms the standard deviation of the sun vector in FL coordinates to the standard deviation of the azimuth angle in degrees
    """
    jacobian = jacobian_sun_fl_to_az_rads(sun_f, sun_l)
    sun_f_var = sun_f_std**2
    sun_l_var = sun_l_std**2
    variances = np.array([sun_f_var, sun_l_var])
    transformed_variances = np.sum((jacobian**2) * variances, axis=0)
    return np.rad2deg(np.sqrt(transformed_variances))


def plot_results_csv(gt_csv: Optional[str] = None):
    """
    This function plots the results csv file
    """
    results = pd.read_csv('results.csv', header=None, names=['image_name', 'sun_f', 'sun_l', 'sun_f_var', 'sun_l_var'])
    # split off the .jpg
    results['image_ind'] = results['image_name'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))
 
    results = results.set_index('image_ind')
    results = results.sort_index()

    plt.figure()
    results['sun_f'].plot()
    results['sun_l'].plot()
    # results['sun_u'].plot()
    # Add a legend
    plt.legend(['sun_f', 'sun_l'])

    plt.figure()
    # Compute a running average of the sun f and sun l
    # and compute local standard deviation
    results['sun_f_avg'] = results['sun_f'].rolling(window=10).mean()
    results['sun_l_avg'] = results['sun_l'].rolling(window=10).mean()
    results['sun_f_std'] = results['sun_f'].rolling(window=10).std()
    results['sun_l_std'] = results['sun_l'].rolling(window=10).std()
    results['sun_f_avg'].plot()
    results['sun_l_avg'].plot()
    # Add the standard deviation as error bars
    plt.fill_between(results.index, results['sun_f_avg'] - results['sun_f_std'], results['sun_f_avg'] + results['sun_f_std'], alpha=0.5)
    plt.fill_between(results.index, results['sun_l_avg'] - results['sun_l_std'], results['sun_l_avg'] + results['sun_l_std'], alpha=0.5)
    az_degs_std = transform_fl_std_to_az_degs_std(results['sun_f'], results['sun_l'], results['sun_f_std'], results['sun_l_std'])
    results['az_degs_std'] = az_degs_std
    plt.legend(['sun_f_avg', 'sun_l_avg'])
    plt.xlabel('Image index')
    plt.ylabel('Sun vector component')
    plt.minorticks_on()
    plt.grid(which='both')

    # Calculate the azimuth and zenith angles from the sun vector
    plt.figure()
    results['az_deg'] = np.rad2deg(np.arctan2(results['sun_l'], results['sun_f']))
    # results['zen_deg'] = np.rad2deg(np.arccos(results['sun_u']))
    results['az_deg'].plot()
    # Add az_degs_std as error bars
    plt.fill_between(results.index, results['az_deg'] - az_degs_std, results['az_deg'] + az_degs_std, alpha=0.5)
    # Load train.csv
    if gt_csv:
        gt_csv = Path(gt_csv)
        if not gt_csv.exists():
            raise FileNotFoundError(f'Could not find {gt_csv}')
        gt_data = pd.read_csv(gt_csv)
        gt_data['az_deg'] = np.rad2deg(np.arctan2(gt_data['sun_l'], gt_data['sun_f']))
        # gt_data['zen_deg'] = np.rad2deg(np.arccos(gt_data['sun_u']))
        gt_data['image_ind'] = gt_data['image_name'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))
        gt_data = gt_data.set_index('image_ind')
        gt_data['az_deg'].plot()
        plt.legend(['predicted', 'actual'])
        plt.tight_layout()
    plt.xlabel('Image index')
    plt.ylabel('Azimuth angle degs')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.title('Azimuth angle of sun vector')
    # Set limits to +- 200 degrees
    plt.ylim([-200, 200])

    plt.figure()
    results['sun_f'].plot()
    results['sun_l'].plot()
    if gt_csv:
        gt_data['sun_f'].plot()
        gt_data['sun_l'].plot()
        plt.legend(['sun_f_pred', 'sun_l_pred', 'sun_f_gt', 'sun_l_gt'])
    else:
        plt.legend(['sun_f_pred', 'sun_l_pred'])
    plt.xlabel('Image index')
    plt.ylabel('Sun vector component')
    plt.minorticks_on()
    plt.grid(which='both')

    # plt.figure()
    # # plt.plot(np.sqrt(results['sun_f']**2 + results['sun_l']**2 + results['sun_u']**2))
    # plt.title('Norm of sun vector')
    # plt.minorticks_on()
    # plt.grid(which='both')

    plt.figure()
    results['az_degs_std'].plot()
    results['az_degs_pred_std'] = transform_fl_std_to_az_degs_std(
        results['sun_f'], 
        results['sun_l'], 
        np.sqrt(results['sun_f_var']), 
        np.sqrt(results['sun_l_std'])
    )
    results['az_degs_pred_std'].plot()
    plt.legend(['inter frame computed', 'predicted'])
    plt.title('Approx angle std dev')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.xlabel('Image index')
    plt.ylabel('Azimuth angle std dev degs')

    plt.show()



@click.command()
@click.option("--model_name", default="3_224_224_resnet_baseline_5_all.pth", help="The name of the model to run")
@click.option("--image_dir", default="evaluation/", help="The directory containing the images to run the model on")
@click.option("--gt_csv", default=None, help="The csv file containing the ground truth sun vectors")
def cli(model_name, image_dir, gt_csv):
    # Remove the evaluation directory
    if Path('example_dataset/evaluation/images').exists():
        for file in Path('example_dataset/evaluation/images').glob('*'):
            file.unlink()
        Path('example_dataset/evaluation/images').rmdir()
    symlink_images_to_evalutation_folder(Path(image_dir))
    
    # Everything before the 3rd underscore is the model size
    model_size = tuple(map(int, model_name.split('_')[:3]))
    run_model_on_images(
        model_name, 
        Path('example_dataset/evaluation'),
        image_size=model_size
    )
    plot_results_csv(gt_csv)


if __name__ == '__main__':
    cli()
    # Example usage:
    # python3 run_model_on_images.py --model_name 1_100_100_miami.pth --image_dir /home/hugo/youtube_odometry/frames/cape_town/ --gt_csv example_dataset/cape_town.csv