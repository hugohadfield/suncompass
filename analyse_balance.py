
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click

# Load a dataset csv file and plot a histogram of the azimuth angles of the sun vectors
# This is to check that the sun vectors are balanced
def analyse_balance(dataset_csv):
    """
    Load a dataset csv file and plot a histogram of the azimuth angles of the sun vectors
    This is to check that the sun vectors are balanced
    """
    dataset = pd.read_csv(dataset_csv)
    sun_vecs_f = dataset['sun_f']
    sun_vecs_l = dataset['sun_l']
    
    # Plot the sun vector azimuth angle
    azimuth_angle_degs = np.rad2deg(np.arctan2(sun_vecs_l, sun_vecs_f))
    # plt.plot(dataset['timestamp_unixus'], azimuth_angle_degs)
    # plt.xlabel('Timestamp')
    # plt.ylabel('Azimuth Angle (degs)')

    print(f"Total number of sun vectors: {len(azimuth_angle_degs)}")

    plt.hist(azimuth_angle_degs, bins=100)
    plt.xlabel('Azimuth Angle (degs)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Azimuth Angles of Sun Vectors')
    plt.show()


@click.command()
@click.option("--dataset_csv", default="manual_labels.csv", help="The csv file containing the dataset")
def cli(dataset_csv):
    analyse_balance(dataset_csv)


if __name__ == "__main__":
    cli()
    # Example usage: python3 analyse_balance.py --dataset_csv example_dataset/train.csv
