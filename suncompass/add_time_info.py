
import click
import json
import os


def get_frame_numbers(frame_folder: str):
    # Get the frame names
    frame_names = os.listdir(frame_folder)

    # Ensure the frame names are not empty
    if not frame_names:
        raise ValueError(f"No frames found in folder: {frame_folder}")

    # Get the frame numbers
    frame_numbers = [int(frame_name.split('.')[0]) for frame_name in frame_names]
    frame_numbers.sort()

    return frame_numbers


def add_time_info(json_file: str, frame_folder: str, frame_rate_fps: int):
    # Ensure the json file exists
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Json file not found: {json_file}")
    
    # Load the json file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Get the start timestamp
    start_timestamp = int(data['timestamp_unixus'])

    # Remove all the frames
    data['frames'] = []

    # Get the frame numbers
    frame_numbers = get_frame_numbers(frame_folder)

    # Add the frame numbers and timestamps
    for frame_number in frame_numbers:
        # Calculate the timestamp
        timestamp_us = int(start_timestamp + (frame_number / frame_rate_fps) * 1e6)
        frame = {
            "timestamp_unixus": timestamp_us,
            "frame_number": frame_number
        }
        data['frames'].append(frame)

    # Save the json file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


@click.command()
@click.argument("json_file")
@click.argument("frame_folder")
@click.argument("frame_rate_fps")
def add_time_info_cli(json_file: str, frame_folder: str, frame_rate_fps: int):
    add_time_info(json_file, frame_folder, int(frame_rate_fps))


if __name__ == "__main__":
    add_time_info_cli()

    # Example usage:
    # python3 add_time_info.py ../meta_info/dublin.json ../frames/dublin 30
