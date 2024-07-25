import cv2
import os
import tqdm
import click


def get_video_info(video_path):
    # Ensure video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    cap.release()
    cv2.destroyAllWindows()

    return fps, frame_count, duration


def get_frames(video_path, output_folder, frame_rate=1, skip_frames=0, max_frames=10000000000):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    _, total_frames, _ = get_video_info(video_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_extracted = 0
    
    # Read frames
    for _ in tqdm.tqdm(range(min(total_frames, max_frames))):
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame
        if frame_count % frame_rate == 0:
            if frame_count >= skip_frames:
                frame_name = f"{output_folder}/{frame_count}.jpg"
                cv2.imwrite(frame_name, frame)
                frames_extracted += 1
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return frames_extracted


@click.command()
@click.argument("video_path")
@click.argument("output_folder")
@click.option("--skip_frames", default=4920, help="Skip frames before extracting frames")
@click.option("--frame_rate", default=10, help="Frame rate to extract frames")
def cli(video_path, output_folder, skip_frames, frame_rate):
    fps, frame_count, duration = get_video_info(video_path)
    print(f"Frame count: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration} seconds")

    frames_extracted = get_frames(video_path, output_folder, frame_rate, skip_frames=skip_frames, max_frames=30_000)
    print(f"Frames extracted: {frames_extracted}")


if __name__ == "__main__":
    # Example usage of the script
    # python3 video_utils.py videos/8BfHVcWR3FI.webm frames/yokohama/ --frame_rate 60
    cli()  # pylint: disable=no-value-for-parameter