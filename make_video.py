import os
import cv2
import numpy as np
from tqdm import tqdm

# --- Helper function for resizing to target_height while keeping aspect ratio ---
def resize_to_target_height(img, target_height):
    """
    Resize 'img' so that its height == target_height, preserving aspect ratio.
    """
    original_height, original_width = img.shape[:2]
    scale_factor = target_height / float(original_height)
    new_width = int(round(original_width * scale_factor))
    resized_img = cv2.resize(
        img, (new_width, target_height), interpolation=cv2.INTER_AREA
    )
    return resized_img

def create_video_from_images(root_image_dir, 
                             output_video_path,
                             start_frame_id=None, 
                             end_frame_id=None,
                             fps=30,
                             target_height=512):
    """
    Reads images from `root_image_dir/contact` and `root_image_dir/rgb_hand`, 
    merges them side-by-side (horizontally), and writes out a video file.

    :param root_image_dir: Path to the root directory containing 'contact' and 'rgb_hand' subdirs
    :param output_video_path: Filename (path) for the resulting video (e.g. 'output.mp4')
    :param start_frame_id: The minimum timestamp (integer) to include (None means no lower bound)
    :param end_frame_id: The maximum timestamp (integer) to include (None means no upper bound)
    :param fps: Frames per second for the output video
    """

    # Directories
    contact_dir = os.path.join(root_image_dir, "contact")
    rgb_hand_dir = os.path.join(root_image_dir, "rgb_hand")
    
    # Safety checks
    if not os.path.isdir(contact_dir):
        raise FileNotFoundError(f"Directory not found: {contact_dir}")
    if not os.path.isdir(rgb_hand_dir):
        raise FileNotFoundError(f"Directory not found: {rgb_hand_dir}")

    # Get image files
    contact_files = [f for f in os.listdir(contact_dir) if f.endswith(".png")]
    rgb_hand_files = [f for f in os.listdir(rgb_hand_dir) if f.endswith(".jpg")]

    # Parse timestamps (strip .jpg, convert to int)
    contact_timestamps = set(int(os.path.splitext(f)[0]) for f in contact_files)
    rgb_hand_timestamps = set(int(os.path.splitext(f)[0]) for f in rgb_hand_files)

    # print(contact_timestamps)
    # print(rgb_hand_timestamps)
    # common_timestamps = []

    # for ts in contact_timestamps:
    #     if (ts in rgb_hand_timestamps):
    #         common_timestamps.append(ts)

    # Find common timestamps
    common_timestamps = sorted(contact_timestamps.intersection(rgb_hand_timestamps))

    # Filter by start_frame_id / end_frame_id
    if start_frame_id is not None:
        common_timestamps = [ts for ts in common_timestamps if ts >= start_frame_id]
    if end_frame_id is not None:
        common_timestamps = [ts for ts in common_timestamps if ts <= end_frame_id]

    if not common_timestamps:
        raise ValueError("No common timestamps found within the specified frame range.")

    # Read the first pair to determine video size
    first_ts = common_timestamps[0]
    first_contact_path = os.path.join(contact_dir, f"{first_ts}.png")
    first_rgb_hand_path = os.path.join(rgb_hand_dir, f"{first_ts}.jpg")

    first_contact_img = cv2.imread(first_contact_path)
    first_rgb_hand_img = cv2.imread(first_rgb_hand_path)

    if first_contact_img is None or first_rgb_hand_img is None:
        raise ValueError(f"Failed to read the first pair of images: {first_ts}.jpg")

    # Resize both images to have the same height (target_height)
    first_contact_img = first_contact_img[10:-10, 10:-10, ...]
    resized_contact_1 = resize_to_target_height(first_contact_img, target_height)
    resized_rgb_hand_1 = resize_to_target_height(first_rgb_hand_img, target_height)

    # Concatenate horizontally to find final frame shape
    first_concat = np.hstack((resized_contact_1, resized_rgb_hand_1))
    height, width, channels = first_concat.shape

    # Create the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write the first frame
    out.write(first_concat)

    # --- Process remaining timestamps ---
    for ts in tqdm(common_timestamps[1:]):
        contact_path = os.path.join(contact_dir, f"{ts}.png")
        rgb_hand_path = os.path.join(rgb_hand_dir, f"{ts}.jpg")

        contact_img = cv2.imread(contact_path)
        rgb_hand_img = cv2.imread(rgb_hand_path)

        contact_img = contact_img[10:-10,10:-10]

        # Make sure both images loaded successfully
        if contact_img is None or rgb_hand_img is None:
            print(f"Warning: skipping {ts} because one of the images could not be read.")
            continue

        # Resize to target_height
        resized_contact = resize_to_target_height(contact_img, target_height)
        resized_rgb_hand = resize_to_target_height(rgb_hand_img, target_height)

        # Horizontally concatenate
        concat_img = np.hstack((resized_contact, resized_rgb_hand))

        # Write frame
        out.write(concat_img)

    # Release video writer
    out.release()
    print(f"Video saved to: {output_video_path}")


if __name__ == "__main__":
    # Example usage:
    # root_image_dir = "/path/to/root_image"
    # output_video_path = "/path/to/output.mp4"
    # create_video_from_images(root_image_dir, output_video_path,
    #                          start_frame_id=100000, end_frame_id=100200, fps=30)

    # Just a placeholder example with dummy paths:
    root_image_dir = "/data/HOT3D_dataset/P0001_f6cc0cc8/images"
    output_video_path = "/data/HOT3D_dataset/P0001_f6cc0cc8/images/output.mp4"
    create_video_from_images(root_image_dir,
                             output_video_path,
                             start_frame_id=None, 
                             end_frame_id=None,
                             fps=15)
