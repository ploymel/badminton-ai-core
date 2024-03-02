import pytube

# Define the link and the destination folder
link = "https://www.youtube.com/watch?v=UzpmfFHsGdo"
destination_folder = "./videos"  # Replace with the path to your desired folder

# Create a YouTube object with the link
yt = pytube.YouTube(link)

# Get the highest resolution stream available
stream = yt.streams.get_highest_resolution()

# Download the video to the specified folder
stream.download(output_path=destination_folder)

print(f"Video has been downloaded to {destination_folder}")
