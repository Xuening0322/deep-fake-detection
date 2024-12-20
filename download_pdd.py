from yt_dlp import YoutubeDL

# Define the data for all 32 videos
video_data = [
    {"ID": "t-00", "Name": "Trump", "Label": "Fake Conservative", "Video Source": "https://youtu.be/TEh1gyFGgfI"},
    {"ID": "t-01", "Name": "Trump", "Label": "Fake Conservative", "Video Source": "https://youtu.be/Nb-7PrqW-tM"},
    {"ID": "t-02", "Name": "Trump", "Label": "Fake Conservative", "Video Source": "https://youtu.be/pTFIvfuRDe4"},
    {"ID": "t-03", "Name": "Trump", "Label": "Fake Conservative", "Video Source": "https://youtu.be/ddoj7KUN-WU"},
    {"ID": "t-04", "Name": "Trump", "Label": "Fake Liberal", "Video Source": "https://youtu.be/DYZBu9jdZbU"},
    {"ID": "t-05", "Name": "Trump", "Label": "Fake Liberal", "Video Source": "https://youtu.be/dW8VpYKwltE"},
    {"ID": "t-06", "Name": "Trump", "Label": "Fake Liberal", "Video Source": "https://youtu.be/MrTu3ndnoAo"},
    {"ID": "t-07", "Name": "Trump", "Label": "Fake Liberal", "Video Source": "https://youtu.be/sWeUJITORko"},
    {"ID": "t-08", "Name": "Trump", "Label": "Real Conservative", "Video Source": "https://youtu.be/aHTuCsAFUpM"},
    {"ID": "t-09", "Name": "Trump", "Label": "Real Conservative", "Video Source": "https://youtu.be/uQVoXKmlqFM"},
    {"ID": "t-10", "Name": "Trump", "Label": "Real Conservative", "Video Source": "https://youtu.be/_ocb1UDflbE"},
    {"ID": "t-11", "Name": "Trump", "Label": "Real Conservative", "Video Source": "https://youtu.be/foHLbGR1atk"},
    {"ID": "t-12", "Name": "Trump", "Label": "Real Liberal", "Video Source": "https://youtu.be/H5fjE2LDLeU"},
    {"ID": "t-13", "Name": "Trump", "Label": "Real Liberal", "Video Source": "https://youtu.be/zGEsqmNKSeE"},
    {"ID": "t-14", "Name": "Trump", "Label": "Real Liberal", "Video Source": "https://youtu.be/IBKmTbRmHI4"},
    {"ID": "t-15", "Name": "Trump", "Label": "Real Liberal", "Video Source": "https://youtu.be/bvyPuJI7rFU"},
    {"ID": "b-00", "Name": "Biden", "Label": "Fake Conservative", "Video Source": "https://youtu.be/JeEUFfHoYVw"},
    {"ID": "b-01", "Name": "Biden", "Label": "Fake Conservative", "Video Source": "https://youtu.be/v3GeWvqx-C4"},
    {"ID": "b-02", "Name": "Biden", "Label": "Fake Conservative", "Video Source": "https://youtu.be/jP6tNR65wF4"},
    {"ID": "b-03", "Name": "Biden", "Label": "Fake Conservative", "Video Source": "https://youtu.be/r_ieNWukEDI"},
    {"ID": "b-04", "Name": "Biden", "Label": "Fake Liberal", "Video Source": "https://youtu.be/tZNCOYSbgiw"},
    {"ID": "b-05", "Name": "Biden", "Label": "Fake Liberal", "Video Source": "https://youtu.be/b9t-ZxGDJBI"},
    {"ID": "b-06", "Name": "Biden", "Label": "Fake Liberal", "Video Source": "https://youtu.be/v8AEivzb6Ng"},
    {"ID": "b-07", "Name": "Biden", "Label": "Fake Liberal", "Video Source": "https://youtu.be/u36HnG0-bBY"},
    {"ID": "b-08", "Name": "Biden", "Label": "Real Conservative", "Video Source": "https://youtu.be/9zlsX7RGPnc"},
    {"ID": "b-09", "Name": "Biden", "Label": "Real Conservative", "Video Source": "https://youtu.be/nJwOhuIl5x4"},
    {"ID": "b-10", "Name": "Biden", "Label": "Real Conservative", "Video Source": "https://youtu.be/wv_62WiuUgQ"},
    {"ID": "b-11", "Name": "Biden", "Label": "Real Conservative", "Video Source": "https://youtu.be/_LRGcONgoxY"},
    {"ID": "b-12", "Name": "Biden", "Label": "Real Liberal", "Video Source": "https://youtu.be/iobG3_FQll0"},
    {"ID": "b-13", "Name": "Biden", "Label": "Real Liberal", "Video Source": "https://youtu.be/DvV7xWMIjQc"},
    {"ID": "b-14", "Name": "Biden", "Label": "Real Liberal", "Video Source": "https://youtu.be/IyrCML0cwMk"},
    {"ID": "b-15", "Name": "Biden", "Label": "Real Liberal", "Video Source": "https://youtu.be/2OYJPNEZhBY"},
]

# Download and save videos with appropriate names
for video in video_data:
    output_name = f"{video['ID']}_{video['Name']}_{video['Label'].replace(' ', '_')}.mp4"
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'pdd_dataset/' + output_name,  # Set output template here
        'quiet': True,
    }
    print(f"Downloading {video['ID']} from {video['Video Source']} as {output_name}...")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video['Video Source']])
        
        
video_paths = [
    "pdd_dataset/t-00_Trump_Fake_Conservative.mp4",
    "pdd_dataset/t-01_Trump_Fake_Conservative.mp4",
    "pdd_dataset/t-02_Trump_Fake_Conservative.mp4",
    "pdd_dataset/t-03_Trump_Fake_Conservative.mp4",
    "pdd_dataset/t-04_Trump_Fake_Liberal.mp4",
    "pdd_dataset/t-05_Trump_Fake_Liberal.mp4",
    "pdd_dataset/t-06_Trump_Fake_Liberal.mp4",
    "pdd_dataset/t-07_Trump_Fake_Liberal.mp4",
    "pdd_dataset/t-08_Trump_Real_Conservative.mp4",
    "pdd_dataset/t-09_Trump_Real_Conservative.mp4",
    "pdd_dataset/t-10_Trump_Real_Conservative.mp4",
    "pdd_dataset/t-11_Trump_Real_Conservative.mp4",
    "pdd_dataset/t-12_Trump_Real_Liberal.mp4",
    "pdd_dataset/t-13_Trump_Real_Liberal.mp4",
    "pdd_dataset/t-14_Trump_Real_Liberal.mp4",
    "pdd_dataset/t-15_Trump_Real_Liberal.mp4",
    "pdd_dataset/b-00_Biden_Fake_Conservative.mp4",
    "pdd_dataset/b-01_Biden_Fake_Conservative.mp4",
    "pdd_dataset/b-02_Biden_Fake_Conservative.mp4",
    "pdd_dataset/b-03_Biden_Fake_Conservative.mp4",
    "pdd_dataset/b-04_Biden_Fake_Liberal.mp4",
    "pdd_dataset/b-05_Biden_Fake_Liberal.mp4",
    "pdd_dataset/b-06_Biden_Fake_Liberal.mp4",
    "pdd_dataset/b-07_Biden_Fake_Liberal.mp4",
    "pdd_dataset/b-08_Biden_Real_Conservative.mp4",
    "pdd_dataset/b-09_Biden_Real_Conservative.mp4",
    "pdd_dataset/b-10_Biden_Real_Conservative.mp4",
    "pdd_dataset/b-11_Biden_Real_Conservative.mp4",
    "pdd_dataset/b-12_Biden_Real_Liberal.mp4",
    "pdd_dataset/b-13_Biden_Real_Liberal.mp4",
    "pdd_dataset/b-14_Biden_Real_Liberal.mp4",
    "pdd_dataset/b-15_Biden_Real_Liberal.mp4"
]

# Define the output file path
output_file_path = "fake_videos.txt"

# Write the relative paths to the text file
with open(output_file_path, "w") as file:
    file.write("\n".join(video_paths))

# Confirm the output file has been created
print(f"Video paths saved to {output_file_path}")
