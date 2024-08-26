import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from moviepy.editor import VideoFileClip
import argparse

class VideoCropperApp:
    def __init__(self, root, video_path=None):
        self.root = root
        self.root.title("Video Cropper")

        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        self.size_label = tk.Label(root, text="Crop Size:")
        self.size_label.pack()
        self.size_entry = tk.Entry(root)
        self.size_entry.pack()
        self.size_entry.insert(0, "100")

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.slider_label = tk.Label(root, text="Select Duration and Playback:")
        self.slider_label.pack()

        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack()

        self.time_slider = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=1000, showvalue=0)
        self.time_slider.pack(side=tk.TOP, fill=tk.X)
        self.time_slider.bind("<B1-Motion>", self.update_playback)

        self.start_time_label = tk.Label(self.slider_frame, text="Start Time: 0")
        self.start_time_label.pack(side=tk.LEFT)
        self.end_time_label = tk.Label(self.slider_frame, text="End Time: 100")
        self.end_time_label.pack(side=tk.LEFT)
        self.playback_time_label = tk.Label(self.slider_frame, text="Playback Time: 0")
        self.playback_time_label.pack(side=tk.LEFT)

        self.next_section_button = tk.Button(root, text="Next Section", command=self.load_next_section)
        self.next_section_button.pack()

        self.save_button = tk.Button(root, text="Save as GIF", command=self.save_gif, state=tk.DISABLED)
        self.save_button.pack()

        self.video_path = video_path
        self.crop_center = (0, 0)
        self.crop_size = 100
        self.frame = None
        self.cropped_frame = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        self.video_clip = None
        self.current_section_start = 0
        self.section_duration = 10  # seconds
        self.overlap_duration = 2  # seconds

        self.start_time = 0
        self.end_time = 100
        self.playback_time = 0

        if self.video_path:
            self.load_video_from_path()

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.load_video_from_path()

    def load_video_from_path(self):
        self.video_clip = VideoFileClip(self.video_path)
        self.current_section_start = 0
        self.load_video_section()

    def load_video_section(self):
        start = self.current_section_start
        end = start + self.section_duration
        if self.video_clip.duration < end:
            end = self.video_clip.duration
        self.frames = []
        for t in range(int(start * self.video_clip.fps), int(end * self.video_clip.fps)):
            self.frames.append(self.video_clip.get_frame(t / self.video_clip.fps))
        self.show_frame(self.frames[0])
        self.update_slider()
        self.save_button.config(state=tk.NORMAL)

    def load_next_section(self):
        self.current_section_start += self.section_duration - self.overlap_duration
        if self.current_section_start < self.video_clip.duration:
            self.load_video_section()

    def update_slider(self):
        max_value = len(self.frames) - 1
        self.time_slider.config(to=max_value)
        self.time_slider.set(0)
        self.start_time = 0
        self.end_time = max_value
        self.playback_time = 0
        self.update_labels()

    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def on_click(self, event):
        self.crop_center = (event.x, event.y)
        self.update_crop()

    def on_drag(self, event):
        self.crop_center = (event.x, event.y)
        self.update_crop()

    def update_crop(self):
        try:
            self.crop_size = int(self.size_entry.get())
        except ValueError:
            self.crop_size = 100  # Default size if input is invalid
        if self.frame is not None:
            x, y = self.crop_center
            size = self.crop_size // 2
            self.cropped_frame = self.frame[y-size:y+size, x-size:x+size]
            self.show_frame(self.frame)
            self.canvas.create_rectangle(x-size, y-size, x+size, y+size, outline="red")

    def update_playback(self, event):
        self.playback_time = self.time_slider.get()
        self.update_labels()
        self.show_frame(self.frames[self.playback_time])

    def update_labels(self):
        self.start_time_label.config(text=f"Start Time: {self.start_time}")
        self.end_time_label.config(text=f"End Time: {self.end_time}")
        self.playback_time_label.config(text=f"Playback Time: {self.playback_time}")

    def save_gif(self):
        if self.video_path and self.crop_center:
            x, y = self.crop_center
            size = self.crop_size // 2
            start_frame = self.start_time
            end_frame = self.end_time
            frames_to_gif = self.frames[start_frame:end_frame+1]

            # Save frames as GIF
            images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_to_gif]
            output_path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF files", "*.gif")])
            if output_path:
                images[0].save(output_path, save_all=True, append_images=images[1:], duration=100, loop=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Cropper App")
    parser.add_argument('video_path', type=str, help='Path to the video file')
    args = parser.parse_args()

    root = tk.Tk()
    app = VideoCropperApp(root, video_path=args.video_path)
    root.mainloop()
