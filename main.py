import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageTk

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Analysis Application")
        self.root.geometry("800x600")  # Set the window size

        # Set up the vertical layout
        self.frame1 = tk.Frame(root, height=int(root.winfo_screenheight() * 0.75))
        self.frame1.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.frame2 = tk.Frame(root, height=int(root.winfo_screenheight() * 0.25))
        self.frame2.pack(side=tk.TOP, fill=tk.X)

        self.frame3 = tk.Frame(root)
        self.frame3.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Frame 1: Video display
        self.video_label = tk.Label(self.frame1)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # Frame 2: Controls
        self.capture_type = tk.StringVar(value="Video")
        self.radio_video = tk.Radiobutton(self.frame2, text="Video", variable=self.capture_type, value="Video", command=self.update_buttons)
        self.radio_real_time = tk.Radiobutton(self.frame2, text="Real Time", variable=self.capture_type, value="Real Time", command=self.update_buttons)
        self.radio_video.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.radio_real_time.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.choose_button = tk.Button(self.frame2, text="Choose Video", command=self.choose_video)
        self.choose_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.start_stop_button = tk.Button(self.frame2, text="Start", command=self.start_stop)
        self.start_stop_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.restart_button = tk.Button(self.frame2, text="Restart", command=self.restart)
        self.restart_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Frame 3: Statistics table
        self.stats_tree = ttk.Treeview(self.frame3, columns=("Statistic", "Value"), show='headings')
        self.stats_tree.heading("Statistic", text="Statistic")
        self.stats_tree.heading("Value", text="Value")
        self.stats_tree.pack(fill=tk.BOTH, expand=True)

        # Variables for video capture and processing
        self.video_source = None
        self.cap = None
        self.running = False
        self.thread = None

        self.display_thread = None
        self.display_running = False

    def update_buttons(self):
        if self.capture_type.get() == "Real Time":
            self.choose_button.config(state=tk.DISABLED)
            self.start_display()
        else:
            self.choose_button.config(state=tk.NORMAL)
            self.stop_display()

    def choose_video(self):
        self.video_source = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_source:
            self.cap = cv2.VideoCapture(self.video_source)

    def start_stop(self):
        if self.running:
            self.running = False
            self.start_stop_button.config(text="Start")
        else:
            self.running = True
            self.start_stop_button.config(text="Stop")
            if self.capture_type.get() == "Real Time":
                self.cap = cv2.VideoCapture(0)
            self.thread = threading.Thread(target=self.process_video)
            self.thread.start()

    def restart(self):
        self.running = False
        self.start_stop_button.config(text="Start")
        self.video_source = None
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

    def process_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize frame to fit in the video label
            frame = cv2.resize(frame, (self.video_label.winfo_width(), self.video_label.winfo_height()))

            # Convert the frame to a format suitable for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            # Simulate processing time
            time.sleep(0.03)

        if self.cap:
            self.cap.release()

        # Display dummy statistics after processing
        self.display_statistics({"Mean Intensity": np.mean(frame), "Frame Count": 100})

    def display_statistics(self, stats):
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        for stat, value in stats.items():
            self.stats_tree.insert("", "end", values=(stat, value))

    def start_display(self):
        if not self.display_running:
            self.display_running = True
            self.display_thread = threading.Thread(target=self.display_video)
            self.display_thread.start()

    def stop_display(self):
        self.display_running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    def display_video(self):
        self.cap = cv2.VideoCapture(0)
        while self.display_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize frame to fit in the video label
            frame = cv2.resize(frame, (self.video_label.winfo_width(), self.video_label.winfo_height()))

            # Convert the frame to a format suitable for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            # Simulate display time
            time.sleep(0.03)

        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
