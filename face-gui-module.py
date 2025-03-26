import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from collections import Counter
from face_detector import FaceMeshDetector

class FaceMeshGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Initialize variables
        self.video_source = 0
        self.is_recording = False
        self.detector = None
        
        # Create main container
        self.container = ttk.Frame(window)
        self.container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create video frame
        self.video_frame = ttk.Frame(self.container)
        self.video_frame.pack(fill='both', expand=True)
        
        # Create canvas for video
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()
        
        # Controls frame
        self.controls_frame = ttk.Frame(self.container)
        self.controls_frame.pack(fill='x', pady=10)
        
        # Session entry
        self.session_label = ttk.Label(self.controls_frame, text="Session ID (subject_activity):")
        self.session_label.pack(side='left', padx=5)
        self.session_entry = ttk.Entry(self.controls_frame)
        self.session_entry.pack(side='left', padx=5)
        
        # Start/Stop button
        self.btn_start = ttk.Button(self.controls_frame, text="Start Recording", command=self.toggle_recording)
        self.btn_start.pack(side='left', padx=5)
        
        # Results window (initially hidden)
        self.results_window = None
        
        # Open video source
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.video_source)
            
        # Update timer
        self.delay = 15
        self.update()
        
        # Protocol for closing the window
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def preprocess_data(self, df):
        """Preprocess the data to correctly handle engagement levels."""
        # Create a new column for engagement level
        df['Engagement'] = None
        
        # First, clean up the 'Class' column - replace 'None' with '0'
        df['Class'] = df['Class'].replace('None', '0')
        
        # Convert the Class column to numeric, coercing errors to NaN
        df.loc[df['Face_Detected'] == 1, 'Engagement'] = pd.to_numeric(
            df.loc[df['Face_Detected'] == 1, 'Class'], 
            errors='coerce'
        ).fillna(0)  # Fill NaN with 0 since these represent Level 0
        
        # Make sure Engagement is numeric
        df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
        
        return df
    
    def create_timeline_plot(self, df, ax):
        """Create an enhanced timeline plot of detections."""
        # Convert timestamp to datetime and sort
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        
        # Preprocess data to handle engagement levels correctly
        df = self.preprocess_data(df)
        
        if not df.empty:
            # Plot only valid engagement levels (where face was detected)
            valid_data = df[df['Face_Detected'] == 1]
            
            # Plot the main line
            ax.plot(valid_data['Timestamp'], 
                   valid_data['Engagement'],
                   '-o',  # Line with dots at each point
                   linewidth=3,
                   markersize=6,
                   alpha=0.8)
            
            # Set y-axis ticks to show engagement levels 0-3
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['Level 0', 'Level 1', 'Level 2', 'Level 3'])
            
            # Set y-axis limits slightly beyond the data range
            ax.set_ylim(-0.2, 3.2)
            
            # Customize the plot
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel('Engagement Level')
            ax.set_title('Engagement Level Timeline')
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Calculate engagement level statistics (only for detected faces)
            level_counts = valid_data['Engagement'].value_counts().sort_index()
            total_detections = len(valid_data)
            
            # Create statistics text
            stats_text = "Engagement Statistics:\n\n"
            for level in range(4):  # 0 to 3
                count = level_counts.get(level, 0)
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                stats_text += f"Level {level}: {count} ({percentage:.1f}%)\n"
            # Place text box in upper left corner
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    def show_results(self):
        """Show results window with detection statistics and enhanced plot."""
        self.results_window = tk.Toplevel(self.window)
        self.results_window.title("Detection Results")
        self.results_window.geometry("1000x700")
        
        csv_path = os.path.join('sessions', self.detector.session_id, f'detection_log_{self.detector.session_id}.csv')
        df = pd.read_csv(csv_path)
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Calculate engagement statistics (only for detected faces)
        valid_data = df[df['Face_Detected'] == 1]
        engagement_counts = valid_data['Engagement'].value_counts()
        if not engagement_counts.empty:
            most_common_level = engagement_counts.index[0]
            most_common_percentage = (engagement_counts[most_common_level] / len(valid_data)) * 100
        else:
            most_common_level = "N/A"
            most_common_percentage = 0
        
        # Create results frame
        results_frame = ttk.Frame(self.results_window)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Show detection summary
        summary_text = f"Most Common Engagement: Level {most_common_level} ({most_common_percentage:.1f}%)\n"

        
        summary_label = ttk.Label(results_frame, 
                                text=summary_text,
                                font=('Helvetica', 12))
        summary_label.pack(pady=10)
        
        # Create figure for plotting
        # Buat figure dengan resolusi lebih tinggi
        fig = Figure(figsize=(10, 6), dpi=150)  # Tambahkan dpi untuk resolusi lebih tinggi
        ax = fig.add_subplot(111)

        
        # Create the enhanced timeline plot
        self.create_timeline_plot(df, ax)
        
        # Add some padding around the plot
        fig.tight_layout()
        
        # Create canvas for matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

    def toggle_recording(self):
        """Toggle between starting and stopping recording."""
        if not self.is_recording:
            session_id = self.session_entry.get()
            if not session_id or '_' not in session_id:
                messagebox.showerror("Error", "Please enter a valid session ID (subject_activity)")
                return
                
            self.detector = FaceMeshDetector(session_id)
            self.is_recording = True
            self.btn_start.config(text="Stop Recording")
        else:
            self.is_recording = False
            self.btn_start.config(text="Start Recording")
            self.show_results()
        
    def update(self):
        """Update video frame."""
        ret, frame = self.vid.read()
        
        if ret:
            if self.is_recording and self.detector:
                frame, bbox, prediction, comp_time = self.detector.process_frame(frame)
            
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(self.delay, self.update)
        
    def on_closing(self):
        """Clean up resources when closing the application."""
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = FaceMeshGUI(root, "Face Mesh Detection")
    root.mainloop()

if __name__ == "__main__":
    main()