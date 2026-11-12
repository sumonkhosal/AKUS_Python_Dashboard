"""AKUS Offline Virtual HMI — Native Python Desktop Dashboard (Tkinter + Matplotlib).

Run directly without Streamlit or a browser:
    python desktop_app.py

The hardware interface is deliberately NOT connected: actions are simulations only.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import tempfile
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "akus_mplconfig"))
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from akus_sim import (
    Simulator,
    OP_MODES,
    TOP_DEG,
    NORMAL_TOP_DEG,
)

ROOT = Path(__file__).parent

# Theme palette matching the original dark industrial dashboard
BG_MAIN = "#0b1220"
BG_SIDEBAR = "#101b2b"
BG_CARD = "#131f30"
BG_INPUT = "#1a2940"
BORDER_COLOR = "#263449"
FG_PRIMARY = "#e6edf7"
FG_HEADING = "#ecf4ff"
FG_MUTED = "#9eb2c9"
FG_KICKER = "#67cbfd"
ACCENT_BLUE = "#1f78cb"
ACCENT_RED = "#d9534f"
ACCENT_GREEN = "#6fe1a2"
ACCENT_WARN = "#ffbb66"


class AkusDesktopApp(tk.Tk):
    """Standalone Tkinter + Matplotlib desktop application for the AKUS virtual bench."""

    def __init__(self):
        super().__init__()
        self.title("AKUS | Virtual Control Bench (Native Python Desktop)")
        self.geometry("1540x940")
        self.minsize(1200, 760)
        self.configure(bg=BG_MAIN)

        self.sim = Simulator()
        self.live_clock_var = tk.BooleanVar(value=True)
        self.speed_var = tk.IntVar(value=12)
        self.elevator_var = tk.BooleanVar(value=self.sim.in_elevator)
        self.plant_var = tk.StringVar(value=self.sim.simulation_type)
        self.last_message_var = tk.StringVar(
            value="Virtual bench initialized (OFFLINE, no EtherCAT/ADS writes)"
        )
        self.wall_clock = time.monotonic()
        self._image_refs: dict[str, ImageTk.PhotoImage] = {}
        self._last_history_len = -1
        self._last_events_len = -1

        self._configure_styles()
        self._build_layout()
        self._refresh_experiment_plan()
        self.lift()
        self.attributes("-topmost", True)
        self.after(250, lambda: self.attributes("-topmost", False))
        self.focus_force()
        self._update_loop()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG_MAIN, foreground=FG_PRIMARY, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG_MAIN)
        style.configure("Sidebar.TFrame", background=BG_SIDEBAR)
        style.configure("Card.TFrame", background=BG_CARD, relief="solid", borderwidth=1)

        style.configure("TLabel", background=BG_MAIN, foreground=FG_PRIMARY, font=("Segoe UI", 10))
        style.configure("Sidebar.TLabel", background=BG_SIDEBAR, foreground=FG_PRIMARY, font=("Segoe UI", 9))
        style.configure("SidebarMuted.TLabel", background=BG_SIDEBAR, foreground=FG_MUTED, font=("Segoe UI", 8))
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG_PRIMARY, font=("Segoe UI", 10))
        style.configure("MetricLabel.TLabel", background=BG_CARD, foreground=FG_MUTED, font=("Segoe UI", 8, "bold"))
        style.configure("MetricValue.TLabel", background=BG_CARD, foreground=FG_HEADING, font=("Segoe UI", 15, "bold"))
        style.configure("Kicker.TLabel", background=BG_MAIN, foreground=FG_KICKER, font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", background=BG_MAIN, foreground=FG_HEADING, font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=BG_MAIN, foreground=FG_MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=BG_MAIN, foreground=FG_HEADING, font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", background=BG_MAIN, foreground=FG_MUTED, font=("Segoe UI", 9))

        style.configure(
            "TButton",
            background="#1d2f47",
            foreground=FG_HEADING,
            bordercolor=BORDER_COLOR,
            focusthickness=0,
            padding=(10, 6),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("TButton", background=[("active", "#294263"), ("disabled", "#141e2d")])

        style.configure(
            "Primary.TButton",
            background=ACCENT_BLUE,
            foreground="#ffffff",
            bordercolor="#2f8ae0",
            padding=(10, 6),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#2889e4")])

        style.configure(
            "Danger.TButton",
            background=ACCENT_RED,
            foreground="#ffffff",
            bordercolor="#e66460",
            padding=(10, 6),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Danger.TButton", background=[("active", "#e0635f")])

        style.configure(
            "TNotebook",
            background=BG_MAIN,
            borderwidth=0,
            tabmargins=[2, 4, 2, 0],
        )
        style.configure(
            "TNotebook.Tab",
            background=BG_SIDEBAR,
            foreground=FG_MUTED,
            padding=(14, 7),
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", ACCENT_BLUE), ("active", "#1c314e")],
            foreground=[("selected", "#ffffff"), ("active", FG_HEADING)],
        )

        style.configure(
            "Treeview",
            background=BG_CARD,
            fieldbackground=BG_CARD,
            foreground=FG_PRIMARY,
            rowheight=26,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#1b2b42",
            foreground=FG_HEADING,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#24466e")], foreground=[("selected", "#ffffff")])

        style.configure("TCheckbutton", background=BG_MAIN, foreground=FG_PRIMARY, font=("Segoe UI", 9))
        style.configure("Sidebar.TCheckbutton", background=BG_SIDEBAR, foreground=FG_PRIMARY, font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", BG_MAIN)])
        style.map("Sidebar.TCheckbutton", background=[("active", BG_SIDEBAR)])

        style.configure("TRadiobutton", background=BG_MAIN, foreground=FG_PRIMARY, font=("Segoe UI", 9))
        style.map("TRadiobutton", background=[("active", BG_MAIN)])

        style.configure(
            "TCombobox",
            fieldbackground=BG_INPUT,
            background=BG_INPUT,
            foreground=FG_PRIMARY,
            arrowcolor=FG_PRIMARY,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG_INPUT)],
            selectbackground=[("readonly", BG_INPUT)],
            selectforeground=[("readonly", FG_PRIMARY)],
        )

        style.configure("TEntry", fieldbackground=BG_INPUT, foreground=FG_PRIMARY, insertcolor=FG_PRIMARY)
        style.configure("TSpinbox", fieldbackground=BG_INPUT, foreground=FG_PRIMARY, arrowcolor=FG_PRIMARY)

    # ------------------------------------------------------------------
    # Helper to invoke simulator actions safely
    # ------------------------------------------------------------------
    def invoke(self, action: str, *args, **kwargs):
        try:
            result = getattr(self.sim, action)(*args, **kwargs)
            self.last_message_var.set(f"Action accepted in simulator: {action}")
            self.msg_banner.configure(fg=FG_MUTED, bg=BG_MAIN)
            self._update_ui_now()
            return result
        except (ValueError, TypeError) as ex:
            self.last_message_var.set(f"⚠ {ex}")
            self.msg_banner.configure(fg=ACCENT_WARN, bg="#2b2118")
            self._update_ui_now()
            return None

    def _load_photo(self, rel_path: str, max_size: tuple[int, int]) -> ImageTk.PhotoImage | None:
        full = ROOT / rel_path
        if not full.exists():
            return None
        try:
            img = Image.open(full)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._image_refs[f"{rel_path}_{max_size}"] = photo
            return photo
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main Layout Construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=290)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Left Sidebar
        self._build_sidebar()

        # 2. Right Main Content Area
        main_frame = ttk.Frame(self, padding=(16, 12))
        main_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(header_frame, text="INDUSTRIAL MOTION · OFFLINE DIGITAL TWIN", style="Kicker.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header_frame, text="AKUS · Two-Axis Carrier Control", style="Title.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(
            header_frame,
            text="TwinCAT-inspired Python Desktop HMI · positions / velocity / acceleration / model / tests / data",
            style="Subtitle.TLabel",
        ).grid(row=2, column=0, sticky="w")

        self.msg_banner = tk.Label(
            header_frame,
            textvariable=self.last_message_var,
            bg=BG_MAIN,
            fg=FG_MUTED,
            font=("Segoe UI", 9, "italic"),
            anchor="w",
            padx=6,
            pady=2,
        )
        self.msg_banner.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        self.fault_banner = tk.Label(
            header_frame,
            text="",
            bg="#441c24",
            fg="#ff9da4",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            padx=10,
            pady=4,
        )

        # Top Metric Bar (6 cards)
        metrics_frame = ttk.Frame(main_frame)
        metrics_frame.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self.metric_vars: dict[str, tk.StringVar] = {}
        metric_defs = [
            ("STATE", "Init"),
            ("HEIGHT", "0.0 mm"),
            ("VELOCITY", "0.00 mm/s"),
            ("ACCELERATION", "1.000 g"),
            ("AXIS SKEW", "0.0°"),
            ("TIME", "0.0 s"),
        ]
        for idx, (label, initial) in enumerate(metric_defs):
            metrics_frame.grid_columnconfigure(idx, weight=1, uniform="metric")
            card = tk.Frame(metrics_frame, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=10, pady=6)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 5, 0))
            ttk.Label(card, text=label, style="MetricLabel.TLabel").pack(anchor="w")
            var = tk.StringVar(value=initial)
            self.metric_vars[label] = var
            ttk.Label(card, textvariable=var, style="MetricValue.TLabel").pack(anchor="w", pady=(2, 0))

        # Middle Live Monitor (Left: Twin-Axis Canvas, Right: Live Status Table)
        monitor_frame = ttk.Frame(main_frame)
        monitor_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        monitor_frame.grid_columnconfigure(0, weight=5)
        monitor_frame.grid_columnconfigure(1, weight=6)

        canvas_container = tk.Frame(monitor_frame, bg=BG_SIDEBAR, highlightbackground=BORDER_COLOR, highlightthickness=1)
        canvas_container.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.carrier_canvas = tk.Canvas(
            canvas_container, width=440, height=290, bg=BG_SIDEBAR, highlightthickness=0
        )
        self.carrier_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.carrier_canvas.bind("<Configure>", lambda _e: self._draw_carrier())

        status_container = tk.Frame(monitor_frame, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        status_container.grid(row=0, column=1, sticky="nsew")
        self.status_tree = ttk.Treeview(
            status_container, columns=("signal", "value"), show="headings", height=10
        )
        self.status_tree.heading("signal", text="Signal", anchor="w")
        self.status_tree.heading("value", text="Value", anchor="w")
        self.status_tree.column("signal", width=190, anchor="w")
        self.status_tree.column("value", width=260, anchor="w")
        self.status_tree.pack(fill="both", expand=True, padx=2, pady=2)

        # Bottom Tabs (8 tabs matching Streamlit app)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=3, column=0, sticky="nsew")

        self.tab_experiment = ttk.Frame(self.notebook, padding=12)
        self.tab_manual = ttk.Frame(self.notebook, padding=12)
        self.tab_tuning = ttk.Frame(self.notebook, padding=12)
        self.tab_ident = ttk.Frame(self.notebook, padding=12)
        self.tab_vacuum = ttk.Frame(self.notebook, padding=12)
        self.tab_scopes = ttk.Frame(self.notebook, padding=12)
        self.tab_diag = ttk.Frame(self.notebook, padding=12)
        self.tab_ref = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.tab_experiment, text="Experiment")
        self.notebook.add(self.tab_manual, text="Manual motion")
        self.notebook.add(self.tab_tuning, text="Controller tuning")
        self.notebook.add(self.tab_ident, text="Identification")
        self.notebook.add(self.tab_vacuum, text="Vacuum")
        self.notebook.add(self.tab_scopes, text="Analysis / Scopes")
        self.notebook.add(self.tab_diag, text="Diagnostics")
        self.notebook.add(self.tab_ref, text="Project reference")

        self._build_tab_experiment()
        self._build_tab_manual()
        self._build_tab_tuning()
        self._build_tab_identification()
        self._build_tab_vacuum()
        self._build_tab_scopes()
        self._build_tab_diagnostics()
        self._build_tab_reference()

    # ------------------------------------------------------------------
    # Sidebar Construction
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        sidebar = tk.Frame(self, bg=BG_SIDEBAR, width=290, padx=14, pady=14)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        title_row = tk.Frame(sidebar, bg=BG_SIDEBAR)
        title_row.pack(fill="x")
        tk.Label(title_row, text="AKUS", fg="#8ed6fa", bg=BG_SIDEBAR, font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(title_row, text=" Control Studio", fg=FG_HEADING, bg=BG_SIDEBAR, font=("Segoe UI", 14, "bold")).pack(side="left")

        badge = tk.Label(
            sidebar,
            text="● SIMULATION ONLY",
            bg="#163a39",
            fg="#8af2cc",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=3,
        )
        badge.pack(anchor="w", pady=(6, 4))

        ttk.Label(
            sidebar,
            text="Derived from the uploaded TwinCAT 3 PLC /\nVisualization / Scope project",
            style="SidebarMuted.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        tk.Frame(sidebar, bg=BORDER_COLOR, height=1).pack(fill="x", pady=6)

        ttk.Label(sidebar, text="Numerical plant", style="Sidebar.TLabel").pack(anchor="w", pady=(4, 2))
        plant_cb = ttk.Combobox(
            sidebar,
            textvariable=self.plant_var,
            values=["Ideal NC trajectory", "Dynamic force model"],
            state="readonly",
        )
        plant_cb.pack(fill="x", pady=(0, 8))
        plant_cb.bind("<<ComboboxSelected>>", self._on_plant_changed)

        self.speed_label = ttk.Label(
            sidebar, text=f"Simulation speed: {self.speed_var.get()}× real time", style="Sidebar.TLabel"
        )
        self.speed_label.pack(anchor="w", pady=(2, 2))
        speed_scale = tk.Scale(
            sidebar,
            from_=1,
            to=60,
            orient="horizontal",
            variable=self.speed_var,
            bg=BG_SIDEBAR,
            fg=FG_PRIMARY,
            highlightthickness=0,
            troughcolor=BG_INPUT,
            activebackground=ACCENT_BLUE,
            showvalue=False,
            command=lambda v: self.speed_label.configure(text=f"Simulation speed: {int(float(v))}× real time"),
        )
        speed_scale.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(
            sidebar,
            text="Live simulation clock",
            variable=self.live_clock_var,
            style="Sidebar.TCheckbutton",
        ).pack(anchor="w", pady=3)

        ttk.Checkbutton(
            sidebar,
            text="Elevator mode (zero gravity)",
            variable=self.elevator_var,
            style="Sidebar.TCheckbutton",
            command=self._on_elevator_toggled,
        ).pack(anchor="w", pady=3)

        ttk.Label(
            sidebar,
            text="Run to completion calculates a full motion immediately without waiting for playback.",
            style="SidebarMuted.TLabel",
            wraplength=260,
        ).pack(anchor="w", pady=(6, 8))

        btn_row = tk.Frame(sidebar, bg=BG_SIDEBAR)
        btn_row.pack(fill="x", pady=(0, 6))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        ttk.Button(btn_row, text="⏹ Stop", command=lambda: self.invoke("stop")).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(btn_row, text="⚠ E-STOP", style="Danger.TButton", command=lambda: self.invoke("emergency_stop")).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )

        ttk.Button(sidebar, text="Reset virtual controller", command=lambda: self.invoke("reset")).pack(
            fill="x", pady=3
        )
        ttk.Button(
            sidebar,
            text="Run sequence to completion",
            style="Primary.TButton",
            command=lambda: self.invoke("finish_operation", 300.0),
        ).pack(fill="x", pady=3)
        ttk.Button(sidebar, text="Advance +5 simulated seconds", command=lambda: self.invoke("fast_forward", 5.0)).pack(
            fill="x", pady=3
        )

        tk.Frame(sidebar, bg=BORDER_COLOR, height=1).pack(fill="x", pady=10)
        ttk.Label(sidebar, text="Source mechanism illustration", style="SidebarMuted.TLabel").pack(anchor="w", pady=(0, 4))

        mech_photo = self._load_photo("assets/linear_axes_carrier.png", (255, 220))
        if mech_photo:
            lbl_img = tk.Label(sidebar, image=mech_photo, bg=BG_SIDEBAR, bd=0)
            lbl_img.pack(anchor="center", pady=2)

        ttk.Label(
            sidebar,
            text="Important: no safety PLC, drives, I/O or real emergency stop are controlled from this app.",
            style="SidebarMuted.TLabel",
            wraplength=260,
        ).pack(anchor="w", side="bottom", pady=(6, 0))

    def _on_plant_changed(self, _event=None):
        self.sim.simulation_type = self.plant_var.get()
        self.sim.event(f"Numerical plant changed to: {self.sim.simulation_type}")

    def _on_elevator_toggled(self):
        self.sim.in_elevator = bool(self.elevator_var.get())
        self._update_ui_now()

    # ------------------------------------------------------------------
    # Carrier Canvas Drawing (Matches original SVG twin-axis view)
    # ------------------------------------------------------------------
    def _draw_carrier(self):
        c = self.carrier_canvas
        c.delete("all")
        w = max(c.winfo_width(), 380)
        h = max(c.winfo_height(), 240)

        top_mm = TOP_DEG * 5.0 / 360.0
        position_mm = self.sim.pos_mm
        skew = self.sim.offset2_deg
        pos_ratio = min(1.0, max(0.0, position_mm / top_mm))

        pad_x = 48
        box_left = pad_x
        box_right = w - pad_x - 24
        box_top = 14
        box_bottom = h - 28

        # Outer housing
        c.create_rectangle(box_left, box_top, box_right, box_bottom, fill="#101b2a", outline="#2c455d", width=1.5)

        rail1_x = box_left + 32
        rail2_x = box_right - 32
        screw1_x = rail1_x + 24
        screw2_x = rail2_x - 24

        # Guide rails
        c.create_rectangle(rail1_x - 8, box_top + 6, rail1_x + 8, box_bottom - 22, fill="#9fb2c4", outline="#6e839a")
        c.create_rectangle(rail2_x - 8, box_top + 6, rail2_x + 8, box_bottom - 22, fill="#9fb2c4", outline="#6e839a")

        # Ball screws
        c.create_line(screw1_x, box_top + 10, screw1_x, box_bottom - 24, fill="#cba43b", width=6)
        c.create_line(screw2_x, box_top + 10, screw2_x, box_bottom - 24, fill="#cba43b", width=6)

        # Horizontal grid lines
        for gy in range(int(box_top + 28), int(box_bottom - 28), 32):
            c.create_line(screw1_x + 4, gy, screw2_x - 4, gy, fill="#263b52", dash=(4, 5))

        # Top TouchProbe switches (green when closed, orange when actuated)
        tp_color = ACCENT_GREEN if self.sim.switch_top_nc else ACCENT_WARN
        c.create_rectangle(rail1_x - 7, box_top + 8, rail1_x + 7, box_top + 18, fill=tp_color, outline="")
        c.create_rectangle(rail2_x - 7, box_top + 8, rail2_x + 7, box_top + 18, fill=tp_color, outline="")

        # Bottom stops
        c.create_rectangle(rail1_x - 7, box_bottom - 34, rail1_x + 7, box_bottom - 24, fill="#c87979", outline="")
        c.create_rectangle(rail2_x - 7, box_bottom - 34, rail2_x + 7, box_bottom - 24, fill="#c87979", outline="")

        # Motors M1 & M2
        c.create_rectangle(rail1_x - 15, box_bottom - 21, rail1_x + 15, box_bottom - 3, fill="#223b60", outline="#477ec0")
        c.create_rectangle(rail2_x - 15, box_bottom - 21, rail2_x + 15, box_bottom - 3, fill="#223b60", outline="#477ec0")
        c.create_text(rail1_x, box_bottom - 12, text="M1", fill="white", font=("Segoe UI", 8, "bold"))
        c.create_text(rail2_x, box_bottom - 12, text="M2", fill="white", font=("Segoe UI", 8, "bold"))

        # Carrier vertical coordinate
        y_top = box_top + 28
        y_bottom = box_bottom - 42
        y_left = y_bottom - pos_ratio * (y_bottom - y_top)
        y_right = y_left - max(-14.0, min(14.0, skew / 3.0))

        # Left & right bearing blocks
        c.create_rectangle(screw1_x - 12, y_left - 13, screw1_x + 12, y_left + 13, fill="#c8dae5", outline="#6e839a")
        c.create_rectangle(screw2_x - 12, y_right - 13, screw2_x + 12, y_right + 13, fill="#c8dae5", outline="#6e839a")

        # Carrier bridge polygon (tilts with skew)
        c.create_polygon(
            screw1_x + 12, y_left - 7,
            screw2_x - 12, y_right - 7,
            screw2_x - 12, y_right + 7,
            screw1_x + 12, y_left + 7,
            fill="#4d90b8",
            outline="#b2e7f7",
            width=1.5,
        )

        # Payload capsule dome on carrier
        mid_x = 0.5 * (screw1_x + screw2_x)
        mid_y = 0.5 * (y_left + y_right)
        c.create_rectangle(mid_x - 24, mid_y - 32, mid_x + 24, mid_y - 7, fill="#8fa7b8", outline="#d6edf6", width=1.5)
        c.create_oval(mid_x - 5, mid_y - 23, mid_x + 5, mid_y - 13, fill="#df6d56", outline="")

        # Right-side height gauge
        gauge_x = box_right + 16
        c.create_text(gauge_x, y_top - 8, text="TOP", fill="#8da2ba", font=("Segoe UI", 7, "bold"))
        c.create_text(gauge_x, y_bottom + 8, text="0", fill="#8da2ba", font=("Segoe UI", 7, "bold"))
        c.create_line(gauge_x, y_top, gauge_x, y_bottom, fill="#415f7e", width=2)
        c.create_oval(gauge_x - 5, y_left - 5, gauge_x + 5, y_left + 5, fill="#4fcbef", outline="#d1ffff")

        # Bottom label
        c.create_text(
            w / 2,
            h - 12,
            text=f"SIMULATED CARRIER · {position_mm:.1f} mm  (Skew: {skew:.1f}°)",
            fill="#bfd1e8",
            font=("Segoe UI", 9, "bold"),
        )

    # ------------------------------------------------------------------
    # Tab 1: Experiment
    # ------------------------------------------------------------------
    def _build_tab_experiment(self):
        f = self.tab_experiment
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        f.grid_columnconfigure(2, weight=1)

        ttk.Label(f, text="Automatic acceleration experiment · Flug", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            f,
            text="Calculation follows MAIN.TcPOU; the alternate total-time formula is corrected to the intended trapezoidal distance.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.exp_g_var = tk.StringVar(value="0.00100")
        self.exp_mode_var = tk.StringVar(value="PositionBeckhoff")
        self.exp_dur_mode_var = tk.StringVar(value="Acceleration time")
        self.exp_t_var = tk.StringVar(value="3.0")
        self.exp_delay_var = tk.StringVar(value="0.0")
        self.exp_lock_var = tk.StringVar(value="8.0")
        self.exp_return_var = tk.BooleanVar(value=True)

        # Column 0
        col0 = ttk.Frame(f)
        col0.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(col0, text="Set acceleration [g]:").pack(anchor="w")
        e_g = ttk.Entry(col0, textvariable=self.exp_g_var)
        e_g.pack(fill="x", pady=(2, 6))
        e_g.bind("<KeyRelease>", lambda _e: self._refresh_experiment_plan())

        ttk.Label(col0, text="Feedback control:").pack(anchor="w")
        ttk.Combobox(
            col0,
            textvariable=self.exp_mode_var,
            values=("PositionBeckhoff", "Position", "Acceleration"),
            state="readonly",
        ).pack(fill="x", pady=(2, 6))

        ttk.Label(col0, text="Time specification:").pack(anchor="w")
        for opt in ("Acceleration time", "Total acceleration + braking time"):
            ttk.Radiobutton(
                col0,
                text=opt,
                variable=self.exp_dur_mode_var,
                value=opt,
                command=self._refresh_experiment_plan,
            ).pack(anchor="w")

        # Column 1
        col1 = ttk.Frame(f)
        col1.grid(row=2, column=1, sticky="nsew", padx=10)
        ttk.Label(col1, text="Requested duration [s]:").pack(anchor="w")
        e_t = ttk.Entry(col1, textvariable=self.exp_t_var)
        e_t.pack(fill="x", pady=(2, 6))
        e_t.bind("<KeyRelease>", lambda _e: self._refresh_experiment_plan())

        ttk.Label(col1, text="Trigger delay [s]:").pack(anchor="w")
        ttk.Entry(col1, textvariable=self.exp_delay_var).pack(fill="x", pady=(2, 6))

        ttk.Label(col1, text="Brake/lock duration [s]:").pack(anchor="w")
        ttk.Entry(col1, textvariable=self.exp_lock_var).pack(fill="x", pady=(2, 6))

        # Column 2
        col2 = ttk.Frame(f)
        col2.grid(row=2, column=2, sticky="nsew", padx=(10, 0))
        ttk.Checkbutton(col2, text="Return automatically to start", variable=self.exp_return_var).pack(
            anchor="w", pady=(4, 8)
        )
        ttk.Label(col2, text="• Travel guard: motion must remain below 44000° - 720° (601.11 mm).", style="Muted.TLabel").pack(
            anchor="w", pady=2
        )
        ttk.Label(col2, text="• Elevator guard: total motion duration ≤ 3.6 s.", style="Muted.TLabel").pack(
            anchor="w", pady=2
        )
        ttk.Button(
            col2,
            text="▶ Start simulated experiment",
            style="Primary.TButton",
            command=self._on_start_experiment,
        ).pack(fill="x", pady=(12, 4))

        # Experiment plan preview metrics
        self.exp_plan_label = ttk.Label(f, text="", style="Kicker.TLabel")
        self.exp_plan_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 8))

        tk.Frame(f, bg=BORDER_COLOR, height=1).grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)

        # Bottom row: Homing & Move to Start
        bot_left = ttk.Frame(f)
        bot_left.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=(0, 10))
        ttk.Label(bot_left, text="Referencing / initialization", style="Section.TLabel").pack(anchor="w")
        self.homing_mode_var = tk.StringVar(value="Beckhoff")
        ttk.Radiobutton(
            bot_left,
            text="Beckhoff NC · SyncPos (20–25)",
            variable=self.homing_mode_var,
            value="Beckhoff",
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            bot_left,
            text="Custom cascade · SyncPosTorque (1–11)",
            variable=self.homing_mode_var,
            value="Custom",
        ).pack(anchor="w", pady=2)
        ttk.Button(
            bot_left,
            text="Home both axes (simulation)",
            style="Primary.TButton",
            command=lambda: self.invoke("start_homing", beckhoff=(self.homing_mode_var.get() == "Beckhoff")),
        ).pack(anchor="w", pady=(6, 0))

        bot_right = ttk.Frame(f)
        bot_right.grid(row=5, column=2, sticky="nsew", padx=(10, 0))
        ttk.Label(bot_right, text="Start position", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            bot_right,
            text="The original procedure searches the upper limit, assigns 44,000°, then returns to 0°.",
            style="Muted.TLabel",
            wraplength=360,
        ).pack(anchor="w", pady=4)
        ttk.Button(
            bot_right,
            text="Move carrier to 0 mm",
            command=lambda: self.invoke("move_to_start"),
        ).pack(anchor="w", pady=(6, 0))

    def _refresh_experiment_plan(self):
        try:
            g_val = float(self.exp_g_var.get())
            t_val = float(self.exp_t_var.get())
            is_total = self.exp_dur_mode_var.get().startswith("Total")
            plan = self.sim.experiment_plan(
                g_val, t_val, total_duration_s=(t_val if is_total else None)
            )
            self.exp_plan_label.configure(
                text=(
                    f"PLAN PREVIEW   |   Peak speed: {plan['v_peak_deg_s']:.1f} °/s   ·   "
                    f"Stroke needed: {plan['distance_mm']:.2f} mm   ·   "
                    f"Accel phase: {plan['t_acc_s']:.3f} s   ·   "
                    f"Decel phase: {plan['t_dec_s']:.3f} s"
                ),
                foreground=FG_KICKER,
            )
        except Exception as ex:
            self.exp_plan_label.configure(text=f"⚠ Plan preview error: {ex}", foreground=ACCENT_WARN)

    def _on_start_experiment(self):
        try:
            g_val = float(self.exp_g_var.get())
            t_val = float(self.exp_t_var.get())
            delay_val = float(self.exp_delay_var.get())
            lock_val = float(self.exp_lock_var.get())
        except ValueError:
            self.last_message_var.set("⚠ Invalid numerical input in Experiment form.")
            return
        is_total = self.exp_dur_mode_var.get().startswith("Total")
        self.invoke(
            "start_experiment",
            g_val,
            t_val,
            self.exp_mode_var.get(),
            total_duration_s=(t_val if is_total else None),
            delay_s=delay_val,
            lock_s=lock_val,
            return_to_start=bool(self.exp_return_var.get()),
        )

    # ------------------------------------------------------------------
    # Tab 2: Manual Motion
    # ------------------------------------------------------------------
    def _build_tab_manual(self):
        f = self.tab_manual
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        f.grid_columnconfigure(2, weight=1)

        ttk.Label(f, text="Manual operation · Einzelbewegung / Jog", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            f,
            text="Position commands use the original motor-degree convention: 360° motor rotation = 5 mm travel.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.man_pos_var = tk.StringVar(value="100.0")
        self.man_vel_var = tk.StringVar(value="2800.0")
        self.man_ctrl_var = tk.StringVar(value="SyncPos")
        self.man_acc_var = tk.StringVar(value="400.0")
        self.man_dec_var = tk.StringVar(value="500.0")

        c0 = ttk.Frame(f)
        c0.grid(row=2, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(c0, text=f"Target carrier position [mm] (0–{NORMAL_TOP_DEG*5/360:.1f}):").pack(anchor="w")
        ttk.Entry(c0, textvariable=self.man_pos_var).pack(fill="x", pady=(2, 6))
        ttk.Label(c0, text="Acceleration [°/s²]:").pack(anchor="w")
        ttk.Entry(c0, textvariable=self.man_acc_var).pack(fill="x", pady=(2, 6))

        c1 = ttk.Frame(f)
        c1.grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Label(c1, text="Velocity [°/s]:").pack(anchor="w")
        ttk.Entry(c1, textvariable=self.man_vel_var).pack(fill="x", pady=(2, 6))
        ttk.Label(c1, text="Deceleration [°/s²]:").pack(anchor="w")
        ttk.Entry(c1, textvariable=self.man_dec_var).pack(fill="x", pady=(2, 6))

        c2 = ttk.Frame(f)
        c2.grid(row=2, column=2, sticky="ew", padx=(8, 0))
        ttk.Label(c2, text="Motion backend:").pack(anchor="w")
        ttk.Combobox(
            c2, textvariable=self.man_ctrl_var, values=("SyncPos", "SyncPosTorque", "SyncVel"), state="readonly"
        ).pack(fill="x", pady=(2, 14))
        ttk.Button(c2, text="▶ Move to position", style="Primary.TButton", command=self._on_move_absolute).pack(
            fill="x", pady=(4, 6)
        )

        tk.Frame(f, bg=BORDER_COLOR, height=1).grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)

        # Bounded Timed Jog
        ttk.Label(f, text="Bounded timed jog", style="Section.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )
        self.jog_vel_var = tk.StringVar(value="700.0")
        self.jog_dur_var = tk.StringVar(value="1.0")

        jc0 = ttk.Frame(f)
        jc0.grid(row=5, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(jc0, text="Jog velocity [°/s]:").pack(anchor="w")
        ttk.Entry(jc0, textvariable=self.jog_vel_var).pack(fill="x", pady=2)

        jc1 = ttk.Frame(f)
        jc1.grid(row=5, column=1, sticky="ew", padx=8)
        ttk.Label(jc1, text="Jog duration [s] (0.05–20):").pack(anchor="w")
        ttk.Entry(jc1, textvariable=self.jog_dur_var).pack(fill="x", pady=2)

        jc2 = ttk.Frame(f)
        jc2.grid(row=5, column=2, sticky="ew", padx=(8, 0))
        ttk.Button(jc2, text="↑ Jog up", command=lambda: self._on_jog(1)).pack(side="left", expand=True, fill="x", padx=(0, 4), pady=(14, 0))
        ttk.Button(jc2, text="↓ Jog down", command=lambda: self._on_jog(-1)).pack(side="left", expand=True, fill="x", padx=(4, 0), pady=(14, 0))

        tk.Frame(f, bg=BORDER_COLOR, height=1).grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)

        # Skew slider
        ttk.Label(f, text="Synchronization illustration (Inject Axis 2 position offset [°])", style="Section.TLabel").grid(
            row=7, column=0, columnspan=2, sticky="w"
        )
        self.skew_warn_lbl = ttk.Label(f, text="", foreground=ACCENT_WARN)
        self.skew_warn_lbl.grid(row=7, column=2, sticky="e")

        self.skew_var = tk.DoubleVar(value=self.sim.offset2_deg)
        skew_scale = tk.Scale(
            f,
            from_=-50.0,
            to=50.0,
            resolution=1.0,
            orient="horizontal",
            variable=self.skew_var,
            bg=BG_MAIN,
            fg=FG_PRIMARY,
            highlightthickness=0,
            troughcolor=BG_INPUT,
            activebackground=ACCENT_BLUE,
            command=self._on_skew_changed,
        )
        skew_scale.grid(row=8, column=0, columnspan=3, sticky="ew")

    def _on_move_absolute(self):
        try:
            pos = float(self.man_pos_var.get())
            vel = float(self.man_vel_var.get())
            acc = float(self.man_acc_var.get())
            dec = float(self.man_dec_var.get())
        except ValueError:
            self.last_message_var.set("⚠ Invalid number in Manual motion fields.")
            return
        self.invoke("move_absolute", pos, vel, acc, dec, self.man_ctrl_var.get())

    def _on_jog(self, direction: int):
        try:
            vel = float(self.jog_vel_var.get())
            dur = float(self.jog_dur_var.get())
        except ValueError:
            self.last_message_var.set("⚠ Invalid number in Jog fields.")
            return
        self.invoke("jog_for", direction, dur, vel)

    def _on_skew_changed(self, val):
        self.sim.offset2_deg = float(val)
        if abs(self.sim.offset2_deg) > 40:
            self.skew_warn_lbl.configure(
                text="⚠ Exceeds GVL synchronization tolerance (40°)!"
            )
        else:
            self.skew_warn_lbl.configure(text="")
        self._draw_carrier()

    # ------------------------------------------------------------------
    # Tab 3: Controller Tuning
    # ------------------------------------------------------------------
    def _build_tab_tuning(self):
        f = self.tab_tuning
        f.grid_columnconfigure(0, weight=3)
        f.grid_columnconfigure(1, weight=2)

        cfg = asdict(self.sim.config)
        self.param_clusters = {
            "Position cascade": [k for k in cfg if k.startswith("pos_")],
            "Acceleration cascade": [k for k in cfg if k.startswith("acc_") and not k.startswith("acc_gradient") and not k.startswith("acc_offset") and not k.startswith("acc_use")],
            "Torque limits": [k for k in cfg if k.startswith("torque_") or k.endswith("_added")],
            "Plant physics": [k for k in cfg if k.startswith("m_") or k.startswith("friction_")],
            "Acceleration calibration": [k for k in cfg if k.startswith("acc_gradient") or k.startswith("acc_offset") or k.startswith("acc_use")],
            "Offline-only feedback assumptions": [k for k in cfg if k.startswith("max_motor") or k.startswith("simulated_")],
        }

        left = ttk.Frame(f)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_columnconfigure(0, weight=1)

        ttk.Label(left, text="ControlConfig · PID + mechanical plant parameters", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Initial values from GVL_Persist.TcGVL. Tuning affects Dynamic force model only.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        top_bar = ttk.Frame(left)
        top_bar.pack(fill="x", pady=4)
        ttk.Label(top_bar, text="Parameter group:").pack(side="left", padx=(0, 8))
        self.param_group_var = tk.StringVar(value="Position cascade")
        grp_cb = ttk.Combobox(
            top_bar,
            textvariable=self.param_group_var,
            values=list(self.param_clusters.keys()),
            state="readonly",
            width=32,
        )
        grp_cb.pack(side="left")
        grp_cb.bind("<<ComboboxSelected>>", lambda _e: self._populate_param_entries())

        self.param_entries_frame = ttk.Frame(left)
        self.param_entries_frame.pack(fill="x", pady=6)
        self.param_entry_vars: dict[str, tk.StringVar] = {}
        self._populate_param_entries()

        btn_bar = ttk.Frame(left)
        btn_bar.pack(fill="x", pady=6)
        ttk.Button(btn_bar, text="Apply parameter changes", style="Primary.TButton", command=self._apply_params).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btn_bar, text="Export JSON...", command=self._export_params_json).pack(side="left", padx=6)
        ttk.Button(btn_bar, text="Import JSON...", command=self._import_params_json).pack(side="left", padx=6)

        # Right side: Controller test + force diagram
        right = ttk.Frame(f)
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text="Simulate controller response test", style="Section.TLabel").pack(anchor="w")

        ct_grid = ttk.Frame(right)
        ct_grid.pack(fill="x", pady=6)
        self.ct_start_var = tk.StringVar(value="550.0")
        self.ct_end_var = tk.StringVar(value="50.0")
        self.ct_vel_var = tk.StringVar(value="2000.0")
        self.ct_acc_var = tk.StringVar(value="1000.0")

        for idx, (lbl, var) in enumerate([
            ("Start [mm]:", self.ct_start_var),
            ("End [mm]:", self.ct_end_var),
            ("Velocity [°/s]:", self.ct_vel_var),
            ("Acceleration [°/s²]:", self.ct_acc_var),
        ]):
            r, c = divmod(idx, 2)
            ct_grid.grid_columnconfigure(c, weight=1)
            cell = ttk.Frame(ct_grid)
            cell.grid(row=r, column=c, sticky="ew", padx=4, pady=2)
            ttk.Label(cell, text=lbl).pack(anchor="w")
            ttk.Entry(cell, textvariable=var).pack(fill="x")

        ttk.Button(
            right,
            text="▶ Run offline controller test",
            style="Primary.TButton",
            command=self._on_run_control_test,
        ).pack(fill="x", padx=4, pady=6)

        force_photo = self._load_photo("assets/force_model.png", (420, 190))
        if force_photo:
            tk.Label(right, image=force_photo, bg=BG_MAIN, bd=0).pack(anchor="center", pady=4)

    def _populate_param_entries(self):
        for child in self.param_entries_frame.winfo_children():
            child.destroy()
        self.param_entry_vars.clear()
        cfg = asdict(self.sim.config)
        keys = self.param_clusters.get(self.param_group_var.get(), [])
        for idx, k in enumerate(keys):
            r, c = divmod(idx, 2)
            self.param_entries_frame.grid_columnconfigure(c, weight=1)
            row_f = ttk.Frame(self.param_entries_frame)
            row_f.grid(row=r, column=c, sticky="ew", padx=6, pady=3)
            ttk.Label(row_f, text=k, width=24).pack(side="left")
            var = tk.StringVar(value=f"{float(cfg[k]):.6g}")
            self.param_entry_vars[k] = var
            ttk.Entry(row_f, textvariable=var, width=14).pack(side="right", fill="x", expand=True)

    def _apply_params(self):
        if self.sim.current or self.sim.operation:
            messagebox.showwarning("Machine Busy", "Stop the virtual machine before editing plant parameters.")
            return
        try:
            changes = {k: float(v.get()) for k, v in self.param_entry_vars.items()}
            if not all(math.isfinite(val) for val in changes.values()):
                raise ValueError("Parameters must be finite numbers.")
            if any(
                val <= 0
                for k, val in changes.items()
                if k in ("m_equivalent_kg", "m_platform_kg", "max_motor_torque_nm")
            ):
                raise ValueError("Masses and motor torque limit must be > 0.")
            for k, val in changes.items():
                setattr(self.sim.config, k, val)
            self.sim.event("Applied virtual parameter group: " + self.param_group_var.get())
            self.last_message_var.set(f"Virtual parameters updated ({self.param_group_var.get()})")
        except ValueError as ex:
            messagebox.showerror("Invalid Parameter", str(ex))

    def _export_params_json(self):
        path = filedialog.asksaveasfilename(
            title="Export ControlConfig JSON",
            defaultextension=".json",
            initialfile="akus_parameters.json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            Path(path).write_text(json.dumps(asdict(self.sim.config), indent=2), encoding="utf-8")
            self.last_message_var.set(f"Exported parameters to {path}")

    def _import_params_json(self):
        path = filedialog.askopenfilename(
            title="Import ControlConfig JSON",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                allowed = asdict(self.sim.config)
                for k, v in data.items():
                    if k in allowed and isinstance(v, (float, int)) and math.isfinite(v):
                        setattr(self.sim.config, k, float(v))
                self.sim.event("Control configuration imported from JSON")
                self._populate_param_entries()
                self.last_message_var.set(f"Imported parameters from {path}")
            except Exception as ex:
                messagebox.showerror("Import Error", str(ex))

    def _on_run_control_test(self):
        try:
            s = float(self.ct_start_var.get())
            e = float(self.ct_end_var.get())
            v = float(self.ct_vel_var.get())
            a = float(self.ct_acc_var.get())
        except ValueError:
            self.last_message_var.set("⚠ Invalid numbers in controller test inputs.")
            return
        self.invoke("start_control_test", s, e, v, a)

    # ------------------------------------------------------------------
    # Tab 4: Identification
    # ------------------------------------------------------------------
    def _build_tab_identification(self):
        f = self.tab_ident
        ttk.Label(f, text="Parameter identification · IdentTrajectory", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            f,
            text="Moves to start, records an excitation trajectory, returns to start and supports CSV export.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        self.id_start_var = tk.StringVar(value="50.0")
        self.id_end_var = tk.StringVar(value="500.0")
        self.id_vel_var = tk.StringVar(value="2400.0")
        self.id_acc_var = tk.StringVar(value="1500.0")

        for idx, (lbl, var) in enumerate([
            ("Start [mm]:", self.id_start_var),
            ("End [mm]:", self.id_end_var),
            ("Velocity [°/s]:", self.id_vel_var),
            ("Acceleration [°/s²]:", self.id_acc_var),
        ]):
            row.grid_columnconfigure(idx, weight=1)
            cell = ttk.Frame(row)
            cell.grid(row=0, column=idx, sticky="ew", padx=4)
            ttk.Label(cell, text=lbl).pack(anchor="w")
            ttk.Entry(cell, textvariable=var).pack(fill="x", pady=2)

        ttk.Button(
            f,
            text="▶ Run simulated identification",
            style="Primary.TButton",
            command=self._on_run_identification,
        ).pack(anchor="w", pady=10)

        eq_card = tk.Frame(f, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=14, pady=12)
        eq_card.pack(fill="x", pady=8)
        ttk.Label(eq_card, text="Plant Dynamic Equations (FB_AkusModel.TcPOU)", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        equations = (
            "  F_motor    = (T₁ + T₂) · (2π / h)          F_drive = F_motor − m_p · g\n"
            "  F_friction = sgn(F_drive) · [ B_c + B_z · (z − z₀)² ] + B_v · ż\n"
            "  z̈         = (F_drive − F_friction) / (m_ers + m_p)"
        )
        tk.Label(eq_card, text=equations, bg=BG_CARD, fg="#8ed6fa", font=("Consolas", 11), justify="left").pack(anchor="w")
        ttk.Label(
            eq_card,
            text="Note: Torque traces are inferred from the force balance, not physical drive measurements.",
            style="Card.TLabel",
            foreground=FG_MUTED,
        ).pack(anchor="w", pady=(8, 0))

    def _on_run_identification(self):
        try:
            s = float(self.id_start_var.get())
            e = float(self.id_end_var.get())
            v = float(self.id_vel_var.get())
            a = float(self.id_acc_var.get())
        except ValueError:
            self.last_message_var.set("⚠ Invalid numbers in Identification form.")
            return
        self.invoke("start_identification", s, e, v, a)

    # ------------------------------------------------------------------
    # Tab 5: Vacuum
    # ------------------------------------------------------------------
    def _build_tab_vacuum(self):
        f = self.tab_vacuum
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)

        ttk.Label(f, text="Vacuum test panel · Ventilsteuerung", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            f,
            text="Offline signal-state panel based on MAIN vacuum I/O symbols. No pump, valve, or sensor is connected.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        left = ttk.Frame(f)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        vac_photo = self._load_photo("assets/vacuum_valves.png", (480, 270))
        if vac_photo:
            tk.Label(left, image=vac_photo, bg=BG_MAIN, bd=0).pack(anchor="w", pady=4)

        right = ttk.Frame(f)
        right.grid(row=2, column=1, sticky="nsew")

        self.vac_vars = {
            "pump_turbo": tk.BooleanVar(value=self.sim.pump_turbo),
            "pump_diaphragm": tk.BooleanVar(value=self.sim.pump_diaphragm),
            "valve_big": tk.BooleanVar(value=self.sim.valve_big),
            "valve_normal": tk.BooleanVar(value=self.sim.valve_normal),
            "valve_diaphragm": tk.BooleanVar(value=self.sim.valve_diaphragm),
        }
        labels = [
            ("pump_turbo", "Turbo pump"),
            ("pump_diaphragm", "Diaphragm pump"),
            ("valve_big", "Big valve"),
            ("valve_normal", "Normal valve"),
            ("valve_diaphragm", "Diaphragm valve"),
        ]
        for key, text in labels:
            ttk.Checkbutton(right, text=text, variable=self.vac_vars[key], command=self._apply_vacuum).pack(
                anchor="w", pady=2
            )

        self.vac_tree = ttk.Treeview(right, columns=("sensor", "state"), show="headings", height=6)
        self.vac_tree.heading("sensor", text="Sensor", anchor="w")
        self.vac_tree.heading("state", text="Value", anchor="w")
        self.vac_tree.column("sensor", width=220, anchor="w")
        self.vac_tree.column("state", width=120, anchor="w")
        self.vac_tree.pack(fill="x", pady=8)
        self._refresh_vacuum_table()

    def _apply_vacuum(self):
        self.sim.set_vacuum(**{k: v.get() for k, v in self.vac_vars.items()})
        self._refresh_vacuum_table()

    def _refresh_vacuum_table(self):
        for item in self.vac_tree.get_children():
            self.vac_tree.delete(item)
        rows = [
            ("bBigValveOpened", str(self.sim.valve_big)),
            ("bBigValveClosed", str(not self.sim.valve_big)),
            ("bNormalValveOpened", str(self.sim.valve_normal)),
            ("bNormalValveClosed", str(not self.sim.valve_normal)),
            ("bDiaphragmValveOpened", str(self.sim.valve_diaphragm)),
            ("bDiaphragmValveClosed", str(not self.sim.valve_diaphragm)),
        ]
        for r in rows:
            self.vac_tree.insert("", "end", values=r)

    # ------------------------------------------------------------------
    # Tab 6: Analysis / Scopes (Embedded Matplotlib)
    # ------------------------------------------------------------------
    def _build_tab_scopes(self):
        f = self.tab_scopes
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)

        presets_path = ROOT / "reference" / "scope_catalog.json"
        self.scope_presets = (
            json.loads(presets_path.read_text(encoding="utf-8")) if presets_path.exists() else {"Default": []}
        )

        top_bar = ttk.Frame(f)
        top_bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(top_bar, text="Preset:").pack(side="left", padx=(0, 4))
        self.scope_preset_var = tk.StringVar(value=list(self.scope_presets.keys())[0])
        preset_cb = ttk.Combobox(
            top_bar,
            textvariable=self.scope_preset_var,
            values=list(self.scope_presets.keys()),
            state="readonly",
            width=28,
        )
        preset_cb.pack(side="left", padx=(0, 12))
        preset_cb.bind("<<ComboboxSelected>>", lambda _e: self._draw_scopes(force=True))

        ttk.Label(top_bar, text="Window [s]:").pack(side="left", padx=(0, 4))
        self.scope_window_var = tk.IntVar(value=60)
        win_spin = ttk.Spinbox(
            top_bar,
            from_=10,
            to=300,
            increment=10,
            textvariable=self.scope_window_var,
            width=6,
            command=lambda: self._draw_scopes(force=True),
        )
        win_spin.pack(side="left", padx=(0, 12))

        self.scope_choices = {
            "Position": ("pos_mm", "Position [mm]", "#5ac8fa"),
            "Velocity": ("vel_mm_s", "Velocity [mm/s]", "#7ce3ab"),
            "Acceleration": ("acc_g", "Acceleration [g]", "#ffbb66"),
            "Torque": ("torque1_nm", "Motor torque [Nm]", "#db9bff"),
            "Motor pos": ("pos1_deg", "Encoder [deg]", "#ff7b89"),
            "Setpoint": ("set_pos_deg", "Setpoint [deg]", "#92e3ff"),
        }
        self.scope_sig_vars: dict[str, tk.BooleanVar] = {}
        for name in self.scope_choices:
            var = tk.BooleanVar(value=name in ("Position", "Velocity", "Acceleration"))
            self.scope_sig_vars[name] = var
            ttk.Checkbutton(
                top_bar,
                text=name,
                variable=var,
                command=lambda: self._draw_scopes(force=True),
            ).pack(side="left", padx=4)

        ttk.Button(top_bar, text="Clear history", command=lambda: self.invoke("clear_history")).pack(
            side="right", padx=4
        )
        ttk.Button(top_bar, text="Export CSV...", style="Primary.TButton", command=self._export_scope_csv).pack(
            side="right", padx=4
        )

        # Matplotlib Figure
        plot_frame = ttk.Frame(f)
        plot_frame.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        self.fig = Figure(figsize=(10, 3.6), dpi=100, facecolor="#101b2a")
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self.plot_canvas, toolbar_frame)
        toolbar.update()

    def _draw_scopes(self, force: bool = False):
        hist = self.sim.history
        if not force and len(hist) == self._last_history_len:
            return
        self._last_history_len = len(hist)

        selected = [k for k, v in self.scope_sig_vars.items() if v.get()]
        self.fig.clear()
        if not selected or len(hist) < 2:
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#101b2a")
            ax.text(
                0.5,
                0.5,
                "Run homing or a motion to populate live scope traces.",
                color=FG_MUTED,
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.plot_canvas.draw_idle()
            return

        try:
            window_s = float(self.scope_window_var.get())
        except ValueError:
            window_s = 60.0

        t_max = hist[-1]["time_s"]
        t_min = t_max - window_s
        recent = [row for row in hist if row["time_s"] >= t_min]
        times = [row["time_s"] for row in recent]

        n_plots = len(selected)
        axes = self.fig.subplots(n_plots, 1, sharex=True)
        if n_plots == 1:
            axes = [axes]

        for ax, name in zip(axes, selected):
            col_key, ylabel, color = self.scope_choices[name]
            y_vals = [row[col_key] for row in recent]
            ax.set_facecolor("#101b2a")
            ax.plot(times, y_vals, color=color, linewidth=1.6)
            if col_key == "pos_mm":
                ax.axhline(TOP_DEG * 5.0 / 360.0, color="#ed7a82", linestyle=":", linewidth=1.0)
            ax.set_ylabel(ylabel, color=FG_PRIMARY, fontsize=8)
            ax.tick_params(colors=FG_MUTED, labelsize=8)
            ax.grid(True, color="#213146", linewidth=0.7)
            for spine in ax.spines.values():
                spine.set_color("#263449")

        axes[-1].set_xlabel("Simulation time [s]", color=FG_PRIMARY, fontsize=9)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.14, hspace=0.18)
        self.plot_canvas.draw_idle()

    def _export_scope_csv(self):
        path = filedialog.asksaveasfilename(
            title="Export Simulated Scope CSV",
            defaultextension=".csv",
            initialfile="akus_simulated_scope.csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if path:
            Path(path).write_bytes(self.sim.csv_bytes())
            self.last_message_var.set(f"Saved CSV trace to {path}")

    # ------------------------------------------------------------------
    # Tab 7: Diagnostics
    # ------------------------------------------------------------------
    def _build_tab_diagnostics(self):
        f = self.tab_diag
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        f.grid_rowconfigure(1, weight=1)

        ttk.Label(f, text="Read-only virtual status", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(f, text="Event log (newest first)", style="Section.TLabel").grid(row=0, column=1, sticky="w")

        self.diag_tree = ttk.Treeview(f, columns=("item", "val"), show="headings", height=10)
        self.diag_tree.heading("item", text="Item", anchor="w")
        self.diag_tree.heading("val", text="Current value", anchor="w")
        self.diag_tree.column("item", width=220, anchor="w")
        self.diag_tree.column("val", width=200, anchor="w")
        self.diag_tree.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=4)

        ev_frame = ttk.Frame(f)
        ev_frame.grid(row=1, column=1, sticky="nsew", pady=4)
        ev_frame.grid_columnconfigure(0, weight=1)
        ev_frame.grid_rowconfigure(0, weight=1)

        self.events_tree = ttk.Treeview(ev_frame, columns=("time_s", "event"), show="headings", height=10)
        self.events_tree.heading("time_s", text="Time [s]", anchor="w")
        self.events_tree.heading("event", text="Event", anchor="w")
        self.events_tree.column("time_s", width=85, anchor="w")
        self.events_tree.column("event", width=420, anchor="w")
        self.events_tree.grid(row=0, column=0, sticky="nsew")

        ev_scroll = ttk.Scrollbar(ev_frame, orient="vertical", command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=ev_scroll.set)
        ev_scroll.grid(row=0, column=1, sticky="ns")

        ttk.Button(f, text="Export event log JSON...", command=self._export_events_json).grid(
            row=2, column=1, sticky="e", pady=4
        )

    def _export_events_json(self):
        path = filedialog.asksaveasfilename(
            title="Export Event Log JSON",
            defaultextension=".json",
            initialfile="akus_simulation_events.json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            Path(path).write_text(json.dumps(self.sim.events, indent=2), encoding="utf-8")
            self.last_message_var.set(f"Exported event log to {path}")

    # ------------------------------------------------------------------
    # Tab 8: Project Reference
    # ------------------------------------------------------------------
    def _build_tab_reference(self):
        f = self.tab_ref
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        ttk.Label(
            f,
            text="Source-to-dashboard coverage & TwinCAT unit constraints",
            style="Section.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        ref_tree = ttk.Treeview(f, columns=("source", "feature", "coverage"), show="headings", height=11)
        ref_tree.heading("source", text="Source", anchor="w")
        ref_tree.heading("feature", text="Feature", anchor="w")
        ref_tree.heading("coverage", text="Current coverage", anchor="w")
        ref_tree.column("source", width=230, anchor="w")
        ref_tree.column("feature", width=390, anchor="w")
        ref_tree.column("coverage", width=360, anchor="w")
        ref_tree.grid(row=1, column=0, sticky="nsew")

        mapping = [
            ("MAIN.TcPOU", "Init / Ready / Moving / ToStart / Ident / CtrlConfig", "Workflow approximation"),
            ("MAIN.TcPOU", "Upward reference search, MC_Home coordinate, return", "Behaviorally simulated"),
            ("MAIN.TcPOU", "Manual position, velocity, timed Jog", "Kinematic approximation"),
            ("MAIN.TcPOU", "Experiment calculation, delay, lock, automatic return", "Core workflow implemented"),
            ("MAIN.TcPOU", "Calibration / single-axis mode", "Not implemented"),
            ("FeedbackControl.TcPOU", "E_OpMode 6–11 mappings", "Labels only; no NC/drive equivalence"),
            ("FB_AkusModel.TcPOU", "Mass, gravity, nonlinear dry/viscous/position friction", "Simplified dynamics option"),
            ("FB_AccIntegrator.TcPOU", "Euler / trapezoid / RK4 selectability", "Not individually replicated"),
            ("GVL_Persist.TcGVL", "Default parameter values", "Imported as editable Python config"),
            ("VISUs/*.TcVIS", "Original page concepts", "Recreated as modern desktop HMI"),
            ("Measurement/*.tcscopex", "Scope channel catalogs", "Catalogs extracted; synthetic traces"),
            ("Beckhoff PLCopen MC_* / EtherCAT", "Real motor drives / torque / homing / E-STOP", "NOT connected for safety"),
        ]
        for row in mapping:
            ref_tree.insert("", "end", values=row)

        ttk.Label(
            f,
            text=(
                "Units & constraints:   1 motor rev = 360° = 5 mm travel   |   "
                "44,000° = 611.11 mm upper limit   |   "
                "Experiment travel guard = 43,280° (601.11 mm)   |   g = 9.80665 m/s²"
            ),
            style="Kicker.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

    # ------------------------------------------------------------------
    # Live Periodic Simulation & UI Refresh (50 ms / 20 FPS)
    # ------------------------------------------------------------------
    def _update_ui_now(self):
        s = self.sim
        self.metric_vars["STATE"].set(s.state)
        self.metric_vars["HEIGHT"].set(f"{s.pos_mm:.1f} mm")
        self.metric_vars["VELOCITY"].set(f"{s.vel_mm_s:.2f} mm/s")
        self.metric_vars["ACCELERATION"].set(f"{s.acc_g:.3f} g")
        self.metric_vars["AXIS SKEW"].set(f"{s.axis_deviation_deg:.1f}°")
        self.metric_vars["TIME"].set(f"{s.t_s:.1f} s")

        if s.fault:
            self.fault_banner.configure(text=f"🚨 FAULT: {s.fault}")
            self.fault_banner.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        else:
            self.fault_banner.grid_forget()

        # Update Status Table
        status_rows = [
            ("Operational state", s.state),
            ("Active step", s.phase),
            ("Mode (E_OpMode)", f"{s.op_mode} = {OP_MODES.get(s.op_mode, '—')}"),
            ("Axis 1 [deg]", f"{s.pos_deg:,.2f}"),
            ("Axis 2 [deg]", f"{s.pos2_deg:,.2f}"),
            ("Upper TouchProbe", "CLOSED (normal)" if s.switch_top_nc else "OPEN (actuated)"),
            ("Reference", "Done" if s.homed else "Required"),
            ("Motors", "Enabled (virtual)" if s.enabled else "Disabled"),
            ("Recording", "ON" if s.recording else "OFF"),
            ("Camera trigger", "ON" if s.camera_on else "OFF"),
        ]
        existing = self.status_tree.get_children()
        if len(existing) == len(status_rows):
            for item_id, row in zip(existing, status_rows):
                self.status_tree.item(item_id, values=row)
        else:
            for item_id in existing:
                self.status_tree.delete(item_id)
            for row in status_rows:
                self.status_tree.insert("", "end", values=row)

        # Draw carrier
        self._draw_carrier()

        # Refresh active tab if Scopes or Diagnostics is visible
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 5:
            self._draw_scopes(force=False)
        elif current_tab == 6:
            self._refresh_diagnostics_tab()

    def _refresh_diagnostics_tab(self):
        snap = self.sim.snapshot()
        diag_rows = [
            ("Controller state", str(snap["state"])),
            ("Reference complete", str(snap["homed"])),
            ("Fault latched", str(snap["e_stop"])),
            ("TouchProbe 1, NC", str(snap["top_probe_nc"])),
            ("TouchProbe 2, NC", str(snap["top_probe_nc"])),
            ("Motor 1 torque, modeled", f"{self.sim.torque1_nm:.3f} Nm"),
            ("Motor 2 torque, modeled", f"{self.sim.torque2_nm:.3f} Nm"),
            ("Position sync error", f"{self.sim.offset2_deg:.2f}°"),
            ("Experiment camera signal", str(self.sim.camera_on)),
            ("Virtual recording", str(self.sim.recording)),
        ]
        existing = self.diag_tree.get_children()
        if len(existing) == len(diag_rows):
            for item_id, row in zip(existing, diag_rows):
                self.diag_tree.item(item_id, values=row)
        else:
            for item_id in existing:
                self.diag_tree.delete(item_id)
            for row in diag_rows:
                self.diag_tree.insert("", "end", values=row)

        if len(self.sim.events) != self._last_events_len:
            self._last_events_len = len(self.sim.events)
            for item_id in self.events_tree.get_children():
                self.events_tree.delete(item_id)
            for ev in reversed(self.sim.events):
                self.events_tree.insert("", "end", values=(f"{ev['time_s']:.3f}", ev["event"]))

    def _update_loop(self):
        now = time.monotonic()
        elapsed = now - self.wall_clock
        self.wall_clock = now
        if self.live_clock_var.get() and (self.sim.current or self.sim.next_step or self.sim.operation):
            self.sim.step(max(0.0, min(elapsed, 0.5)) * float(self.speed_var.get()))
        self._update_ui_now()
        self.after(60, self._update_loop)


if __name__ == "__main__":
    app = AkusDesktopApp()
    app.mainloop()
