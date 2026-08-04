"""
plassys_master_gui.py

The combined, final app: one window with

  LEFT  : Live Angle Tester (type alpha/theta, see the wafer move in
          real time, now with THREE reference conventions: load/mill/
          deposition) + Recipe Design Parameters + "Generate Recipe"
  RIGHT : A notebook with three tabs --
            "Junction Design"    -- live proportional preview of the
                junction geometry (fingers, adhesion plates, patches,
                Nb electrodes) as you type dimensions/angles, plus the
                overlap shadow-evaporation calculator. Comes FIRST so
                the geometry is confirmed before generating a recipe.
            "Chamber & Recipe"   -- top-down + side chamber views
                (updating live) with a resolved-orientation readout,
                and the generated recipe below it -- click any step's
                angle to load it into the Live Angle Tester.
            "Junction Formation" -- a step-by-step schematic slideshow
                of how the junction physically forms, including mill-
                phase highlighting of which Nb region is being milled,
                persistent "already deposited" / "already milled"
                state, and oxidation-step labeling.

All rotation math is in rotation_core.py, all chamber drawing is in
plotting.py, all recipe logic is in recipe_model.py, and the junction
schematic drawing is in junction_view.py. This file is presentation only.

Run with:  python plassys_master_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from rotation_core import (
    compute_orientation,
    preset_load,
    preset_mill,
    preset_deposition,
    alpha_internal_to_display,
    WaferOrientation,
)
from plotting import (
    draw_top_down, draw_side_view, BG_COLOR, TEXT_COLOR,
    FLAT_MARKER_COLOR, LOCAL_X_COLOR, LOCAL_Y_COLOR, NORMAL_COLOR,
)
from recipe_model import (
    DesignParameters,
    SidewallMill,
    generate_recipe_steps,
    format_recipe_text,
    compute_alpha_from_linewidth,
    compute_overlap_shadow_bounds,
)
from junction_view import draw_junction_step, draw_junction_preview

PANEL_COLOR = "#14151b"
ACCENT_COLOR = "#7fb8ff"
STAR_COLOR = "#ffd166"
WARN_COLOR = "#ff8f6b"
DIM_COLOR = "#8a8d9e"


class SidewallRow(ttk.Frame):
    """One editable (alpha, theta, label) row for a sidewall mill entry."""

    def __init__(self, parent, on_remove, alpha="60", theta="0", label=""):
        super().__init__(parent)
        self.on_remove = on_remove

        ttk.Label(self, text="alpha:").pack(side="left")
        self.alpha_entry = ttk.Entry(self, width=6)
        self.alpha_entry.insert(0, alpha)
        self.alpha_entry.pack(side="left", padx=(2, 8))

        ttk.Label(self, text="theta:").pack(side="left")
        self.theta_entry = ttk.Entry(self, width=6)
        self.theta_entry.insert(0, theta)
        self.theta_entry.pack(side="left", padx=(2, 8))

        ttk.Label(self, text="label:").pack(side="left")
        self.label_entry = ttk.Entry(self, width=12)
        self.label_entry.insert(0, label)
        self.label_entry.pack(side="left", padx=(2, 8))

        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        ttk.Button(self, text="x", width=2, command=self._remove).pack(side="left")

    def _remove(self):
        self.on_remove(self)

    def get_values(self):
        alpha = float(self.alpha_entry.get())
        theta = float(self.theta_entry.get())
        label = self.label_entry.get().strip()
        return SidewallMill(alpha=alpha, theta=theta, label=label)


class MasterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Plassys PVD / Ion-Mill — Wafer Orientation & Recipe Generator")
        self.root.geometry("1760x1000")
        self.root.configure(bg=BG_COLOR)

        self.reference_var = tk.StringVar(value="load")
        self.current_orientation: WaferOrientation = preset_load()
        self.sidewall_rows = []
        self._last_recipe_text = ""
        self._suppress_live_update = False
        self._geometry_confirmed = False

        # Junction-formation slideshow state
        self._jf_steps = []       # full ordered list of RecipeStep
        self._jf_index = 0
        self._jf_params = None

        self._build_style()
        self._build_layout()
        self._on_live_change()
        self._update_preview()

    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 12, "bold"))
        style.configure("Dim.TLabel", background=BG_COLOR, foreground=DIM_COLOR, font=("Segoe UI", 8))
        style.configure("TButton", font=("Segoe UI", 10), padding=5)
        style.configure("TCheckbutton", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TRadiobutton", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 9))
        style.configure("TEntry", font=("Consolas", 11))
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_COLOR, foreground=TEXT_COLOR,
                        padding=(14, 6), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", BG_COLOR)],
                  foreground=[("selected", ACCENT_COLOR)])

    # ------------------------------------------------------------------
    # TOP-LEVEL LAYOUT: left (scrollable controls) | right (notebook)
    # ------------------------------------------------------------------
    def _build_layout(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # ---- LEFT: scrollable control column ----
        left_outer = ttk.Frame(main, width=430)
        left_outer.pack(side="left", fill="y", padx=(0, 8))
        left_outer.pack_propagate(False)

        canvas = tk.Canvas(left_outer, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        left = ttk.Frame(canvas)
        left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=left, anchor="nw", width=410)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_live_tester(left)
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=12)
        self._build_recipe_inputs(left)

        # ---- RIGHT: notebook -- Junction Design, Chamber & Recipe, Junction Formation ----
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        design_tab = ttk.Frame(self.notebook)
        chamber_tab = ttk.Frame(self.notebook)
        junction_tab = ttk.Frame(self.notebook)
        self.notebook.add(design_tab, text="Junction Design")
        self.notebook.add(chamber_tab, text="Chamber & Recipe")
        self.notebook.add(junction_tab, text="Junction Formation")

        self._build_design_tab(design_tab)
        self._build_chamber_tab(chamber_tab)
        self._build_junction_tab(junction_tab)

    # ------------------------------------------------------------------
    # LEFT: Live Angle Tester
    # ------------------------------------------------------------------
    def _build_live_tester(self, parent):
        ttk.Label(parent, text="Live Angle Tester", style="Header.TLabel").pack(anchor="w", pady=(4, 8))
        ttk.Label(parent, text="Type a value and the wafer updates instantly.", style="Dim.TLabel").pack(anchor="w")

        ref_frame = ttk.Frame(parent)
        ref_frame.pack(fill="x", pady=(8, 6))
        ttk.Radiobutton(ref_frame, text="Load ref (0=load, 90=mill)", variable=self.reference_var,
                         value="load", command=self._on_live_change).pack(anchor="w")
        ttk.Radiobutton(ref_frame, text="Mill ref (0=mill, -90=load, 90=dep)", variable=self.reference_var,
                         value="mill", command=self._on_live_change).pack(anchor="w")
        ttk.Radiobutton(ref_frame, text="Deposition ref (0=dep, -90=mill, -180=load)",
                         variable=self.reference_var,
                         value="deposition", command=self._on_live_change).pack(anchor="w")
        ttk.Label(parent, text="(deposition ref: UNVERIFIED whether the machine really\n"
                               "re-zeros alpha a 2nd time here -- test and confirm.)",
                  style="Dim.TLabel", justify="left").pack(anchor="w", pady=(2, 0))

        alpha_frame = ttk.Frame(parent)
        alpha_frame.pack(fill="x", pady=(6, 4))
        ttk.Label(alpha_frame, text="Alpha (deg):").pack(side="left")
        self.live_alpha_entry = ttk.Entry(alpha_frame, width=10)
        self.live_alpha_entry.insert(0, "0")
        self.live_alpha_entry.pack(side="right")
        self.live_alpha_entry.bind("<KeyRelease>", self._on_live_change)

        theta_frame = ttk.Frame(parent)
        theta_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(theta_frame, text="Theta (deg):").pack(side="left")
        self.live_theta_entry = ttk.Entry(theta_frame, width=10)
        self.live_theta_entry.insert(0, "0")
        self.live_theta_entry.pack(side="right")
        self.live_theta_entry.bind("<KeyRelease>", self._on_live_change)

        preset_frame = ttk.Frame(parent)
        preset_frame.pack(fill="x", pady=(0, 6))
    # ------------------------------------------------------------------
    # LEFT: Recipe Design Parameters
    # ------------------------------------------------------------------
    def _build_recipe_inputs(self, parent):
        ttk.Label(parent, text="Recipe Design Parameters", style="Header.TLabel").pack(anchor="w", pady=(0, 8))

        ttk.Label(parent, text="Deposition alpha (fixed for batch):").pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(2, 6))
        self.alpha_dep_entry = ttk.Entry(row, width=10)
        self.alpha_dep_entry.insert(0, "30")
        self.alpha_dep_entry.pack(side="left")
        ttk.Button(row, text="send to tester", command=self._send_dep_alpha_to_tester).pack(side="left", padx=6)

        ttk.Label(parent, text="-- or compute alpha from line width --", style="Dim.TLabel").pack(anchor="w", pady=(2, 2))
        calc_row1 = ttk.Frame(parent)
        calc_row1.pack(fill="x", pady=2)
        ttk.Label(calc_row1, text="LW (nm):").pack(side="left")
        self.lw_entry = ttk.Entry(calc_row1, width=8)
        self.lw_entry.pack(side="left", padx=(2, 10))
        ttk.Label(calc_row1, text="PMMA (nm):").pack(side="left")
        self.pmma_entry = ttk.Entry(calc_row1, width=8)
        self.pmma_entry.insert(0, "875")
        self.pmma_entry.pack(side="left", padx=(2, 0))

        calc_row2 = ttk.Frame(parent)
        calc_row2.pack(fill="x", pady=2)
        ttk.Label(calc_row2, text="margin (nm):").pack(side="left")
        self.margin_entry = ttk.Entry(calc_row2, width=8)
        self.margin_entry.insert(0, "100")
        self.margin_entry.pack(side="left", padx=(2, 10))
        ttk.Button(calc_row2, text="Compute -> alpha", command=self._compute_alpha).pack(side="left")

        pmgi_row = ttk.Frame(parent)
        pmgi_row.pack(fill="x", pady=(6, 8))
        ttk.Label(pmgi_row, text="PMGI thickness (nm):").pack(side="left")
        self.pmgi_entry = ttk.Entry(pmgi_row, width=8)
        self.pmgi_entry.insert(0, "600")
        self.pmgi_entry.pack(side="right")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

        # ---- Finger geometry ----
        ttk.Label(parent, text="Finger geometry", style="Header.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(parent, text="Angle = direction FROM the crossing TO this finger's\n"
                               "own Nb electrode (0=right, 90=up, 180=left, 270=down, ...).",
                  style="Dim.TLabel", justify="left").pack(anchor="w", pady=(0, 6))

        h_row = ttk.Frame(parent)
        h_row.pack(fill="x", pady=2)
        ttk.Label(h_row, text="Finger 1 angle (deg):").pack(side="left")
        self.h_angle_entry = ttk.Entry(h_row, width=7)
        self.h_angle_entry.insert(0, "0")
        self.h_angle_entry.pack(side="right")
        self.h_angle_entry.bind("<KeyRelease>", self._update_preview)

        v_row = ttk.Frame(parent)
        v_row.pack(fill="x", pady=2)
        ttk.Label(v_row, text="Finger 2 angle (deg):").pack(side="left")
        self.v_angle_entry = ttk.Entry(v_row, width=7)
        self.v_angle_entry.insert(0, "270")
        self.v_angle_entry.pack(side="right")
        self.v_angle_entry.bind("<KeyRelease>", self._update_preview)

        # New perpendicularity validation warning readout placement
        self.perp_validation_label = tk.Label(parent, text="", bg=BG_COLOR, fg=WARN_COLOR,
                                              font=("Segoe UI", 9, "bold"), justify="left", anchor="w")
        self.perp_validation_label.pack(fill="x", pady=(2, 4))

        first_row = ttk.Frame(parent)
        first_row.pack(fill="x", pady=(6, 2))
        ttk.Label(first_row, text="Deposit first:").pack(side="left")
        self.deposit_first_var = tk.StringVar(value="horizontal")
        ttk.Radiobutton(first_row, text="Finger 1", variable=self.deposit_first_var,
                         value="horizontal", command=self._update_preview).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(first_row, text="Finger 2", variable=self.deposit_first_var,
                         value="vertical", command=self._update_preview).pack(side="left")
        ttk.Label(parent, text="(the OTHER finger gets the connect+overlap pair)",
                  style="Dim.TLabel").pack(anchor="w", pady=(0, 6))

        hlw_row = ttk.Frame(parent)
        hlw_row.pack(fill="x", pady=2)
        ttk.Label(hlw_row, text="Finger 1 LW (nm):").pack(side="left")
        self.h_lw_entry = ttk.Entry(hlw_row, width=8)
        self.h_lw_entry.insert(0, "200")
        self.h_lw_entry.pack(side="right")
        self.h_lw_entry.bind("<KeyRelease>", self._update_preview)

        vlw_row = ttk.Frame(parent)
        vlw_row.pack(fill="x", pady=2)
        ttk.Label(vlw_row, text="Finger 2 LW (nm):").pack(side="left")
        self.v_lw_entry = ttk.Entry(vlw_row, width=8)
        self.v_lw_entry.insert(0, "220")
        self.v_lw_entry.pack(side="right")
        self.v_lw_entry.bind("<KeyRelease>", self._update_preview)

        hlen_row = ttk.Frame(parent)
        hlen_row.pack(fill="x", pady=2)
        ttk.Label(hlen_row, text="Finger 1 length (um):").pack(side="left")
        self.h_len_entry = ttk.Entry(hlen_row, width=8)
        self.h_len_entry.insert(0, "3")
        self.h_len_entry.pack(side="right")
        self.h_len_entry.bind("<KeyRelease>", self._update_preview)

        vlen_row = ttk.Frame(parent)
        vlen_row.pack(fill="x", pady=(2, 8))
        ttk.Label(vlen_row, text="Finger 2 length (um):").pack(side="left")
        self.v_len_entry = ttk.Entry(vlen_row, width=8)
        self.v_len_entry.insert(0, "3")
        self.v_len_entry.pack(side="right")
        self.v_len_entry.bind("<KeyRelease>", self._update_preview)

        ttk.Label(parent, text="Adhesion plates (same dims, both fingers):",
                  style="Dim.TLabel").pack(anchor="w")
        plate_row = ttk.Frame(parent)
        plate_row.pack(fill="x", pady=(2, 8))
        ttk.Label(plate_row, text="width (um):").pack(side="left")
        self.plate_w_entry = ttk.Entry(plate_row, width=6)
        self.plate_w_entry.insert(0, "1.0")
        self.plate_w_entry.pack(side="left", padx=(2, 10))
        self.plate_w_entry.bind("<KeyRelease>", self._update_preview)
        ttk.Label(plate_row, text="height (um):").pack(side="left")
        self.plate_h_entry = ttk.Entry(plate_row, width=6)
        self.plate_h_entry.insert(0, "1.0")
        self.plate_h_entry.pack(side="left", padx=(2, 0))
        self.plate_h_entry.bind("<KeyRelease>", self._update_preview)

        overlap_row = ttk.Frame(parent)
        overlap_row.pack(fill="x", pady=(0, 8))
        ttk.Label(overlap_row, text="Target overlap (um):").pack(side="left")
        self.overlap_entry = ttk.Entry(overlap_row, width=8)
        self.overlap_entry.insert(0, "1.5")
        self.overlap_entry.pack(side="right")

        # ---- Patch-integrated (reveal-on-check) ----
        self.patch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Patch-integrated design", variable=self.patch_var,
                         command=self._on_patch_toggle).pack(anchor="w", pady=(4, 2))

        self.patch_frame = ttk.Frame(parent)
        # Not packed yet -- revealed by _on_patch_toggle when checked.

        pc_row = ttk.Frame(self.patch_frame)
        pc_row.pack(fill="x", pady=2)
        ttk.Label(pc_row, text="Number of patches:").pack(side="left")
        self.patch_count_entry = ttk.Entry(pc_row, width=6)
        self.patch_count_entry.insert(0, "3")
        self.patch_count_entry.pack(side="right")
        self.patch_count_entry.bind("<KeyRelease>", self._update_preview)

        psp_row = ttk.Frame(self.patch_frame)
        psp_row.pack(fill="x", pady=2)
        ttk.Label(psp_row, text="Spacing between patches (um):").pack(side="left")
        self.patch_spacing_entry = ttk.Entry(psp_row, width=6)
        self.patch_spacing_entry.insert(0, "1.0")
        self.patch_spacing_entry.pack(side="right")
        self.patch_spacing_entry.bind("<KeyRelease>", self._update_preview)

        pdim_row = ttk.Frame(self.patch_frame)
        pdim_row.pack(fill="x", pady=2)
        ttk.Label(pdim_row, text="Patch width (um):").pack(side="left")
        self.patch_w_entry = ttk.Entry(pdim_row, width=6)
        self.patch_w_entry.insert(0, "0.5")
        self.patch_w_entry.pack(side="left", padx=(2, 10))
        self.patch_w_entry.bind("<KeyRelease>", self._update_preview)
        ttk.Label(pdim_row, text="height (um):").pack(side="left")
        self.patch_h_entry = ttk.Entry(pdim_row, width=6)
        self.patch_h_entry.insert(0, "1.5")
        self.patch_h_entry.pack(side="left", padx=(2, 0))
        self.patch_h_entry.bind("<KeyRelease>", self._update_preview)

        pangle_row = ttk.Frame(self.patch_frame)
        pangle_row.pack(fill="x", pady=(2, 4))
        ttk.Label(pangle_row, text="Patch angle (deg):").pack(side="left")
        self.patch_angle_entry = ttk.Entry(pangle_row, width=6)
        self.patch_angle_entry.insert(0, "45")
        self.patch_angle_entry.pack(side="right")
        self.patch_angle_entry.bind("<KeyRelease>", self._update_preview)
        ttk.Label(self.patch_frame,
                  text="Adds a 3rd Al deposition (2 sub-steps, angle/angle+180),\n"
                       "additive to both finger steps -- not a replacement.",
                  style="Dim.TLabel", justify="left").pack(anchor="w", pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(parent, text="Ion milling", style="Header.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(parent, text="Top mill: alpha always 0 (full top-surface contact).",
                  style="Dim.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Sidewall mills (your choice, add as needed):").pack(anchor="w", pady=(6, 2))

        self.sidewall_container = ttk.Frame(parent)
        self.sidewall_container.pack(fill="x")
        ttk.Button(parent, text="+ Add sidewall mill", command=self._add_sidewall_row).pack(anchor="w", pady=6)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

        wait_row = ttk.Frame(parent)
        wait_row.pack(fill="x", pady=(0, 4))
        ttk.Label(wait_row, text="Final wait: 20 minutes (fixed)").pack(side="left")
        ttk.Label(parent, text="Arbitrary constant, not user-editable -- no derivation behind this number.",
                  style="Dim.TLabel").pack(anchor="w")

        ttk.Label(parent, text="Notes:").pack(anchor="w", pady=(6, 2))
        self.notes_text = tk.Text(parent, width=40, height=3, bg=PANEL_COLOR, fg=TEXT_COLOR,
                                   font=("Segoe UI", 9), relief="flat", padx=6, pady=6)
        self.notes_text.pack(fill="x")

        ttk.Button(parent, text="Generate Recipe \u2193", command=self._generate_recipe).pack(
            fill="x", pady=(14, 20))

        self._add_sidewall_row(alpha="60", theta="0", label="sidewall 1")
        self._add_sidewall_row(alpha="60", theta="270", label="sidewall 2")

    def _on_patch_toggle(self):
        if self.patch_var.get():
            self.patch_frame.pack(fill="x", pady=(2, 6))
        else:
            self.patch_frame.pack_forget()
        self._update_preview()

    def _add_sidewall_row(self, alpha="60", theta="0", label=""):
        row = SidewallRow(self.sidewall_container, self._remove_sidewall_row, alpha=alpha, theta=theta, label=label)
        row.pack(anchor="w", pady=2)
        self.sidewall_rows.append(row)

    def _remove_sidewall_row(self, row):
        row.destroy()
        self.sidewall_rows.remove(row)

    def _compute_alpha(self):
        try:
            lw = float(self.lw_entry.get())
            pmma = float(self.pmma_entry.get())
            margin = float(self.margin_entry.get())
            alpha = compute_alpha_from_linewidth(lw, pmma, margin)
            self.alpha_dep_entry.delete(0, tk.END)
            self.alpha_dep_entry.insert(0, f"{alpha:.4f}")
        except ValueError as e:
            messagebox.showerror("Cannot compute alpha", str(e))
        except Exception:
            messagebox.showerror("Cannot compute alpha", "Check that line width and PMMA thickness are numeric.")

    def _send_dep_alpha_to_tester(self):
        """Load the deposition alpha into the Live Angle Tester (mill reference) for a quick look."""
        try:
            alpha = float(self.alpha_dep_entry.get())
        except ValueError:
            return
        self.reference_var.set("mill")
        self.live_alpha_entry.delete(0, tk.END)
        self.live_alpha_entry.insert(0, f"{alpha:g}")
        self._on_live_change()

    # ------------------------------------------------------------------
    # RIGHT TAB 1: Junction Design (live preview + overlap calculator)
    # ------------------------------------------------------------------
    def _build_design_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="both", expand=True)

        left_col = ttk.Frame(top, width=380)
        left_col.pack(side="left", fill="y", padx=(0, 8))
        left_col.pack_propagate(False)

        ttk.Label(left_col, text="Junction Design", style="Header.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(left_col,
                  text="Live preview of your junction geometry from the\n"
                       "parameters on the far left. Confirm here before\n"
                       "generating a recipe.",
                  style="Dim.TLabel", justify="left").pack(anchor="w", pady=(0, 10))

        ttk.Label(left_col, text="Overlap shadow-evaporation calculator", style="Header.TLabel").pack(
            anchor="w", pady=(4, 4))
        ttk.Label(left_col,
                  text="dx = (t_PMMA + t_PMGI) x tan(alpha)  -- min-overlap bound\n"
                       "dy = t_PMGI x tan(alpha)              -- max-overlap bound\n"
                       "Uses the alpha / PMMA / PMGI values from the left panel.",
                  style="Dim.TLabel", justify="left").pack(anchor="w", pady=(0, 6))

        ttk.Button(left_col, text="Compute overlap bounds", command=self._compute_overlap_bounds).pack(
            anchor="w", pady=(0, 6))
        self.overlap_result_label = tk.Label(left_col, text="dx = -- um   dy = -- um   avg = -- um",
                                              bg=PANEL_COLOR, fg=TEXT_COLOR, font=("Consolas", 10),
                                              justify="left", anchor="w", padx=8, pady=8)
        self.overlap_result_label.pack(fill="x", pady=(0, 4))

        ttk.Separator(left_col, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left_col, text="JJ area (Finger 1 LW x Finger 2 LW):", style="Dim.TLabel").pack(anchor="w")
        self.area_label = tk.Label(left_col, text="-- um^2", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                    font=("Consolas", 11), anchor="w", padx=8, pady=6)
        self.area_label.pack(fill="x", pady=(2, 12))

        self.confirm_button = ttk.Button(left_col, text="Confirm Junction Design",
                                          command=self._toggle_design_lock)
        self.confirm_button.pack(fill="x", pady=(0, 4))
        self.confirm_status_label = tk.Label(left_col, text="Not yet confirmed.", bg=PANEL_COLOR,
                                              fg=WARN_COLOR, font=("Segoe UI", 9), anchor="w", padx=8, pady=6)
        self.confirm_status_label.pack(fill="x")

        right_col = ttk.Frame(top)
        right_col.pack(side="left", fill="both", expand=True)
        self.design_fig = Figure(figsize=(7, 7), dpi=100, facecolor=BG_COLOR)
        self.design_ax = self.design_fig.add_subplot(1, 1, 1)
        self.design_canvas = FigureCanvasTkAgg(self.design_fig, master=right_col)
        self.design_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _compute_overlap_bounds(self):
        try:
            alpha = float(self.alpha_dep_entry.get())
            pmma = float(self.pmma_entry.get())
            pmgi = float(self.pmgi_entry.get())
            
            # Read overlap string in micrometers and convert it to nanometers (* 1000)
            try:
                raw_val = self.overlap_entry.get()
                chosen_overlap_nm = float(raw_val) * 1000.0 if raw_val.strip() else 1500.0
            except ValueError:
                chosen_overlap_nm = 1500.0

            # Compute actual bounds
            dx, dy, min_overlap, max_overlap, is_safe = compute_overlap_shadow_bounds(
                pmma, pmgi, alpha, chosen_overlap_nm=chosen_overlap_nm
            )
            
            # Save recommended target (dx + a safety margin, e.g., 1.50)
            self._recommended_safe_overlap = max(1.5, round(dx + 0.65, 2))
            
            # Build a short, non-clipping text layout string
            text_lines = [
                f"dx = {dx:.2f} \u00b5m | dy = {dy:.2f} \u00b5m",
                f"Worst-case Min: {min_overlap:.2f} \u00b5m"
            ]
            
            # Add a clean, brief conditional warning tag at the bottom
            if not is_safe:
                text_lines.append("[WARNING: Negative Overlap!]")
                self.overlap_result_label.configure(foreground="#ff4d4d") # Red alert
            else:
                self.overlap_result_label.configure(foreground=TEXT_COLOR) # Standard
                
            self.overlap_result_label.configure(text="\n".join(text_lines))
                     
        except ValueError:
            messagebox.showerror("Cannot compute", "Alpha, PMMA thickness, and PMGI thickness must be numeric.")

    def _flash_generate_button(self, count=0):
        if not getattr(self, "_design_confirmed", False):
            # Revert to standard look when unlocked or reset
            self.style.configure("Flashing.TButton", background="SystemButtonFace", foreground="black")
            self.generate_recipe_btn.configure(style="TButton")
            return

        # Toggle styles to override system themes cleanly
        if count % 2 == 0:
            self.style.configure("Flashing.TButton", background="#ffd166", foreground="#1e1f26")
        else:
            self.style.configure("Flashing.TButton", background="#7fb8ff", foreground="#1e1f26")

        self.generate_recipe_btn.configure(style="Flashing.TButton")
        self.root.after(500, lambda: self._flash_generate_button(count + 1))

    def _toggle_design_lock(self):
        if not hasattr(self, "_design_confirmed"):
            self._design_confirmed = False

        if not self._design_confirmed:
            self._design_confirmed = True
            self.confirm_button.configure(text="Unlock & Re-edit") # Match your variable name
            self.confirm_status_label.configure(text="Design Confirmed! Ready to generate.", fg="#8fd694")
            state_setting = "disabled"
            self._flash_generate_button()
        else:
            self._design_confirmed = False
            self.confirm_button.configure(text="Confirm Junction Design") # Match your variable name
            self.confirm_status_label.configure(text="Not yet confirmed.", fg=WARN_COLOR)
            state_setting = "normal"

        # Toggles entry inputs on the far left panel
        self.overlap_entry.configure(state=state_setting)
        self.pmma_entry.configure(state=state_setting)
        self.pmgi_entry.configure(state=state_setting)
        self.alpha_dep_entry.configure(state=state_setting)
    

    def _confirm_design(self):
        self._geometry_confirmed = True
        self.confirm_status_label.configure(text="\u2713 Confirmed -- ready to generate a recipe.", fg=ACCENT_COLOR)

    def _update_preview(self, event=None):
        """Live-redraws the Junction Design preview from current field values.
        Validates perpendicularity and dynamically alters confirmation button permissions."""
        try:
            h_ang_str = self.h_angle_entry.get().strip()
            v_ang_str = self.v_angle_entry.get().strip()
            
            # Reset error status and freeze buttons if the fields are empty or mid-typing
            if not h_ang_str or not v_ang_str or h_ang_str in ["-", "."] or v_ang_str in ["-", "."]:
                self.perp_validation_label.configure(text="")
                self.confirm_button.configure(state="disabled")
                return

            h_ang = float(h_ang_str)
            v_ang = float(v_ang_str)
            
            # Check Modulo 180 perpendicularity condition (difference must be 90)
            angular_diff = abs(h_ang - v_ang) % 180
            if abs(angular_diff - 90.0) > 1e-3:
                target_choice_1 = (h_ang + 90) % 360
                target_choice_2 = (h_ang + 270) % 360
                
                self.perp_validation_label.configure(
                    text=f"Invalid input: angle must be {target_choice_1:g}° or {target_choice_2:g}°\n"
                         f"to be perpendicular to Finger 1 ({h_ang:g}°).",
                    fg=WARN_COLOR
                )
                self.confirm_button.configure(state="disabled")
                return
            else:
                self.perp_validation_label.configure(text="✓ Angles are perpendicular", fg=ACCENT_COLOR)
                self.confirm_button.configure(state="normal")

            params = self._collect_recipe_params(for_preview=True)
        except ValueError:
            self.confirm_button.configure(state="disabled")
            return
            
        draw_junction_preview(self.design_ax, params)
        self.design_canvas.draw()
        area = params.jj_area_um2()
        if area is not None:
            self.area_label.configure(text=f"{area:.4f} um^2  "
                                            f"({params.horizontal_lw_nm:g} nm x {params.vertical_lw_nm:g} nm)")
        
        if self._geometry_confirmed:
            self._geometry_confirmed = False
            self.confirm_status_label.configure(text="Geometry changed -- please re-confirm.", fg=WARN_COLOR)

    # ------------------------------------------------------------------
    # RIGHT TAB 2: Chamber & Recipe
    # ------------------------------------------------------------------
    def _build_chamber_tab(self, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(9, 4.6), dpi=100, facecolor=BG_COLOR)
        self.ax_top = self.fig.add_subplot(1, 2, 1)
        self.ax_side = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.readout_label = tk.Label(parent, text="", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                       font=("Consolas", 10), justify="left", anchor="w", padx=10, pady=8)
        self.readout_label.pack(fill="x", pady=(6, 6))

        out_header = ttk.Frame(parent)
        out_header.pack(fill="x")
        ttk.Label(out_header, text="Generated Recipe", style="Header.TLabel").pack(side="left")
        ttk.Label(out_header, text="  (click a step's angle line to preview it above)",
                  style="Dim.TLabel").pack(side="left")
        ttk.Button(out_header, text="Save as .txt", command=self._save_txt).pack(side="right")

        self.output_text = tk.Text(parent, bg=PANEL_COLOR, fg=TEXT_COLOR,
                                    font=("Consolas", 10), relief="flat", padx=12, pady=10,
                                    spacing1=1, spacing3=4, wrap="word", height=16, cursor="arrow")
        self.output_text.pack(fill="both", expand=True, pady=(6, 0))
        self.output_text.tag_configure("star", foreground=STAR_COLOR, font=("Consolas", 10, "bold"))
        self.output_text.tag_configure("header", foreground=ACCENT_COLOR, font=("Consolas", 12, "bold"))
        self.output_text.tag_configure("detail", foreground=DIM_COLOR, font=("Consolas", 9))
        self.output_text.tag_configure("warn", foreground=WARN_COLOR, font=("Consolas", 9))
        self.output_text.tag_configure("clickable", underline=True)
        self.output_text.insert(tk.END, "Set your parameters on the left, confirm the Junction Design "
                                         "tab, then click \"Generate Recipe\" to see the recipe here.")
        self.output_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # RIGHT TAB 3: Junction Formation slideshow
    # ------------------------------------------------------------------
    def _build_junction_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="both", expand=True)

        list_frame = ttk.Frame(top, width=340)
        list_frame.pack(side="left", fill="y", padx=(0, 8))
        list_frame.pack_propagate(False)

        ttk.Label(list_frame, text="Recipe Steps", style="Header.TLabel").pack(anchor="w", pady=(4, 4))
        ttk.Label(list_frame, text="Click a step, or use Prev/Next below.",
                  style="Dim.TLabel").pack(anchor="w", pady=(0, 6))

        self.jf_listbox = tk.Listbox(list_frame, bg=PANEL_COLOR, fg=TEXT_COLOR,
                                      font=("Consolas", 9), selectbackground=STAR_COLOR,
                                      selectforeground="#1e1f26", relief="flat",
                                      activestyle="none", highlightthickness=0)
        self.jf_listbox.pack(fill="both", expand=True)
        self.jf_listbox.bind("<<ListboxSelect>>", self._on_jf_listbox_select)

        right_col = ttk.Frame(top)
        right_col.pack(side="left", fill="both", expand=True)

        self.jf_fig = Figure(figsize=(7, 7), dpi=100, facecolor=BG_COLOR)
        self.jf_ax = self.jf_fig.add_subplot(1, 1, 1)
        self.jf_canvas = FigureCanvasTkAgg(self.jf_fig, master=right_col)
        self.jf_canvas.get_tk_widget().pack(fill="both", expand=True)

        nav_frame = ttk.Frame(right_col)
        nav_frame.pack(fill="x", pady=(6, 4))
        ttk.Button(nav_frame, text="\u25c0 Prev", command=self._jf_prev).pack(side="left", padx=4)
        self.jf_step_label = ttk.Label(nav_frame, text="No recipe generated yet.", style="Header.TLabel")
        self.jf_step_label.pack(side="left", expand=True)
        ttk.Button(nav_frame, text="Next \u25b6", command=self._jf_next).pack(side="right", padx=4)

        self.jf_detail_label = tk.Label(right_col, text="", bg=PANEL_COLOR, fg=TEXT_COLOR,
                                         font=("Segoe UI", 9), justify="left", anchor="w",
                                         wraplength=700, padx=10, pady=8)
        self.jf_detail_label.pack(fill="x")

    # ------------------------------------------------------------------
    # Live angle tester logic
    # ------------------------------------------------------------------
    def _on_live_change(self, event=None):
        if self._suppress_live_update:
            return
        try:
            alpha_val = float(self.live_alpha_entry.get())
            theta_val = float(self.live_theta_entry.get())
        except ValueError:
            return  # user is mid-typing (e.g. "-" or ""); just wait for more input

        reference = self.reference_var.get()
        orientation = compute_orientation(alpha_val, theta_val, reference=reference)
        self.current_orientation = orientation
        self._redraw_plots(orientation)
        self._update_readout(orientation)

    def _apply_preset(self, name: str):
        self._suppress_live_update = True
        if name == "load":
            self.reference_var.set("load")
            self._set_entry(self.live_alpha_entry, "0")
            self._set_entry(self.live_theta_entry, "0")
        elif name == "mill":
            self.reference_var.set("mill")
            self._set_entry(self.live_alpha_entry, "0")
            self._set_entry(self.live_theta_entry, "0")
        elif name == "deposition":
            self.reference_var.set("deposition")
            self._set_entry(self.live_alpha_entry, "0")
            self._set_entry(self.live_theta_entry, "0")
        self._suppress_live_update = False
        self._on_live_change()

    @staticmethod
    def _set_entry(entry, value):
        entry.delete(0, tk.END)
        entry.insert(0, value)

    def _redraw_plots(self, o: WaferOrientation):
        draw_top_down(self.ax_top, o)
        draw_side_view(self.ax_side, o)
        self.canvas.draw()

    def _update_readout(self, o: WaferOrientation):
        alpha_load = alpha_internal_to_display(o.alpha_internal, "load")
        alpha_mill = alpha_internal_to_display(o.alpha_internal, "mill")
        alpha_dep = alpha_internal_to_display(o.alpha_internal, "deposition")
        preset_label = self._nearest_preset_label(o)
        corrected_note = "  [auto-corrected +180]" if o.autocorrected else ""
        text = (
            f"alpha(load)={alpha_load:6.2f}  alpha(mill)={alpha_mill:6.2f}  "
            f"alpha(dep)={alpha_dep:6.2f}  theta={o.theta:6.2f}"
            f"{corrected_note}   |  nearest preset: {preset_label}\n"
            f"normal=({o.normal_vector[0]:+.3f}, {o.normal_vector[1]:+.3f}, {o.normal_vector[2]:+.3f})"
            f"   flat=({o.flat_vector[0]:+.3f}, {o.flat_vector[1]:+.3f}, {o.flat_vector[2]:+.3f})"
        )
        self.readout_label.configure(text=text)

    def _nearest_preset_label(self, o: WaferOrientation) -> str:
        presets = {"LOAD": preset_load(), "MILL": preset_mill(), "DEPOSITION": preset_deposition()}
        best_name, best_dist = None, None
        for name, p in presets.items():
            d = abs(p.alpha_internal - o.alpha_internal) + min(abs(p.theta - o.theta), 360 - abs(p.theta - o.theta))
            if best_dist is None or d < best_dist:
                best_dist, best_name = d, name
        if best_dist is not None and best_dist < 1.0:
            return f"{best_name} (exact)"
        elif best_dist is not None and best_dist < 15.0:
            return f"near {best_name}"
        return "(custom)"

    # ------------------------------------------------------------------
    # Recipe generation + clickable output
    # ------------------------------------------------------------------
    def _collect_recipe_params(self, for_preview=False) -> DesignParameters:
        sidewalls = [row.get_values() for row in self.sidewall_rows] if not for_preview else []

        def opt_float(entry):
            s = entry.get().strip()
            return float(s) if s else None

        return DesignParameters(
            alpha_deposition=float(self.alpha_dep_entry.get()),
            horizontal_finger_angle=float(self.h_angle_entry.get()),
            vertical_finger_angle=float(self.v_angle_entry.get()),
            deposit_first=self.deposit_first_var.get(),
            horizontal_lw_nm=opt_float(self.h_lw_entry),
            vertical_lw_nm=opt_float(self.v_lw_entry),
            horizontal_length_um=float(self.h_len_entry.get()),
            vertical_length_um=float(self.v_len_entry.get()),
            adhesion_plate_width_um=float(self.plate_w_entry.get()),
            adhesion_plate_height_um=float(self.plate_h_entry.get()),
            overlap_target_um=float(self.overlap_entry.get()),
            patch_integrated=self.patch_var.get(),
            patch_count=int(float(self.patch_count_entry.get())) if self.patch_var.get() else 3,
            patch_spacing_um=float(self.patch_spacing_entry.get()) if self.patch_var.get() else 1.0,
            patch_width_um=float(self.patch_w_entry.get()) if self.patch_var.get() else 0.5,
            patch_height_um=float(self.patch_h_entry.get()) if self.patch_var.get() else 1.5,
            patch_angle=float(self.patch_angle_entry.get()) if self.patch_var.get() else 45.0,
            sidewall_mills=sidewalls,
            wait_minutes=20.0,
            pmma_thickness_nm=opt_float(self.pmma_entry),
            pmgi_thickness_nm=opt_float(self.pmgi_entry),
            line_width_nm=opt_float(self.lw_entry),
            safety_margin_nm=float(self.margin_entry.get()) if self.margin_entry.get().strip() else 100.0,
            notes=self.notes_text.get("1.0", tk.END).strip() if not for_preview else "",
        )

    def _generate_recipe(self):
        # FIXED: Changed from _geometry_confirmed to match our new variable name
        if not getattr(self, "_design_confirmed", False):
            proceed = messagebox.askyesno(
                "Junction design not confirmed",
                "You haven't confirmed the Junction Design tab yet, so this recipe's "
                "geometry hasn't been visually checked. Generate anyway?")
            if not proceed:
                self.notebook.select(0)
                return

        try:
            params = self._collect_recipe_params()
        except ValueError:
            messagebox.showerror("Invalid input", "One or more recipe fields are not valid numbers.")
            return

        

        steps, warnings = generate_recipe_steps(params)
        self._render_recipe_output(steps, warnings, params)
        self._populate_junction_tab(steps, params)

    def _render_recipe_output(self, steps, warnings, params: DesignParameters):
        t = self.output_text
        t.configure(state="normal")
        t.delete("1.0", tk.END)
        for tag in t.tag_names():
            if tag.startswith("step_"):
                t.tag_delete(tag)

        t.insert(tk.END, "PLASSYS RECIPE", "header")
        if params.patch_integrated:
            t.insert(tk.END, f"  (patch-integrated: +3rd Al deposition, patches at "
                              f"{params.patch_angle:g}/{(params.patch_angle + 180) % 360:g} deg)\n", "detail")
        else:
            t.insert(tk.END, "\n")
        t.insert(tk.END, f"Deposition alpha (fixed for batch): {params.alpha_deposition:.2f} deg\n")
        area = params.jj_area_um2()
        if area is not None:
            t.insert(tk.END, f"JJ area: {area:.4f} um^2 "
                              f"({params.horizontal_lw_nm:g} nm Finger 1 x {params.vertical_lw_nm:g} nm Finger 2)\n")
        t.insert(tk.END, "\n")

        for i, s in enumerate(steps):
            tag = "star" if s.starred else None
            marker = "\u2605 " if s.starred else "    "
            if s.alpha is not None and s.theta is not None:
                angle_str = f"alpha={s.alpha:.2f}, theta={s.theta:.2f}"
                click_tag = f"step_{i}"
                t.insert(tk.END, f"{marker}[{s.index}] {s.title}   (", tag)
                t.insert(tk.END, angle_str, (tag, "clickable", click_tag) if tag else ("clickable", click_tag))
                t.insert(tk.END, ")\n", tag)
                t.tag_bind(click_tag, "<Button-1>",
                           lambda e, a=s.alpha, th=s.theta, title=s.title: self._load_step_into_tester(a, th, title))
                t.tag_bind(click_tag, "<Enter>", lambda e: t.configure(cursor="hand2"))
                t.tag_bind(click_tag, "<Leave>", lambda e: t.configure(cursor="arrow"))
            else:
                t.insert(tk.END, f"{marker}[{s.index}] {s.title}\n", tag)
            t.insert(tk.END, f"        {s.detail}\n\n", "detail")

        if warnings:
            t.insert(tk.END, "WARNINGS / THINGS TO VERIFY\n", "header")
            for w in warnings:
                t.insert(tk.END, f"  - {w}\n", "warn")
            t.insert(tk.END, "\n")

        if params.notes:
            t.insert(tk.END, "Notes:\n", "header")
            t.insert(tk.END, f"  {params.notes}\n")

        t.configure(state="disabled")
        self._last_recipe_text = format_recipe_text(steps, warnings, params)

    def _load_step_into_tester(self, alpha: float, theta: float, title: str):
        """Clicking a recipe step's angle loads it into the Live Angle Tester
        with the correct reference base (deposition vs mill)."""
        self._suppress_live_update = True
        
        # FIXED: Check the title string to determine the reference mode
        if "dep" in title.lower() or "deposit" in title.lower():
            self.reference_var.set("deposition")
        else:
            self.reference_var.set("mill")
            
        self._set_entry(self.live_alpha_entry, f"{alpha:g}")
        self._set_entry(self.live_theta_entry, f"{theta:g}")
        self._suppress_live_update = False
        self._on_live_change()

    def _save_txt(self):
        if not self._last_recipe_text:
            messagebox.showinfo("Nothing to save", "Generate a recipe first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt")],
                                             initialfile="plassys_recipe.txt")
        if path:
            with open(path, "w") as f:
                f.write(self._last_recipe_text)
            messagebox.showinfo("Saved", f"Recipe saved to:\n{path}")

    # ------------------------------------------------------------------
    # Junction Formation tab logic
    # ------------------------------------------------------------------
    def _populate_junction_tab(self, steps, params: DesignParameters):
        self._jf_steps = list(steps)
        self._jf_params = params
        self._jf_index = 0

        self.jf_listbox.delete(0, tk.END)
        for s in self._jf_steps:
            star = "\u2605 " if s.starred else "   "
            self.jf_listbox.insert(tk.END, f"{star}[{s.index}] {s.title}")

        self._jf_render_current()

    def _jf_render_current(self):
        if not self._jf_steps:
            return
        self._jf_index = max(0, min(self._jf_index, len(self._jf_steps) - 1))
        step = self._jf_steps[self._jf_index]

        self.jf_listbox.selection_clear(0, tk.END)
        self.jf_listbox.selection_set(self._jf_index)
        self.jf_listbox.see(self._jf_index)

        draw_junction_step(self.jf_ax, self._jf_params, self._jf_steps, self._jf_index)
        self.jf_canvas.draw()

        star = "\u2605 " if step.starred else ""
        self.jf_step_label.configure(
            text=f"Step {self._jf_index + 1} / {len(self._jf_steps)} -- {star}[{step.index}] {step.title}")

        if step.alpha is not None and step.theta is not None:
            angle_line = f"alpha={step.alpha:.2f} deg, theta={step.theta:.2f} deg\n\n"
        else:
            angle_line = ""
        self.jf_detail_label.configure(text=f"{angle_line}{step.detail}")

    def _jf_prev(self):
        if not self._jf_steps:
            return
        self._jf_index -= 1
        self._jf_render_current()

    def _jf_next(self):
        if not self._jf_steps:
            return
        self._jf_index += 1
        self._jf_render_current()

    def _on_jf_listbox_select(self, event=None):
        sel = self.jf_listbox.curselection()
        if not sel:
            return
        self._jf_index = sel[0]
        self._jf_render_current()


def main():
    root = tk.Tk()
    app = MasterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
