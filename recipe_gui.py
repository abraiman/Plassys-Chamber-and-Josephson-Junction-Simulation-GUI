"""
recipe_gui.py

Two-part GUI for generating a Plassys recipe from junction design
parameters: an Input Parameters tab and a Generated Recipe tab, linked
by a "Generate Recipe" button. All computation lives in recipe_model.py;
this file is purely presentation.

Run with:  python recipe_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from recipe_model import (
    DesignParameters,
    SidewallMill,
    generate_recipe_steps,
    format_recipe_text,
    compute_alpha_from_linewidth,
)

BG_COLOR = "#1e1f26"
PANEL_COLOR = "#14151b"
TEXT_COLOR = "#e8e8ef"
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
        self.label_entry = ttk.Entry(self, width=14)
        self.label_entry.insert(0, label)
        self.label_entry.pack(side="left", padx=(2, 8))

        ttk.Button(self, text="Remove", command=self._remove).pack(side="left")

    def _remove(self):
        self.on_remove(self)

    def get_values(self):
        alpha = float(self.alpha_entry.get())
        theta = float(self.theta_entry.get())
        label = self.label_entry.get().strip()
        return SidewallMill(alpha=alpha, theta=theta, label=label)


class RecipeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Plassys Recipe Generator")
        self.root.geometry("1000x780")
        self.root.configure(bg=BG_COLOR)

        self._build_style()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.input_tab = ttk.Frame(self.notebook)
        self.output_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.input_tab, text="1. Input Parameters")
        self.notebook.add(self.output_tab, text="2. Generated Recipe")

        self.sidewall_rows = []

        self._build_input_tab()
        self._build_output_tab()

    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 13, "bold"))
        style.configure("Dim.TLabel", background=BG_COLOR, foreground=DIM_COLOR, font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=5)
        style.configure("TCheckbutton", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TEntry", font=("Consolas", 11))
        style.configure("TNotebook", background=BG_COLOR)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(12, 6))

    # ------------------------------------------------------------------
    # INPUT TAB
    # ------------------------------------------------------------------
    def _build_input_tab(self):
        canvas = tk.Canvas(self.input_tab, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.input_tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        pad = dict(padx=14, pady=6)

        # --- Deposition alpha section ---
        ttk.Label(scroll_frame, text="Deposition Alpha (fixed for whole batch)", style="Header.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", **pad)

        ttk.Label(scroll_frame, text="Alpha (deg):").grid(row=1, column=0, sticky="e", **pad)
        self.alpha_dep_entry = ttk.Entry(scroll_frame, width=10)
        self.alpha_dep_entry.insert(0, "30")
        self.alpha_dep_entry.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(scroll_frame, text="-- or compute from line width --", style="Dim.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=14)

        ttk.Label(scroll_frame, text="Line width LW (nm):").grid(row=3, column=0, sticky="e", **pad)
        self.lw_entry = ttk.Entry(scroll_frame, width=10)
        self.lw_entry.grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(scroll_frame, text="PMMA thickness (nm):").grid(row=3, column=2, sticky="e", **pad)
        self.pmma_entry = ttk.Entry(scroll_frame, width=10)
        self.pmma_entry.insert(0, "875")
        self.pmma_entry.grid(row=3, column=3, sticky="w", **pad)

        ttk.Label(scroll_frame, text="Safety margin (nm):").grid(row=4, column=0, sticky="e", **pad)
        self.margin_entry = ttk.Entry(scroll_frame, width=10)
        self.margin_entry.insert(0, "100")
        self.margin_entry.grid(row=4, column=1, sticky="w", **pad)

        ttk.Button(scroll_frame, text="Compute alpha -> fill above", command=self._compute_alpha).grid(
            row=4, column=2, columnspan=2, sticky="w", **pad)

        ttk.Label(scroll_frame, text="PMGI thickness (nm, documentation only):").grid(
            row=5, column=0, sticky="e", **pad)
        self.pmgi_entry = ttk.Entry(scroll_frame, width=10)
        self.pmgi_entry.insert(0, "600")
        self.pmgi_entry.grid(row=5, column=1, sticky="w", **pad)

        ttk.Separator(scroll_frame, orient="horizontal").grid(row=6, column=0, columnspan=4, sticky="ew", pady=10)

        # --- Finger geometry section ---
        ttk.Label(scroll_frame, text="Junction Finger Geometry", style="Header.TLabel").grid(
            row=7, column=0, columnspan=4, sticky="w", **pad)

        ttk.Label(scroll_frame, text="Base finger angle (deg):").grid(row=8, column=0, sticky="e", **pad)
        self.base_angle_entry = ttk.Entry(scroll_frame, width=10)
        self.base_angle_entry.insert(0, "0")
        self.base_angle_entry.grid(row=8, column=1, sticky="w", **pad)
        ttk.Label(scroll_frame, text="(as drawn in KLayout; 0 = horizontal)", style="Dim.TLabel").grid(
            row=8, column=2, columnspan=2, sticky="w")

        self.patch_var = tk.BooleanVar(value=False)
        patch_check = ttk.Checkbutton(scroll_frame, text="Patch-integrated design", variable=self.patch_var,
                                       command=self._on_patch_toggle)
        patch_check.grid(row=9, column=0, columnspan=2, sticky="w", **pad)

        self.top_angle_label = ttk.Label(scroll_frame, text="Top finger angle (deg):")
        self.top_angle_label.grid(row=10, column=0, sticky="e", **pad)
        self.top_angle_entry = ttk.Entry(scroll_frame, width=10)
        self.top_angle_entry.insert(0, "90")
        self.top_angle_entry.grid(row=10, column=1, sticky="w", **pad)
        self.top_angle_note = ttk.Label(scroll_frame, text="(auto: 90 normal / 45 if patch-integrated -- editable)",
                                         style="Dim.TLabel")
        self.top_angle_note.grid(row=10, column=2, columnspan=2, sticky="w")

        ttk.Label(scroll_frame, text="Target overlap length (um):").grid(row=11, column=0, sticky="e", **pad)
        self.overlap_entry = ttk.Entry(scroll_frame, width=10)
        self.overlap_entry.insert(0, "1.5")
        self.overlap_entry.grid(row=11, column=1, sticky="w", **pad)
        ttk.Label(scroll_frame, text="(documentation only, not used in angle math)", style="Dim.TLabel").grid(
            row=11, column=2, columnspan=2, sticky="w")

        ttk.Separator(scroll_frame, orient="horizontal").grid(row=12, column=0, columnspan=4, sticky="ew", pady=10)

        # --- Ion milling section ---
        ttk.Label(scroll_frame, text="Ion Milling", style="Header.TLabel").grid(
            row=13, column=0, columnspan=4, sticky="w", **pad)
        ttk.Label(scroll_frame, text="Top mill: alpha is always 0 (full top-surface contact).",
                  style="Dim.TLabel").grid(row=14, column=0, columnspan=4, sticky="w", padx=14)

        ttk.Label(scroll_frame, text="Sidewall mills (your choice of alpha/theta, add as many as needed):").grid(
            row=15, column=0, columnspan=4, sticky="w", **pad)

        self.sidewall_container = ttk.Frame(scroll_frame)
        self.sidewall_container.grid(row=16, column=0, columnspan=4, sticky="w", padx=14)

        ttk.Button(scroll_frame, text="+ Add sidewall mill", command=self._add_sidewall_row).grid(
            row=17, column=0, sticky="w", **pad)

        ttk.Separator(scroll_frame, orient="horizontal").grid(row=18, column=0, columnspan=4, sticky="ew", pady=10)

        # --- Timing / notes ---
        ttk.Label(scroll_frame, text="Timing & Notes", style="Header.TLabel").grid(
            row=19, column=0, columnspan=4, sticky="w", **pad)

        ttk.Label(scroll_frame, text="Final wait (minutes):").grid(row=20, column=0, sticky="e", **pad)
        self.wait_entry = ttk.Entry(scroll_frame, width=10)
        self.wait_entry.insert(0, "20")
        self.wait_entry.grid(row=20, column=1, sticky="w", **pad)

        ttk.Label(scroll_frame, text="Notes:").grid(row=21, column=0, sticky="ne", **pad)
        self.notes_text = tk.Text(scroll_frame, width=50, height=3, bg=PANEL_COLOR, fg=TEXT_COLOR,
                                   font=("Segoe UI", 9), relief="flat", padx=6, pady=6)
        self.notes_text.grid(row=21, column=1, columnspan=3, sticky="w", **pad)

        ttk.Button(scroll_frame, text="Generate Recipe ->", command=self._generate).grid(
            row=22, column=0, columnspan=4, sticky="ew", padx=14, pady=(16, 20))

        # seed with the two sidewalls from the known-good recipe
        self._add_sidewall_row(alpha="60", theta="0", label="sidewall 1")
        self._add_sidewall_row(alpha="60", theta="270", label="sidewall 2")

    def _on_patch_toggle(self):
        auto_angle = "45" if self.patch_var.get() else "90"
        self.top_angle_entry.delete(0, tk.END)
        self.top_angle_entry.insert(0, auto_angle)

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

    def _collect_params(self) -> DesignParameters:
        sidewalls = [row.get_values() for row in self.sidewall_rows]

        def opt_float(entry):
            s = entry.get().strip()
            return float(s) if s else None

        return DesignParameters(
            alpha_deposition=float(self.alpha_dep_entry.get()),
            base_finger_angle=float(self.base_angle_entry.get()),
            patch_integrated=self.patch_var.get(),
            top_finger_angle=float(self.top_angle_entry.get()) if self.top_angle_entry.get().strip() else None,
            overlap_target_um=float(self.overlap_entry.get()),
            sidewall_mills=sidewalls,
            wait_minutes=float(self.wait_entry.get()),
            pmma_thickness_nm=opt_float(self.pmma_entry),
            pmgi_thickness_nm=opt_float(self.pmgi_entry),
            line_width_nm=opt_float(self.lw_entry),
            safety_margin_nm=float(self.margin_entry.get()) if self.margin_entry.get().strip() else 100.0,
            notes=self.notes_text.get("1.0", tk.END).strip(),
        )

    def _generate(self):
        try:
            params = self._collect_params()
        except ValueError:
            messagebox.showerror("Invalid input", "One or more fields are not valid numbers. Please check your entries.")
            return

        steps, warnings = generate_recipe_steps(params)
        self._render_output(steps, warnings, params)
        self.notebook.select(self.output_tab)

    # ------------------------------------------------------------------
    # OUTPUT TAB
    # ------------------------------------------------------------------
    def _build_output_tab(self):
        top_bar = ttk.Frame(self.output_tab)
        top_bar.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(top_bar, text="Generated Recipe", style="Header.TLabel").pack(side="left")
        ttk.Button(top_bar, text="Save as .txt", command=self._save_txt).pack(side="right")

        self.output_text = tk.Text(self.output_tab, bg=PANEL_COLOR, fg=TEXT_COLOR,
                                    font=("Consolas", 11), relief="flat", padx=14, pady=14,
                                    spacing1=1, spacing3=4, wrap="word")
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.output_text.tag_configure("star", foreground=STAR_COLOR, font=("Consolas", 11, "bold"))
        self.output_text.tag_configure("header", foreground=ACCENT_COLOR, font=("Consolas", 13, "bold"))
        self.output_text.tag_configure("detail", foreground=DIM_COLOR, font=("Consolas", 9))
        self.output_text.tag_configure("warn", foreground=WARN_COLOR, font=("Consolas", 10))
        self.output_text.insert(tk.END, "Fill in parameters on the Input tab, then click "
                                         "\"Generate Recipe ->\" to see the recipe here.")
        self.output_text.configure(state="disabled")
        self._last_recipe_text = ""

    def _render_output(self, steps, warnings, params: DesignParameters):
        t = self.output_text
        t.configure(state="normal")
        t.delete("1.0", tk.END)

        t.insert(tk.END, "PLASSYS RECIPE", "header")
        if params.patch_integrated:
            t.insert(tk.END, "  (patch-integrated: top finger 45/225 deg)\n", "detail")
        else:
            t.insert(tk.END, "\n")
        t.insert(tk.END, f"Deposition alpha (fixed for batch): {params.alpha_deposition:.2f} deg\n")
        if params.pmma_thickness_nm:
            t.insert(tk.END, f"PMMA thickness: {params.pmma_thickness_nm:g} nm   "
                              f"Safety margin: {params.safety_margin_nm:g} nm\n")
        t.insert(tk.END, "\n")

        for s in steps:
            tag = "star" if s.starred else None
            marker = "\u2605 " if s.starred else "    "
            if s.alpha is not None and s.theta is not None:
                angle_str = f"  (alpha={s.alpha:.2f} deg, theta={s.theta:.2f} deg)"
            else:
                angle_str = ""
            header_line = f"{marker}[{s.index}] {s.title}{angle_str}\n"
            if tag:
                t.insert(tk.END, header_line, tag)
            else:
                t.insert(tk.END, header_line)
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


def main():
    root = tk.Tk()
    app = RecipeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
