"""
Main GUI

Panels
  1.Simulation  
  2.Genetic Algorun 
  3.Swarm (PSO) 
  4.Neural Network 
  5.Decision Tree  
  6.Analysis 
  7.Parameters
  8.Pop. Analysis (Predator-prey, stability, diversity charts)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading, queue, time, math, random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from simulation     import Ecosystem, EcosystemParams, run_simulation, get_season, SEASON_COLORS
from genetic_algorithm  import GeneticAlgorithm
from particle_swarm     import ParticleSwarmOptimisation
from ml_models          import (NeuralNetworkPredictor, DecisionTreeAnalyzer,
                                collect_training_data)
from analysis           import (compare_algorithms, parameter_sensitivity,
                                hemisphere_comparison, evolutionary_dynamics_report,
                                dt_insights_report, scalability_test, GRID_SIZES,
                                nn_accuracy_report)

# Palette & Fonts
BG      = '#F0F4F8'
PANEL   = '#FFFFFF'
ACCENT1 = '#4CAF50'
ACCENT2 = '#2196F3'
ACCENT3 = '#FF9800'
ACCENT4 = '#9C27B0'
DANGER  = '#F44336'
TEXT    = '#263238'
SUBTEXT = '#607D8B'
BORDER  = '#CFD8DC'
SUCCESS = '#1B5E20'

FONT_H1    = ('Segoe UI', 16, 'bold')
FONT_H2    = ('Segoe UI', 12, 'bold')
FONT_BODY  = ('Segoe UI', 10)
FONT_SMALL = ('Segoe UI', 9)
FONT_MONO  = ('Consolas', 9)

ENTITY_EMOJI = {'plant': '🌿', 'herbivore_M': '🐄', 'herbivore_F': '🐐',
                'carnivore_M': '🦁', 'carnivore_F': '🐯'}
ENTITY_COLOR = {'plant': '#43A047', 'herbivore': '#FB8C00', 'carnivore': '#E53935'}

# Habitat Presets
_D  = EcosystemParams.DEFAULTS.copy()
_B  = EcosystemParams.BOUNDS
_N  = EcosystemParams.PARAM_NAMES

def _idx(name): return _N.index(name)
def _clip(v, i): return float(np.clip(v, _B[i, 0], _B[i, 1]))

def _make_preset(mods: dict) -> np.ndarray:
    vec = _D.copy()
    for name, val in mods.items():
        i = _idx(name)
        vec[i] = _clip(val, i)
    return vec

PRESETS = {
    '🏞️  Normal (Default)': _D.copy(),

    '🌴  Tropical Forest': _make_preset({
        'plant_spread_spring':  _clip(_D[_idx('plant_spread_spring')]  * 1.30, _idx('plant_spread_spring')),
        'plant_spread_summer':  _clip(_D[_idx('plant_spread_summer')]  * 1.30, _idx('plant_spread_summer')),
        'initial_herbivores':   _clip(_D[_idx('initial_herbivores')]   * 1.40, _idx('initial_herbivores')),
        'carn_vision':          _clip(_D[_idx('carn_vision')]          + 2,    _idx('carn_vision')),
        'plant_max_age':        _clip(_D[_idx('plant_max_age')]        * 1.20, _idx('plant_max_age')),
        'energy_from_plant':    _clip(_D[_idx('energy_from_plant')]    * 1.15, _idx('energy_from_plant')),
    }),

    '🏜️  Desert / Arid': _make_preset({
        'plant_spread_spring':  _clip(_D[_idx('plant_spread_spring')]  * 0.50, _idx('plant_spread_spring')),
        'plant_spread_summer':  _clip(_D[_idx('plant_spread_summer')]  * 0.50, _idx('plant_spread_summer')),
        'plant_spread_autumn':  _clip(_D[_idx('plant_spread_autumn')]  * 0.50, _idx('plant_spread_autumn')),
        'plant_spread_winter':  _clip(_D[_idx('plant_spread_winter')]  * 0.50, _idx('plant_spread_winter')),
        'energy_from_plant':    _clip(_D[_idx('energy_from_plant')]    * 0.70, _idx('energy_from_plant')),
        'initial_plants':       _clip(_D[_idx('initial_plants')]       * 0.60, _idx('initial_plants')),
        'initial_herbivores':   _clip(_D[_idx('initial_herbivores')]   * 0.60, _idx('initial_herbivores')),
        'initial_carnivores':   _clip(_D[_idx('initial_carnivores')]   * 0.60, _idx('initial_carnivores')),
        'herb_move_cost':       _clip(_D[_idx('herb_move_cost')]       * 1.50, _idx('herb_move_cost')),
        'carn_move_cost':       _clip(_D[_idx('carn_move_cost')]       * 1.50, _idx('carn_move_cost')),
    }),

    '❄️  Arctic Region': _make_preset({
        'plant_spread_spring':  _clip(_D[_idx('plant_spread_spring')]  * 0.60, _idx('plant_spread_spring')),
        'plant_spread_summer':  _clip(_D[_idx('plant_spread_summer')]  * 0.60, _idx('plant_spread_summer')),
        'plant_spread_autumn':  _clip(_D[_idx('plant_spread_autumn')]  * 1.20, _idx('plant_spread_autumn')),
        'plant_winter_death':   _clip(_D[_idx('plant_winter_death')]   * 0.50, _idx('plant_winter_death')),
        'herb_start_energy':    _clip(_D[_idx('herb_start_energy')]    * 1.30, _idx('herb_start_energy')),
        'carn_start_energy':    _clip(_D[_idx('carn_start_energy')]    * 1.30, _idx('carn_start_energy')),
        'herb_max_age':         _clip(_D[_idx('herb_max_age')]         * 0.85, _idx('herb_max_age')),
    }),
}

# Helpers
def _card(parent, **kw):
    f = tk.Frame(parent, bg=PANEL, relief='flat', bd=0, **kw)
    f.configure(highlightbackground=BORDER, highlightthickness=1)
    return f

def _label(parent, text, font=FONT_BODY, fg=TEXT, bg=PANEL, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)

def _btn(parent, text, cmd, color=ACCENT1, fg='white', width=18, **kw):
    b = tk.Button(parent, text=text, command=cmd, bg=color, fg=fg,
                  font=FONT_BODY, relief='flat', cursor='hand2',
                  activebackground=color, activeforeground=fg,
                  padx=8, pady=5, width=width, **kw)
    return b

def _section(parent, title, bg=PANEL):
    f = tk.Frame(parent, bg=bg)
    tk.Label(f, text=title, font=FONT_H2, fg=ACCENT1, bg=bg).pack(anchor='w', pady=(6,2))
    tk.Frame(f, bg=BORDER, height=1).pack(fill='x', pady=(0,6))
    return f

def _embed_fig(parent, fig, row=0, col=0, rs=1, cs=1):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().grid(row=row, column=col, rowspan=rs, columnspan=cs,
                                 sticky='nsew', padx=4, pady=4)
    return canvas

def _scrollable_frame(parent):
    outer = tk.Frame(parent, bg=PANEL)
    canvas  = tk.Canvas(outer, bg=PANEL, highlightthickness=0)
    vsb     = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    inner   = tk.Frame(canvas, bg=PANEL)
    inner.bind('<Configure>',
               lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=inner, anchor='nw')
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side='left', fill='both', expand=True)
    vsb.pack(side='right', fill='y')
    # Mouse-wheel scroll
    def _on_wheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
    canvas.bind_all('<MouseWheel>', _on_wheel)
    return outer, inner


# Main Application
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('🌿  AI-Optimized Ecosystem Simulation  –  IBA Karachi  CSE-307')
        self.configure(bg=BG)
        self.state('zoomed')
        self.minsize(1200, 700)

        self._q           = queue.Queue()
        self._eco         : Ecosystem | None     = None
        self._sim_running = False
        self._sim_speed   = 300
        self._paused      = False
        self._ga          = GeneticAlgorithm()
        self._pso         = ParticleSwarmOptimisation()
        self._nn          = NeuralNetworkPredictor()
        self._dt          = DecisionTreeAnalyzer()
        self._train_X     : np.ndarray | None = None
        self._train_y     : np.ndarray | None = None
        self._best_params : EcosystemParams    = EcosystemParams()

        # Per-parameter tkinter variables (for Parameter tab)
        self._param_vars       : dict[str, tk.DoubleVar]  = {}
        self._param_entry_vars : dict[str, tk.StringVar]  = {}
        for i, name in enumerate(EcosystemParams.PARAM_NAMES):
            val = float(EcosystemParams.DEFAULTS[i])
            self._param_vars[name]       = tk.DoubleVar(value=val)
            self._param_entry_vars[name] = tk.StringVar(value=f'{val:.3g}')

        self._build_header()
        self._build_notebook()
        self._poll_queue()

    
    # Param helpers  (used by every tab that needs current custom params)
    def _get_custom_params(self) -> EcosystemParams:
        vec = np.array([self._param_vars[n].get()
                        for n in EcosystemParams.PARAM_NAMES], dtype=float)
        return EcosystemParams.from_vector(vec)

    def _sync_sliders_to_params(self, params: EcosystemParams):
        # Push values from params object into all 24 slider/entry vars.
        vec = params.to_vector()
        for i, name in enumerate(EcosystemParams.PARAM_NAMES):
            val = float(vec[i])
            self._param_vars[name].set(val)
            self._param_entry_vars[name].set(f'{val:.4g}')

    def _apply_preset(self, preset_name: str):
        vec = PRESETS.get(preset_name, _D.copy())
        self._sync_sliders_to_params(EcosystemParams.from_vector(vec))
        # Reset sim so new params take effect
        self._reset_sim()

    # Header
    def _build_header(self):
        h = tk.Frame(self, bg=ACCENT1, pady=10)
        h.pack(fill='x')
        tk.Label(h, text='🌿  AI-Optimized Ecosystem Simulation',
                 font=('Segoe UI', 18, 'bold'), fg='white', bg=ACCENT1).pack(side='left', padx=18)
        tk.Label(h, text='CSE-307  |  IBA Karachi  |  Spring 2026',
                 font=FONT_BODY, fg='#C8E6C9', bg=ACCENT1).pack(side='right', padx=18)

    # Notebook
    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook',       background=BG, borderwidth=0)
        style.configure('TNotebook.Tab',   background='#E8F5E9', foreground=TEXT,
                        font=FONT_BODY, padding=[12, 5])
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT1)],
                  foreground=[('selected', 'white')])

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=10, pady=(6, 10))

        tabs = [
            ('🌍  Simulation',      self._build_sim_tab),
            ('🧬  Genetic Algo',    self._build_ga_tab),
            ('🐝  Swarm (PSO)',      self._build_pso_tab),
            ('🧠  Neural Network',  self._build_nn_tab),
            ('🌳  Decision Tree',   self._build_dt_tab),
            ('📊  Analysis',         self._build_analysis_tab),
            ('⚙️  Parameters',      self._build_params_tab),
            ('📈  Pop. Analysis',   self._build_popanalysis_tab),
        ]
        for label, builder in tabs:
            frame = tk.Frame(nb, bg=BG)
            nb.add(frame, text=label)
            builder(frame)

    # TAB 1 – Live Simulation
    def _build_sim_tab(self, parent):
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=2)
        parent.rowconfigure(1, weight=1) 

        # Grid canva
        grid_card = _card(parent)
        grid_card.grid(row=0, column=0, sticky='nsew', padx=(8,4), pady=8)
        grid_card.rowconfigure(1, weight=3)  
        grid_card.rowconfigure(0, weight=0)  
        grid_card.columnconfigure(0, weight=1)

        self._season_lbl = _label(grid_card, '  Season: —  |  Month: 0',
                                   font=FONT_H2, bg=PANEL, fg=ACCENT1)
        self._season_lbl.grid(row=0, column=0, sticky='w', padx=10, pady=(8,2))

        self._canvas = tk.Canvas(grid_card, bg='#E8F5E9', highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky='nsew', padx=8, pady=8)
        self._canvas.bind('<Configure>', lambda e: self._redraw_grid())

        # Right pane
        ctrl = _card(parent)
        ctrl.grid(row=0, column=1, sticky='nsew', padx=(4,8), pady=8)

        # Scrollable right panel
        ctrl_outer, ctrl_inner = _scrollable_frame(ctrl)
        ctrl_outer.pack(fill='both', expand=True)
        C = ctrl_inner   # alias for brevity

        # Config section
        _label(C, '⚙  Configuration', FONT_H2, ACCENT1).pack(anchor='w', padx=10, pady=(10,4))
        tk.Frame(C, bg=BORDER, height=1).pack(fill='x', padx=10)

        def _row(lbl, widget_factory):
            f = tk.Frame(C, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
            _label(f, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            w = widget_factory(f); w.pack(side='right')
            return w

        self._gs_var  = tk.StringVar(value='20')
        self._hem_var = tk.StringVar(value='N')
        self._spd_var = tk.IntVar(value=300)

        _row('Grid Size:',  lambda f: ttk.Combobox(f, textvariable=self._gs_var,
             values=['10','20','30','40'], width=6, state='readonly'))
        _row('Hemisphere:', lambda f: ttk.Combobox(f, textvariable=self._hem_var,
             values=['N','S'], width=6, state='readonly'))

        sf = tk.Frame(C, bg=PANEL); sf.pack(fill='x', padx=10, pady=3)
        _label(sf, 'Speed:', FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
        tk.Scale(sf, from_=50, to=1000, resolution=50, orient='horizontal',
                 variable=self._spd_var, bg=PANEL, highlightthickness=0, length=110,
                 command=lambda v: setattr(self, '_sim_speed', int(v))).pack(side='right')

        # Habitat Preset dropdown
        tk.Frame(C, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)
        _label(C, '🌏  Habitat Preset', FONT_SMALL, SUBTEXT).pack(anchor='w', padx=10)
        self._preset_var = tk.StringVar(value='🏞️  Normal (Default)')
        preset_cb = ttk.Combobox(C, textvariable=self._preset_var,
                                  values=list(PRESETS.keys()),
                                  width=22, state='readonly')
        preset_cb.pack(padx=10, pady=3, fill='x')
        preset_cb.bind('<<ComboboxSelected>>',
                       lambda e: self._apply_preset(self._preset_var.get()))

        # Use optimised params
        self._use_best = tk.BooleanVar(value=False)
        tk.Checkbutton(C, text='Use AI-Optimised Params',
                       variable=self._use_best, bg=PANEL,
                       font=FONT_SMALL, fg=TEXT,
                       activebackground=PANEL).pack(anchor='w', padx=12, pady=2)

        # Control button
        for txt, cmd, col in [
            ('▶  Start Simulation', self._start_sim,   ACCENT1),
            ('⏸  Pause / Resume',   self._toggle_pause, ACCENT3),
            ('⏹  Stop',              self._stop_sim,    DANGER),
            ('🔄  Reset',             self._reset_sim,   ACCENT2),
        ]:
            _btn(C, txt, cmd, col).pack(padx=10, pady=3, fill='x')

        # Population stats
        tk.Frame(C, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        _label(C, '📊  Live Population', FONT_H2, ACCENT1).pack(anchor='w', padx=10)
        self._stat_labels = {}
        for key, lbl in [('plants','🌿 Plants'), ('herbivores','🐄 Herbivores'),
                          ('carnivores','🦁 Carnivores'), ('month','📅 Month')]:
            f = tk.Frame(C, bg=PANEL); f.pack(fill='x', padx=10, pady=2)
            _label(f, lbl+':', FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            sv = tk.StringVar(value='—')
            self._stat_labels[key] = sv
            _label(f, '', FONT_SMALL, TEXT, bg=PANEL, textvariable=sv).pack(side='right')

        # Enhanced Monthly Statistic
        tk.Frame(C, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)
        _label(C, '📋  Monthly Details', FONT_H2, ACCENT2).pack(anchor='w', padx=10)

        self._detail_vars = {}
        detail_fields = [
            ('births_plants',   '🌱 Plant births'),
            ('births_herbs',    '🐄 Herb births'),
            ('births_carns',    '🦁 Carn births'),
            ('deaths_herbs',    '💀 Herb deaths'),
            ('deaths_carns',    '💀 Carn deaths'),
            ('food_index',      '🌿/🐄 Food index'),
            ('avg_herb_energy', '⚡ Avg herb energy'),
            ('avg_carn_energy', '⚡ Avg carn energy'),
            ('herb_gender',     '👥 Herb M:F ratio'),
            ('carn_gender',     '👥 Carn M:F ratio'),
        ]
        for key, lbl in detail_fields:
            f = tk.Frame(C, bg=PANEL); f.pack(fill='x', padx=10, pady=1)
            _label(f, lbl+':', FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            sv = tk.StringVar(value='—')
            self._detail_vars[key] = sv
            _label(f, '', FONT_SMALL, TEXT, bg=PANEL, textvariable=sv).pack(side='right')

        # Legen
        tk.Frame(C, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        _label(C, 'Legend', FONT_SMALL, SUBTEXT).pack(anchor='w', padx=10)
        for sym, desc in [('🌿','Plant'), ('🐄','Herbivore ♂'),
                          ('🐐','Herbivore ♀'), ('🦁','Carnivore ♂'), ('🐯','Carnivore ♀')]:
            f = tk.Frame(C, bg=PANEL); f.pack(anchor='w', padx=14, pady=1)
            tk.Label(f, text=sym, font=('Segoe UI', 11), bg=PANEL).pack(side='left')
            _label(f, f'  {desc}', FONT_SMALL, TEXT, bg=PANEL).pack(side='left')

        # Population graph
        pop_card = _card(parent)
        pop_card.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=8, pady=(0,8))
        pop_card.columnconfigure(0, weight=1); pop_card.rowconfigure(0, weight=1)

        self._pop_fig = Figure(figsize=(8, 1.5), dpi=90, facecolor=PANEL)
        self._pop_ax  = self._pop_fig.add_subplot(111)
        self._pop_ax.set_facecolor('#F9FBE7')
        self._pop_fig.tight_layout(pad=1.2)
        self._pop_canvas = _embed_fig(pop_card, self._pop_fig)
        self._pop_hist = {'plants':[], 'herbivores':[], 'carnivores':[], 'months':[]}

    # Simulation helpers
    def _make_eco(self):
        gs  = int(self._gs_var.get())
        hm  = self._hem_var.get()
        p   = self._best_params if self._use_best.get() else self._get_custom_params()
        return Ecosystem(p, gs, hm)

    def _start_sim(self):
        if self._sim_running: return
        self._eco = self._make_eco()
        self._pop_hist = {'plants':[], 'herbivores':[], 'carnivores':[], 'months':[]}
        self._sim_running = True; self._paused = False
        self._sim_loop()

    def _toggle_pause(self):
        self._paused = not self._paused

    def _stop_sim(self):
        self._sim_running = False

    def _reset_sim(self):
        self._stop_sim()
        self._eco = None
        self._canvas.delete('all')
        self._pop_ax.clear(); self._pop_canvas.draw()
        for k in self._stat_labels:    self._stat_labels[k].set('—')
        for k in self._detail_vars:    self._detail_vars[k].set('—')
        self._season_lbl.config(text='  Season: —  |  Month: 0')

    def _sim_loop(self):
        if not self._sim_running: return
        if self._paused:
            self.after(200, self._sim_loop); return
        if not self._eco or not self._eco.is_alive():
            self._sim_running = False
            if self._eco:
                messagebox.showinfo('Simulation Ended',
                    f'Ecosystem ended at month {self._eco.month}.\n'
                    f'Plants: {len(self._eco.plants)}  '
                    f'Herbivores: {len(self._eco.herbivores)}  '
                    f'Carnivores: {len(self._eco.carnivores)}')
            return
        stats = self._eco.step()
        self._redraw_grid()
        self._update_pop_graph(stats)
        self._update_stats(stats)
        self._update_detail_stats(stats)
        self.after(self._sim_speed, self._sim_loop)

    def _redraw_grid(self):
        if not self._eco: return
        c = self._canvas
        c.delete('all')
        W = c.winfo_width(); H = c.winfo_height()
        gs = self._eco.grid_size
        if W < 10 or H < 10: return
        cs = max(15, min(W // gs, H // gs))
        ox = (W - cs * gs) // 2; oy = (H - cs * gs) // 2

        season = get_season(self._eco.month, self._eco.hemisphere)
        c.configure(bg={'Spring':'#E8F5E9','Summer':'#FFFDE7',
                        'Autumn':'#FFF3E0','Winter':'#E3F2FD'}[season])

        for i in range(gs + 1):
            c.create_line(ox + i*cs, oy, ox + i*cs, oy + gs*cs, fill='#C8E6C9', width=1)
            c.create_line(ox, oy + i*cs, ox + gs*cs, oy + i*cs, fill='#C8E6C9', width=1)

        font_size  = max(7, min(cs - 4, 18))
        emoji_font = ('Segoe UI Emoji', font_size)

        for (r, col_idx, kind, gender) in self._eco.get_snapshot():
            x = ox + col_idx * cs + cs // 2
            y = oy + r        * cs + cs // 2
            key   = kind if kind == 'plant' else f'{kind}_{gender}'
            emoji = ENTITY_EMOJI.get(key, '?')
            c.create_text(x, y, text=emoji, font=emoji_font, anchor='center')

        sc = SEASON_COLORS.get(season, '#EEE')
        c.create_rectangle(ox, oy - 20, ox + 140, oy, fill=sc, outline='', tags='badge')
        c.create_text(ox + 70, oy - 10,
                      text=f'  {season}  –  Month {self._eco.month}',
                      font=FONT_SMALL, fill=TEXT, tags='badge')

    def _update_pop_graph(self, stats):
        h = self._pop_hist
        h['months'].append(stats['month'])
        h['plants'].append(stats['plants'])
        h['herbivores'].append(stats['herbivores'])
        h['carnivores'].append(stats['carnivores'])
        ax = self._pop_ax; ax.clear(); ax.set_facecolor('#F9FBE7')
        months = h['months']
        ax.plot(months, h['plants'],     color=ACCENT1, lw=1.5, label='Plants')
        ax.plot(months, h['herbivores'], color=ACCENT3, lw=1.5, label='Herbivores')
        ax.plot(months, h['carnivores'], color=DANGER,  lw=1.5, label='Carnivores')
        ax.set_xlabel('Month', fontsize=8); ax.set_ylabel('Population', fontsize=8)
        ax.legend(fontsize=7, loc='upper right'); ax.tick_params(labelsize=7)
        self._pop_fig.tight_layout(pad=1); self._pop_canvas.draw_idle()

    def _update_stats(self, stats):
        for k in ('plants', 'herbivores', 'carnivores'):
            self._stat_labels[k].set(str(stats[k]))
        self._stat_labels['month'].set(str(stats['month']))
        season = stats['season']
        self._season_lbl.config(
            text=f"  Season: {season}  |  Month: {stats['month']}",
            bg=SEASON_COLORS.get(season, PANEL))

    def _update_detail_stats(self, stats):
# Populate the enhanced Monthly Details panel.
        dv = self._detail_vars
        dv['births_plants'].set(str(stats.get('plant_births', 0)))
        dv['births_herbs'].set (str(stats.get('herb_births',  0)))
        dv['births_carns'].set (str(stats.get('carn_births',  0)))
        dv['deaths_herbs'].set (str(stats.get('herb_deaths',  0)))
        dv['deaths_carns'].set (str(stats.get('carn_deaths',  0)))

        if self._eco:
            herbs  = [h for h in self._eco.herbivores if h.alive]
            carns  = [c for c in self._eco.carnivores if c.alive]
            plants = [p for p in self._eco.plants     if p.alive]

            food_idx = len(plants) / max(len(herbs), 1)
            dv['food_index'].set(f'{food_idx:.2f}')

            if herbs:
                avg_he = np.mean([h.energy for h in herbs])
                hm = sum(1 for h in herbs if h.gender=='M')
                hf = len(herbs) - hm
                dv['avg_herb_energy'].set(f'{avg_he:.1f}')
                dv['herb_gender'].set(f'M:{hm}  F:{hf}')
            else:
                dv['avg_herb_energy'].set('—'); dv['herb_gender'].set('—')

            if carns:
                avg_ce = np.mean([c.energy for c in carns])
                cm = sum(1 for c in carns if c.gender=='M')
                cf = len(carns) - cm
                dv['avg_carn_energy'].set(f'{avg_ce:.1f}')
                dv['carn_gender'].set(f'M:{cm}  F:{cf}')
            else:
                dv['avg_carn_energy'].set('—'); dv['carn_gender'].set('—')

    # TAB 2 – Genetic Algorithm
    def _build_ga_tab(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        left = _card(parent)
        left.grid(row=0, column=0, sticky='nsew', padx=(8,4), pady=8)
        _label(left, '🧬  Genetic Algorithm', FONT_H2, ACCENT1).pack(anchor='w', padx=10, pady=(10,2))
        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)

        def _sl(p, lbl, var, lo, hi, res=1):
            f = tk.Frame(p, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
            _label(f, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            tk.Scale(f, from_=lo, to=hi, resolution=res, orient='horizontal',
                     variable=var, bg=PANEL, highlightthickness=0, length=130).pack(side='right')

        self._ga_pop  = tk.IntVar(value=30); self._ga_gen = tk.IntVar(value=25)
        self._ga_mut  = tk.DoubleVar(value=0.10)
        self._ga_gs   = tk.StringVar(value='20'); self._ga_hem = tk.StringVar(value='N')

        _sl(left, 'Population Size', self._ga_pop, 10, 60)
        _sl(left, 'Generations',     self._ga_gen, 5,  50)
        f = tk.Frame(left, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
        _label(f, 'Mutation Rate', FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
        tk.Scale(f, from_=0.01, to=0.30, resolution=0.01, orient='horizontal',
                 variable=self._ga_mut, bg=PANEL, highlightthickness=0,
                 length=130).pack(side='right')
        for lbl, var, vals in [('Grid Size', self._ga_gs, ['10','20','30','40']),
                                ('Hemisphere', self._ga_hem, ['N','S'])]:
            f2 = tk.Frame(left, bg=PANEL); f2.pack(fill='x', padx=10, pady=3)
            _label(f2, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            ttk.Combobox(f2, textvariable=var, values=vals, width=6,
                         state='readonly').pack(side='right')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        self._ga_progress = ttk.Progressbar(left, mode='determinate', length=180)
        self._ga_progress.pack(padx=10, pady=4)
        self._ga_status = _label(left, 'Ready', FONT_SMALL, SUBTEXT)
        self._ga_status.pack(anchor='w', padx=10)

        for txt, cmd, col in [
            ('▶  Run GA',            self._run_ga,     ACCENT1),
            ('⏹  Stop',               self._stop_ga,    DANGER),
            ('✅  Apply Best Params', self._apply_ga,   ACCENT2),
        ]:
            _btn(left, txt, cmd, col, width=22).pack(padx=10, pady=4, fill='x')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        _label(left, 'Best Fitness', FONT_SMALL, SUBTEXT).pack(anchor='w', padx=10)
        self._ga_best_var = tk.StringVar(value='—')
        _label(left, '', FONT_H1, ACCENT1, bg=PANEL,
               textvariable=self._ga_best_var).pack(anchor='w', padx=10)

        right = _card(parent)
        right.grid(row=0, column=1, sticky='nsew', padx=(4,8), pady=8)
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        self._ga_fig    = Figure(figsize=(8, 6), dpi=90, facecolor=PANEL)
        self._ga_ax_fit = self._ga_fig.add_subplot(2, 2, 1)
        self._ga_ax_div = self._ga_fig.add_subplot(2, 2, 2)
        self._ga_ax_pop = self._ga_fig.add_subplot(2, 1, 2)
        for ax in [self._ga_ax_fit, self._ga_ax_div, self._ga_ax_pop]:
            ax.set_facecolor('#F3E5F5')
        self._ga_fig.tight_layout(pad=2)
        self._ga_canvas = _embed_fig(right, self._ga_fig, rs=2, cs=2)
        self._ga_best_hist = []; self._ga_avg_hist = []; self._ga_div_hist = []

    def _run_ga(self):
        def worker():
            gs   = int(self._ga_gs.get())
            hem  = self._ga_hem.get()
            gens = self._ga_gen.get()
            self._ga = GeneticAlgorithm(
                population_size=self._ga_pop.get(),
                mutation_rate=self._ga_mut.get(),
                grid_size=gs, hemisphere=hem,
                callback=lambda info: self._q.put(('ga_step', info)))
            self._ga_best_hist.clear(); self._ga_avg_hist.clear(); self._ga_div_hist.clear()
            self._ga_progress['maximum'] = gens
            self._ga.run(gens)
            self._q.put(('ga_done', None))
        self._ga_status.config(text='Running…', fg=ACCENT3)
        threading.Thread(target=worker, daemon=True).start()

    def _stop_ga(self):
        self._ga.stop()
        self._ga_status.config(text='Stopped', fg=DANGER)

    def _apply_ga(self):
        if not self._ga.best_params:
            messagebox.showwarning('No Results', 'Run GA first.'); return
        self._best_params = self._ga.best_params
        self._sync_sliders_to_params(self._best_params)   # ← sync all 24 sliders
        self._preset_var.set('🏞️  Normal (Default)')       # reset preset label
        messagebox.showinfo('GA Params Applied',
            '✅ Best GA parameters applied to all 24 parameter sliders!\n'
            'Enable "Use AI-Optimised Params" in the Simulation tab to use them.')

    def _update_ga_chart(self, info):
        self._ga_best_hist.append(info['best_fitness'])
        self._ga_avg_hist.append(info['avg_fitness'])
        self._ga_div_hist.append(info['diversity'])
        self._ga_best_var.set(f"{info['best_fitness']:.4f}")
        self._ga_status.config(
            text=f"Gen {info['generation']}  |  Best: {info['best_fitness']:.4f}", fg=TEXT)
        self._ga_progress['value'] = info['generation']
        gens = list(range(1, len(self._ga_best_hist) + 1))

        ax1 = self._ga_ax_fit; ax1.clear(); ax1.set_facecolor('#F3E5F5')
        ax1.plot(gens, self._ga_best_hist, color=ACCENT1, lw=2, label='Best')
        ax1.plot(gens, self._ga_avg_hist,  color=ACCENT2, lw=1.5, ls='--', label='Average')
        ax1.set_title('Fitness over Generations', fontsize=9)
        ax1.legend(fontsize=7); ax1.tick_params(labelsize=7)

        ax2 = self._ga_ax_div; ax2.clear(); ax2.set_facecolor('#F3E5F5')
        ax2.plot(gens, self._ga_div_hist, color=ACCENT4, lw=2)
        ax2.set_title('Population Diversity', fontsize=9); ax2.tick_params(labelsize=7)

        ax3 = self._ga_ax_pop; ax3.clear(); ax3.set_facecolor('#F3E5F5')
        if info.get('best_params'):
            vec  = info['best_params'].to_vector()
            bnd  = EcosystemParams.BOUNDS
            norm = (vec - bnd[:,0]) / (bnd[:,1] - bnd[:,0])
            short = [l[:14] for l in EcosystemParams.LABELS]
            ax3.barh(short, norm, color=ACCENT1, alpha=0.7)
            ax3.set_xlim(0, 1)
            ax3.set_title('Best Params (normalised)', fontsize=9)
            ax3.tick_params(labelsize=6)

        self._ga_fig.tight_layout(pad=1.5)
        self._ga_canvas.draw_idle()

    # TAB 3 – Particle Swarm
    def _build_pso_tab(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        left = _card(parent)
        left.grid(row=0, column=0, sticky='nsew', padx=(8,4), pady=8)
        _label(left, '🐝  Particle Swarm Optimisation', FONT_H2, ACCENT2).pack(
            anchor='w', padx=10, pady=(10,2))
        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)

        self._pso_n    = tk.IntVar(value=25); self._pso_iter = tk.IntVar(value=25)
        self._pso_w    = tk.DoubleVar(value=0.7)
        self._pso_c1   = tk.DoubleVar(value=1.5); self._pso_c2 = tk.DoubleVar(value=1.5)
        self._pso_gs   = tk.StringVar(value='20'); self._pso_hem = tk.StringVar(value='N')

        def _sl(p, lbl, var, lo, hi, res=1):
            f = tk.Frame(p, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
            _label(f, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            tk.Scale(f, from_=lo, to=hi, resolution=res, orient='horizontal',
                     variable=var, bg=PANEL, highlightthickness=0, length=130).pack(side='right')

        _sl(left, 'Particles',      self._pso_n,    10, 60)
        _sl(left, 'Iterations',     self._pso_iter, 5,  50)
        _sl(left, 'Inertia (w)',    self._pso_w,    0.2, 0.99, 0.01)
        _sl(left, 'Cognitive (c1)', self._pso_c1,   0.5, 3.0, 0.1)
        _sl(left, 'Social (c2)',    self._pso_c2,   0.5, 3.0, 0.1)
        for lbl, var, vals in [('Grid Size', self._pso_gs, ['10','20','30','40']),
                                ('Hemisphere', self._pso_hem, ['N','S'])]:
            f = tk.Frame(left, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
            _label(f, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            ttk.Combobox(f, textvariable=var, values=vals, width=6,
                         state='readonly').pack(side='right')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        self._pso_progress = ttk.Progressbar(left, mode='determinate', length=180)
        self._pso_progress.pack(padx=10, pady=4)
        self._pso_status = _label(left, 'Ready', FONT_SMALL, SUBTEXT)
        self._pso_status.pack(anchor='w', padx=10)

        for txt, cmd, col in [
            ('▶  Run PSO',            self._run_pso,    ACCENT2),
            ('⏹  Stop',               self._stop_pso,   DANGER),
            ('✅  Apply Best Params', self._apply_pso,  ACCENT1),
        ]:
            _btn(left, txt, cmd, col, width=22).pack(padx=10, pady=4, fill='x')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        _label(left, 'Best Fitness', FONT_SMALL, SUBTEXT).pack(anchor='w', padx=10)
        self._pso_best_var = tk.StringVar(value='—')
        _label(left, '', FONT_H1, ACCENT2, bg=PANEL,
               textvariable=self._pso_best_var).pack(anchor='w', padx=10)

        right = _card(parent)
        right.grid(row=0, column=1, sticky='nsew', padx=(4,8), pady=8)
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        self._pso_fig  = Figure(figsize=(8, 6), dpi=90, facecolor=PANEL)
        self._pso_ax1  = self._pso_fig.add_subplot(2, 2, 1)
        self._pso_ax2  = self._pso_fig.add_subplot(2, 2, 2)
        self._pso_ax3  = self._pso_fig.add_subplot(2, 1, 2)
        for ax in [self._pso_ax1, self._pso_ax2, self._pso_ax3]:
            ax.set_facecolor('#E3F2FD')
        self._pso_fig.tight_layout(pad=2)
        self._pso_canvas = _embed_fig(right, self._pso_fig)
        self._pso_best_hist = []; self._pso_avg_hist = []; self._pso_div_hist = []

    def _run_pso(self):
        def worker():
            iters = self._pso_iter.get()
            self._pso = ParticleSwarmOptimisation(
                n_particles=self._pso_n.get(),
                w_start=self._pso_w.get(),
                c1=self._pso_c1.get(), c2=self._pso_c2.get(),
                grid_size=int(self._pso_gs.get()),
                hemisphere=self._pso_hem.get(),
                callback=lambda info: self._q.put(('pso_step', info)))
            self._pso_best_hist.clear(); self._pso_avg_hist.clear(); self._pso_div_hist.clear()
            self._pso_progress['maximum'] = iters
            self._pso.run(iters)
            self._q.put(('pso_done', None))
        self._pso_status.config(text='Running…', fg=ACCENT3)
        threading.Thread(target=worker, daemon=True).start()

    def _stop_pso(self):
        self._pso.stop(); self._pso_status.config(text='Stopped', fg=DANGER)

    def _apply_pso(self):
        if not self._pso.best_params:
            messagebox.showwarning('No Results', 'Run PSO first.'); return
        self._best_params = self._pso.best_params
        self._sync_sliders_to_params(self._best_params)   # ← sync all 24 sliders
        self._preset_var.set('🏞️  Normal (Default)')
        messagebox.showinfo('PSO Params Applied',
            '✅ Best PSO parameters applied to all 24 parameter sliders!\n'
            'Enable "Use AI-Optimised Params" in the Simulation tab to use them.')

    def _update_pso_chart(self, info):
        self._pso_best_hist.append(info['best_fitness'])
        self._pso_avg_hist.append(info['avg_fitness'])
        self._pso_div_hist.append(info['diversity'])
        self._pso_best_var.set(f"{info['best_fitness']:.4f}")
        self._pso_status.config(
            text=f"Iter {info['iteration']}  |  Best: {info['best_fitness']:.4f}", fg=TEXT)
        self._pso_progress['value'] = info['iteration']
        iters = list(range(1, len(self._pso_best_hist) + 1))

        self._pso_ax1.clear(); self._pso_ax1.set_facecolor('#E3F2FD')
        self._pso_ax1.plot(iters, self._pso_best_hist, color=ACCENT2, lw=2, label='Best')
        self._pso_ax1.plot(iters, self._pso_avg_hist,  color=ACCENT3, lw=1.5, ls='--', label='Avg')
        self._pso_ax1.set_title('Fitness Convergence', fontsize=9)
        self._pso_ax1.legend(fontsize=7); self._pso_ax1.tick_params(labelsize=7)

        self._pso_ax2.clear(); self._pso_ax2.set_facecolor('#E3F2FD')
        self._pso_ax2.plot(iters, self._pso_div_hist, color=ACCENT4, lw=2)
        self._pso_ax2.set_title('Swarm Diversity', fontsize=9)
        self._pso_ax2.tick_params(labelsize=7)

        self._pso_ax3.clear(); self._pso_ax3.set_facecolor('#E3F2FD')
        if len(self._pso_best_hist) > 1:
            improvement = np.diff([0] + self._pso_best_hist)
            self._pso_ax3.bar(iters, improvement, color=ACCENT2, alpha=0.6)
            self._pso_ax3.set_title('Fitness Improvement per Iteration', fontsize=9)
            self._pso_ax3.tick_params(labelsize=7)

        self._pso_fig.tight_layout(pad=1.5); self._pso_canvas.draw_idle()

    # TAB 4 – Neural Network
    def _build_nn_tab(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        left = _card(parent)
        left.grid(row=0, column=0, sticky='nsew', padx=(8,4), pady=8)
        _label(left, '🧠  Neural Network Predictor', FONT_H2, ACCENT4).pack(
            anchor='w', padx=10, pady=(10,2))
        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)

        self._nn_samples = tk.IntVar(value=80); self._nn_epochs = tk.IntVar(value=60)
        self._nn_lr      = tk.DoubleVar(value=0.001)
        self._nn_gs      = tk.StringVar(value='20'); self._nn_hem = tk.StringVar(value='N')

        def _sl(p, lbl, var, lo, hi, res=1):
            f = tk.Frame(p, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
            _label(f, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            tk.Scale(f, from_=lo, to=hi, resolution=res, orient='horizontal',
                     variable=var, bg=PANEL, highlightthickness=0, length=130).pack(side='right')

        _sl(left, 'Training Samples', self._nn_samples, 30, 200)
        _sl(left, 'Epochs',           self._nn_epochs,  20, 150)
        f = tk.Frame(left, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
        _label(f, 'Learning Rate', FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
        tk.Scale(f, from_=0.0001, to=0.01, resolution=0.0001, orient='horizontal',
                 variable=self._nn_lr, bg=PANEL, highlightthickness=0,
                 length=130).pack(side='right')
        for lbl, var, vals in [('Grid Size', self._nn_gs, ['10','20','30','40']),
                                ('Hemisphere', self._nn_hem, ['N','S'])]:
            f2 = tk.Frame(left, bg=PANEL); f2.pack(fill='x', padx=10, pady=3)
            _label(f2, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            ttk.Combobox(f2, textvariable=var, values=vals, width=6,
                         state='readonly').pack(side='right')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        self._nn_progress = ttk.Progressbar(left, mode='determinate', length=180)
        self._nn_progress.pack(padx=10, pady=4)
        self._nn_status = _label(left, 'Ready – collect data first', FONT_SMALL, SUBTEXT)
        self._nn_status.pack(anchor='w', padx=10)

        for txt, cmd, col in [
            ('📦  Collect Data',         self._nn_collect,         ACCENT3),
            ('▶  Train Network',          self._nn_train,           ACCENT4),
            ('🧪  Test on Current Params', self._nn_test_compare,   ACCENT2),
            ('🔮  Predict Current',        self._nn_predict_current, ACCENT1),
        ]:
            _btn(left, txt, cmd, col, width=22).pack(padx=10, pady=3, fill='x')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        _label(left, 'Metrics', FONT_SMALL, SUBTEXT).pack(anchor='w', padx=10)
        self._nn_metric_vars = {}
        for k in ('MAE', 'R²', 'NN Predict', 'Actual (60m)', 'Δ Error'):
            f3 = tk.Frame(left, bg=PANEL); f3.pack(fill='x', padx=10, pady=2)
            _label(f3, k+':', FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            sv = tk.StringVar(value='—')
            self._nn_metric_vars[k] = sv
            _label(f3, '', FONT_SMALL, TEXT, bg=PANEL, textvariable=sv).pack(side='right')

        right = _card(parent)
        right.grid(row=0, column=1, sticky='nsew', padx=(4,8), pady=8)
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        self._nn_fig = Figure(figsize=(8, 6), dpi=90, facecolor=PANEL)
        self._nn_ax1 = self._nn_fig.add_subplot(2, 2, 1)
        self._nn_ax2 = self._nn_fig.add_subplot(2, 2, 2)
        self._nn_ax3 = self._nn_fig.add_subplot(2, 1, 2)
        for ax in [self._nn_ax1, self._nn_ax2, self._nn_ax3]:
            ax.set_facecolor('#F3E5F5')
        self._nn_fig.tight_layout(pad=2)
        self._nn_canvas = _embed_fig(right, self._nn_fig)

    def _nn_collect(self):
        def worker():
            n   = self._nn_samples.get()
            gs  = int(self._nn_gs.get())
            hem = self._nn_hem.get()
            self._nn_progress['maximum'] = n

            def cb(i, total):
                self._q.put(('nn_progress', dict(i=i, total=total,
                              msg=f'Collecting {i}/{total}…')))

            X, y_months, y_binary = collect_training_data(n, gs, hem, 120, cb)
            self._train_X = X; self._train_y = y_months
            self._q.put(('nn_collected', dict(n=n)))
        self._nn_status.config(text='Collecting…', fg=ACCENT3)
        threading.Thread(target=worker, daemon=True).start()

    def _nn_train(self):
        if self._train_X is None:
            messagebox.showwarning('No Data', 'Please collect training data first.'); return

        def worker():
            X, y = self._train_X, self._train_y
            split = int(len(X) * 0.8)
            Xtr, Xval = X[:split], X[split:]
            ytr, yval = y[:split], y[split:]
            self._nn = NeuralNetworkPredictor(
                lr=self._nn_lr.get(), epochs=self._nn_epochs.get(),
                callback=lambda info: self._q.put(('nn_epoch', info)))
            self._nn_progress['maximum'] = self._nn_epochs.get()
            self._nn.train(Xtr, ytr, Xval, yval)
            metrics = self._nn.evaluate(Xval, yval)
            self._q.put(('nn_trained', dict(metrics=metrics,
                                            y_true=yval, y_pred=metrics['predictions'])))
        self._nn_status.config(text='Training…', fg=ACCENT4)
        threading.Thread(target=worker, daemon=True).start()

    def _nn_predict_current(self):
        if not self._nn.trained:
            messagebox.showwarning('Not Trained', 'Please train the network first.'); return
        pred = self._nn.predict_single(self._best_params)
        self._nn_metric_vars['NN Predict'].set(f'{pred:.1f} months')
        messagebox.showinfo('Prediction',
            f'Predicted survival for current best params:\n\n'
            f'≈ {pred:.1f} months ({pred/12:.1f} years)')

    def _nn_test_compare(self):
        if not self._nn.trained:
            messagebox.showwarning('Not Trained', 'Train the NN first.'); return

        def worker():
            params = self._get_custom_params()
            gs     = int(self._nn_gs.get())
            hem    = self._nn_hem.get()
            # Actual simulation
            actual_months, _, _ = run_simulation(params, gs, hem, 60)
            # NN prediction
            nn_pred = self._nn.predict_single(params)
            self._q.put(('nn_compare', dict(actual=actual_months, predicted=nn_pred)))

        self._nn_status.config(text='Running actual simulation for comparison…', fg=ACCENT3)
        threading.Thread(target=worker, daemon=True).start()

    def _update_nn_charts(self, event, data):
        if event == 'nn_epoch':
            self._nn_progress['value'] = data['epoch']
            self._nn_status.config(text=f"Epoch {data['epoch']}  loss={data['loss']:.4f}", fg=TEXT)
            self._nn_ax1.clear(); self._nn_ax1.set_facecolor('#F3E5F5')
            self._nn_ax1.plot(self._nn.loss_history, color=ACCENT4, lw=2, label='Train Loss')
            if self._nn.val_loss_history:
                self._nn_ax1.plot(self._nn.val_loss_history, color=ACCENT2, lw=1.5,
                                   ls='--', label='Val Loss')
            self._nn_ax1.set_title('Training Loss', fontsize=9)
            self._nn_ax1.legend(fontsize=7); self._nn_ax1.tick_params(labelsize=7)
            self._nn_fig.tight_layout(pad=1.5); self._nn_canvas.draw_idle()

        elif event == 'nn_trained':
            m = data['metrics']
            self._nn_metric_vars['MAE'].set(f"{m['mae']:.2f} months")
            self._nn_metric_vars['R²'].set(f"{m['r2']:.4f}")
            self._nn_status.config(text='Training complete!', fg=ACCENT1)

            yt = np.array(data['y_true']); yp = data['y_pred']

            self._nn_ax2.clear(); self._nn_ax2.set_facecolor('#F3E5F5')
            self._nn_ax2.scatter(yt, yp, alpha=0.5, color=ACCENT4, s=20)
            mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            self._nn_ax2.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
            self._nn_ax2.set_xlabel('Actual', fontsize=8)
            self._nn_ax2.set_ylabel('Predicted', fontsize=8)
            self._nn_ax2.set_title(f'Actual vs Predicted  (R²={m["r2"]:.3f})', fontsize=9)
            self._nn_ax2.tick_params(labelsize=7)

            self._nn_ax3.clear(); self._nn_ax3.set_facecolor('#F3E5F5')
            errors = np.abs(yt - yp)
            self._nn_ax3.hist(errors, bins=15, color=ACCENT4, alpha=0.7, edgecolor='white')
            self._nn_ax3.axvline(m['mae'], color='red', lw=2, linestyle='--',
                                  label=f'MAE={m["mae"]:.1f}')
            self._nn_ax3.set_title('Absolute Error Distribution', fontsize=9)
            self._nn_ax3.legend(fontsize=7); self._nn_ax3.tick_params(labelsize=7)
            self._nn_fig.tight_layout(pad=1.5); self._nn_canvas.draw_idle()

        elif event == 'nn_compare':
            actual    = data['actual']
            predicted = data['predicted']
            delta     = abs(actual - predicted)
            pct_err   = delta / max(actual, 1) * 100
            color     = SUCCESS if pct_err < 20 else ('orange' if pct_err < 40 else 'red')

            self._nn_metric_vars['NN Predict'].set(f'{predicted:.1f} months')
            self._nn_metric_vars['Actual (60m)'].set(f'{actual} months')
            self._nn_metric_vars['Δ Error'].set(f'{delta:.1f} m  ({pct_err:.0f}%)')
            self._nn_status.config(text=f'Compare done  –  Δ={delta:.1f} months ({pct_err:.0f}%)',
                                   fg=color)

            # Draw side-by-side bar
            self._nn_ax3.clear(); self._nn_ax3.set_facecolor('#F3E5F5')
            bars = self._nn_ax3.bar(['NN Prediction', 'Actual Simulation'],
                                     [predicted, actual],
                                     color=[ACCENT4, ACCENT1], alpha=0.85, width=0.4)
            self._nn_ax3.set_title(f'NN Prediction vs Actual Run  (Δ={delta:.1f} months)',
                                    fontsize=9)
            self._nn_ax3.set_ylabel('Survival Months')
            for bar, v in zip(bars, [predicted, actual]):
                self._nn_ax3.text(bar.get_x() + bar.get_width()/2,
                                   bar.get_height() + 0.5,
                                   f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
            err_color = ACCENT1 if pct_err < 20 else ('orange' if pct_err < 40 else DANGER)
            self._nn_ax3.set_facecolor('#F3E5F5')
            tk.Label(self._nn_canvas.get_tk_widget().master,
                     text=f'  {"✅ CLOSE" if pct_err<20 else "⚠️ FAR"}  ({pct_err:.0f}% error)',
                     fg=err_color, bg=PANEL, font=FONT_SMALL).pack()
            self._nn_fig.tight_layout(pad=1.5); self._nn_canvas.draw_idle()

    # TAB 5 – Decision Tree
    def _build_dt_tab(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        left = _card(parent)
        left.grid(row=0, column=0, sticky='nsew', padx=(8,4), pady=8)
        _label(left, '🌳  Decision Tree Analyser', FONT_H2, ACCENT3).pack(
            anchor='w', padx=10, pady=(10,2))
        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)

        self._dt_depth = tk.IntVar(value=5); self._dt_minlf = tk.IntVar(value=8)
        for lbl, var, lo, hi in [('Max Depth', self._dt_depth, 2, 10),
                                  ('Min Leaf Samples', self._dt_minlf, 2, 20)]:
            f = tk.Frame(left, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
            _label(f, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            tk.Scale(f, from_=lo, to=hi, resolution=1, orient='horizontal',
                     variable=var, bg=PANEL, highlightthickness=0, length=130).pack(side='right')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        self._dt_status = _label(left, 'Needs training data (collect from NN tab)',
                                  FONT_SMALL, SUBTEXT)
        self._dt_status.pack(anchor='w', padx=10, pady=4)

        for txt, cmd, col in [
            ('▶  Train Decision Tree', self._dt_train,      ACCENT3),
            ('📋  Show Rules',          self._dt_show_rules, ACCENT2),
        ]:
            _btn(left, txt, cmd, col, width=22).pack(padx=10, pady=4, fill='x')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        _label(left, 'Accuracy', FONT_SMALL, SUBTEXT).pack(anchor='w', padx=10)
        self._dt_acc_var = tk.StringVar(value='—')
        _label(left, '', FONT_H1, ACCENT3, bg=PANEL,
               textvariable=self._dt_acc_var).pack(anchor='w', padx=10)

        right = _card(parent)
        right.grid(row=0, column=1, sticky='nsew', padx=(4,8), pady=8)
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        self._dt_fig = Figure(figsize=(8, 6), dpi=90, facecolor=PANEL)
        self._dt_ax1 = self._dt_fig.add_subplot(1, 2, 1)
        self._dt_ax2 = self._dt_fig.add_subplot(1, 2, 2)
        for ax in [self._dt_ax1, self._dt_ax2]:
            ax.set_facecolor('#FFF8E1')
        self._dt_fig.tight_layout(pad=2)
        self._dt_canvas = _embed_fig(right, self._dt_fig)

    def _dt_train(self):
        if self._train_X is None:
            messagebox.showwarning('No Data',
                'Please collect training data in the Neural Network tab first.')
            return

        def worker():
            try:
                # Use ALL data but limit features for speed
                X = self._train_X.copy()
                y = (self._train_y > 60).astype(int)
                
                # CRITICAL: Limit to 150 samples for fast training
                if len(X) > 150:
                    np.random.seed(42)
                    indices = np.random.choice(len(X), 150, replace=False)
                    X = X[indices]
                    y = y[indices]
                
                # 80/20 split
                split = int(len(X) * 0.8)
                Xtr, Xte = X[:split], X[split:]
                ytr, yte = y[:split], y[split:]
                
                # Create and train tree
                self._dt = DecisionTreeAnalyzer(
                    max_depth=self._dt_depth.get(),
                    min_samples_leaf=self._dt_minlf.get())
                
                # Update status
                self._q.put(('dt_progress', 'Training decision tree... please wait 5-10 seconds'))
                
                # Train (this is the slow part)
                self._dt.train(Xtr, ytr)
                
                # Evaluate
                metrics = self._dt.evaluate(Xte, yte)
                report = dt_insights_report(self._dt, Xte, yte)
                
                self._q.put(('dt_trained', dict(metrics=metrics, report=report)))
                
            except Exception as e:
                self._q.put(('dt_error', str(e)))
        
        self._dt_status.config(text='Training... (this may take a moment)', fg=ACCENT3)
        threading.Thread(target=worker, daemon=True).start()

    def _dt_show_rules(self):
        if not self._dt.trained:
            messagebox.showwarning('Not Trained', 'Please train the decision tree first.'); return
        win = tk.Toplevel(self); win.title('Decision Tree Rules')
        win.configure(bg=PANEL); win.geometry('750x540')
        txt = scrolledtext.ScrolledText(win, font=FONT_MONO, bg='#FAFAFA',
                                         fg=TEXT, relief='flat', wrap='word')
        txt.pack(fill='both', expand=True, padx=12, pady=12)
        txt.insert('end', '=== DECISION TREE SURVIVAL RULES ===\n\n')
        txt.insert('end', f'Tree Accuracy: {self._dt.accuracy*100:.1f}%\n\n')
        txt.insert('end', '─' * 60 + '\n\n')
        for rule in self._dt.rules:
            txt.insert('end', rule + '\n\n')
        txt.config(state='disabled')

    def _update_dt_charts(self, data):
        report = data['report']
        acc    = report['accuracy']
        self._dt_acc_var.set(f"{acc*100:.1f}%")
        self._dt_status.config(text=f'Accuracy: {acc*100:.1f}%', fg=ACCENT1)

        top_feats   = report['top_features']
        names       = [f[0][:16] for f in top_feats]
        importances = [f[1] for f in top_feats]

        self._dt_ax1.clear(); self._dt_ax1.set_facecolor('#FFF8E1')
        colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, DANGER,
                  '#00BCD4', '#795548', '#607D8B'][:len(names)]
        self._dt_ax1.barh(names[::-1], importances[::-1], color=colors[::-1], alpha=0.85)
        self._dt_ax1.set_title('Feature Importance', fontsize=9)
        self._dt_ax1.tick_params(labelsize=7)

        yt = np.array(report['y_true']); yp = np.array(report['predictions'])
        tp = int(((yp==1)&(yt==1)).sum()); fp = int(((yp==1)&(yt==0)).sum())
        fn = int(((yp==0)&(yt==1)).sum()); tn = int(((yp==0)&(yt==0)).sum())
        cm = np.array([[tp, fp],[fn, tn]])

        self._dt_ax2.clear(); self._dt_ax2.set_facecolor('#FFF8E1')
        self._dt_ax2.imshow(cm, cmap='YlOrBr')
        self._dt_ax2.set_xticks([0,1]); self._dt_ax2.set_yticks([0,1])
        self._dt_ax2.set_xticklabels(['Pred Sur.','Pred Col.'], fontsize=8)
        self._dt_ax2.set_yticklabels(['Sur.','Col.'], fontsize=8)
        for i in range(2):
            for j in range(2):
                self._dt_ax2.text(j, i, str(cm[i,j]), ha='center', va='center',
                                   fontsize=14, fontweight='bold',
                                   color='white' if cm[i,j]>cm.max()/2 else TEXT)
        self._dt_ax2.set_title('Confusion Matrix', fontsize=9)
        self._dt_fig.tight_layout(pad=1.5); self._dt_canvas.draw_idle()

    # TAB 6 – Analysis (6 Research Experiments)
    def _build_analysis_tab(self, parent):
        parent.columnconfigure(0, weight=1); parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        left = _card(parent)
        left.grid(row=0, column=0, sticky='nsew', padx=(8,4), pady=8)
        _label(left, '📊  Research Experiments', FONT_H2, ACCENT2).pack(
            anchor='w', padx=10, pady=(10,2))
        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)

        self._an_gs  = tk.StringVar(value='20'); self._an_hem = tk.StringVar(value='N')
        self._an_tri = tk.IntVar(value=12)

        for lbl, var, vals in [('Grid Size', self._an_gs, ['10','20','30','40']),
                                ('Hemisphere', self._an_hem, ['N','S'])]:
            f = tk.Frame(left, bg=PANEL); f.pack(fill='x', padx=10, pady=3)
            _label(f, lbl, FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
            ttk.Combobox(f, textvariable=var, values=vals, width=6,
                         state='readonly').pack(side='right')

        f2 = tk.Frame(left, bg=PANEL); f2.pack(fill='x', padx=10, pady=3)
        _label(f2, 'Trials (hemi compare)', FONT_SMALL, SUBTEXT, bg=PANEL).pack(side='left')
        tk.Scale(f2, from_=5, to=30, resolution=1, orient='horizontal',
                 variable=self._an_tri, bg=PANEL, highlightthickness=0, length=100).pack(side='right')

        tk.Frame(left, bg=BORDER, height=1).pack(fill='x', padx=10, pady=6)
        self._an_progress = ttk.Progressbar(left, mode='determinate', length=180)
        self._an_progress.pack(padx=10, pady=4)
        self._an_status = _label(left, 'Select an experiment', FONT_SMALL, SUBTEXT)
        self._an_status.pack(anchor='w', padx=10)

        experiments = [
            ('Exp 1: GA vs PSO',              self._exp_ga_pso),
            ('Exp 2: Sensitivity Analysis',    self._exp_sensitivity),
            ('Exp 3: NN Accuracy',             self._exp_nn_accuracy),
            ('Exp 4: Hemisphere Comparison',   self._exp_hemisphere),
            ('Exp 5: Evolutionary Dynamics',   self._exp_evo_dynamics),
            ('Exp 6: DT Rule Extraction',      self._exp_dt_rules),
            ('Scalability (All Grid Sizes)',   self._exp_scalability),
        ]
        for txt, cmd in experiments:
            _btn(left, txt, cmd, ACCENT2, width=24).pack(padx=10, pady=3, fill='x')

        right = _card(parent)
        right.grid(row=0, column=1, sticky='nsew', padx=(4,8), pady=8)
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        self._an_fig    = Figure(figsize=(8, 6.5), dpi=90, facecolor=PANEL)
        self._an_canvas = FigureCanvasTkAgg(self._an_fig, master=right)
        self._an_canvas.draw()
        self._an_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

    # Experiment runners
    def _exp_ga_pso(self):
        if not self._ga.best_fitness_history or not self._pso.best_fitness_history:
            messagebox.showwarning('Run Algorithms First',
                'Please run both GA and PSO first (tabs 2 & 3).'); return
        data = compare_algorithms(self._ga.best_fitness_history,
                                  self._pso.best_fitness_history)
        self._an_fig.clear()
        ax1 = self._an_fig.add_subplot(1, 2, 1)
        ax2 = self._an_fig.add_subplot(1, 2, 2)
        ga_h = data['ga_history']; pso_h = data['pso_history']
        ax1.plot(range(1,len(ga_h)+1),  ga_h,  color=ACCENT1, lw=2, label='GA')
        ax1.plot(range(1,len(pso_h)+1), pso_h, color=ACCENT2, lw=2, label='PSO')
        ax1.set_title('Fitness: GA vs PSO', fontsize=10); ax1.legend()
        labels = ['GA Best','PSO Best','GA Final','PSO Final']
        vals   = [data['ga_best'],data['pso_best'],data['ga_final'],data['pso_final']]
        bars   = ax2.bar(labels, vals, color=[ACCENT1,ACCENT2,ACCENT1,ACCENT2], alpha=0.8)
        ax2.set_title(f'Final Results  (Winner: {data["winner"]})', fontsize=10); ax2.set_ylim(0,1)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f'{v:.3f}', ha='center', fontsize=8)
        self._an_fig.tight_layout(pad=2); self._an_canvas.draw_idle()
        self._an_status.config(text=f'Winner: {data["winner"]}  GA conv={data["ga_convergence"]}  PSO conv={data["pso_convergence"]}', fg=TEXT)

    def _exp_sensitivity(self):
        def worker():
            gs  = int(self._an_gs.get()); hem = self._an_hem.get()
            n   = len(EcosystemParams.PARAM_NAMES) * 6
            self._an_progress['maximum'] = n
            def cb(done, total):
                self._q.put(('an_progress', dict(done=done, total=total, msg=f'Sensitivity {done}/{total}')))
            result = parameter_sensitivity(EcosystemParams(), gs, hem, 6, cb)
            self._q.put(('an_sensitivity', result))
        self._an_status.config(text='Running sensitivity…', fg=ACCENT3)
        threading.Thread(target=worker, daemon=True).start()

    def _draw_sensitivity(self, result):
        ranked = result['ranked'][:10]
        names  = [v[1]['label'][:16] for v in ranked]
        svals  = [v[1]['sensitivity'] for v in ranked]
        self._an_fig.clear(); ax = self._an_fig.add_subplot(1, 1, 1)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))[::-1]
        bars = ax.barh(names[::-1], svals[::-1], color=colors, alpha=0.85)
        ax.set_title('Parameter Sensitivity (survival month range)', fontsize=11)
        ax.set_xlabel('Survival Range (months)'); ax.tick_params(labelsize=8)
        for bar, v in zip(bars, svals[::-1]):
            ax.text(v+0.3, bar.get_y()+bar.get_height()/2, f'{v:.1f}', va='center', fontsize=8)
        self._an_fig.tight_layout(pad=2); self._an_canvas.draw_idle()
        self._an_status.config(text=f'Most sensitive: {ranked[0][1]["label"]}', fg=TEXT)

    def _exp_nn_accuracy(self):
        if not self._nn.trained:
            messagebox.showwarning('Not Trained', 'Please train the Neural Network first.'); return
        if self._train_X is None: return
        split = int(len(self._train_X)*0.8)
        Xte=self._train_X[split:]; yte=self._train_y[split:]
        yp=self._nn.predict(Xte); report=nn_accuracy_report(yte, yp)
        self._an_fig.clear()
        ax1=self._an_fig.add_subplot(1,2,1); ax2=self._an_fig.add_subplot(1,2,2)
        ax1.scatter(yte,yp,alpha=0.5,color=ACCENT4,s=25)
        mn,mx=min(yte.min(),yp.min()),max(yte.max(),yp.max())
        ax1.plot([mn,mx],[mn,mx],'r--',lw=2)
        ax1.set_xlabel('Actual (months)',fontsize=9); ax1.set_ylabel('Predicted (months)',fontsize=9)
        ax1.set_title(f'NN Prediction  (R²={report["r2"]:.3f})',fontsize=10)
        errors=np.abs(yte-yp)
        ax2.hist(errors,bins=15,color=ACCENT4,alpha=0.7,edgecolor='white')
        ax2.axvline(report['mae'],color='red',lw=2,ls='--',label=f'MAE={report["mae"]:.1f}')
        ax2.set_title('Absolute Error Distribution',fontsize=10); ax2.legend(fontsize=8)
        self._an_fig.tight_layout(pad=2); self._an_canvas.draw_idle()
        self._an_status.config(text=f'MAE={report["mae"]:.2f}m  RMSE={report["rmse"]:.2f}  R²={report["r2"]:.3f}',fg=TEXT)

    def _exp_hemisphere(self):
        def worker():
            n=self._an_tri.get(); gs=int(self._an_gs.get())
            self._an_progress['maximum']=n*2
            def cb(done,total):
                self._q.put(('an_progress',dict(done=done,total=total,msg=f'Hemisphere {done}/{total}')))
            result=hemisphere_comparison(n,gs,120,cb)
            self._q.put(('an_hemisphere',result))
        self._an_status.config(text='Comparing hemispheres…',fg=ACCENT3)
        threading.Thread(target=worker,daemon=True).start()

    def _draw_hemisphere(self, data):
        nm=data['north_months']; sm=data['south_months']
        self._an_fig.clear()
        ax1=self._an_fig.add_subplot(1,2,1); ax2=self._an_fig.add_subplot(1,2,2)
        ax1.boxplot([nm,sm],labels=['North','South'],patch_artist=True,
                   boxprops=dict(facecolor='#B3E5FC'),medianprops=dict(color='red',lw=2))
        ax1.set_title('Survival: North vs South',fontsize=10); ax1.set_ylabel('Months')
        cats=['Mean','Std','Max']
        n_vals=[data['north_mean'],data['north_std'],data['north_max']]
        s_vals=[data['south_mean'],data['south_std'],data['south_max']]
        x=np.arange(len(cats)); w=0.35
        ax2.bar(x-w/2,n_vals,w,label='North',color=ACCENT2,alpha=0.8)
        ax2.bar(x+w/2,s_vals,w,label='South',color=ACCENT3,alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(cats)
        ax2.set_title('Comparison Metrics',fontsize=10); ax2.legend(fontsize=8)
        self._an_fig.tight_layout(pad=2); self._an_canvas.draw_idle()
        self._an_status.config(text=f'North μ={data["north_mean"]:.1f}m  South μ={data["south_mean"]:.1f}m',fg=TEXT)

    def _exp_evo_dynamics(self):
        if not self._ga.best_fitness_history:
            messagebox.showwarning('Run GA First','Please run the Genetic Algorithm first.'); return
        report=evolutionary_dynamics_report(self._ga.best_fitness_history,
                                            self._ga.avg_fitness_history,
                                            self._ga.diversity_history)
        self._an_fig.clear()
        ax1=self._an_fig.add_subplot(2,1,1); ax2=self._an_fig.add_subplot(2,1,2)
        gens=list(range(1,len(report['best_history'])+1))
        ax1.plot(gens,report['best_history'],color=ACCENT1,lw=2,label='Best Fitness')
        ax1.plot(gens,report['avg_history'],color=ACCENT2,lw=1.5,ls='--',label='Avg Fitness')
        ax1.axvline(report['exploitation_start'],color='red',lw=1.5,ls=':',
                    label=f'Exploitation starts gen {report["exploitation_start"]}')
        ax1.set_title('Evolutionary Dynamics – Fitness Progression',fontsize=10)
        ax1.legend(fontsize=8); ax1.set_ylabel('Fitness')
        ax2.plot(gens,report['diversity_history'],color=ACCENT4,lw=2)
        ax2.fill_between(gens,report['diversity_history'],alpha=0.2,color=ACCENT4)
        ax2.axvline(report['exploitation_start'],color='red',lw=1.5,ls=':')
        ax2.set_title('Population Diversity over Generations',fontsize=10)
        ax2.set_xlabel('Generation'); ax2.set_ylabel('Diversity')
        self._an_fig.tight_layout(pad=2); self._an_canvas.draw_idle()
        self._an_status.config(text=f'Improvement: +{report["total_improvement"]:.4f}  Converged: {report["converged"]}  Final diversity: {report["final_diversity"]:.4f}',fg=TEXT)

    def _exp_dt_rules(self):
        if not self._dt.trained:
            messagebox.showwarning('Not Trained','Please train the Decision Tree first.'); return
        top=self._dt.top_features(8)
        names=[f[0][:18] for f in top]; vals=[f[1] for f in top]
        self._an_fig.clear(); ax=self._an_fig.add_subplot(1,1,1)
        cmap=plt.cm.Greens(np.linspace(0.4,0.9,len(names)))
        ax.barh(names[::-1],vals[::-1],color=cmap[::-1],alpha=0.9)
        ax.set_title(f'Decision Tree Feature Importance  (Accuracy={self._dt.accuracy*100:.1f}%)',fontsize=10)
        ax.set_xlabel('Importance Score'); ax.tick_params(labelsize=8)
        for i,(name,v) in enumerate(zip(names[::-1],vals[::-1])):
            ax.text(v+0.005,i,f'{v:.3f}',va='center',fontsize=8)
        self._an_fig.tight_layout(pad=2); self._an_canvas.draw_idle()
        self._an_status.config(text=f'Top predictor: {top[0][0]}  Accuracy: {self._dt.accuracy*100:.1f}%',fg=TEXT)

    def _exp_scalability(self):
        def worker():
            hem=self._an_hem.get()
            self._an_progress['maximum']=len(GRID_SIZES)
            def cb(done,total):
                self._q.put(('an_progress',dict(done=done,total=total,
                              msg=f'Grid {GRID_SIZES[done-1]}×{GRID_SIZES[done-1]}')))
            # FIX: Use properly-scaled random params for each grid size
            results = {}
            for i, gs in enumerate(GRID_SIZES):
                scaled = EcosystemParams.random_params(gs)
                months, fit, _ = run_simulation(scaled, gs, hem, 60)
                results[gs] = dict(months=months, fitness=fit, grid_size=gs,
                                   area=gs*gs, label=f'{gs}×{gs}')
                cb(i+1, len(GRID_SIZES))
            self._q.put(('an_scalability', results))
        self._an_status.config(text='Testing grid scalability…',fg=ACCENT3)
        threading.Thread(target=worker,daemon=True).start()

    def _draw_scalability(self, data):
        gs_labels=[f'{v["label"]}\n({v["area"]} km²)' for v in data.values()]
        months=[v['months'] for v in data.values()]
        fitnesses=[v['fitness'] for v in data.values()]
        self._an_fig.clear()
        ax1=self._an_fig.add_subplot(1,2,1); ax2=self._an_fig.add_subplot(1,2,2)
        colors=[ACCENT1,ACCENT2,ACCENT3,ACCENT4]
        ax1.bar(gs_labels,months,color=colors,alpha=0.8,edgecolor='white')
        ax1.set_title('Survival Months by Grid Size',fontsize=10); ax1.set_ylabel('Months survived')
        for x,v in zip(ax1.patches,months):
            ax1.text(x.get_x()+x.get_width()/2,v+0.5,str(v),ha='center',fontsize=9)
        ax2.bar(gs_labels,fitnesses,color=colors,alpha=0.8,edgecolor='white')
        ax2.set_title('Fitness Score by Grid Size',fontsize=10)
        ax2.set_ylabel('Fitness'); ax2.set_ylim(0,1)
        for x,v in zip(ax2.patches,fitnesses):
            ax2.text(x.get_x()+x.get_width()/2,v+0.01,f'{v:.3f}',ha='center',fontsize=9)
        self._an_fig.tight_layout(pad=2); self._an_canvas.draw_idle()
        self._an_status.config(text='Scalability test complete',fg=ACCENT1)

    # TAB 7 – Parameter Controls  (ALL 24 PARAMETERS)
    def _build_params_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)

        # Top toolbar
        toolbar = tk.Frame(parent, bg=BG, pady=6)
        toolbar.grid(row=0, column=0, sticky='ew', padx=8)

        _label(toolbar, '⚙️  Manual Parameter Controls  –  All 24 Ecosystem Parameters',
               FONT_H2, ACCENT1, bg=BG).pack(side='left', padx=8)

        for txt, cmd, col in [
            ('↺  Reset to Default',    self._params_reset_default, ACCENT2),
            ('✅  Apply to Simulation', self._params_apply_sim,    ACCENT1),
            ('📤  Export to GA/PSO',    self._params_export,        ACCENT3),
        ]:
            _btn(toolbar, txt, cmd, col, width=20).pack(side='right', padx=4)

        # Preset selector in toolbar
        _label(toolbar, 'Preset:', FONT_SMALL, SUBTEXT, bg=BG).pack(side='right', padx=(8,2))
        self._param_preset_var = tk.StringVar(value='🏞️  Normal (Default)')
        cb = ttk.Combobox(toolbar, textvariable=self._param_preset_var,
                          values=list(PRESETS.keys()), width=20, state='readonly')
        cb.pack(side='right', padx=2)
        cb.bind('<<ComboboxSelected>>', lambda e: self._apply_preset(self._param_preset_var.get()))

        # Scrollable parameters body
        body_card = _card(parent)
        body_card.grid(row=1, column=0, sticky='nsew', padx=8, pady=(0,8))
        body_card.columnconfigure(0, weight=1); body_card.rowconfigure(0, weight=1)

        outer, inner = _scrollable_frame(body_card)
        outer.grid(row=0, column=0, sticky='nsew')

        # Column headers
        header = tk.Frame(inner, bg='#E8F5E9')
        header.pack(fill='x', padx=4, pady=(4,0))
        for text, w in [('Parameter', 28), ('Min', 6), ('Value (Slider)', 32), ('Max', 6), ('Input', 10)]:
            _label(header, text, FONT_SMALL, TEXT, bg='#E8F5E9', width=w, anchor='center').pack(side='left', padx=2)

        # One row per parameter
        for i, name in enumerate(EcosystemParams.PARAM_NAMES):
            lo, hi    = float(EcosystemParams.BOUNDS[i, 0]), float(EcosystemParams.BOUNDS[i, 1])
            label_txt = EcosystemParams.LABELS[i]
            row_bg    = PANEL if i % 2 == 0 else '#F8F9FA'

            row = tk.Frame(inner, bg=row_bg)
            row.pack(fill='x', padx=4, pady=1)

            # Label
            _label(row, label_txt, FONT_SMALL, TEXT, bg=row_bg, width=28,
                   anchor='w').pack(side='left', padx=(6,2))

            # Min value
            _label(row, f'{lo:.3g}', FONT_SMALL, SUBTEXT, bg=row_bg,
                   width=6, anchor='e').pack(side='left', padx=2)

            # Slider
            is_int  = i in EcosystemParams.INT_IDX
            res     = 1 if is_int else (hi - lo) / 200
            var     = self._param_vars[name]
            entry_v = self._param_entry_vars[name]

            def _on_slider(val, n=name, ev=entry_v, is_i=is_int):
                fval = float(val)
                ev.set(str(int(round(fval))) if is_i else f'{fval:.4g}')

            slider = tk.Scale(row, from_=lo, to=hi, resolution=res,
                              orient='horizontal', variable=var,
                              bg=row_bg, highlightthickness=0, length=220,
                              showvalue=False, command=_on_slider)
            slider.pack(side='left', padx=4)

            # Max value
            _label(row, f'{hi:.3g}', FONT_SMALL, SUBTEXT, bg=row_bg,
                   width=6, anchor='w').pack(side='left', padx=2)

            # Numeric entry
            def _on_entry(event, n=name, v=var, ev=entry_v, lo=lo, hi=hi, is_i=is_int):
                try:
                    val = float(ev.get())
                    val = max(lo, min(hi, val))
                    if is_i: val = int(round(val))
                    v.set(val)
                    ev.set(str(int(val)) if is_i else f'{val:.4g}')
                except ValueError:
                    pass

            entry = tk.Entry(row, textvariable=entry_v, font=FONT_MONO,
                             width=10, bg='#F5F5F5', relief='flat',
                             highlightbackground=BORDER, highlightthickness=1)
            entry.pack(side='left', padx=(4,6))
            entry.bind('<Return>', _on_entry)
            entry.bind('<FocusOut>', _on_entry)

    def _params_reset_default(self):
        self._sync_sliders_to_params(EcosystemParams())
        self._preset_var.set('🏞️  Normal (Default)')
        self._param_preset_var.set('🏞️  Normal (Default)')

    def _params_apply_sim(self):
        self._reset_sim()   # eco will be recreated with new params on next Start
        messagebox.showinfo('Applied',
            '✅ Custom parameters applied!\nPress ▶ Start Simulation to run with these values.')

    def _params_export(self):
        params = self._get_custom_params()
        self._best_params = params
        messagebox.showinfo('Exported',
            '✅ Current parameter values exported as "Best Params".\n'
            'Enable "Use AI-Optimised Params" in the Simulation tab.')

    # TAB 8 – Population Analysis
    def _build_popanalysis_tab(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)

        # Toolbar
        tb = tk.Frame(parent, bg=BG, pady=6); tb.grid(row=0, column=0, sticky='ew', padx=8)
        _label(tb, '📈  Population Analysis  –  Deep Ecosystem Insights',
               FONT_H2, ACCENT2, bg=BG).pack(side='left', padx=8)
        _btn(tb, '🔄  Refresh Charts', self._refresh_popanalysis, ACCENT2, width=20).pack(side='right', padx=8)

        # Chart card
        card = _card(parent)
        card.grid(row=1, column=0, sticky='nsew', padx=8, pady=(0,8))
        card.columnconfigure(0, weight=1); card.rowconfigure(0, weight=1)

        self._pa_fig    = Figure(figsize=(12, 7), dpi=90, facecolor=PANEL)
        self._pa_canvas = FigureCanvasTkAgg(self._pa_fig, master=card)
        self._pa_canvas.draw()
        self._pa_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

        # Status
        self._pa_status = _label(card, 'Run a simulation first, then click Refresh Charts.',
                                  FONT_SMALL, SUBTEXT)
        self._pa_status.pack(anchor='w', padx=10, pady=4)

    def _refresh_popanalysis(self):
        if not self._eco or not self._eco.stats_history:
            messagebox.showwarning('No Data', 'Run a simulation first, then click Refresh.'); return

        history = self._eco.stats_history
        months  = [s['month']      for s in history]
        plants  = [s['plants']     for s in history]
        herbs   = [s['herbivores'] for s in history]
        carns   = [s['carnivores'] for s in history]

        self._pa_fig.clear()
        self._pa_fig.suptitle('Ecosystem Population Analysis', fontsize=13, fontweight='bold')

        # Graph A: Predator-Prey Dynamics
        ax1 = self._pa_fig.add_subplot(2, 3, (1, 2))
        ax1.plot(months, herbs,  color=ACCENT3, lw=2,   label='Herbivores 🐄')
        ax1.plot(months, carns,  color=DANGER,  lw=2,   label='Carnivores 🦁')
        ax1.fill_between(months, herbs, alpha=0.12, color=ACCENT3)
        ax1.fill_between(months, carns, alpha=0.12, color=DANGER)
        # Annotate local peaks (predator-prey cycles)
        herb_arr = np.array(herbs); carn_arr = np.array(carns)
        for i in range(1, len(herb_arr)-1):
            if herb_arr[i] > herb_arr[i-1] and herb_arr[i] > herb_arr[i+1] and herb_arr[i] > np.mean(herb_arr)*1.2:
                ax1.annotate('↑', (months[i], herb_arr[i]), fontsize=9, color=ACCENT3,
                             xytext=(0, 4), textcoords='offset points', ha='center')
        ax1.set_title('A  Predator-Prey Population Dynamics', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Month'); ax1.set_ylabel('Population Count')
        ax1.legend(fontsize=8); ax1.set_facecolor('#FFFDE7')
        ax1.tick_params(labelsize=8)

        # Phase portrait inset (optional)
        if len(herbs) > 5:
            ax1_ins = ax1.inset_axes([0.72, 0.5, 0.26, 0.45])
            ax1_ins.plot(herbs, carns, color='#6A1B9A', lw=1, alpha=0.7)
            ax1_ins.scatter([herbs[-1]], [carns[-1]], color='red', s=20, zorder=5)
            ax1_ins.set_xlabel('Herb', fontsize=6); ax1_ins.set_ylabel('Carn', fontsize=6)
            ax1_ins.set_title('Phase', fontsize=7); ax1_ins.tick_params(labelsize=6)

        # Graph B: Population Stabilit
        ax2 = self._pa_fig.add_subplot(2, 3, 3)
        window = 12
        if len(herbs) >= window:
            rolling_std  = [np.std(herbs[max(0,i-window):i+1]) for i in range(len(herbs))]
            max_std      = max(rolling_std) if max(rolling_std) > 0 else 1
            stability    = [1 - (s / max_std) for s in rolling_std]
            # Color by stability level
            for i in range(len(months)-1):
                col = ACCENT1 if stability[i] > 0.6 else (ACCENT3 if stability[i] > 0.3 else DANGER)
                ax2.plot(months[i:i+2], stability[i:i+2], color=col, lw=2)
            ax2.axhline(0.3, color=ACCENT3, lw=1, ls='--', label='Warning threshold')
            ax2.axhline(0.6, color=ACCENT1, lw=1, ls='--', label='Stable threshold')
            ax2.set_ylim(0, 1.05)
            ax2.fill_between(months, stability, alpha=0.08, color=ACCENT2)
            ax2.set_title('B  Population Stability Score\n(12-month rolling)', fontsize=9, fontweight='bold')
            ax2.set_xlabel('Month'); ax2.set_ylabel('Stability (0=unstable, 1=stable)')
            ax2.legend(fontsize=7); ax2.tick_params(labelsize=7)
            final_stab = stability[-1]
            ax2.set_facecolor('#F3F9F3' if final_stab > 0.5 else '#FFF3E0')
        else:
            ax2.text(0.5, 0.5, 'Need ≥12 months\nof data', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=11, color=SUBTEXT)
            ax2.set_title('B  Population Stability', fontsize=9, fontweight='bold')

        # Graph C1: Gender Ratio (Pie charts)
        ax3 = self._pa_fig.add_subplot(2, 3, 4)
        if self._eco:
            live_h = [h for h in self._eco.herbivores if h.alive]
            hm = sum(1 for h in live_h if h.gender=='M')
            hf = len(live_h) - hm
            live_c = [c for c in self._eco.carnivores if c.alive]
            cm_ = sum(1 for c in live_c if c.gender=='M')
            cf  = len(live_c) - cm_

            angles_h = [hm, hf] if hm + hf > 0 else [1, 1]
            angles_c = [cm_, cf] if cm_ + cf > 0 else [1, 1]

            ax3.pie(angles_h, labels=[f'♂ {hm}', f'♀ {hf}'], autopct='%1.0f%%',
                    colors=[ACCENT2, '#F48FB1'], startangle=90,
                    wedgeprops=dict(width=0.5), textprops=dict(fontsize=8))
            ax3.set_title(f'C1  Herbivore Gender Ratio\n(Total: {len(live_h)})',
                          fontsize=9, fontweight='bold')

            ax4 = self._pa_fig.add_subplot(2, 3, 5)
            ax4.pie(angles_c, labels=[f'♂ {cm_}', f'♀ {cf}'], autopct='%1.0f%%',
                    colors=[ACCENT3, '#CE93D8'], startangle=90,
                    wedgeprops=dict(width=0.5), textprops=dict(fontsize=8))
            ax4.set_title(f'C2  Carnivore Gender Ratio\n(Total: {len(live_c)})',
                          fontsize=9, fontweight='bold')

        # Graph C3: Shannon Biodiversity Index
        ax5 = self._pa_fig.add_subplot(2, 3, 6)
        shannon_scores = []
        for s in history:
            total = s['plants'] + s['herbivores'] + s['carnivores']
            if total == 0:
                shannon_scores.append(0); continue
            probs = [s['plants']/total, s['herbivores']/total, s['carnivores']/total]
            H = -sum(p * math.log(p+1e-9) for p in probs if p > 0) / math.log(3)
            shannon_scores.append(H)

        ax5.plot(months, shannon_scores, color=ACCENT4, lw=2)
        ax5.fill_between(months, shannon_scores, alpha=0.15, color=ACCENT4)
        ax5.set_ylim(0, 1.05)
        ax5.axhline(0.8, color=ACCENT1, lw=1, ls='--', label='High diversity')
        ax5.axhline(0.5, color=ACCENT3, lw=1, ls='--', label='Medium diversity')
        ax5.set_title('C3  Shannon Biodiversity Index\n(0=monoculture, 1=balanced)',
                      fontsize=9, fontweight='bold')
        ax5.set_xlabel('Month'); ax5.set_ylabel('Diversity Index')
        ax5.legend(fontsize=7); ax5.tick_params(labelsize=7)
        ax5.set_facecolor('#F3E5F5')

        if shannon_scores:
            final_H = shannon_scores[-1]
            quality = '🟢 High' if final_H > 0.8 else ('🟡 Medium' if final_H > 0.5 else '🔴 Low')
            self._pa_status.config(
                text=f'Month {months[-1]}  |  Shannon Index: {final_H:.3f}  ({quality})  '
                     f'|  Herbs: {herbs[-1]}  Carns: {carns[-1]}',
                fg=ACCENT1 if final_H > 0.6 else DANGER)

        self._pa_fig.tight_layout(rect=[0, 0, 1, 0.96], pad=2)
        self._pa_canvas.draw_idle()

    # Queue poller  (thread-safe GUI updates)
    def _poll_queue(self):
        try:
            while True:
                event, data = self._q.get_nowait()

                if event == 'ga_step':
                    self._update_ga_chart(data)
                elif event == 'ga_done':
                    self._ga_status.config(text='GA complete! ✅', fg=ACCENT1)

                elif event == 'pso_step':
                    self._update_pso_chart(data)
                elif event == 'pso_done':
                    self._pso_status.config(text='PSO complete! ✅', fg=ACCENT1)

                elif event in ('nn_epoch', 'nn_trained', 'nn_compare'):
                    self._update_nn_charts(event, data)
                elif event == 'nn_progress':
                    self._nn_progress['value'] = data['i']
                    self._nn_status.config(text=data['msg'], fg=ACCENT3)
                elif event == 'nn_collected':
                    self._nn_status.config(
                        text=f'{data["n"]} samples collected. Ready to train. ✅', fg=ACCENT1)
                    self._nn_progress['value'] = self._nn_progress['maximum']

                elif event == 'dt_trained':
                    self._update_dt_charts(data)
                    self._dt_status.config(text='Tree trained! ✅', fg=ACCENT1)

                elif event == 'an_progress':
                    self._an_progress['value'] = data['done']
                    self._an_status.config(text=data['msg'], fg=ACCENT3)
                elif event == 'an_sensitivity':
                    self._draw_sensitivity(data)
                elif event == 'an_hemisphere':
                    self._draw_hemisphere(data)
                elif event == 'an_scalability':
                    self._draw_scalability(data)
                elif event == 'dt_progress':
                    self._dt_status.config(text=data, fg=ACCENT3)

                elif event == 'dt_error':
                    self._dt_status.config(text=f'Error: {data}', fg=DANGER)
                    messagebox.showerror('DT Error', f'Training failed:\n{data}')

        except queue.Empty:
            pass
        self.after(80, self._poll_queue)


if __name__ == '__main__':
    app = App()
    app.mainloop()
