# -*- coding: utf-8 -*-
import numpy as np
import taichi as ti
import pyvista as pv
from scipy.special import gammaln
import matplotlib.pyplot as plt
import sys # Folosit pentru a afișa progresul

# --- Partea 1: Configurare și Inițializare ---

# Numărul total de particule
N_PARTICLES = 100
# Numărul de cadre inițiale în care sunt afișate doar particulele
INITIAL_PARTICLE_FRAMES = 60

# Numărul total de cadre de salvat în animație
TOTAL_FRAMES = 360 # (12 secunde la 30 fps)
# Numele fișierului de ieșire
OUTPUT_FILENAME = "simulatie_entropie_rotatie.mp4"

# Vitezele inițiale aleatorii
MIN_VEL = -1.0
MAX_VEL = 1.0

# Parametrii fizici și de simulare
CUBE_SIZE = 1.0
DT = 0.001 
N_SUBSTEPS = 5

# Grila împărțită în 3x3x3 = 27 de cubulețe
GRID_DIV = 3

# Constanta lui Boltzmann
K_B = 1.380649e-23 

# Inițializare Taichi
try:
    ti.init(arch=ti.gpu)
    print("Backend Taichi: GPU")
except:
    ti.init(arch=ti.cpu)
    print("Backend Taichi: CPU")

# --- Partea 2: Structuri de Date și Kernel-uri Taichi ---

positions = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
cubulet_counts = ti.field(dtype=ti.i32, shape=(GRID_DIV, GRID_DIV, GRID_DIV))

@ti.kernel
def initialize_particles():
    for i in range(N_PARTICLES):
        positions[i] = ti.Vector([ti.random() for _ in range(3)]) * CUBE_SIZE
        vel_range = MAX_VEL - MIN_VEL
        velocities[i] = ti.Vector([ti.random() for _ in range(3)]) * vel_range + MIN_VEL

@ti.kernel
def update():
    for i in range(N_PARTICLES):
        pos = positions[i] + velocities[i] * DT
        for dim in ti.static(range(3)):
            if pos[dim] < 0 or pos[dim] > CUBE_SIZE:
                velocities[i][dim] *= -1
                pos[dim] = ti.max(0, ti.min(CUBE_SIZE, pos[dim]))
        positions[i] = pos

@ti.kernel
def count_particles():
    cubulet_counts.fill(0)
    for i in range(N_PARTICLES):
        index = ti.cast(positions[i] / (CUBE_SIZE / GRID_DIV), ti.i32)
        index = ti.max(0, ti.min(GRID_DIV - 1, index))
        ti.atomic_add(cubulet_counts[index], 1)

# --- Partea 3: Funcția de Calcul a Entropiei ---

def calculate_entropy_per_cell_proportional(counts_array):
    counts_flat = counts_array.flatten()
    return gammaln(counts_flat + 1)

# --- Partea 4: Bucla Principală de Simulare și Randare ---

# --- Partea 4: Bucla Principală de Simulare și Randare ---

def main():
    initialize_particles()

    # --- Configurare Scenă PyVista ---
    
    # Inițializăm Plotter-ul fără fereastră (off_screen)
    plotter = pv.Plotter(window_size=[1024, 768], off_screen=True)
    
    # Conturul cubului principal
    cube_outline = pv.Cube(center=(CUBE_SIZE/2, CUBE_SIZE/2, CUBE_SIZE/2),
                           x_length=CUBE_SIZE, y_length=CUBE_SIZE, z_length=CUBE_SIZE)
    plotter.add_mesh(cube_outline, style='wireframe', color='white', line_width=3)

    # Grila pentru cele 27 de cubulețe
    x = np.linspace(0, CUBE_SIZE, GRID_DIV + 1)
    y = np.linspace(0, CUBE_SIZE, GRID_DIV + 1)
    z = np.linspace(0, CUBE_SIZE, GRID_DIV + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid = pv.StructuredGrid(xx, yy, zz)
    
    grid_actor = plotter.add_mesh(grid, scalars=np.zeros(grid.n_cells), cmap='coolwarm_r',
                                 clim=[0, gammaln(6 + 1)], 
                                 scalar_bar_args={'title': 'Entropie (Albastru=Mare, Roșu=Mică)', 'color': 'black'},
                                 opacity=0)
    
    grid.cell_data['entropy_scalars'] = np.zeros(grid.n_cells)
    grid.set_active_scalars('entropy_scalars')

    # Particulele vizibile
    particle_points = pv.PolyData(positions.to_numpy())
    plotter.add_points(particle_points, color='yellow', point_size=6, render_points_as_spheres=True)

    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.0) 
    
    # --- Bucla de animație pentru salvare ---
    
    # **CORECTIE:** Adaugă un render inițial aici
    # Acest apel forțează inițializarea `plotter.camera` înainte de a intra în buclă.
    print("Inițializare scenă (render inițial)...")
    plotter.render()
    
    print(f"Se pregătește salvarea animației în {OUTPUT_FILENAME}...")
    print(f"Acest proces va genera {TOTAL_FRAMES} cadre.")
    
    # Calculăm unghiul de rotație per cadru
    # Va face o rotație completă (360 grade) pe parcursul întregii animații
    rotation_angle_per_frame = 360.0 / TOTAL_FRAMES
    
    # Deschide fișierul video
    plotter.open_movie(OUTPUT_FILENAME, framerate=30) 
    
    text_actor = None 

    # Rulează bucla pentru numărul total de cadre
    for frame_count in range(TOTAL_FRAMES):
        
        # --- Logica de simulare ---
        
        for _ in range(N_SUBSTEPS):
            update()
        
        particle_points.points = positions.to_numpy()
        text_content = f"Timp: {frame_count * DT * N_SUBSTEPS:.2f}s (Cadru: {frame_count})"
        
        if frame_count >= INITIAL_PARTICLE_FRAMES:
            count_particles()
            counts_np = cubulet_counts.to_numpy()
            entropy_values = calculate_entropy_per_cell_proportional(counts_np)
            
            if grid_actor.GetProperty().GetOpacity() == 0:
                grid_actor.GetProperty().SetOpacity(0.5)
            
            grid.cell_data['entropy_scalars'] = entropy_values
            grid.Modified()
            
            total_entropy = K_B * (gammaln(N_PARTICLES + 1) - np.sum(gammaln(counts_np[counts_np > 0] + 1)))
            text_content += f"\nEntropie Totală: {total_entropy:.3e} J/K"
        
        if text_actor:
            plotter.remove_actor(text_actor, render=False)
        
        text_actor = plotter.add_text(text_content, position='upper_left',
                                      font_size=12, render=False)
        
        # --- Sfârșitul logicii de simulare ---
        
        # **** MODIFICAREA CHEIE ESTE AICI ****
        # În loc de .azimuth(...), folosim +=
        plotter.camera.azimuth += rotation_angle_per_frame

        # Randează cadrul curent
        plotter.render()
        
        # Scrie cadrul în fișierul video
        plotter.write_frame()
        
        # Afișează progresul în consolă (cu suprascriere)
        progress = (frame_count + 1) / TOTAL_FRAMES
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f"\rProgres: |{bar}| {progress:.1%} ({frame_count + 1}/{TOTAL_FRAMES})")
        sys.stdout.flush()

    # Închide fișierul video după ce bucla s-a terminat
    plotter.close()

    print(f"\n\nSimulare încheiată.")
    print(f"Animația a fost salvată cu succes în {OUTPUT_FILENAME}")

main()