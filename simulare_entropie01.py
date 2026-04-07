# -*- coding: utf-8 -*-
import numpy as np
import taichi as ti
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from scipy.special import gammaln
import matplotlib.pyplot as plt

# --- Partea 1: Configurare și Inițializare ---

# Numărul total de particule (Crescut pentru efect statistic și vizual mai bun)
N_PARTICLES = 1500
# Numărul de cadre inițiale în care sunt afișate doar particulele
INITIAL_PARTICLE_FRAMES = 60

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

ti.init(arch=ti.gpu)
print("Backend Taichi: GPU")

# --- Partea 2: Structuri de Date și Kernel-uri Taichi ---

positions = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
cubulet_counts = ti.field(dtype=ti.i32, shape=(GRID_DIV, GRID_DIV, GRID_DIV))

@ti.kernel
def initialize_particles():
    for i in range(N_PARTICLES):
        # Plasăm particulele inițial într-un colț (ex: o porțiune din cub)
        # pentru a vizualiza creșterea entropiei pe măsură ce gazul se extinde
        # din starea ordonată spre starea de echilibru (dezordonată).
        positions[i] = ti.Vector([ti.random() for _ in range(3)]) * (CUBE_SIZE / 2.5)
        vel_range = MAX_VEL - MIN_VEL
        velocities[i] = ti.Vector([ti.random() for _ in range(3)]) * vel_range + MIN_VEL

@ti.kernel
def update():
    for i in range(N_PARTICLES):
        pos = positions[i] + velocities[i] * DT
        for dim in ti.static(range(3)):
            if pos[dim] < 0 or pos[dim] > CUBE_SIZE:
                velocities[i][dim] *= -1
                pos[dim] = ti.max(0.0, ti.min(CUBE_SIZE, pos[dim]))
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

def main():
    initialize_particles()

    # --- Configurare Scenă PyVista ---
    
    plotter = BackgroundPlotter(window_size=(1024, 768))
    
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
    
    # Ajustare clim: entropia medie la echilibru este când particulele sunt distribuite uniform
    # Folosim o valoare de max ~ 1.5x peste medie pentru a avea variații de culoare frumoase.
    avg_particles_per_cell = N_PARTICLES / (GRID_DIV**3)
    max_expected_entropy = gammaln(avg_particles_per_cell * 1.5 + 1)
    
    grid_actor = plotter.add_mesh(grid, scalars=np.zeros(grid.n_cells), cmap='coolwarm_r',
                                  clim=[0, max_expected_entropy], 
                                  scalar_bar_args={'title': 'Contributie Entropie', 'color': 'white'},
                                  opacity=0)
    
    grid.cell_data['entropy_scalars'] = np.zeros(grid.n_cells)
    grid.set_active_scalars('entropy_scalars')

    # Particule mai mici și de o culoare modernă (cyan/turcoaz)
    particle_points = pv.PolyData(positions.to_numpy())
    plotter.add_points(particle_points, color='#00ffcc', point_size=4, render_points_as_spheres=True)

    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.0) 
    
    print("Pornire simulare interactivă...")
    
    frame_count = 0
    text_actor = None 

    def simulation_callback():
        nonlocal frame_count, text_actor, grid_actor, particle_points, grid
        
        for _ in range(N_SUBSTEPS):
            update()
        
        particle_points.points = positions.to_numpy()
        text_content = f"Timp: {frame_count * DT * N_SUBSTEPS:.2f}s"
        
        if frame_count >= INITIAL_PARTICLE_FRAMES:
            count_particles()
            counts_np = cubulet_counts.to_numpy()
            entropy_values = calculate_entropy_per_cell_proportional(counts_np)
            
            if grid_actor.GetProperty().GetOpacity() == 0:
                grid_actor.GetProperty().SetOpacity(0.35) # Opacitate mai subtilă pentru a vedea particulele
            
            grid.cell_data['entropy_scalars'] = entropy_values
            grid.Modified()
            
            # Entropia totală S = k_B * ln(W)
            total_entropy = K_B * (gammaln(N_PARTICLES + 1) - np.sum(gammaln(counts_np[counts_np > 0] + 1)))
            
            # Formatăm entropia mai curat
            text_content += f"\nEntropie: {total_entropy * 1e23:.2f} x 10^-23 J/K"
        
        if text_actor:
            plotter.remove_actor(text_actor, render=False)
        
        text_actor = plotter.add_text(text_content, position='upper_left',
                                      font_size=14, render=False, color='white')
        
        frame_count += 1
        plotter.render()

    # Adăugăm callback-ul pentru a rula continuu
    plotter.add_callback(simulation_callback, interval=16)

    # --- Buton de Ieșire ---
    def exit_simulation(state):
        plotter.close()
        
    plotter.add_checkbox_button_widget(exit_simulation, value=False,
                                       color_on='red', color_off='darkred',
                                       position=(10, 10), size=30)
    plotter.add_text("Iesire (Click sau X)", position=(50, 15), font_size=10, color='white')

    # În loc de 'input()', lăsăm plotter-ul să ruleze până când fereastra este închisă
    print("Simularea rulează în fereastra separată. Închideți fereastra sau apăsați butonul de Ieșire pentru a opri.")
    
    try:
        plotter.app.exec_()
    except Exception:
        pass

    print("Simulare încheiată.")

if __name__ == "__main__":
    main()