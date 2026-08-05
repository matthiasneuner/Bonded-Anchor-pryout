import pyvista as pv
import numpy as np
import argparse
import math
import os
import vtk
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*algorithm.*")

def draw_dynamic_chart(current_time, plotter, active_chart, has_csv, csv_disp, csv_load, csv_time, t_max):
    if not has_csv:
        return

    if active_chart is not None:
        plotter.remove_chart(active_chart)

    active_chart = pv.Chart2D(size=(0.35, 0.35), loc=(0.62, 0.62))
    active_chart.background_color = (1.0, 1.0, 1.0, 0.7)

    active_chart.x_label = "displacement (mm)"
    active_chart.y_label = "reaction force(kN)"

    active_chart.x_axis.label_size = 30
    active_chart.y_axis.label_size = 30
    active_chart.x_axis.tick_label_size = 30
    active_chart.y_axis.tick_label_size = 30

    idx_max = np.searchsorted(csv_time, t_max)
    if idx_max >= len(csv_time):
        idx_max = len(csv_time) - 1

    csv_disp_clipped = csv_disp[:idx_max+1]
    csv_load_clipped = csv_load[:idx_max+1]

    csv_max_disp = csv_disp_clipped.max()
    csv_max_load = csv_load_clipped.max()
    
    active_chart.line(csv_disp_clipped, csv_load_clipped, color="grey", width=4.0)
    active_chart.x_axis.behavior = "fixed"
    active_chart.x_axis.range = [0.0, csv_max_disp * 1.05]
    active_chart.y_axis.behavior = "fixed"
    active_chart.y_axis.range = [0.0, csv_max_load * 1.05]

    idx = np.searchsorted(csv_time, current_time)
    if idx >= len(csv_time):
        idx = len(csv_time) - 1
    if idx > idx_max:
        idx = idx_max

    mask_progress = np.arange(len(csv_disp_clipped)) <= idx
    if np.any(mask_progress):
        active_chart.line(csv_disp_clipped[mask_progress], csv_load_clipped[mask_progress], color="red", width=6.0)

    current_disp_val = csv_disp[idx]
    current_load = csv_load[idx]
    active_chart.scatter(np.array([current_disp_val]), np.array([current_load]), color="red", size=25)

    plotter.add_chart(active_chart)
    return active_chart

def extract_block(multiblock, name):
    for i in range(multiblock.n_blocks):
        block_name = multiblock.get_block_name(i)
        if block_name == name:
            return multiblock[i]
        if isinstance(multiblock[i], pv.MultiBlock):
            res = extract_block(multiblock[i], name)
            if res is not None:
                return res
    raise ValueError(f"Block '{name}' not found in the dataset.")

def try_get_block(multiblock, name):
    try:
        return extract_block(multiblock, name)
    except ValueError:
        return None

def find_bc_block(multiblock, name):
    target = name.upper()
    
    def search_exact(block):
        for i in range(block.n_blocks):
            block_name = block.get_block_name(i)
            if block_name is not None and block_name.upper() == target:
                return block[i]
            if isinstance(block[i], pv.MultiBlock):
                res = search_exact(block[i])
                if res is not None:
                    return res
        return None

    exact_res = search_exact(multiblock)
    if exact_res is not None:
        return exact_res
        
    def search_substring(block):
        for i in range(block.n_blocks):
            block_name = block.get_block_name(i)
            if block_name is not None and target in block_name.upper():
                return block[i]
            if isinstance(block[i], pv.MultiBlock):
                res = search_substring(block[i])
                if res is not None:
                    return res
        return None

    return search_substring(multiblock)

def create_bc_glyphs(nodes_mesh, direction_vector, size=8.0):
    if nodes_mesh is None or nodes_mesh.n_points == 0:
        return None
    
    dir_np = np.array(direction_vector, dtype=float)
    dir_len = np.linalg.norm(dir_np)
    if dir_len > 1e-8:
        dir_np = dir_np / dir_len
    
    cone_geom = pv.Cone(
        center=-dir_np * size / 2.0,
        direction=dir_np,
        height=size,
        radius=size * 0.3,
        resolution=12
    )
    
    glyphs = nodes_mesh.glyph(geom=cone_geom, orient=False, scale=False)
    return glyphs

def get_clean_quad_surface(mesh):
    try:
        return mesh.extract_surface(nonlinear_subdivision=0, algorithm='dataset_surface')
    except (TypeError, ValueError):
        try:
            return mesh.extract_surface(nonlinear_subdivision=0)
        except Exception:
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputData(mesh)
            surface_filter.SetNonlinearSubdivisionLevel(0)
            surface_filter.Update()
            return pv.wrap(surface_filter.GetOutput())

cached_frames = {}
def get_raw_data_at_time(t, reader, t_min_orig, t_max_orig, time_fraction, warp_factor):
    if t not in cached_frames:
        reader.set_active_time_value(t)
        cached_frames[t] = reader.read()
    return cached_frames[t]

def get_cinematic_state(f, default_pos, default_focal, back_pos, back_focal):
    opacities = {"concrete": 0.0, "mortar": 0.0, "plate": 0.0, "anchor": 0.0, "washer": 0.0, "nut": 0.0, "bc": 0.0, "back_bc": 0.0}
    cam_pos = default_pos
    cam_focal = default_focal
    annotation = ""
    anchor_offset_y = 0.0
    
    if f < 45:
        progress = f / 44.0
        opacities["concrete"] = 0.5 * (1.0 - math.cos(progress * math.pi)) * 0.9999
        annotation = "concrete slab"
    elif f < 60:
        opacities["concrete"] = 0.9999
        annotation = "concrete slab"
        
    elif f < 105:
        progress = (f - 60) / 44.0
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.5 * (1.0 - math.cos(progress * math.pi)) * 0.9999
        annotation = "chemical mortar"
    elif f < 120:
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        annotation = "chemical mortar"
        
    elif f < 150:
        progress = (f - 120) / 29.0
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.5 * (1.0 - math.cos(progress * math.pi)) * 0.9999
        annotation = "anchor plate"
    elif f < 160:
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        annotation = "anchor plate"
        
    elif f < 190:
        progress = (f - 160) / 29.0
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        t = 0.5 * (1.0 - math.cos(progress * math.pi))
        opacities["anchor"] = t * 0.9999
        opacities["washer"] = t * 0.9999
        opacities["nut"] = t * 0.9999
        anchor_offset_y = 120.0
        annotation = "anchor, washer, nut"
    elif f < 200:
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        anchor_offset_y = 120.0
        annotation = "anchor, washer, nut"
        
    elif f < 230:
        progress = (f - 200) / 29.0
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        anchor_offset_y = 120.0 - (0.5 * (1.0 - math.cos(progress * math.pi)) * 120.0)
        annotation = "anchor, washer, nut"
    elif f < 240:
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        anchor_offset_y = 0.0
        annotation = "anchor, washer, nut"
        
    elif f < 270:
        progress = (f - 240) / 29.0
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        opacities["bc"] = 0.5 * (1.0 - math.cos(progress * math.pi)) * 0.9999
        annotation = "boundary conditions"
    elif f < 285:
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        opacities["bc"] = 0.9999
        annotation = "boundary conditions"

    elif f < 330:
        progress = (f - 285) / 44.0
        t = 0.5 * (1.0 - math.cos(progress * math.pi))
        cam_pos = tuple(p0 + t * (p1 - p0) for p0, p1 in zip(default_pos, back_pos))
        cam_focal = tuple(f0 + t * (f1 - f0) for f0, f1 in zip(default_focal, back_focal))
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        opacities["bc"] = 0.9999
        opacities["back_bc"] = 0.5 * (1.0 - math.cos(progress * math.pi)) * 0.9999
        annotation = "back support BCs"
    elif f < 345:
        cam_pos = back_pos
        cam_focal = back_focal
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        opacities["bc"] = 0.9999
        opacities["back_bc"] = 0.9999
        annotation = "back support BCs"

    elif f < 390:
        progress = (f - 345) / 44.0
        t = 0.5 * (1.0 - math.cos(progress * math.pi))
        cam_pos = tuple(p0 + t * (p1 - p0) for p0, p1 in zip(back_pos, default_pos))
        cam_focal = tuple(f0 + t * (f1 - f0) for f0, f1 in zip(back_focal, default_focal))
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        opacities["bc"] = 0.9999
        opacities["back_bc"] = 0.9999
        annotation = "boundary conditions"
    else:
        opacities["concrete"] = 0.9999
        opacities["mortar"] = 0.9999
        opacities["plate"] = 0.9999
        opacities["anchor"] = 0.9999
        opacities["washer"] = 0.9999
        opacities["nut"] = 0.9999
        opacities["bc"] = 0.9999
        opacities["back_bc"] = 0.9999
        annotation = "boundary conditions"

    return opacities, cam_pos, cam_focal, annotation, anchor_offset_y


def main():
    def interpolate_mesh(mesh_a, mesh_b, weight, out_mesh=None):
        if out_mesh is None:
            out_mesh = mesh_a.copy(deep=True)
        out_mesh.points = mesh_a.points + weight * (mesh_b.points - mesh_a.points)
    
        if disp_var in mesh_b.point_data:
            a_disp = mesh_a.point_data[disp_var] if disp_var in mesh_a.point_data else np.zeros_like(mesh_b.point_data[disp_var])
            out_mesh.point_data[disp_var] = a_disp + weight * (mesh_b.point_data[disp_var] - a_disp)
    
        if rf_var in mesh_b.point_data:
            a_rf = mesh_a.point_data[rf_var] if rf_var in mesh_a.point_data else np.zeros_like(mesh_b.point_data[rf_var])
            out_mesh.point_data[rf_var] = a_rf + weight * (mesh_b.point_data[rf_var] - a_rf)
    
        if raw_damage_var in mesh_a.point_data and raw_damage_var in mesh_b.point_data:
            interp_kappa = mesh_a.point_data[raw_damage_var] + weight * (mesh_b.point_data[raw_damage_var] - mesh_a.point_data[raw_damage_var])
            out_mesh.point_data[raw_damage_var] = interp_kappa
            out_mesh.point_data[mapped_damage_var] = 1.0 - np.exp(-interp_kappa / damage_scale_factor)
            
        if "S" in mesh_b.cell_data:
            a_s = mesh_a.cell_data["S"] if "S" in mesh_a.cell_data else np.zeros_like(mesh_b.cell_data["S"])
            out_mesh.cell_data["S"] = a_s + weight * (mesh_b.cell_data["S"] - a_s)
    
        return out_mesh

    parser = argparse.ArgumentParser(description="Offscreen render structural FEA results with PyVista (Cinematic Intro) for Pryout.")
    parser.add_argument("filename", type=str, help="Path to the EnSight .case file")
    parser.add_argument("--steel-vm", action="store_true", help="Plot von Mises stress on the steel anchor (cell data S)")
    parser.add_argument("-t", "--threshold", type=float, default=0.8,
                        help="Damage threshold for the fracture surface (default: 0.8)")
    parser.add_argument("-w", "--warp", type=float, default=1.0,
                        help="Displacement magnification factor (default: 1.0)")
    parser.add_argument("-g", "--glyphscale", type=float, default=0.002,
                        help="Scale factor for the reaction force arrows (default: 0.002)")
    parser.add_argument("-p", "--platethick", type=float, default=10.0,
                        help="Thickness of the rigid plates in negative normal direction (default: 10.0)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output MP4 filename (default: same as case file name with _pryout suffix)")
    parser.add_argument("-l", "--label", type=str, default="",
                        help="Custom text label to display on the animation")
    parser.add_argument("-c", "--csv", type=str, default="",
                        help="Path to CSV file (defaults to matching .case filename)")
    parser.add_argument("-m", "--maxtime", type=float, default=None,
                        help="Maximum time until results are considered")
    parser.add_argument("-b", "--bcsize", type=float, default=12.0,
                        help="Height of the boundary condition support cones (default: 8.0)")
    parser.add_argument("--hide-bc", action="store_true",
                        help="Hide boundary condition support cones in the visualization")
    parser.add_argument("--no-intro", action="store_true", help="Skip the cinematic intro and jump straight to deformation")
    parser.add_argument("-d", "--maxdisp", type=float, default=None,
                        help="Maximum displacement (mm) to consider in the animation")
    args = parser.parse_args()

    filename = args.filename
    warp_factor = args.warp
    damage_threshold = args.threshold
    glyph_scale = args.glyphscale
    plate_thickness = args.platethick
    bc_size = args.bcsize
    show_bc = not args.hide_bc

    output_file = args.output
    if output_file is None:
        base = "".join(os.path.basename(filename).split(".")[:-1])
        suffix = "_pryout"
        if args.no_intro:
            suffix += "_nointro"
        if args.steel_vm:
            suffix += "_vm"
        suffix += f"_w{args.warp}_t{args.threshold}"
        if args.maxdisp is not None:
            suffix += f"_d{args.maxdisp}"
        if args.maxtime is not None:
            suffix += f"_m{args.maxtime}"
        output_file = base + suffix + ".mp4"
    custom_label = args.label
    max_time_arg = args.maxtime
    max_disp_arg = args.maxdisp
    no_intro = args.no_intro

    csv_file = args.csv if args.csv else filename.replace(".case", ".csv")

    disp_var = "nodeDisplacements"
    raw_damage_var = "nonlocalDamage"
    mapped_damage_var = "damage"
    rf_var = "nodeReactionForces" 
    damage_scale_factor = 0.0037

    block_names = {
        "concrete": "ASSEMBLY_CONCRETE-1_CONCRETE",
        "mortar": "ASSEMBLY_STEEL-1_MORTAR",
        "anchor": "ASSEMBLY_STEEL-1_ANCHOR",
        "plate": "ASSEMBLY_STEEL-1_PLATE",
        "washer": "ASSEMBLY_STEEL-1_WASHER",
        "nut": "ASSEMBLY_STEEL-1_NUT",
        "load_nodes": "NSET_ASSEMBLY_STEEL-1_PLATE_LEFT_LOADING" 
    }

    bc_configs = [
        {
            "block": "NSET_ASSEMBLY_CONCRETE-1_LEFT",
            "constraints": ["X"]
        },
        {
            "block": "NSET_ASSEMBLY_CONCRETE-1_RIGHT",
            "constraints": ["X"]
        },
        {
            "block": "NSET_ASSEMBLY_CONCRETE-1_BOTTOM",
            "constraints": ["Y"]
        },
        {
            "block": "Z_SYMM",
            "constraints": ["Z"]
        }
    ]

    back_support_bc_config = {
        "block": "NSET_ASSEMBLY_CONCRETE-1_BACK",
        "constraints": ["X"]
    }

    sargs = dict(
        vertical=True,
        position_x=0.03,
        position_y=0.03,
        height=0.5,
        width=0.08,
        title_font_size=30,   
        label_font_size=30,
        fmt="%.1f"
    )

    sargs_vm = dict(
        title="von Mises (MPa)",
        vertical=True,
        position_x=0.11,
        position_y=0.03,
        height=0.5,
        width=0.08,
        title_font_size=30,   
        label_font_size=30,
        fmt="%.0f"
    )

    force_arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

    reader = pv.get_reader(filename)
    available_times = np.array(reader.time_values)

    t_min_orig = available_times.min()
    t_max_orig = available_times.max()
    t_max = min(t_max_orig, max_time_arg) if max_time_arg is not None else t_max_orig
    t_max += 1e-4

    time_fraction = (t_max - t_min_orig) / (t_max_orig - t_min_orig) if t_max_orig > t_min_orig else 1.0

    plotter = pv.Plotter(window_size=[1280*2, 960*2], off_screen=False)
    plotter.open_movie(output_file, framerate=30)
    plotter.set_background("white")
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)

    if custom_label:
        plotter.add_text(custom_label, position="upper_left", font_size=30, color="black")

    plotter.add_axes(viewport=(0.0, 0.8, 0.2, 1.0))

    has_csv = os.path.exists(csv_file)
    if has_csv:
        try:
            with open(csv_file, 'r') as f:
                header = f.readline().strip().split(',')
            header = [h.strip().lower() for h in header]

            time_idx = 0
            load_idx = 1
            disp_idx = 2

            for idx, name in enumerate(header):
                if 'time' in name:
                    time_idx = idx
                    break
            for idx, name in enumerate(header):
                if any(k in name for k in ['disp', 'u1', 'u2', 'u3', 'displ', 'u_']):
                    disp_idx = idx
                    break
            for idx, name in enumerate(header):
                if idx != time_idx and idx != disp_idx:
                    if any(k in name for k in ['rf', 'force', 'load', 'f1', 'f2', 'f3']):
                        load_idx = idx
                        break

            csv_data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
            csv_time = csv_data[:, time_idx]
            csv_load = csv_data[:, load_idx] * 2.0 / 1000.
            csv_disp = csv_data[:, disp_idx]

            if np.abs(csv_disp).max() < 0.1:
                csv_disp = csv_disp * 1000.0

        except Exception as e:
            csv_data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
            csv_time = csv_data[:, 0]
            csv_load = csv_data[:, 1] * 2.0 / 1000.
            csv_disp = csv_data[:, 2]
    else:
        csv_time = None

    if has_csv and len(csv_time) > 0:
        csv_disp = np.abs(csv_disp)
        csv_load = np.abs(csv_load)
        for i in range(1, len(csv_time)):
            if csv_time[i] < csv_time[i-1]:
                csv_time[i:] += csv_time[i-1]

    active_chart = None

    if max_disp_arg is not None:
        if has_csv:
            matching_indices = np.where(np.abs(csv_disp) >= max_disp_arg)[0]
            if len(matching_indices) > 0:
                t_max_disp = csv_time[matching_indices[0]]

                if t_max_disp > t_min_orig:
                    t_max = min(t_max, t_max_disp)
        else:
            try:
                init_data = get_raw_data_at_time(t_min_orig, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)
                load_nodes_ref = extract_block(init_data, block_names["load_nodes"])
                all_nodes_ref = extract_block(init_data, "ALL")
                load_pts = load_nodes_ref.points
                all_pts = all_nodes_ref.points
                load_indices_temp = []
                for pt in load_pts:
                    diff = np.linalg.norm(all_pts - pt, axis=1)
                    idx = np.argmin(diff)
                    load_indices_temp.append(idx)

                t_max_disp = None
                for t in available_times:
                    temp_data = get_raw_data_at_time(t, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)
                    if "ALL" in temp_data.keys() and disp_var in temp_data["ALL"].point_data:
                        all_disp = temp_data["ALL"].point_data[disp_var]
                        avg_disp = np.abs(np.mean(all_disp[load_indices_temp, 0]))
                        if avg_disp >= max_disp_arg:
                            t_max_disp = t
                            break
                if t_max_disp is not None:
                    if t_max_disp > t_min_orig:
                        t_max = min(t_max, t_max_disp)
            except Exception as e:
                pass

    active_chart = None

    load_indices = None
    disp_component_idx = 0
    max_disp_val = 0.0

    if has_csv:
        try:
            init_data = get_raw_data_at_time(t_min_orig, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)
            load_nodes_ref = extract_block(init_data, block_names["load_nodes"])
            all_nodes_ref = extract_block(init_data, "ALL")

            load_pts = load_nodes_ref.points
            all_pts = all_nodes_ref.points

            load_indices = []
            for pt in load_pts:
                diff = np.linalg.norm(all_pts - pt, axis=1)
                idx = np.argmin(diff)
                load_indices.append(idx)

            final_data = get_raw_data_at_time(t_max, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)
            if "ALL" in final_data.keys() and disp_var in final_data["ALL"].point_data:
                all_disp = final_data["ALL"].point_data[disp_var]
                final_disp_vec = np.mean(all_disp[load_indices], axis=0)
                csv_disp_max = csv_disp.max()
                disp_component_idx = int(np.argmin(np.abs(np.abs(final_disp_vec) - csv_disp_max)))
                max_disp_val = np.abs(final_disp_vec[disp_component_idx])
            else:
                max_disp_val = csv_disp.max()
        except Exception as e:
            max_disp_val = csv_disp.max() if has_csv else 0.0

    if max_disp_arg is not None:
        max_disp_val = max_disp_arg

    global_max_rf = 0.0
    for t in available_times[available_times <= t_max + 1e-4]:
        temp_data = get_raw_data_at_time(t, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)
        try:
            temp_load = extract_block(temp_data, block_names["load_nodes"])
            if rf_var in temp_load.point_data:
                rf_vecs = np.array(temp_load.point_data[rf_var])
                if rf_vecs.ndim == 1:
                    rf_vecs = rf_vecs.reshape(-1, 3)
                current_sum = np.sum(np.abs(rf_vecs[:, 0]))
                if current_sum > global_max_rf:
                    global_max_rf = current_sum
        except ValueError:
            pass

    if global_max_rf == 0.0: 
        global_max_rf = 1.0  

    init_data = get_raw_data_at_time(t_min_orig, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)

    concrete_init = try_get_block(init_data, block_names["concrete"])
    mortar_init = try_get_block(init_data, block_names["mortar"])
    anchor_init = try_get_block(init_data, block_names["anchor"])
    plate_init = try_get_block(init_data, block_names["plate"])
    washer_init = try_get_block(init_data, block_names["washer"])
    nut_init = try_get_block(init_data, block_names["nut"])

    clean_concrete = get_clean_quad_surface(concrete_init) if concrete_init else None
    if clean_concrete:
        if raw_damage_var in clean_concrete.point_data:
            clean_concrete.point_data[mapped_damage_var] = 1.0 - np.exp(-clean_concrete.point_data[raw_damage_var] / damage_scale_factor)
        else:
            clean_concrete.point_data[mapped_damage_var] = np.zeros(clean_concrete.n_points)

    clean_mortar = get_clean_quad_surface(mortar_init) if mortar_init else None
    clean_anchor = get_clean_quad_surface(anchor_init) if anchor_init else None
    clean_plate = get_clean_quad_surface(plate_init) if plate_init else None
    clean_washer = get_clean_quad_surface(washer_init) if washer_init else None
    clean_nut = get_clean_quad_surface(nut_init) if nut_init else None

    if clean_concrete:
        cam_dist = clean_concrete.length * 0.8
    else:
        cam_dist = 500.0

    default_pos = (cam_dist, cam_dist, cam_dist)
    default_focal = (0.0, 0.0, 0.0)

    _rot_matrix = np.array([
        [ 0.0,  0.0, -1.0],
        [ 0.0,  1.0,  0.0],
        [ 1.0,  0.0,  0.0]
    ])
    if clean_concrete:
        _cb = clean_concrete.bounds
        _rot_center = np.array([_cb[0], 0.0, _cb[5]]) 
    else:
        _rot_center = np.zeros(3)

    back_pos   = tuple(_rot_center + _rot_matrix @ (np.array(default_pos)   - _rot_center))
    back_focal = list(_rot_center + _rot_matrix @ (np.array(default_focal) - _rot_center))
    if clean_concrete:
        _cb = clean_concrete.bounds
        back_focal[2] = 0.5 * (_cb[4] + _cb[5])
    back_focal = tuple(back_focal)

    if not no_intro:
        intro_frames = 405
        for f in range(intro_frames):
            opacities, cam_pos, cam_focal, annotation, anchor_offset_y = get_cinematic_state(
                f, default_pos, default_focal, back_pos, back_focal)

            plotter.camera.position = cam_pos
            plotter.camera.focal_point = cam_focal
            plotter.camera.up = (0.0, 1.0, 0.0)

            if clean_concrete is not None:
                plotter.add_mesh(clean_concrete, scalars=mapped_damage_var, cmap="coolwarm", clim=[0.0, 1.0],
                                 opacity=opacities["concrete"], show_edges=True, edge_color="black",
                                 scalar_bar_args=sargs, show_scalar_bar=False, reset_camera=False, name="concrete_mesh")

            if clean_mortar is not None:
                plotter.add_mesh(clean_mortar, color="#3CB371", show_edges=True, edge_color="black",
                                 opacity=opacities["mortar"], reset_camera=False, name="mortar_mesh")

            if clean_plate is not None:
                plotter.add_mesh(clean_plate, color="silver", show_edges=True, edge_color="black",
                                 opacity=opacities["plate"], reset_camera=False, name="plate_mesh")

            if clean_anchor is not None:
                a_mesh = clean_anchor.copy(deep=True) if anchor_offset_y > 0 else clean_anchor
                if anchor_offset_y > 0: a_mesh.points += np.array([0.0, 1.0, 0.0]) * anchor_offset_y
                if args.steel_vm and "S" in a_mesh.cell_data:
                    s = a_mesh.cell_data["S"]
                    if s.shape[1] >= 6:
                        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2 + 6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)))
                    else:
                        vm = np.zeros(s.shape[0])
                    a_mesh.cell_data["von_Mises"] = vm
                    plotter.add_mesh(a_mesh, scalars="von_Mises", cmap="copper_r", show_edges=True, edge_color="black",
                                     opacity=opacities["anchor"], reset_camera=False, name="anchor_mesh", preference="cell",
                                     clim=[0.0, 1080.0], scalar_bar_args=sargs_vm, show_scalar_bar=False)
                else:
                    plotter.add_mesh(a_mesh, color="silver", show_edges=True, edge_color="black",
                                     opacity=opacities["anchor"], reset_camera=False, name="anchor_mesh")

            if clean_washer is not None:
                w_mesh = clean_washer.copy(deep=True) if anchor_offset_y > 0 else clean_washer
                if anchor_offset_y > 0: w_mesh.points += np.array([0.0, 1.0, 0.0]) * anchor_offset_y
                plotter.add_mesh(w_mesh, color="darkgrey", show_edges=True, edge_color="black",
                                 opacity=opacities["washer"], reset_camera=False, name="washer_mesh")

            if clean_nut is not None:
                n_mesh = clean_nut.copy(deep=True) if anchor_offset_y > 0 else clean_nut
                if anchor_offset_y > 0: n_mesh.points += np.array([0.0, 1.0, 0.0]) * anchor_offset_y
                plotter.add_mesh(n_mesh, color="darkgrey", show_edges=True, edge_color="black",
                                 opacity=opacities["nut"], reset_camera=False, name="nut_mesh")

            if show_bc and opacities["bc"] > 0.0:
                for bc_cfg in bc_configs:
                    bc_mesh = find_bc_block(init_data, bc_cfg["block"])
                    if bc_mesh is not None:
                        for constr in bc_cfg["constraints"]:
                            if constr == "X":
                                dir_vec = (1, 0, 0)
                                color = "red"
                                name = f"bc_{bc_cfg['block']}_X"
                            elif constr == "Y":
                                dir_vec = (0, 1, 0)
                                color = "green"
                                name = f"bc_{bc_cfg['block']}_Y"
                            elif constr == "Z":
                                dir_vec = (0, 0, 1)
                                color = "blue"
                                name = f"bc_{bc_cfg['block']}_Z"
                            else:
                                continue

                            glyphs = create_bc_glyphs(bc_mesh, dir_vec, size=bc_size)
                            if glyphs is not None:
                                plotter.add_mesh(glyphs, color=color, show_edges=False,
                                                 opacity=opacities["bc"], reset_camera=False, name=name)
            else:
                for bc_cfg in bc_configs:
                    for constr in bc_cfg["constraints"]:
                        try:
                            plotter.remove_actor(f"bc_{bc_cfg['block']}_{constr}")
                        except Exception:
                            pass

            if show_bc and opacities["back_bc"] > 0.0:
                bc_mesh = find_bc_block(init_data, back_support_bc_config["block"])
                if bc_mesh is not None:
                    for constr in back_support_bc_config["constraints"]:
                        if constr == "X":
                            dir_vec = (1, 0, 0)
                            color = "red"
                            name = f"bc_{back_support_bc_config['block']}_X"
                        elif constr == "Y":
                            dir_vec = (0, 1, 0)
                            color = "green"
                            name = f"bc_{back_support_bc_config['block']}_Y"
                        elif constr == "Z":
                            dir_vec = (0, 0, 1)
                            color = "blue"
                            name = f"bc_{back_support_bc_config['block']}_Z"
                        else:
                            continue
                        glyphs = create_bc_glyphs(bc_mesh, dir_vec, size=bc_size)
                        if glyphs is not None:
                            plotter.add_mesh(glyphs, color=color, show_edges=False,
                                             opacity=opacities["back_bc"], reset_camera=False, name=name)
            else:
                for constr in back_support_bc_config["constraints"]:
                    try:
                        plotter.remove_actor(f"bc_{back_support_bc_config['block']}_{constr}")
                    except Exception:
                        pass

            if annotation:
                actor = plotter.add_text(annotation, position=(0.5, 0.90), font_size=30, color="black",
                                          name="cinematic_annotation", viewport=True)
                actor.GetTextProperty().SetJustificationToCentered()
            else:
                try:
                    plotter.remove_actor("cinematic_annotation")
                except Exception:
                    pass

            plotter.write_frame()

        try:
            plotter.remove_actor("cinematic_annotation")
        except Exception:
            pass

        _sym_meshes = []
        if clean_concrete is not None:
            _sym_meshes.append(("sym_concrete", clean_concrete.reflect((0, 0, 1), point=(0, 0, 0)),
                                dict(scalars=mapped_damage_var, cmap="coolwarm", clim=[0.0, 1.0],
                                     show_edges=True, edge_color="black", scalar_bar_args=sargs,
                                     show_scalar_bar=False)))
        if clean_mortar is not None:
            _sym_meshes.append(("sym_mortar", clean_mortar.reflect((0, 0, 1), point=(0, 0, 0)),
                                dict(color="#3CB371", show_edges=True, edge_color="black")))
        if clean_plate is not None:
            _sym_meshes.append(("sym_plate", clean_plate.reflect((0, 0, 1), point=(0, 0, 0)),
                                dict(color="silver", show_edges=True, edge_color="black")))
        if clean_anchor is not None:
            sym_anchor = clean_anchor.reflect((0, 0, 1), point=(0, 0, 0))
            if args.steel_vm and "S" in sym_anchor.cell_data:
                s = sym_anchor.cell_data["S"]
                if s.shape[1] >= 6:
                    vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2 + 6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)))
                else:
                    vm = np.zeros(s.shape[0])
                sym_anchor.cell_data["von_Mises"] = vm
                _sym_meshes.append(("sym_anchor", sym_anchor,
                                    dict(scalars="von_Mises", cmap="copper_r", show_edges=True, edge_color="black", preference="cell", clim=[0.0, 1080.0], scalar_bar_args=sargs_vm, show_scalar_bar=False)))
            else:
                _sym_meshes.append(("sym_anchor", sym_anchor,
                                    dict(color="silver", show_edges=True, edge_color="black")))
        if clean_washer is not None:
            _sym_meshes.append(("sym_washer", clean_washer.reflect((0, 0, 1), point=(0, 0, 0)),
                                dict(color="darkgrey", show_edges=True, edge_color="black")))
        if clean_nut is not None:
            _sym_meshes.append(("sym_nut", clean_nut.reflect((0, 0, 1), point=(0, 0, 0)),
                                dict(color="darkgrey", show_edges=True, edge_color="black")))

        _sym_fade_in  = 30
        _sym_hold     = 15
        _sym_fade_out = 30
        _sym_total    = _sym_fade_in + _sym_hold + _sym_fade_out

        plotter.camera.position   = default_pos
        plotter.camera.focal_point = default_focal
        plotter.camera.up = (0.0, 1.0, 0.0)

        for _s in range(_sym_total):
            if _s < _sym_fade_in:
                progress = _s / (_sym_fade_in - 1)
                _alpha = 0.5 * (1.0 - math.cos(progress * math.pi)) * 0.9999
            elif _s < _sym_fade_in + _sym_hold:
                _alpha = 0.9999
            else:
                progress = (_s - _sym_fade_in - _sym_hold) / (_sym_fade_out - 1)
                _alpha = 0.5 * (1.0 + math.cos(progress * math.pi)) * 0.9999
            _alpha = float(np.clip(_alpha, 0.0, 0.9999))

            for _sym_name, _sym_mesh, _sym_kwargs in _sym_meshes:
                plotter.add_mesh(_sym_mesh, opacity=_alpha, reset_camera=False,
                                 name=_sym_name, **_sym_kwargs)

            _ann = plotter.add_text("Z-symmetry: only half the specimen modeled",
                                     position=(0.5, 0.90), font_size=30, color="black",
                                     name="sym_annotation", viewport=True)
            _ann.GetTextProperty().SetJustificationToCentered()

            plotter.write_frame()

        for _sym_name, _, _ in _sym_meshes:
            try:
                plotter.remove_actor(_sym_name)
            except Exception:
                pass
        try:
            plotter.remove_actor("sym_annotation")
        except Exception:
            pass

        if clean_concrete is not None:
            plotter.add_mesh(clean_concrete, scalars=mapped_damage_var, cmap="coolwarm", clim=[0.0, 1.0],
                             opacity=0.9999, show_edges=True, edge_color="black",
                             scalar_bar_args=sargs, show_scalar_bar=True, reset_camera=False, name="concrete_mesh")

    time_steps = np.linspace(t_min_orig, t_max, 200)

    first_frame = True
    all_time_max_stress = 0.0

    for i, t in enumerate(time_steps):
        idx_after = np.searchsorted(available_times, t)
        idx_before = max(0, idx_after - 1)
        if idx_after >= len(available_times):
            idx_after = len(available_times) - 1

        t_before = available_times[idx_before]
        t_after = available_times[idx_after]
        weight = (t - t_before) / (t_after - t_before) if t_after != t_before else 0.0

        data_before = get_raw_data_at_time(t_before, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)
        data_after = get_raw_data_at_time(t_after, reader, t_min_orig, t_max_orig, time_fraction, warp_factor)

        concrete_a = try_get_block(data_before, block_names["concrete"])
        concrete_b = try_get_block(data_after, block_names["concrete"])
        mortar_a = try_get_block(data_before, block_names["mortar"])
        mortar_b = try_get_block(data_after, block_names["mortar"])
        
        anchor_a = try_get_block(data_before, block_names["anchor"])
        anchor_b = try_get_block(data_after, block_names["anchor"])
        plate_a = try_get_block(data_before, block_names["plate"])
        plate_b = try_get_block(data_after, block_names["plate"])
        washer_a = try_get_block(data_before, block_names["washer"])
        washer_b = try_get_block(data_after, block_names["washer"])
        nut_a = try_get_block(data_before, block_names["nut"])
        nut_b = try_get_block(data_after, block_names["nut"])
        
        load_nodes_a = try_get_block(data_before, block_names["load_nodes"])
        load_nodes_b = try_get_block(data_after, block_names["load_nodes"])
        if not hasattr(plotter, '_prealloc_load_nodes_w') and load_nodes_a:
            plotter._prealloc_load_nodes_w = load_nodes_a.copy(deep=True)
        load_nodes_w = interpolate_mesh(load_nodes_a, load_nodes_b, weight, getattr(plotter, '_prealloc_load_nodes_w', None)) if load_nodes_a else None
        if load_nodes_w and disp_var in load_nodes_w.point_data:
            load_nodes_w.points += load_nodes_w.point_data[disp_var] * warp_factor

        if not hasattr(plotter, '_prealloc_concrete_w') and concrete_a:
            plotter._prealloc_concrete_w = concrete_a.copy(deep=True)
        if not hasattr(plotter, '_prealloc_mortar_w') and mortar_a:
            plotter._prealloc_mortar_w = mortar_a.copy(deep=True)
        if not hasattr(plotter, '_prealloc_anchor_w') and anchor_a:
            plotter._prealloc_anchor_w = anchor_a.copy(deep=True)
        if not hasattr(plotter, '_prealloc_plate_w') and plate_a:
            plotter._prealloc_plate_w = plate_a.copy(deep=True)
        if not hasattr(plotter, '_prealloc_washer_w') and washer_a:
            plotter._prealloc_washer_w = washer_a.copy(deep=True)
        if not hasattr(plotter, '_prealloc_nut_w') and nut_a:
            plotter._prealloc_nut_w = nut_a.copy(deep=True)

        concrete_w = interpolate_mesh(concrete_a, concrete_b, weight, getattr(plotter, '_prealloc_concrete_w', None)) if concrete_a else None
        mortar_w = interpolate_mesh(mortar_a, mortar_b, weight, getattr(plotter, '_prealloc_mortar_w', None)) if mortar_a else None
        anchor_w = interpolate_mesh(anchor_a, anchor_b, weight, getattr(plotter, '_prealloc_anchor_w', None)) if anchor_a else None
        plate_w = interpolate_mesh(plate_a, plate_b, weight, getattr(plotter, '_prealloc_plate_w', None)) if plate_a else None
        washer_w = interpolate_mesh(washer_a, washer_b, weight, getattr(plotter, '_prealloc_washer_w', None)) if washer_a else None
        nut_w = interpolate_mesh(nut_a, nut_b, weight, getattr(plotter, '_prealloc_nut_w', None)) if nut_a else None

        if concrete_w and disp_var in concrete_w.point_data:
            concrete_w.points += concrete_w.point_data[disp_var] * warp_factor
        if mortar_w and disp_var in mortar_w.point_data:
            mortar_w.points += mortar_w.point_data[disp_var] * warp_factor
        if anchor_w and disp_var in anchor_w.point_data:
            anchor_w.points += anchor_w.point_data[disp_var] * warp_factor
        if plate_w and disp_var in plate_w.point_data:
            plate_w.points += plate_w.point_data[disp_var] * warp_factor
        if washer_w and disp_var in washer_w.point_data:
            washer_w.points += washer_w.point_data[disp_var] * warp_factor
        if nut_w and disp_var in nut_w.point_data:
            nut_w.points += nut_w.point_data[disp_var] * warp_factor

        clean_concrete_w = get_clean_quad_surface(concrete_w) if concrete_w else None
        clean_mortar_w = get_clean_quad_surface(mortar_w) if mortar_w else None
        clean_anchor_w = get_clean_quad_surface(anchor_w) if anchor_w else None
        clean_plate_w = get_clean_quad_surface(plate_w) if plate_w else None
        clean_washer_w = get_clean_quad_surface(washer_w) if washer_w else None
        clean_nut_w = get_clean_quad_surface(nut_w) if nut_w else None

        if clean_concrete_w:
            plotter.add_mesh(clean_concrete_w, scalars=mapped_damage_var, cmap="coolwarm", clim=[0.0, 1.0], show_edges=True, edge_color="black", scalar_bar_args=sargs, opacity=0.9999, reset_camera=False, name="concrete_mesh")
        if clean_mortar_w:
            plotter.add_mesh(clean_mortar_w, color="#3CB371", show_edges=True, edge_color="black", opacity=0.9999, reset_camera=False, name="mortar_mesh")
            
        if clean_plate_w:
            plotter.add_mesh(clean_plate_w, color="silver", show_edges=True, edge_color="black", opacity=0.9999, reset_camera=False, name="plate_mesh")
        if clean_washer_w:
            plotter.add_mesh(clean_washer_w, color="darkgrey", show_edges=True, edge_color="black", opacity=0.9999, reset_camera=False, name="washer_mesh")
        if clean_nut_w:
            plotter.add_mesh(clean_nut_w, color="darkgrey", show_edges=True, edge_color="black", opacity=0.9999, reset_camera=False, name="nut_mesh")

        if clean_anchor_w:
            if args.steel_vm and "S" in clean_anchor_w.cell_data:
                s = clean_anchor_w.cell_data["S"]
                if s.shape[1] >= 6:
                    vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2 + 6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)))
                else:
                    vm = np.zeros(s.shape[0])
                clean_anchor_w.cell_data["von_Mises"] = vm
                plotter.add_mesh(clean_anchor_w, scalars="von_Mises", cmap="copper_r", show_edges=True, edge_color="black", opacity=0.9999, reset_camera=False, name="anchor_mesh", preference="cell", clim=[0.0, 1080.0], scalar_bar_args=sargs_vm)
                
                if "S" in anchor_w.cell_data and anchor_w.cell_data["S"].shape[1] >= 6:
                    s_vol = anchor_w.cell_data["S"]
                    vm_vol = np.sqrt(0.5 * ((s_vol[:, 0] - s_vol[:, 1])**2 + (s_vol[:, 1] - s_vol[:, 2])**2 + (s_vol[:, 2] - s_vol[:, 0])**2 + 6.0 * (s_vol[:, 3]**2 + s_vol[:, 4]**2 + s_vol[:, 5]**2)))
                    max_idx = np.argmax(vm_vol)
                    max_loc = anchor_w.cell_centers().points[max_idx]
                    max_val = vm_vol[max_idx]
                    all_time_max_stress = max(all_time_max_stress, max_val)
                    
                    if max_val > 0.1:
                        try:
                            plotter.remove_actor("max_stress_marker")
                            plotter.remove_actor("max_stress_line")
                        except Exception:
                            pass
                        label_text = f"Current Max: {max_val:.0f} MPa\nAll-time Peak: {all_time_max_stress:.0f} MPa"
                        plotter.add_point_labels([max_loc], [label_text], point_size=10, point_color="red", text_color="red", name="max_stress_label", font_size=30, shape_color="white", shape_opacity=0.6, always_visible=True)
            else:
                plotter.add_mesh(clean_anchor_w, color="silver", show_edges=True, edge_color="black", opacity=0.9999, reset_camera=False, name="anchor_mesh")

        if show_bc:
            for bc_cfg in bc_configs + [back_support_bc_config]:
                try:
                    bc_a = find_bc_block(data_before, bc_cfg["block"])
                    bc_b = find_bc_block(data_after, bc_cfg["block"])
                    if bc_a is not None and bc_b is not None:
                        if not hasattr(plotter, '_prealloc_bc_w'):
                            plotter._prealloc_bc_w = {}
                        if bc_cfg["block"] not in plotter._prealloc_bc_w:
                            plotter._prealloc_bc_w[bc_cfg["block"]] = bc_a.copy(deep=True)

                        bc_w = interpolate_mesh(bc_a, bc_b, weight, plotter._prealloc_bc_w[bc_cfg["block"]])
                        if disp_var in bc_w.point_data:
                            bc_w.points += bc_w.point_data[disp_var] * warp_factor

                        for constr in bc_cfg["constraints"]:
                            if constr == "X":
                                dir_vec = (1, 0, 0)
                                color = "red"
                                name = f"bc_{bc_cfg['block']}_X"
                            elif constr == "Y":
                                dir_vec = (0, 1, 0)
                                color = "green"
                                name = f"bc_{bc_cfg['block']}_Y"
                            elif constr == "Z":
                                dir_vec = (0, 0, 1)
                                color = "blue"
                                name = f"bc_{bc_cfg['block']}_Z"
                            else:
                                continue

                            glyphs = create_bc_glyphs(bc_w, dir_vec, size=bc_size)
                            if glyphs is not None:
                                plotter.add_mesh(glyphs, color=color, show_edges=False,
                                                 reset_camera=False, name=name)
                except Exception:
                    pass

        if load_nodes_w is not None and rf_var in load_nodes_w.point_data:
            rf_vectors = np.array(load_nodes_w.point_data[rf_var])
            if rf_vectors.ndim == 1:
                rf_vectors = rf_vectors.reshape(-1, 3)

            rf_x_vectors = np.zeros_like(rf_vectors)
            rf_x_vectors[:, 0] = rf_vectors[:, 0]

            rf_x_mag = np.abs(rf_vectors[:, 0])

            nset_center = np.mean(load_nodes_w.points, axis=0)
            nset_center[2] = 0.0

            summed_rf_x = np.sum(rf_x_vectors, axis=0).reshape(1, 3)
            current_frame_force = np.sum(rf_x_mag)

            if current_frame_force > 1e-5:
                single_point_mesh = pv.PolyData(nset_center.reshape(1, 3))
                single_point_mesh.point_data["rf_x_vectors"] = summed_rf_x

                rf_glyphs = single_point_mesh.glyph(
                    orient="rf_x_vectors", 
                    scale="rf_x_vectors", 
                    factor=glyph_scale, 
                    geom=force_arrow 
                )

                force_color_array = np.full(rf_glyphs.n_points, current_frame_force)

                plotter.add_mesh(rf_glyphs, scalars=force_color_array, cmap="jet", clim=[0.0, global_max_rf],
                                 show_scalar_bar=False, reset_camera=False, name="rf_glyphs")

        if has_csv and load_indices is not None:
            try:
                all_a = try_get_block(data_before, "ALL")
                all_b = try_get_block(data_after, "ALL")
                disp_before = all_a.point_data[disp_var][load_indices]
                disp_after = all_b.point_data[disp_var][load_indices]
                disp_w = disp_before + weight * (disp_after - disp_before)
                current_disp_val = np.abs(np.mean(disp_w[:, disp_component_idx]))
            except Exception:
                current_disp_val = 0.0

        active_chart = draw_dynamic_chart(t, plotter, active_chart, has_csv, csv_disp, csv_load, csv_time, t_max)

        if first_frame:
            cam_dist = clean_concrete.length * 0.8
            plotter.camera.position = (cam_dist, cam_dist, cam_dist)
            plotter.camera.focal_point = (0.0, 0.0, 0.0)
            plotter.camera.up = (0.0, 1.0, 0.0)
            first_frame = False
        else:
            plotter.camera.zoom(1.001)

        plotter.write_frame()

    if concrete_w:
        smooth_fracture = concrete_w.contour(isosurfaces=[damage_threshold], scalars=mapped_damage_var)
    else:
        smooth_fracture = pv.PolyData()
    mirrored_fracture = smooth_fracture.reflect((0, 0, 1), point=(0, 0, 0))

    plotter.add_mesh(smooth_fracture, scalars=mapped_damage_var, cmap="coolwarm", clim=[0.0, 1.0],
                     opacity=0.0, show_edges=False,
                     scalar_bar_args=sargs, reset_camera=False, name="fractured_mesh")
    plotter.add_mesh(mirrored_fracture, scalars=mapped_damage_var, cmap="coolwarm", clim=[0.0, 1.0],
                     opacity=0.0, show_edges=False,
                     scalar_bar_args=sargs, reset_camera=False, name="mirrored_mesh")

    fade_steps = 60
    for step in range(fade_steps + 1):
        alpha_clipped = step / fade_steps
        alpha_full = 1.0 - (0.8 * alpha_clipped)

        if "concrete_mesh" in plotter.actors:
            plotter.actors["concrete_mesh"].GetProperty().SetOpacity(alpha_full)
        if "fractured_mesh" in plotter.actors:
            plotter.actors["fractured_mesh"].GetProperty().SetOpacity(alpha_clipped)
        if "mirrored_mesh" in plotter.actors:
            plotter.actors["mirrored_mesh"].GetProperty().SetOpacity(alpha_clipped)

        if show_bc:
            all_bc_names = (
                [f"bc_{bc_cfg['block']}_{c}" for bc_cfg in bc_configs for c in bc_cfg["constraints"]]
                + [f"bc_{back_support_bc_config['block']}_{c}" for c in back_support_bc_config["constraints"]]
            )
            for name in all_bc_names:
                if name in plotter.actors:
                    plotter.actors[name].GetProperty().SetOpacity(alpha_full)

        active_chart = draw_dynamic_chart(t_max, plotter, active_chart, has_csv, csv_disp, csv_load, csv_time, t_max)
        plotter.write_frame()

    all_bc_names = (
        [f"bc_{bc_cfg['block']}_{c}" for bc_cfg in bc_configs for c in bc_cfg["constraints"]]
        + [f"bc_{back_support_bc_config['block']}_{c}" for c in back_support_bc_config["constraints"]]
    )
    for _name in all_bc_names:
        try:
            plotter.remove_actor(_name)
        except Exception:
            pass

    rotation_frames = 240
    total_target_azimuth = 360.0
    total_target_elevation = -25.0

    azimuth_angles = []
    elevation_angles = []

    for i in range(1, rotation_frames + 1):
        progress_prev = (i - 1) / rotation_frames
        progress_curr = i / rotation_frames

        entropy_prev = 0.5 * (1.0 - math.cos(progress_prev * math.pi))
        entropy_curr = 0.5 * (1.0 - math.cos(progress_curr * math.pi))

        azi_delta = total_target_azimuth * (entropy_curr - entropy_prev)
        ele_delta = total_target_elevation * (entropy_curr - entropy_prev)

        azimuth_angles.append(azi_delta)
        elevation_angles.append(ele_delta)

    for i, (azi_step, ele_step) in enumerate(zip(azimuth_angles, elevation_angles)):
        plotter.camera.azimuth += azi_step
        plotter.camera.elevation += ele_step

        active_chart = draw_dynamic_chart(t_max, plotter, active_chart, has_csv, csv_disp, csv_load, csv_time, t_max)
        plotter.write_frame()

    plotter.close()

if __name__ == "__main__":
    main()
