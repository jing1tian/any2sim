"""Build a Blender scene from the Scene IR (runs inside the `render` env, bpy 5.0).

The .blend is a compiled artifact: geometry, transforms, cameras and lights come from scene/scene.yaml; textures
from the asset files the IR references (or the appearance stage's albedo textures). The timeline mirrors the
video: frame f = video frame f, the source camera is keyed per frame, objects are keyed at their rest-state poses
and hidden while they move (their pose is unknown there).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def reset():
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_cycles(samples=128, width=1280, height=720, denoise=True):
    import bpy
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    selected_device = "CPU"
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == dev_type]
            if devs:
                selected_device = dev_type
                for d in prefs.devices:
                    d.use = d.type == dev_type
                break
        except TypeError:
            continue
    sc.cycles.device = "GPU" if selected_device != "CPU" else "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = denoise
    sc.render.resolution_x, sc.render.resolution_y = width, height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.display_settings.display_device = "sRGB"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    return selected_device


def import_obj(path, name):
    import bpy
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")   # keep coordinates as written
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if len(new) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
    ob = new[0]
    for slot in list(ob.data.materials):             # materials come from the Scene IR, not from the MTL
        if slot is not None:
            bpy.data.materials.remove(slot)
    ob.data.materials.clear()
    ob.name = name
    ob.data.name = name
    return ob


def load_image(path, colorspace="sRGB"):
    import bpy
    im = bpy.data.images.load(str(path), check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def material_principled(name, image=None, roughness=0.6, specular=0.0, color=(0.8, 0.8, 0.8, 1)):
    import bpy
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = "albedo"
        tex.image = image
        tex.interpolation = "Linear"
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = color
    return m


def material_emission(name, image, strength=1.0):
    import bpy
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    em.inputs["Strength"].default_value = strength
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def assign(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def set_matrix(ob, T):
    from mathutils import Matrix
    ob.matrix_world = Matrix(np.asarray(T).tolist())


def add_camera(name, K, W, H):
    import bpy
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cd.lens = float(K[0, 0]) * 36.0 / W
    cd.shift_x = -(float(K[0, 2]) - W / 2) / W
    cd.shift_y = (float(K[1, 2]) - H / 2) / W
    cd.clip_start, cd.clip_end = 0.01, 50.0
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = float(K[1, 1] / K[0, 0]) if abs(K[1, 1] / K[0, 0] - 1) > 1e-3 else 1.0
    return cam


def set_cv_pose(cam, T_world_cvcam):
    set_matrix(cam, np.asarray(T_world_cvcam) @ CV_TO_GL)


def look_at(eye, target, up=(0, 0, 1)):
    """T_world_cam (OpenCV optical frame) of a camera at eye looking at target."""
    eye, target, up = map(lambda x: np.asarray(x, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def set_camera_background(color):
    """Camera rays that miss everything see `color` (e.g. the backdrop's mean colour); lighting is unchanged."""
    import bpy
    nt = bpy.context.scene.world.node_tree
    bg = nt.nodes.get("Background")
    out = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
    lp = nt.nodes.new("ShaderNodeLightPath")
    cam_bg = nt.nodes.new("ShaderNodeBackground")
    cam_bg.inputs["Color"].default_value = (*color, 1)
    cam_bg.inputs["Strength"].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg.outputs["Background"], mix.inputs[1])
    nt.links.new(cam_bg.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def add_lights(lights):
    """lights: Scene IR light list: {type: area|sun|point, T_world_light | position+target, size, strength_w,
    color} and {type: world, color, strength}."""
    import bpy
    sc = bpy.context.scene
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    bg.inputs["Strength"].default_value = 0.0
    made = []
    for L in lights:
        if L["type"] == "world":
            bg.inputs["Color"].default_value = (*L.get("color", (1, 1, 1)), 1)
            bg.inputs["Strength"].default_value = float(L.get("strength", 0.3))
            continue
        ld = bpy.data.lights.new(L.get("name", L["type"]), L["type"].upper())
        if L["type"] == "area":
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = L.get("size", [1.0, 1.0])
        ld.energy = float(L.get("strength_w", 100.0))
        ld.color = L.get("color", (1, 1, 1))
        if hasattr(ld, "use_shadow"):
            ld.use_shadow = True
        ob = bpy.data.objects.new(L.get("name", L["type"]), ld)
        sc.collection.objects.link(ob)
        T = look_at(L["position"], L.get("target", [0, 0, 0]), up=(0, 1, 0))
        set_matrix(ob, T @ CV_TO_GL)                  # lights emit along their local -z, like cameras
        made.append(ob)
    return made


def material_from_spec(name, spec: dict, scene_dir: Path):
    """Scene IR material -> Blender material (principled | emission; constant or texture; optional metre tiling)."""
    import bpy
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    shader = spec.get("shader", "principled")
    if shader == "emission":
        sh = nt.nodes.new("ShaderNodeEmission")
        sh.inputs["Strength"].default_value = float(spec.get("emission_strength", 1.0))
        color_in = sh.inputs["Color"]
        nt.links.new(sh.outputs["Emission"], out.inputs["Surface"])
    else:
        sh = nt.nodes.new("ShaderNodeBsdfPrincipled")
        sh.inputs["Roughness"].default_value = float(spec.get("roughness", 0.6))
        sh.inputs["Metallic"].default_value = float(spec.get("metallic", 0.0))
        if "Specular IOR Level" in sh.inputs:
            sh.inputs["Specular IOR Level"].default_value = float(spec.get("specular", 0.0))
        color_in = sh.inputs["Base Color"]
        nt.links.new(sh.outputs["BSDF"], out.inputs["Surface"])
    tex = spec.get("base_color_texture")
    proc = spec.get("procedural")
    hsv = spec.get("hsv_shift")
    gr = spec.get("grime")
    if gr and float(gr.get("amount", 0)) > 0:
        # (render-only) stains and wear: two noise layers darken the albedo where they exceed a threshold
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        sc_ = float(gr.get("scale", 15.0))
        mp.inputs["Scale"].default_value = (sc_, sc_, sc_)
        mp.inputs["Location"].default_value = (float(gr.get("seed", 0.0)), 0.37 * float(gr.get("seed", 0.0)), 0.0)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Detail"].default_value = 8.0
        nz.inputs["Roughness"].default_value = 0.65
        nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.52
        ramp.color_ramp.elements[1].position = 0.75
        nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = float(gr["amount"])
        nt.links.new(ramp.outputs["Color"], mul.inputs[0])
        nt.links.new(mul.outputs["Value"], mix.inputs["Factor"])
        mix.inputs["B"].default_value = (*gr.get("color", [0.25, 0.22, 0.2]), 1)
        nt.links.new(mix.outputs["Result"], color_in)
        color_in = mix.inputs["A"]
    if hsv and any(abs(float(x)) > 1e-6 for x in hsv):
        hs = nt.nodes.new("ShaderNodeHueSaturation")       # (render-only) albedo variation of a synthesis variant
        hs.inputs["Hue"].default_value = (0.5 + float(hsv[0]) / 360.0) % 1.0
        hs.inputs["Saturation"].default_value = max(0.0, 1.0 + float(hsv[1]))
        hs.inputs["Value"].default_value = max(0.0, 1.0 + float(hsv[2]))
        nt.links.new(hs.outputs["Color"], color_in)
        color_in = hs.inputs["Color"]
    if proc:
        # (render-only) procedural albedo in object coordinates (metres), two colours through a ramp
        kinds = {"noise": "ShaderNodeTexNoise", "wave": "ShaderNodeTexWave", "checker": "ShaderNodeTexChecker",
                 "voronoi": "ShaderNodeTexVoronoi", "brick": "ShaderNodeTexBrick", "magic": "ShaderNodeTexMagic"}
        tn = nt.nodes.new(kinds[proc.get("type", "noise")])
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        s_ = float(proc.get("scale", 10.0))
        mp.inputs["Scale"].default_value = (s_, s_, s_)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], tn.inputs["Vector"])
        if "Detail" in tn.inputs and proc.get("detail") is not None:
            tn.inputs["Detail"].default_value = float(proc["detail"])
        cols = proc.get("colors") or [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]
        out_name = "Fac" if "Fac" in tn.outputs else ("Distance" if "Distance" in tn.outputs else "Color")
        if proc.get("type") == "checker":
            tn.inputs["Color1"].default_value = (*cols[0], 1)
            tn.inputs["Color2"].default_value = (*cols[1], 1)
            tn.inputs["Scale"].default_value = 1.0
            nt.links.new(tn.outputs["Color"], color_in)
        else:
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = (*cols[0], 1)
            ramp.color_ramp.elements[1].color = (*cols[1], 1)
            nt.links.new(tn.outputs[out_name], ramp.inputs["Fac"])
            nt.links.new(ramp.outputs["Color"], color_in)
        tex = None
    if tex:
        im = load_image(scene_dir / tex, "sRGB")
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.name = "albedo"
        tn.image = im
        tn.interpolation = "Linear"
        if spec.get("tile_m"):
            tn.extension = "REPEAT"
            tc = nt.nodes.new("ShaderNodeTexCoord")
            mp = nt.nodes.new("ShaderNodeMapping")
            s_ = 1.0 / float(spec["tile_m"])
            mp.inputs["Scale"].default_value = (s_, s_, 1.0)
            nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
            nt.links.new(mp.outputs["Vector"], tn.inputs["Vector"])
        nt.links.new(tn.outputs["Color"], color_in)
    elif not proc:
        color_in.default_value = (*spec.get("base_color", [0.8, 0.8, 0.8]), 1.0)
    if shader != "emission":
        # Scalar maps are data, not sRGB colours. Share the albedo's UV convention.
        for key, socket in (("roughness_texture", "Roughness"), ("normal_texture", "Normal")):
            if not spec.get(key):
                continue
            tn = nt.nodes.new("ShaderNodeTexImage")
            tn.name = key
            tn.image = load_image(scene_dir / spec[key], "Non-Color")
            if spec.get("tile_m"):
                tc = nt.nodes.new("ShaderNodeTexCoord")
                mp = nt.nodes.new("ShaderNodeMapping")
                s_ = 1.0 / float(spec["tile_m"])
                mp.inputs["Scale"].default_value = (s_, s_, 1.0)
                nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
                nt.links.new(mp.outputs["Vector"], tn.inputs["Vector"])
            value = tn.outputs["Color"]
            if key == "roughness_texture" and "roughness_map_scale" in spec:
                mul = nt.nodes.new("ShaderNodeMath")
                mul.operation = "MULTIPLY"
                mul.use_clamp = True
                mul.inputs[1].default_value = float(spec["roughness_map_scale"])
                nt.links.new(value, mul.inputs[0])
                value = mul.outputs[0]
            if key == "normal_texture":
                normal = nt.nodes.new("ShaderNodeNormalMap")
                normal.inputs["Strength"].default_value = float(spec.get("normal_strength", 1.0))
                nt.links.new(value, normal.inputs["Color"])
                value = normal.outputs["Normal"]
            nt.links.new(value, sh.inputs[socket])
    return m


def add_dressing(dressing: dict | None) -> dict:
    """Render-only clutter of a synthesis variant: loose cables (bevelled curves) and shadow casters (rectangles the
    camera cannot see that only cast shadows)."""
    import bpy
    out = {"cables": [], "casters": []}
    for k, c in enumerate((dressing or {}).get("cables") or []):
        cu = bpy.data.curves.new(f"dressing_cable_{k}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = float(c["radius"])
        cu.bevel_resolution = 3
        sp = cu.splines.new("NURBS")
        sp.points.add(len(c["points"]) - 1)
        for i, q in enumerate(c["points"]):
            sp.points[i].co = (*q, 1.0)
        sp.use_endpoint_u = True
        sp.order_u = 4
        ob = bpy.data.objects.new(f"dressing_cable_{k}", cu)
        bpy.context.scene.collection.objects.link(ob)
        cu.materials.append(material_principled(f"dressing_cable_{k}", roughness=0.5, specular=0.35,
                                                color=(*c.get("color", [0.01] * 3), 1)))
        out["cables"].append(ob)
    for k, sc in enumerate((dressing or {}).get("shadow_casters") or []):
        me = bpy.data.meshes.new(f"dressing_caster_{k}")
        a, b_ = sc["size"]
        me.from_pydata([(-a / 2, -b_ / 2, 0), (a / 2, -b_ / 2, 0), (a / 2, b_ / 2, 0), (-a / 2, b_ / 2, 0)], [], [(0, 1, 2, 3)])
        ob = bpy.data.objects.new(f"dressing_caster_{k}", me)
        bpy.context.scene.collection.objects.link(ob)
        n = np.asarray(sc["normal"], float)
        T = look_at(np.asarray(sc["center"]) + n, sc["center"], up=(0, 0, 1) if abs(n[2]) < 0.95 else (0, 1, 0))
        Rz = np.eye(4)
        Rz[:2, :2] = [[np.cos(sc.get("yaw", 0)), -np.sin(sc.get("yaw", 0))], [np.sin(sc.get("yaw", 0)), np.cos(sc.get("yaw", 0))]]
        set_matrix(ob, T @ Rz)
        ob.visible_camera = False                       # shadows only
        ob.visible_glossy = False
        ob.visible_diffuse = False
        out["casters"].append(ob)
    return out


def build(project: Path, scene: dict, cams: dict | None = None, lights: list | None = None, samples: int = 128,
          entities: list | None = None):
    """Create the Scene IR in the current (empty) Blender session. Returns (objects by name, source camera, device)."""
    import bpy
    from ..scene_ir import camera_poses, pose_at
    sdir = project / "scene"
    c = scene["cameras"]["source_cam"]
    W, H = c["width"], c["height"]
    dev = setup_cycles(samples, W, H)
    obs = {}
    for name, e in scene["entities"].items():
        if entities is not None and name not in entities:
            continue
        ob = import_obj(sdir / e["geometry"]["visual"]["obj_uri"], name)
        mat = e.get("material") or {}
        assign(ob, material_from_spec(name, mat, sdir))
        if mat.get("shader") == "emission":
            ob.visible_shadow = False
        set_matrix(ob, pose_at(e, None))
        obs[name] = ob
    cam = add_camera("source_cam", np.asarray(c["K"]), W, H)
    set_cv_pose(cam, camera_poses(sdir, scene)[0])
    bpy.context.scene.camera = cam
    add_lights(lights if lights is not None else scene.get("lights") or [])
    return obs, cam, dev


def import_robot_link_visual(path: Path, name: str):
    """GLB (y-up, standard glTF) or STL/OBJ in the link frame -> list of new mesh objects."""
    import bpy
    before = set(bpy.data.objects)
    if path.suffix.lower() in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif path.suffix.lower() == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        if o.type == "MESH" and not o.name.startswith(name):
            o.name = f"{name}:{o.name}"
    return new


def add_robots(project: Path, scene: dict) -> dict:
    """Every Scene IR robot as one empty per link (posed by FK each frame) with its render visuals parented:
    GLB visuals from the bundle (robot.yaml visuals) or the URDF visual meshes with robot.yaml appearance."""
    import bpy
    import fnmatch
    import mathutils
    from ..scene_ir import robot_bundle
    out = {}
    for rid, r in (scene.get("robots") or {}).items():
        b = robot_bundle(project / "scene", scene, rid)
        vis = b.profile.get("visuals") or {}
        app = b.profile.get("appearance") or {}
        links = {}
        mats = {}
        for ln, L in b.model["links"].items():
            e = bpy.data.objects.new(f"{rid}/{ln}", None)
            e.empty_display_size = 0.01
            bpy.context.scene.collection.objects.link(e)
            links[ln] = e
            v = vis.get(ln) or {}
            if v.get("hidden"):
                continue
            parts = []
            if v.get("obj"):                               # textured OBJ (robot_texture: albedo from real images)
                parts = [o for o in import_robot_link_visual(b.root / v["obj"], f"{rid}/{ln}") if o.type == "MESH"]
                classes = b.profile.get("materials") or {}        # robot_appearance material classes
                for o in parts:
                    for slot in o.material_slots:
                        bsdf = slot.material.node_tree.nodes.get("Principled BSDF") if slot.material and slot.material.use_nodes else None
                        if bsdf is None:
                            continue
                        mt = classes.get(slot.material.name.split(".")[0]) or v.get("material") or {}
                        if mt.get("base_color") is not None and not bsdf.inputs["Base Color"].is_linked:
                            bsdf.inputs["Base Color"].default_value = (*mt["base_color"], 1)
                        bsdf.inputs["Roughness"].default_value = float(mt.get("roughness", 0.4))
                        if "Specular IOR Level" in bsdf.inputs:
                            bsdf.inputs["Specular IOR Level"].default_value = float(mt.get("specular", 0.5))
                        bsdf.inputs["Metallic"].default_value = float(mt.get("metallic", 0.0))
            elif v.get("glb"):
                parts = [o for o in import_robot_link_visual(b.root / v["glb"], f"{rid}/{ln}") if o.parent is None]
                if v.get("tint"):                                   # robot_appearance: albedo scale from real images
                    for o in [p_ for p_ in parts] + [c for p_ in parts for c in p_.children_recursive]:
                        if o.type == "MESH":
                            for slot in o.material_slots:
                                tint_material(slot.material, v["tint"])
            else:
                spec = next((a for pat, a in app.items() if fnmatch.fnmatch(ln, pat)), None)
                for k, g in enumerate(L["visuals"]):
                    if g["type"] != "mesh":
                        import bmesh
                        me = bpy.data.meshes.new(f"{rid}/{ln}_prim{k}")
                        bm = bmesh.new()
                        if g["type"] == "box":
                            bmesh.ops.create_cube(bm, size=1.0)
                            bmesh.ops.scale(bm, vec=g["size"], verts=bm.verts)
                        elif g["type"] == "cylinder":
                            bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=g["radius"],
                                                  radius2=g["radius"], depth=g["length"])
                        else:
                            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=g["radius"])
                        bm.to_mesh(me)
                        bm.free()
                        ob = bpy.data.objects.new(f"{rid}/{ln}_prim{k}", me)
                        bpy.context.scene.collection.objects.link(ob)
                        new = [ob]
                    else:
                        new = [o for o in import_robot_link_visual(Path(g["uri"]), f"{rid}/{ln}") if o.type == "MESH"]
                    key = (spec or {}).get("name") or ln
                    if key not in mats:
                        sp = spec or {"base_color": [c ** 2.2 for c in (g.get("rgba") or [0.8, 0.8, 0.8])[:3]], "roughness": 0.5}
                        m = material_principled(f"{rid}/{key}", roughness=float(sp.get("roughness", 0.5)),
                                                specular=float(sp.get("specular", 0.5)),
                                                color=(*sp.get("base_color", [0.8, 0.8, 0.8]), 1))
                        bsdf = m.node_tree.nodes.get("Principled BSDF")
                        bsdf.inputs["Metallic"].default_value = float(sp.get("metallic", 0.0))
                        mats[key] = m
                    for o in new:
                        if o.data.materials:
                            o.data.materials.clear()
                        o.data.materials.append(mats[key])
                        o.matrix_world = mathutils.Matrix(np.asarray(g["T"]).tolist()) @ o.matrix_world
                    parts += new
            for o in parts:
                o.parent = e
                o.matrix_parent_inverse.identity()
        cables = add_cables(f"{rid}/cable", b.profile.get("cables") or {})
        out[rid] = {"bundle": b, "links": links, "T_world_base": np.asarray(r["T_world_base"], float),
                    "q0": {**b.init_q(), **(r.get("initial_q") or {})}, "cables": cables}
    return out


CABLE_POINTS = 48

_TINTED = set()


def tint_material(m, tint):
    """Multiply a material's base colour (constant or texture) by a linear RGB tint (once per material)."""
    if m is None or not m.use_nodes or m.name in _TINTED:
        return
    nt = m.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    inp = bsdf.inputs["Base Color"]
    if inp.is_linked:
        src = inp.links[0].from_socket
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(src, mix.inputs["A"])
        mix.inputs["B"].default_value = (*tint, 1)
        nt.links.new(mix.outputs["Result"], inp)
    else:
        c = inp.default_value
        inp.default_value = (c[0] * tint[0], c[1] * tint[1], c[2] * tint[2], c[3])
    _TINTED.add(m.name)


def add_cables(prefix: str, specs: dict) -> dict:
    """One bevelled poly curve per bundle cable (robot/cables.py); shapes are set by pose_robots."""
    import bpy
    out = {}
    for name, spec in specs.items():
        cu = bpy.data.curves.new(f"{prefix}:{name}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = float(spec.get("radius_m", 0.0035))
        cu.bevel_resolution = 3
        cu.use_fill_caps = True
        sp = cu.splines.new("POLY")
        sp.points.add(CABLE_POINTS - 1)
        ob = bpy.data.objects.new(f"{prefix}:{name}", cu)
        bpy.context.scene.collection.objects.link(ob)
        mt = spec.get("material") or {}
        m = material_principled(f"{prefix}:{name}", roughness=float(mt.get("roughness", 0.5)),
                                specular=float(mt.get("specular", 0.35)), color=(*mt.get("base_color", [0.015] * 3), 1))
        cu.materials.append(m)
        out[name] = {"spec": spec, "obj": ob}
    return out


def pose_cables(R: dict, link_T: dict, frame: int | None = None):
    from ..robot import cables as CB
    for c in (R.get("cables") or {}).values():
        P = CB.centerline(c["spec"], link_T, n=CABLE_POINTS)
        pts = c["obj"].data.splines[0].points
        for i, p in enumerate(P):
            pts[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
            if frame is not None:
                pts[i].keyframe_insert("co", frame=frame)


def pose_robots(robots: dict, q_by_robot: dict | None = None):
    """Place every link empty at its FK pose (q_by_robot: {rid: {joint: value}}; default the initial q)."""
    from ..robot import urdf as U
    for rid, R in robots.items():
        q = dict(R["q0"])
        q.update((q_by_robot or {}).get(rid) or {})
        T = U.fk(R["bundle"].model, q, R["T_world_base"])
        for ln, e in R["links"].items():
            set_matrix(e, T[ln])
        pose_cables(R, T)


def _linear_animation(data):
    """Set sampled motion to linear interpolation, including Blender 4.4+ action slots."""
    anim = data.animation_data
    if not anim or not anim.action:
        return
    action = anim.action
    curves = list(getattr(action, "fcurves", []))
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    for curve in curves:
        for point in curve.keyframe_points:
            point.interpolation = "CONSTANT" if curve.data_path == "hide_render" else "LINEAR"


def _keyframe_pose(ob, frame, previous):
    """Key a pose without quaternion sign flips across adjacent samples."""
    q = ob.rotation_quaternion.copy()
    if ob in previous and q.dot(previous[ob]) < 0:
        q.negate()
    ob.rotation_quaternion = q
    previous[ob] = q.copy()
    ob.keyframe_insert("location", frame=frame)
    ob.keyframe_insert("rotation_quaternion", frame=frame)


def keyframe_motion(scene: dict, obs: dict, robots: dict, motion: dict, n_frames: int, rid: str | None):
    """Key achieved poses with continuous rotations for shutter-time interpolation."""
    import bpy
    from ..scene_ir import motion_at
    traj = {k for k, e in scene["entities"].items() if (e.get("trajectory") or {}).get("motion")}
    animated = [obs[n] for n in traj if n in obs]
    animated += [e for R in (robots or {}).values() for e in R["links"].values()]
    for ob in animated:
        # Do this before assigning matrices: Euler branch choices must not enter the animation.
        ob.rotation_mode = "QUATERNION"
    previous = {}
    for f in range(n_frames):
        q, objs = motion_at(motion, f)
        if robots:
            pose_robots(robots, {rid or next(iter(robots)): q})
            for R in robots.values():
                for e in R["links"].values():
                    _keyframe_pose(e, f, previous)
                for c in (R.get("cables") or {}).values():
                    for pt in c["obj"].data.splines[0].points:
                        pt.keyframe_insert("co", frame=f)
        for n in traj:
            if n in obs and n in objs:
                set_matrix(obs[n], objs[n])
                obs[n].hide_render = False
                _keyframe_pose(obs[n], f, previous)
                obs[n].keyframe_insert("hide_render", frame=f)
    for ob in animated:
        _linear_animation(ob)
    for R in (robots or {}).values():
        for c in (R.get("cables") or {}).values():
            _linear_animation(c["obj"].data)
    bpy.context.scene.frame_set(0)


def pose_for_frame(scene: dict, obs: dict, cam, T_world_cam, frame: int):
    """Camera at the frame's pose; rigid entities at their rest pose, hidden when moving (pose unknown)."""
    from ..scene_ir import pose_at
    set_cv_pose(cam, T_world_cam)
    for name, ob in obs.items():
        T = pose_at(scene["entities"][name], frame)
        ob.hide_render = T is None
        if T is not None:
            set_matrix(ob, T)


def keyframe_timeline(scene: dict, obs: dict, cam, poses):
    """Timeline = source video frames: camera keyed per frame, rigid entities keyed per frame (hidden while moving)."""
    import bpy
    sc = bpy.context.scene
    n = len(poses)
    sc.frame_start, sc.frame_end = 0, n - 1
    rigid = {k: v for k, v in obs.items() if scene["entities"][k].get("capability") == "rigid"
             and not (scene["entities"][k].get("trajectory") or {}).get("motion")}   # trajectories: keyframe_motion
    for f in range(n):
        pose_for_frame(scene, rigid, cam, poses[f], f)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        for ob in rigid.values():
            ob.keyframe_insert("location", frame=f)
            ob.keyframe_insert("rotation_euler", frame=f)
            ob.keyframe_insert("hide_render", frame=f)
    for ob in [cam] + list(rigid.values()):
        if ob.animation_data and ob.animation_data.action:
            for fc in getattr(ob.animation_data.action, "fcurves", []):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
    sc.frame_set(0)


def render_to(path):
    import bpy
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return str(path)


def bake_image(ob, name, w, h, kind, out_path, margin=0):
    """Bake one map for one object into its UV layout. kind: E (diffuse irradiance, white albedo),
    POS (object-space position), NRM (object-space normal), COV (UV coverage)."""
    import bpy
    img = bpy.data.images.new(name, w, h, float_buffer=True, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    mat = ob.data.materials[0]
    nt = mat.node_tree
    orig_links = []
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    tmp = []
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    tmp.append(node)
    nt.nodes.active = node
    bsdf = nt.nodes.get("Principled BSDF")
    if kind in ("POS", "NRM", "COV"):
        orig_links = [(l.from_socket, l.to_socket) for l in out.inputs["Surface"].links]
        em = nt.nodes.new("ShaderNodeEmission")
        tmp.append(em)
        if kind == "POS":
            tc = nt.nodes.new("ShaderNodeTexCoord")
            tmp.append(tc)
            nt.links.new(tc.outputs["Object"], em.inputs["Color"])
        elif kind == "NRM":
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            vt = nt.nodes.new("ShaderNodeVectorTransform")
            vt.vector_type, vt.convert_from, vt.convert_to = "NORMAL", "WORLD", "OBJECT"
            tmp += [geo, vt]
            nt.links.new(geo.outputs["Normal"], vt.inputs["Vector"])
            nt.links.new(vt.outputs["Vector"], em.inputs["Color"])
        else:
            em.inputs["Color"].default_value = (1, 1, 1, 1)
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        btype, passes = "EMIT", set()
    else:
        btype, passes = "DIFFUSE", {"DIRECT", "INDIRECT"}
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.context.scene.cycles.samples = 64 if kind == "E" else 1
    if btype == "EMIT":
        bpy.ops.object.bake(type="EMIT", margin=margin, use_clear=True)
    else:
        bpy.ops.object.bake(type="DIFFUSE", pass_filter=passes, margin=margin, use_clear=True)
    img.filepath_raw = str(out_path)
    img.file_format = "OPEN_EXR"
    img.save()
    for n in tmp:
        nt.nodes.remove(n)
    if orig_links:
        for a, b in orig_links:
            nt.links.new(a, b)
    if bsdf is not None and kind in ("POS", "NRM", "COV"):
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bpy.data.images.remove(img)
    return str(out_path)
