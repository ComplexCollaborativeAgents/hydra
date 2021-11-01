from worlds.polycraft_world import *
import matplotlib.pyplot as plt


def visualize_game_map(state: PolycraftState, ax=None, set_limits = True):
    # Fixing random state for reproducibility

    is_3d = True
    show_cells = True
    show_steve = True
    show_entities = True
    print_results = False

    steve_scatter = None
    cells_scatter = None
    entities_scatter = None

    # Create plot if needed
    if ax is None:
        fig = plt.figure()
        if is_3d:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()

    xs = []
    ys = []
    zs = []
    types = []
    type_to_cells = dict()
    for cell_id, cell_attr in state.game_map.items():
        x, y, z = cell_id.split(",")
        xs.append(int(x))
        ys.append(int(y))
        zs.append(int(z))
        cell_type = cell_attr['name']
        types.append(cell_type)

        if cell_type not in type_to_cells:
            type_to_cells[cell_type] = [[],[],[]]

        type_to_cells[cell_type][0].append(int(x))
        type_to_cells[cell_type][1].append(int(y))
        type_to_cells[cell_type][2].append(int(z))

    if set_limits:
        ax.set_xlim(min(xs), max(xs))
        if is_3d:
            ax.set_ylim(min(ys), max(ys))
            ax.set_zlim(min(zs), max(zs))
        else:
            ax.set_ylim(min(zs), max(zs))

    if show_cells:
        if print_results:
            print(f'x values:{min(xs)}, {max(xs)}')
            print(f'y values:{min(ys)}, {max(ys)}')
            print(f'z values:{min(zs)}, {max(zs)}')
            print('Cell types:')
            for cell_type in set(types):
                print(f'\t {cell_type}')

        type_to_marker = {
            "minecraft:bedrock": "$b$",
            "polycraft:plastic_chest": "$p$",
            "minecraft:log": "$l$",
            "minecraft:wooden_door": "$d$",
            "minecraft:air": "$a$",
            "polycraft:tree_tap": "$t$",
            "minecraft:crafting_table": "$c$",
            "polycraft:block_of_platinum": "$b$",
            "minecraft:diamond_ore": "$D$",
        }

        for cell_type, cells in type_to_cells.items():
            if cell_type=="minecraft:air":
                continue
            m = type_to_marker[cell_type]
            xs = cells[0]
            ys = cells[1]
            zs = cells[2]

            if is_3d:
                cells_scatter = ax.scatter(xs, ys, zs, marker=m)
            else:
                cells_scatter = ax.scatter(xs, zs, marker=m)


    # Steve's location
    if show_steve:
        player_obj = state.location
        steve_x = player_obj['pos'][0]
        steve_y = player_obj['pos'][1]
        steve_z = player_obj['pos'][2]
        if print_results:
            print(f'Steve at {steve_x, steve_y, steve_z}')

        if is_3d:
            steve_scatter = ax.scatter([steve_x], [steve_y], [steve_z], marker="*")
        else:
            steve_scatter = ax.scatter([steve_x], [steve_z], marker="*")

    # Other entities
    entities_scatters = []
    if show_entities:
        for entity_id, entity_attr in state.entities.items():
            x, y, z = entity_attr['pos']
            if print_results:
                print(f"Entity {entity_id} at {x,y,z}")

            if is_3d:
                scatter= ax.scatter([x], [y], [z], marker="$E$")
            else:
                scatter = ax.scatter([x], [z], marker="$E$")
            entities_scatters.append(scatter)

    ax.set_xlabel('X Label')
    if is_3d:
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    else:
        ax.set_ylabel('Z Label')

    plt.show()


    return ax, steve_scatter, cells_scatter, entities_scatters

def create_game_map_matrix(state: PolycraftState):
    xs = []
    zs = []
    loc_to_type = dict()
    for cell_id, cell_attr in state.game_map.items():
        x, _, z = cell_id.split(",")
        xs.append(int(x))
        zs.append(int(z))
        cell_type = cell_attr['name']


    min_x= min(xs)
    max_x = max(xs)
    min_z = min(zs)
    max_z = max(zs)

def visualize_2d_game_map(state: PolycraftState):
    # Fixing random state for reproducibility
    show_cells = True
    show_steve = True
    show_entities = True
    print_results = False

    steve_scatter = None
    cells_scatter = None
    entities_scatter = None

    # Create the data matrix
    xs = []
    zs = []
    types = []
    type_to_cells = dict()
    for cell_id, cell_attr in state.game_map.items():
        x, _, z = cell_id.split(",")
        xs.append(int(x))
        zs.append(int(z))
        cell_type = cell_attr['name']
        types.append(cell_type)

        if cell_type not in type_to_cells:
            type_to_cells[cell_type] = [[],[],[]]

        type_to_cells[cell_type][0].append(int(x))
        type_to_cells[cell_type][1].append(int(y))
        type_to_cells[cell_type][2].append(int(z))

    if set_limits:
        ax.set_xlim(min(xs), max(xs))
        if is_3d:
            ax.set_ylim(min(ys), max(ys))
            ax.set_zlim(min(zs), max(zs))
        else:
            ax.set_ylim(min(zs), max(zs))

    if show_cells:
        if print_results:
            print(f'x values:{min(xs)}, {max(xs)}')
            print(f'y values:{min(ys)}, {max(ys)}')
            print(f'z values:{min(zs)}, {max(zs)}')
            print('Cell types:')
            for cell_type in set(types):
                print(f'\t {cell_type}')

        type_to_marker = {
            "minecraft:bedrock": "$b$",
            "polycraft:plastic_chest": "$p$",
            "minecraft:log": "$l$",
            "minecraft:wooden_door": "$d$",
            "minecraft:air": "$a$",
            "polycraft:tree_tap": "$t$",
            "minecraft:crafting_table": "$c$",
            "polycraft:block_of_platinum": "$b$",
            "minecraft:diamond_ore": "$D$",
        }

        for cell_type, cells in type_to_cells.items():
            if cell_type=="minecraft:air":
                continue
            m = type_to_marker[cell_type]
            xs = cells[0]
            ys = cells[1]
            zs = cells[2]

            if is_3d:
                cells_scatter = ax.scatter(xs, ys, zs, marker=m)
            else:
                cells_scatter = ax.scatter(xs, zs, marker=m)


    # Steve's location
    if show_steve:
        player_obj = state.location
        steve_x = player_obj['pos'][0]
        steve_y = player_obj['pos'][1]
        steve_z = player_obj['pos'][2]
        if print_results:
            print(f'Steve at {steve_x, steve_y, steve_z}')

        if is_3d:
            steve_scatter = ax.scatter([steve_x], [steve_y], [steve_z], marker="*")
        else:
            steve_scatter = ax.scatter([steve_x], [steve_z], marker="*")

    # Other entities
    entities_scatters = []
    if show_entities:
        for entity_id, entity_attr in state.entities.items():
            x, y, z = entity_attr['pos']
            if print_results:
                print(f"Entity {entity_id} at {x,y,z}")

            if is_3d:
                scatter= ax.scatter([x], [y], [z], marker="$E$")
            else:
                scatter = ax.scatter([x], [z], marker="$E$")
            entities_scatters.append(scatter)

    ax.set_xlabel('X Label')
    if is_3d:
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    else:
        ax.set_ylabel('Z Label')

    plt.show()


    return ax, steve_scatter, cells_scatter, entities_scatters

def print_game_map(state: PolycraftState, ax=None, set_limits = True):
    # Fixing random state for reproducibility

    is_3d = True
    show_cells = True
    show_steve = True
    show_entities = True
    print_results = False

    steve_scatter = None
    cells_scatter = None
    entities_scatter = None

    # Create plot if needed
    if ax is None:
        fig = plt.figure()
        if is_3d:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()

    xs = []
    ys = []
    zs = []
    types = []
    type_to_cells = dict()
    for cell_id, cell_attr in state.game_map.items():
        x, y, z = cell_id.split(",")
        xs.append(int(x))
        ys.append(int(y))
        zs.append(int(z))
        cell_type = cell_attr['name']
        types.append(cell_type)

        if cell_type not in type_to_cells:
            type_to_cells[cell_type] = [[],[],[]]

        type_to_cells[cell_type][0].append(int(x))
        type_to_cells[cell_type][1].append(int(y))
        type_to_cells[cell_type][2].append(int(z))

    if set_limits:
        ax.set_xlim(min(xs), max(xs))
        if is_3d:
            ax.set_ylim(min(ys), max(ys))
            ax.set_zlim(min(zs), max(zs))
        else:
            ax.set_ylim(min(zs), max(zs))

    if show_cells:
        if print_results:
            print(f'x values:{min(xs)}, {max(xs)}')
            print(f'y values:{min(ys)}, {max(ys)}')
            print(f'z values:{min(zs)}, {max(zs)}')
            print('Cell types:')
            for cell_type in set(types):
                print(f'\t {cell_type}')

        type_to_marker = {
            "minecraft:bedrock": "$b$",
            "polycraft:plastic_chest": "$p$",
            "minecraft:log": "$l$",
            "minecraft:wooden_door": "$d$",
            "minecraft:air": "$a$",
            "polycraft:tree_tap": "$t$",
            "minecraft:crafting_table": "$c$",
            "polycraft:block_of_platinum": "$b$",
            "minecraft:diamond_ore": "$D$",
        }

        for cell_type, cells in type_to_cells.items():
            if cell_type=="minecraft:air":
                continue
            m = type_to_marker[cell_type]
            xs = cells[0]
            ys = cells[1]
            zs = cells[2]

            if is_3d:
                cells_scatter = ax.scatter(xs, ys, zs, marker=m)
            else:
                cells_scatter = ax.scatter(xs, zs, marker=m)


    # Steve's location
    if show_steve:
        player_obj = state.location
        steve_x = player_obj['pos'][0]
        steve_y = player_obj['pos'][1]
        steve_z = player_obj['pos'][2]
        if print_results:
            print(f'Steve at {steve_x, steve_y, steve_z}')

        if is_3d:
            steve_scatter = ax.scatter([steve_x], [steve_y], [steve_z], marker="*")
        else:
            steve_scatter = ax.scatter([steve_x], [steve_z], marker="*")

    # Other entities
    entities_scatters = []
    if show_entities:
        for entity_id, entity_attr in state.entities.items():
            x, y, z = entity_attr['pos']
            if print_results:
                print(f"Entity {entity_id} at {x,y,z}")

            if is_3d:
                scatter= ax.scatter([x], [y], [z], marker="$E$")
            else:
                scatter = ax.scatter([x], [z], marker="$E$")
            entities_scatters.append(scatter)

    ax.set_xlabel('X Label')
    if is_3d:
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    else:
        ax.set_ylabel('Z Label')

    plt.show()


    return ax, steve_scatter, cells_scatter, entities_scatters