import os, json
import numpy as np
import random
import argparse
from tqdm import tqdm
from template import get_sentence
from collections import defaultdict
import datetime
import signal
import string
from collections import deque
import itertools

# action_name = ['one step down', 'one step up', 'one step left', 'one step right']
# ac = {0: 1, 1: 0, 2: 3, 3: 2}
action_candidate = [
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
    [-1, -1],
    [1, 1],
    [-1, 1],
    [1, -1],
]
# down,   top,   left,    right, bottom-left, top-right, top-left, bottom-right
# down=0, top=1, left=2,  right=3, bottom-left=4, top-right=5, top-left=6, bottom-right=7

action_name = {
    1: "top",
    0: "down",
    3: "right",
    2: "left",
    5: "top-right",
    6: "top-left",
    7: "bottom-right",
    4: "bottom-left",
}


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Search function timed out!")


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def initialize_counter():
    counters = {
        "left": 0,
        "right": 0,
        "above": 0,
        "below": 0,
        "upper-right": 0,
        "lower-right": 0,
        "lower-left": 0,
        "upper-left": 0,
    }

    return counters


def check_counter(counter, proportion_size):
    if (
        counter["left"] >= proportion_size
        and counter["right"] >= proportion_size
        and counter["above"] >= proportion_size
        and counter["below"] >= proportion_size
        and counter["upper-right"] >= proportion_size
        and counter["lower-right"] >= proportion_size
        and counter["lower-left"] >= proportion_size
        and counter["upper-left"] >= proportion_size
    ):
        return True
    else:
        return False


def update_counter(counters, answer, proportion_size):
    update = False
    if answer in counters:
        if counters[answer] < proportion_size:
            counters[answer] += 1
        else:
            update = True
    return counters, update


def search(from_point, to_point, n, existing_points, path=None, depth=0, max_depth=18):
    """
    Finds a path from `from_point` to `to_point` that excludes all points in `existing_points`
    and has a length of at least len(existing_points) + 1.

    Args:
        from_point (list): Starting coordinate [x, y].
        to_point (list): Target coordinate [x, y].
        existing_points (list): List of coordinates to exclude from the path.
        path (list): Current path being explored.

    Returns:
        list: Valid path or None if no path exists.
    """

    if depth > max_depth:
        # print(f"Max depth reached at {from_point} -> {to_point}")
        return None

    # Initialize path if not provided
    if path is None:
        path = [from_point]

    # Compute the minimum required path length
    m = len(existing_points)
    min_length = n + 1
    # max_length = 2*m
    if n > 1:
        if n >= 7:
            max_length = int(1.5 * n)
        else:
            max_length = 2 * n
    else:
        max_length = 2 * n + 1

    # Base case: Check if we are at the destination and the path length is valid
    if len(path) >= max_length:
        if from_point == to_point:
            return path
        return None

    if from_point == to_point:
        if len(path) >= min_length:
            return path
        return None

    # Extract the current coordinates
    x, y = from_point

    # List all neighbors
    neighbors = [
        [x - 1, y],  # left
        [x + 1, y],  # right
        [x, y + 1],  # up
        [x, y - 1],  # down
        [x - 1, y - 1],  # bottom-left
        [x + 1, y - 1],  # bottom-right
        [x - 1, y + 1],  # top-left
        [x + 1, y + 1],  # top-right
    ]

    # Explore all neighbors
    for neighbor in neighbors:
        if (
            neighbor not in path and neighbor not in existing_points
        ):  # Avoid revisiting points and excluded points
            result = search(
                neighbor, to_point, n, existing_points, path + [neighbor], depth + 1
            )
            if result:  # If a valid path is found, return it
                return result

    # If no path is found, return None
    return None


from collections import deque


def bidirectional_search(from_point, to_point, n, existing_points):
    """
    Performs bidirectional search from `from_point` to `to_point` while avoiding `existing_points`
    and ensuring the path length is at least `n + 1`.

    Args:
        from_point (list): Starting coordinate [x, y].
        to_point (list): Target coordinate [x, y].
        n (int): Minimum number of steps required in the path (path length >= n+1).
        existing_points (list): List of coordinates to exclude from the path.

    Returns:
        list: Path connecting `from_point` and `to_point` with length at least `n+1`, or None if no path exists.
    """

    # If start and end points are the same, return immediately
    if from_point == to_point:
        return [from_point]

    # Convert existing points list to a set of tuples for fast lookup
    existing_points_set = {tuple(point) for point in existing_points}

    # Convert from_point and to_point to tuples
    from_point = tuple(from_point)
    to_point = tuple(to_point)

    # Define possible moves
    directions = [
        (-1, 0),
        (1, 0),
        (0, 1),
        (0, -1),  # Left, Right, Up, Down
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),  # Bottom-Left, Bottom-Right, Top-Left, Top-Right
    ]

    # BFS queues for forward and backward search
    forward_queue = deque([[from_point]])
    backward_queue = deque([[to_point]])

    # Visited nodes and paths
    visited_from = {from_point: [from_point]}
    visited_to = {to_point: [to_point]}

    while forward_queue and backward_queue:
        # Expand forward search
        path = forward_queue.popleft()
        last_node = path[-1]

        for dx, dy in directions:
            neighbor = (last_node[0] + dx, last_node[1] + dy)

            if neighbor not in visited_from and neighbor not in existing_points_set:
                new_path = path + [
                    neighbor
                ]  # Keep using tuples for internal consistency
                visited_from[neighbor] = new_path
                forward_queue.append(new_path)

                # Check if paths meet and path length >= n+1 and the path has unique nodes
                if neighbor in visited_to:
                    full_path = merge_paths(
                        visited_from[neighbor], visited_to[neighbor], n + 1, directions
                    )
                    if full_path:
                        return full_path

        # Expand backward search
        path = backward_queue.popleft()
        last_node = path[-1]

        for dx, dy in directions:
            neighbor = (last_node[0] + dx, last_node[1] + dy)

            if neighbor not in visited_to and neighbor not in existing_points_set:
                new_path = path + [
                    neighbor
                ]  # Keep using tuples for internal consistency
                visited_to[neighbor] = new_path
                backward_queue.append(new_path)

                # Check if paths meet and path length >= n+1
                if neighbor in visited_from:
                    full_path = merge_paths(
                        visited_from[neighbor], visited_to[neighbor], n + 1, directions
                    )
                    if full_path:
                        return full_path

    return None  # No valid path found


def merge_paths(path_from, path_to, n, directions):
    """
    Ensures the merged path:
    1. Meets the length constraint (>= n+1).
    2. Has no repeated nodes.
    3. Maintains connectivity (each step is a valid move).

    Args:
        path_from (list): Path from the start node.
        path_to (list): Path from the goal node.
        n (int): Minimum number of steps required.
        directions (list): Valid move directions.

    Returns:
        list: Unique, valid merged path or None if the path is not contiguous.
    """

    # Reverse the backward path for correct ordering
    path_to.reverse()

    # Create a merged path while ensuring uniqueness and connectivity
    merged_path = path_from[:]  # Copy forward path
    seen = set(path_from)  # Track visited nodes

    for node in path_to:
        if node not in seen:
            # Ensure node is a valid neighbor of the last node in merged_path
            last_node = merged_path[-1]
            if any(
                (last_node[0] + dx, last_node[1] + dy) == node for dx, dy in directions
            ):
                merged_path.append(node)
                seen.add(node)
            else:
                # Reject this path and continue searching
                return None

    # Ensure path length constraint is met
    if len(merged_path) >= n + 1:
        return [list(p) for p in merged_path]  # Convert tuples back to lists

    return None  # Path is too short

# down=0, top=1, left=2,  right=3, bottom-left=4, top-right=5, top-left=6, bottom-right=7


def get_action(location1, location2):
    if location1[0] == location2[0] and location1[1] + 1 == location2[1]:
        return 1  # top
    if location1[0] == location2[0] and location1[1] - 1 == location2[1]:
        return 0  # down
    if location1[0] + 1 == location2[0] and location1[1] == location2[1]:
        return 3  # right
    if location1[0] - 1 == location2[0] and location1[1] == location2[1]:
        return 2  # left
    if location1[0] + 1 == location2[0] and location1[1] + 1 == location2[1]:
        return 5  # top-right
    if location1[0] - 1 == location2[0] and location1[1] + 1 == location2[1]:
        return 6  # top-left
    if location1[0] + 1 == location2[0] and location1[1] - 1 == location2[1]:
        return 7  # bottom-right
    if location1[0] - 1 == location2[0] and location1[1] - 1 == location2[1]:
        return 4  # bottom-left


def add_disconnected_noise(entity_name, length, story):

    disconnected_noise = {}
    diconnected_length = length
    disconnected_entity = entity_name
    for i in range(diconnected_length):
        agent_1 = disconnected_entity[i]
        agent_2 = disconnected_entity[i + 1]
        action = np.random.randint(0, 8)
        noise_sentence = get_sentence(agent_2, action, agent_1)
        disconnected_noise[agent_2 + ":" + agent_1] = action_name[action]
        story.append(noise_sentence)
    return story, disconnected_noise


def add_irrelevant_noise(entity_name, length, d, story):

    irrelevant_noise = {}
    irrelevant_length = length
    irrelevant_entity = entity_name
    for i in range(irrelevant_length):
        agent_1 = random.choice(list(d))
        agent_2 = irrelevant_entity.pop()
        d[agent_2] = "pseudo_coordinate"
        action = np.random.randint(0, 8)

        noise_sentence = get_sentence(agent_2, action, agent_1)
        irrelevant_noise[agent_2 + ":" + agent_1] = action_name[action]
        story.append(noise_sentence)
    return story, irrelevant_noise


def add_supporting_noise(cur_entity, d, a, nhop, story):
    entity = cur_entity
    total_edge = 0
    att = 100
    existing_points_temp = []
    supporting_nodes = {}

    temp = list(d.items())

    # iterating both key and values
    for key, value in d.items():
        existing_points_temp.append(value)
    existing_points = existing_points_temp.copy()

    # Choose two nodes whose distance is at least 2

    # print("before while total_edge < 2 or total_edge > 7: {}".format(datetime.datetime.now()))
    # while total_edge < 2 or total_edge > 7: # for k = 1 to 10, 20
    # while total_edge < 2 or total_edge > 13: # k = 50

    # print("int(0.2*nhop): {}".format(int(0.2*nhop)))
    # while total_edge < 2 or total_edge > 17: # k = 100
    # while total_edge < 2 or total_edge > (if nhop < 10 0.7 else int(0.2 * nhop)):  # k = 100
    while total_edge < 2 or total_edge > int((0.7 if nhop <= 20 else 0.2) * nhop):
        random.shuffle(entity)
        noise_1 = entity[0]
        noise_2 = entity[1]
        noise_1_idx = [idx for idx, key in enumerate(temp) if key[0] == noise_1]
        noise_2_idx = [idx for idx, key in enumerate(temp) if key[0] == noise_2]

        from_point = [d[noise_2][0], d[noise_2][1]]
        to_point = [d[noise_1][0], d[noise_1][1]]

        x_diff = int(d[noise_2][0] - d[noise_1][0])
        y_diff = int(d[noise_2][1] - d[noise_1][1])
        total_edge = np.abs(x_diff) + np.abs(y_diff)

        att -= 1
        if att == 0:
            return story, 0, supporting_nodes

    nnodes = np.abs(noise_2_idx[0] - noise_1_idx[0])
    existing_points.remove(from_point)
    existing_points.remove(to_point)

    signal.signal(signal.SIGALRM, timeout_handler)  # Set signal handler
    signal.alarm(300)  # 300 seconds = 5 minutes

    try:
        # path = search(from_point, to_point, nnodes, existing_points)
        path = bidirectional_search(from_point, to_point, nnodes, existing_points)
        signal.alarm(0)  # Cancel the alarm if search finishes in time

        actions = []

        if path:
            for j in range(len(path) - 1):
                actions.append(get_action(path[j], path[j + 1]))
            remaining_entity = a[
                nhop + 1 : nhop + len(path) - 1
            ].copy()  # The number of nodes used is (total_edge-1)

        else:
            remaining_entity = a[nhop + 1 : nhop - 1].copy()

        if len(remaining_entity) > 0:
            cur_node = noise_2
            next_node = remaining_entity[0]

            supporting_nodes[cur_node] = path[0]

            for edge_id in range(len(actions)):

                supporting_nodes[next_node] = path[edge_id + 1]

                sentence = get_sentence(next_node, actions[edge_id], cur_node)
                story.append(sentence)

                if edge_id < (
                    len(actions) - 1
                ):  # if edge = total_edge-1, no change is needed
                    if edge_id < len(actions) - 2:
                        cur_node = next_node
                        next_node = remaining_entity[edge_id + 1]
                    elif edge_id == (len(actions) - 2):
                        cur_node = next_node
                        next_node = noise_1
                    else:
                        print("error")

    except TimeoutException:
        print("Search function took too long, returning None")
        return story, -1, supporting_nodes

    return story, len(actions) - 1, supporting_nodes


def generate_one_story(nhop, noise=True):
    """
    nhop: the maximum value of the potential number of reasoning steps.
    noise: whether add noise into samples
    """

    story = []
    d = {}
    supporting_noise = {}
    disconnected_noise = {}
    irrelevant_noise = {}

    d_inv = defaultdict(list)
    current_position = [0, 0]
    # candidates = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ','DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DJ', 'DK', 'DL', 'DM', 'DN', 'DO', 'DP', 'DQ', 'DR', 'DS', 'DT', 'DU', 'DV', 'DW', 'DX', 'DY', 'DZ', 'EA', 'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EJ', 'EK', 'EL', 'EM', 'EN', 'EO', 'EP', 'EQ', 'ER', 'ES', 'ET', 'EU', 'EV', 'EW', 'EX', 'EY', 'EZ']

    # candidates = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ','DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DJ', 'DK', 'DL', 'DM', 'DN', 'DO', 'DP', 'DQ', 'DR', 'DS', 'DT', 'DU', 'DV', 'DW', 'DX', 'DY', 'DZ', 'EA', 'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EJ', 'EK', 'EL', 'EM', 'EN', 'EO', 'EP', 'EQ', 'ER', 'ES', 'ET', 'EU', 'EV', 'EW', 'EX', 'EY', 'EZ', 'FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FJ', 'FK', 'FL', 'FM', 'FN', 'FO', 'FP', 'FQ', 'FR', 'FS', 'FT', 'FU', 'FV', 'FW', 'FX', 'FY', 'FZ', 'GA', 'GB', 'GC', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GJ', 'GK', 'GL', 'GM', 'GN', 'GO', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GU', 'GV', 'GW', 'GX', 'GY', 'GZ', 'HA', 'HB', 'HC', 'HD', 'HE', 'HF', 'HG', 'HH', 'HI', 'HJ', 'HK', 'HL', 'HM', 'HN', 'HO', 'HP', 'HQ', 'HR', 'HS', 'HT', 'HU', 'HV', 'HW', 'HX', 'HY', 'HZ','IA', 'IB', 'IC', 'ID', 'IE', 'IF', 'IG', 'IH', 'II', 'IJ', 'IK', 'IL', 'IM', 'IN', 'IO', 'IP', 'IQ', 'IR', 'IS', 'IT', 'IU', 'IV', 'IW', 'IX', 'IY', 'IZ']

    candidates = ["XA", "XB", "XD", "XE", "XF", "XG", "XH", "XJ", "XK", "XM", "XN", "XP", "XQ", "XR", "XS", "XT", "XU", "XW", "XY", "XZ", "XAB", "XAC", "XAD", "XAE", "XAF", "XAG", "XAH", "XAI", "XAJ", "XAK", "XAL", "XAM", "XAN", "XAO", "XAP", "XAQ", "XAR", "XAU", "XAV", "XAW", "XAX", "XAY", "XAZ", "XBA", "XBC", "XBD", "XBE", "XBF", "XBG", "XBH", "XBI", "XBJ", "XBK", "XBL", "XBM", "XBN", "XBO", "XBP", "XBQ", "XBR", "XBS", "XBT", "XBU", "XBV", "XBW", "XBX", "XBY", "XBZ", "XCA", "XCB", "XCC", "XCD", "XCE", "XCF", "XCG", "XCH", "XCI", "XCJ", "XCK", "XCL", "XCM", "XCN", "XCO", "XCP", "XCQ", "XCR", "XCS", "XCT", "XCU", "XCV", "XCW", "XCX", "XCY", "XCZ", "XDA", "XDB", "XDC", "XDD", "XDE", "XDF", "XDG", "XDH", "XDI", "XDJ", "XDK", "XDL", "XDM", "XDN", "XDO", "XDP", "XDQ", "XDR", "XDS", "XDT", "XDU", "XDV", "XDW", "XDX", "XDY", "XDZ", "XEA", "XEB", "XEC", "XED", "XEE", "XEF", "XEG", "XEH", "XEI", "XEJ", "XEK", "XEL", "XEM", "XEN", "XEO", "XEP", "XEQ", "XER", "XES", "XET", "XEU", "XEV", "XEW", "XEX", "XEY", "XEZ", "XFA", "XFB", "XFC", "XFD", "XFE", "XFF", "XFG", "XFH", "XFI", "XFJ", "XFK", "XFL", "XFM", "XFN", "XFO", "XFP", "XFQ", "XFR", "XFS", "XFT", "XFU", "XFV", "XFW", "XFX", "XFY", "XFZ", "XGA", "XGB", "XGC", "XGD", "XGE", "XGF", "XGG", "XGH", "XGI", "XGJ", "XGK", "XGL", "XGM", "XGN", "XGO", "XGP", "XGQ", "XGR", "XGS", "XGT", "XGU", "XGV", "XGW", "XGX", "XGY", "XGZ", "XHA", "XHB", "XHC", "XHD", "XHE", "XHF", "XHG", "XHH", "XHI", "XHJ", "XHK", "XHL", "XHM", "XHN", "XHO", "XHP", "XHQ", "XHR", "XHS", "XHT", "XHU", "XHV", "XHW", "XHX", "XHY", "XHZ", "XIA", "XIB", "XIC", "XID", "XIE", "XIF", "XIG", "XIH", "XIJ", "XIK", "XIL", "XIM", "XIO", "XIP", "XIQ", "XIR", "XIU", "XIW", "XIY", "XIZ", "XJA", "XJB", "XJC", "XJD", "XJE", "XJF", "XJG", "XJH", "XJI", "XJK", "XJL", "XJM", "XJN", "XJO", "XJP", "XJQ", "XJR", "XJS", "XJT", "XJU", "XJV", "XJW", "XJX", "XJY", "XJX",]
    ### Generate a story and its answer based on recored coordinates ###
    random.shuffle(candidates)
    entity = candidates[: nhop + 1]

    d[entity[0]] = current_position.copy()
    d_inv[str(0) + "_" + str(0)].append(entity[0])
    for i in range(nhop):
        agent_1 = entity[i]
        agent_2 = entity[i + 1]
        flag = False
        count = 0
        mychoices = random.sample(range(0, 8), 8)
        while not flag and count < len(mychoices):
            action = mychoices[count]
            temp_x = current_position[0] + action_candidate[action][0]
            temp_y = current_position[1] + action_candidate[action][1]

            if str(temp_x) + "_" + str(temp_y) not in d_inv:
                current_position[0] += action_candidate[action][0]
                current_position[1] += action_candidate[action][1]
                d[agent_2] = current_position.copy()
                flag = True
                d_inv[str(current_position[0]) + "_" + str(current_position[1])].append(
                    agent_2
                )
            count = count + 1

        if count >= len(mychoices):
            return "", "", "", (0, 0, 0), {}, {}, {}, {}

        sentence = get_sentence(agent_2, action, agent_1)
        #if agent_2 == agent_1:
        #    print("redundant: {}".format(sentence))

        story.append(sentence)

    # random.shuffle(entity)
    # ask_1 = entity.pop()
    # ask_2 = entity.pop()

    ask_1 = entity[0]
    ask_2 = entity[-1]
    difference = [0, 0]
    difference[0] = d[ask_2][0] - d[ask_1][0]
    difference[1] = d[ask_2][1] - d[ask_1][1]

    # Divide into 8 regions
    if difference[1] > 0:
        if difference[0] < 0:
            answer = "upper-left"
        elif difference[0] > 0:
            answer = "upper-right"
        else:
            answer = "above"
    elif difference[1] == 0:
        if difference[0] < 0:
            answer = "left"
        elif difference[0] == 0:
            answer = ""
        else:
            answer = "right"
    else:
        if difference[0] < 0:
            answer = "lower-left"
        elif difference[0] > 0:
            answer = "lower-right"
        else:
            answer = "below"
    q = "What is the relation of the agent {} to the agent {}?".format(ask_2, ask_1)

    ### Add distracting noise into a story ###
    if noise:
        # Determine the number of nodes used for each type of noise
        supporting_nodes = 0
        num_edge_used = 0
        if (
            nhop > 3
        ):  # Generate supporting noise when there are more than three nodes in the original story
            story, num_edge_used, supporting_noise = add_supporting_noise(
                entity, d, candidates, nhop, story
            )
            if num_edge_used != 0:
                supporting_nodes = num_edge_used - 1
        try:
            disconnected_nodes = random.randint(2, int((nhop + 1) / 3) + 1)
        except:
            disconnected_nodes = 2
        irrelevant_nodes = random.randint(1, int((nhop + 1) / 3) + 1)

        assert disconnected_nodes + irrelevant_nodes + supporting_nodes + (
            nhop + 1
        ) <= len(candidates)

        if num_edge_used != 0:
            disconnected_entity = candidates[
                nhop + num_edge_used : nhop + num_edge_used + disconnected_nodes + 1
            ]
            irrelevant_entity = candidates[
                nhop
                + num_edge_used
                + disconnected_nodes
                + 1 : nhop
                + num_edge_used
                + disconnected_nodes
                + 1
                + irrelevant_nodes
            ]
        else:
            disconnected_entity = candidates[nhop + 1 : nhop + 2 + disconnected_nodes]
            irrelevant_entity = candidates[
                nhop
                + 2
                + disconnected_nodes : nhop
                + 2
                + disconnected_nodes
                + irrelevant_nodes
            ]
        story, disconnected_noise = add_disconnected_noise(
            disconnected_entity, disconnected_nodes - 1, story
        )
        story, irrelevant_noise = add_irrelevant_noise(
            irrelevant_entity, irrelevant_nodes, d, story
        )

        return (
            story,
            q,
            answer,
            (supporting_nodes, disconnected_nodes, irrelevant_nodes),
            d,
            supporting_noise,
            disconnected_noise,
            irrelevant_noise,
        )

    return (
        story,
        q,
        answer,
        (0, 0, 0),
        d,
        supporting_noise,
        disconnected_noise,
        irrelevant_noise,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generator")
    # parser.add_argument('--nhop', type=int, default=10,
    #                     help='number of reasoning hops')
    parser.add_argument(
        "--seed", type=int, default=111, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument("--output_path", type=str, default="../../data/raw/DecompSR_100K")
    parser.add_argument("--proportion_size", type=int, default=125)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # train_size_set = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
    train_size_set = [
        50000
    ] * 100  # [sample_size per hop]*nhops ; train_size_set = 1600 for nhops = 100; train_size_set = 1200 for nhops = 50, train_size_set = 1050 for nhops = 20
    test_size = 50000
    valid_size = 1000
    # train_size_set = [10]*10
    # test_size  = 10
    # valid_size = 10

    print("building datasets...")
    check_path(args.output_path)

    check_path(os.path.join(args.output_path, "clean"))
    check_path(os.path.join(args.output_path, "noise"))
    statistics = []
    # for nhop in range(1, 11):
    # for nhop in range(100, 101):
    for nhop in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
        print("Building dataset with {} hops".format(nhop))

        length = 8
        print("building test datasets...")
        with open(
            os.path.join(
                args.output_path, "clean/qa{}_test_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_clean_shuffle, open(
            os.path.join(
                args.output_path, "clean/qa{}_test_no_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_clean_no_shuffle, open(
            os.path.join(
                args.output_path, "noise/qa{}_test_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_noise_shuffle, open(
            os.path.join(
                args.output_path, "noise/qa{}_test_no_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_noise_no_shuffle:
            s_noise = 0
            d_noise = 0
            i_noise = 0
            # left = 0, right = 0, above = 0, below = 0, upper_right = 0, lower_right = 0, lower_left = 0, upper_left = 0
            counter = initialize_counter()
            for test_index in range(test_size):

                # if left >= args.proportion_size and right >= args.proportion_size and above >= args.proportion_size and below >= args.proportion_size and upper_right >= args.proportion_size and lower_right >= args.proportion_size and lower_left >= args.proportion_size and upper_left >= args.proportion_size:
                #    break

                if check_counter(counter, args.proportion_size):
                    break

                line = 1
                (
                    story,
                    q,
                    a,
                    noise_node_num,
                    original,
                    supporting_noise,
                    disconnected_noise,
                    irrelevant_noise,
                ) = generate_one_story(nhop)

                if (not story and not q and not a) or (not a):
                    test_size = test_size + 1
                    continue

                counters, update = update_counter(counter, a, args.proportion_size)
                if update:
                    continue

                s_noise += noise_node_num[0]
                d_noise += noise_node_num[1]
                i_noise += noise_node_num[2]

                clean_story = story[:nhop].copy()

                myList = []

                for i in range(len(clean_story)):
                    myList.append(str(line) + " " + clean_story[i] + "\n")
                    line += 1
                # myList.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')

                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )
                datum = {
                    "ID": random_string,
                    "data": "".join(myList),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "nhop": nhop,
                    "clean": True,
                    "shuffled": False,
                }

                f_clean_no_shuffle.write(json.dumps(datum) + "\n")

                random.shuffle(clean_story)

                myList = []
                line = 1
                for i in range(len(clean_story)):
                    myList.append(str(line) + " " + clean_story[i] + "\n")
                    line += 1
                # myList.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')

                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )
                datum = {
                    "ID": random_string,
                    "data": "".join(myList),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "nhop": nhop,
                    "clean": True,
                    "shuffled": True,
                }

                f_clean_shuffle.write(json.dumps(datum) + "\n")

                line = 1
                myList2 = []
                for i in range(len(story)):
                    myList2.append(str(line) + " " + story[i] + "\n")
                    line += 1
                # myList2.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')
                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )

                datum = {
                    "ID": random_string,
                    "data": "".join(myList2),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "supporting_noise": supporting_noise,
                    "disconnected_noise": disconnected_noise,
                    "irrelevant_noise": irrelevant_noise,
                    "nhop": nhop,
                    "clean": False,
                    "shuffled": False,
                }

                f_noise_no_shuffle.write(json.dumps(datum) + "\n")
                random.shuffle(story)

                line = 1
                myList2 = []
                for i in range(len(story)):
                    myList2.append(str(line) + " " + story[i] + "\n")
                    line += 1
                # myList2.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')
                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )

                datum = {
                    "ID": random_string,
                    "data": "".join(myList2),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "supporting_noise": supporting_noise,
                    "disconnected_noise": disconnected_noise,
                    "irrelevant_noise": irrelevant_noise,
                    "nhop": nhop,
                    "clean": False,
                    "shuffled": True,
                }

                f_noise_shuffle.write(json.dumps(datum) + "\n")

            statistics.append(
                "Test nhop {}: the average number of noise nodes is ({}, {}, {})".format(
                    nhop, s_noise / test_size, d_noise / test_size, i_noise / test_size
                )
            )
            statistics.append(
                "Total average: {}".format((s_noise + d_noise + i_noise) / test_size)
            )

        print("building train datasets...")
        with open(
            os.path.join(
                args.output_path, "clean/qa{}_train_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_clean_shuffle, open(
            os.path.join(
                args.output_path, "clean/qa{}_train_no_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_clean_no_shuffle, open(
            os.path.join(
                args.output_path, "noise/qa{}_train_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_noise_shuffle, open(
            os.path.join(
                args.output_path, "noise/qa{}_train_no_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_noise_no_shuffle:
            train_size = train_size_set[nhop - 1]
            s_noise = 0
            d_noise = 0
            i_noise = 0

            counter = initialize_counter()
            for i in range(train_size):
                if check_counter(counter, args.proportion_size):
                    break
                line = 1
                (
                    story,
                    q,
                    a,
                    noise_node_num,
                    original,
                    supporting_noise,
                    disconnected_noise,
                    irrelevant_noise,
                ) = generate_one_story(nhop)

                print("answer: {}".format(a))
                if (not story and not q and not a) or (not a):
                    train_size = train_size + 1
                    continue

                counter, update = update_counter(counter, a, args.proportion_size)
                if update:
                    continue

                s_noise += noise_node_num[0]
                d_noise += noise_node_num[1]
                i_noise += noise_node_num[2]

                clean_story = story[:nhop].copy()

                myList = []

                for i in range(len(clean_story)):
                    myList.append(str(line) + " " + clean_story[i] + "\n")
                    line += 1
                # myList.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')

                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )
                datum = {
                    "ID": random_string,
                    "data": "".join(myList),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "nhop": nhop,
                    "clean": True,
                    "shuffled": False,
                }

                f_clean_no_shuffle.write(json.dumps(datum) + "\n")

                random.shuffle(clean_story)

                myList = []
                line = 1
                for i in range(len(clean_story)):
                    myList.append(str(line) + " " + clean_story[i] + "\n")
                    line += 1
                # myList.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')

                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )
                datum = {
                    "ID": random_string,
                    "data": "".join(myList),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "nhop": nhop,
                    "clean": True,
                    "shuffled": True,
                }

                f_clean_shuffle.write(json.dumps(datum) + "\n")

                line = 1
                myList2 = []
                for i in range(len(story)):
                    myList2.append(str(line) + " " + story[i] + "\n")
                    line += 1
                # myList2.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')
                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )

                datum = {
                    "ID": random_string,
                    "data": "".join(myList2),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "supporting_noise": supporting_noise,
                    "disconnected_noise": disconnected_noise,
                    "irrelevant_noise": irrelevant_noise,
                    "nhop": nhop,
                    "clean": False,
                    "shuffled": False,
                }

                f_noise_no_shuffle.write(json.dumps(datum) + "\n")

                random.shuffle(story)

                line = 1
                myList2 = []
                for i in range(len(story)):
                    myList2.append(str(line) + " " + story[i] + "\n")
                    line += 1
                # myList2.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')
                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )

                datum = {
                    "ID": random_string,
                    "data": "".join(myList2),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "supporting_noise": supporting_noise,
                    "disconnected_noise": disconnected_noise,
                    "irrelevant_noise": irrelevant_noise,
                    "nhop": nhop,
                    "clean": False,
                    "shuffled": True,
                }

                f_noise_shuffle.write(json.dumps(datum) + "\n")

            statistics.append(
                "Train nhop {}: the average number of noise nodes is ({}, {}, {})".format(
                    nhop,
                    s_noise / train_size,
                    d_noise / train_size,
                    i_noise / train_size,
                )
            )
            statistics.append(
                "Total average: {}".format((s_noise + d_noise + i_noise) / train_size)
            )

        print("building valid datasets...")
        with open(
            os.path.join(
                args.output_path, "clean/qa{}_valid_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_clean_shuffle, open(
            os.path.join(
                args.output_path, "clean/qa{}_valid_no_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_clean_no_shuffle, open(
            os.path.join(
                args.output_path, "noise/qa{}_valid_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_noise_shuffle, open(
            os.path.join(
                args.output_path, "noise/qa{}_valid_no_shuffle.jsonl".format(nhop)
            ),
            "w",
        ) as f_noise_no_shuffle:
            s_noise = 0
            d_noise = 0
            i_noise = 0

            counter = initialize_counter()
            for _ in range(valid_size):

                if check_counter(counter, int(args.proportion_size * 0.1)):
                    break

                line = 1
                (
                    story,
                    q,
                    a,
                    noise_node_num,
                    original,
                    supporting_noise,
                    disconnected_noise,
                    irrelevant_noise,
                ) = generate_one_story(nhop)


                if (not story and not q and not a) or (not a):
                    valid_size = valid_size + 1
                    print("valid_size: {}".format(valid_size))
                    continue

                # counter, update = update_counter(counter, a, int(args.proportion_size*0.1))
                counter, update = update_counter(
                    counter, a, int(args.proportion_size * 0.1)
                )

                if update:
                    continue

                s_noise += noise_node_num[0]
                d_noise += noise_node_num[1]
                i_noise += noise_node_num[2]

                clean_story = story[:nhop].copy()

                myList = []
                for i in range(len(clean_story)):
                    myList.append(str(line) + " " + clean_story[i] + "\n")
                    line += 1
                # myList.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')

                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )
                datum = {
                    "ID": random_string,
                    "data": "".join(myList),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "nhop": nhop,
                    "clean": True,
                    "shuffled": False,
                }

                f_clean_no_shuffle.write(json.dumps(datum) + "\n")

                random.shuffle(clean_story)

                myList = []
                line = 1
                for i in range(len(clean_story)):
                    myList.append(str(line) + " " + clean_story[i] + "\n")
                    line += 1
                # myList.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')

                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )
                datum = {
                    "ID": random_string,
                    "data": "".join(myList),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "nhop": nhop,
                    "clean": True,
                    "shuffled": True,
                }

                f_clean_shuffle.write(json.dumps(datum) + "\n")

                line = 1
                myList2 = []
                for i in range(len(story)):
                    myList2.append(str(line) + " " + story[i] + "\n")
                    line += 1
                # myList2.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')
                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )

                datum = {
                    "ID": random_string,
                    "data": "".join(myList2),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "supporting_noise": supporting_noise,
                    "disconnected_noise": disconnected_noise,
                    "irrelevant_noise": irrelevant_noise,
                    "nhop": nhop,
                    "clean": False,
                    "shuffled": False,
                }

                f_noise_no_shuffle.write(json.dumps(datum) + "\n")

                random.shuffle(story)

                line = 1
                myList2 = []
                for i in range(len(story)):
                    myList2.append(str(line) + " " + story[i] + "\n")
                    line += 1
                # myList2.append(str(line) + ' ' + q+'\t'+a+'\t'+str(1)+'\n')
                random_string = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )

                datum = {
                    "ID": random_string,
                    "data": "".join(myList2),
                    "question": q,
                    "answer": a,
                    "original": original,
                    "supporting_noise": supporting_noise,
                    "disconnected_noise": disconnected_noise,
                    "irrelevant_noise": irrelevant_noise,
                    "nhop": nhop,
                    "clean": False,
                    "shuffled": True,
                }

                f_noise_shuffle.write(json.dumps(datum) + "\n")

            statistics.append(
                "Valid nhop {}: the average number of noise nodes is ({}, {}, {})".format(
                    nhop,
                    s_noise / valid_size,
                    d_noise / valid_size,
                    i_noise / valid_size,
                )
            )
            statistics.append(
                "Total average: {}\n".format((s_noise + d_noise + i_noise) / valid_size)
            )

    with open(os.path.join(args.output_path, "statistic.txt"), "w") as f:
        for line in statistics:
            f.write(line)
            f.write("\n")

    print("Finished.")
