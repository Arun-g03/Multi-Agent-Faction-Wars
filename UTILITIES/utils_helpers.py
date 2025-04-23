"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from UTILITIES.utils_logger import Logger
import UTILITIES.utils_config as utils_config

logger = Logger(log_file="utils_helpers.txt", log_level=logging.DEBUG)


#    ____  ____   ___  _____ ___ _     ___ _   _  ____
#   |  _ \|  _ \ / _ \|  ___|_ _| |   |_ _| \ | |/ ___|
#   | |_) | |_) | | | | |_   | || |    | ||  \| | |  _
#   |  __/|  _ <| |_| |  _|  | || |___ | || |\  | |_| |
#   |_|   |_| \_\\___/|_|   |___|_____|___|_| \_|\____|
#


def profile_function(
        func,
        output_file=f"Profiling_Stats/profile_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.prof",
        *
        args,
        **kwargs):
    """
    Profiles a function and saves both execution time and function call reports.
    - The nested function `_save_profile_results` handles sorting & saving.
    - The final report combines both execution time and call count results.
    """
    print(f"[DEBUG] Profiling started for {func.__name__}...")

    # Create the Profiling_Stats directory if it doesn't exist
    import os
    os.makedirs("Profiling_Stats", exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        result = func(*args, **kwargs)  # Run the function
    except Exception as e:
        print(f"[ERROR] Profiling interrupted due to an exception: {e}")
        result = None
    finally:
        profiler.disable()

        def _save_profile_results(sort_by):
            """Sorts and captures profiling data into a string."""
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats(sort_by)
            stats.print_stats(40)
            return stream.getvalue()

        #  Get profiling reports
        time_report = _save_profile_results("tottime")
        call_report = _save_profile_results("ncalls")

        #  Combine results into a single file
        with open(output_file, "w") as f:
            f.write("[PROFILING RESULTS - SORTED BY EXECUTION TIME]\n")
            f.write(time_report)
            f.write("\n\n[PROFILING RESULTS - SORTED BY FUNCTION CALLS]\n")
            f.write(call_report)

        print(f"[INFO] Combined profiling results saved to {output_file}")

        return result


#    _____ ___ _   _ ____     ____ _     ___  ____  _____ ____ _____    __
#   |  ___|_ _| \ | |  _ \   / ___| |   / _ \/ ___|| ____/ ___|_   _|  / _ \| __ )    | | ____/ ___|_   _|
#   | |_   | ||  \| | | | | | |   | |  | | | \___ \|  _| \___ \ | |   | | | |  _ \ _  | |  _|| |     | |
#   |  _|  | || |\  | |_| | | |___| |__| |_| |___) | |___ ___) || |   | |_| | |_) | |_| | |__| |___  | |
#   |_|   |___|_| \_|____/   \____|_____\___/|____/|_____|____/ |_|    \___/|____/ \___/|_____\____| |_|
#


def find_closest_actor(entities, entity_type=None, requester=None):
    """
    Finds the closest entity to the requester.
    Supports both dict-style (with 'location') and object-style (with .x and .y) entities.
    """
    if not entities:
        requester_name = getattr(requester, "role", "HQ")
        logger.log_msg(
            f"{requester_name} found no valid {entity_type or 'actor'}.",
            level=logging.WARNING)
        return None

    closest_entity = None
    min_distance = float('inf')

    requester_x = getattr(requester, "x", None)
    requester_y = getattr(requester, "y", None)
    if requester_x is None or requester_y is None:
        return None  # Ensure requester has coordinates

    for entity in entities:
        try:
            if isinstance(entity, dict) and "location" in entity:
                entity_x, entity_y = entity["location"]
            elif hasattr(entity, "x") and hasattr(entity, "y"):
                entity_x, entity_y = entity.x, entity.y
            else:
                continue  # Skip invalid entries
        except Exception as e:
            logger.log_msg(
                f"Skipping invalid entity: {entity}. Error: {e}",
                level=logging.WARNING)
            continue

        dist = ((requester_x - entity_x) ** 2 +
                (requester_y - entity_y) ** 2) ** 0.5
        if dist < min_distance:
            min_distance = dist
            closest_entity = entity

    if closest_entity:
        requester_name = getattr(requester, "role", "HQ")
        logger.log_msg(
            f"{requester_name} found closest {entity_type}: {closest_entity}",
            level=logging.INFO)

    return closest_entity


#     ____                           _                   _                     __               __            _   _
#    / ___| ___ _ __   ___ _ __ __ _| |_ ___    ___ ___ | | ___  _   _ _ __   / _| ___  _ __   / _| __ _  ___| |_(_) ___  _ __
#   | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \  / __/ _ \| |/ _ \| | | | '__| | |_ / _ \| '__| | |_ / _` |/ __| __| |/ _ \| '_ \
#   | |_| |  __/ | | |  __/ | | (_| | ||  __/ | (_| (_) | | (_) | |_| | |    |  _| (_) | |    |  _| (_| | (__| |_| | (_) | | | |
#    \____|\___|_| |_|\___|_|  \__,_|\__\___|  \___\___/|_|\___/ \__,_|_|    |_|  \___/|_|    |_|  \__,_|\___|\__|_|\___/|_| |_|
#


def generate_random_colour(used_colours=None, min_distance=100):
    """
    Generate a random RGB colour that is visually distinct from previously used colours
    and avoids black, white, and grassy greens.

    Args:
        used_colours (list): A list of previously generated colours to avoid similarity.
        min_distance (int): Minimum Euclidean distance between colours.

    Returns:
        tuple: A new (R, G, B) colour.
    """
    if used_colours is None:
        used_colours = []

    def euclidean_distance(colour1, colour2):
        """Calculate the Euclidean distance between two RGB colours."""
        return math.sqrt(
            sum((c1 - c2) ** 2 for c1, c2 in zip(colour1, colour2)))

    def is_valid_colour(colour):
        """Ensure the colour is not too dark, too bright, or grassy green."""
        r, g, b = colour

        # Avoid very dark (black-like) or very bright (white-like) colours
        brightness = (r + g + b) / 3
        if brightness < 50 or brightness > 200:
            return False

        # Avoid grassy greens (high green, low red/blue)
        if 60 <= g <= 200 and r < 100 and b < 100:
            return False

        return True

    while True:
        # Generate a random colour
        colour = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))

        # Ensure the colour is valid and distinct
        if (is_valid_colour(colour) and all(euclidean_distance(
                colour, used) >= min_distance for used in used_colours)):
            used_colours.append(colour)  # Add to the list of used colours
            return colour



