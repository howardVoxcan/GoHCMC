import json
import logging
import traceback
from urllib.parse import urlencode

from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

from location.models import Location
from trip.TSP import Graph, distance
from trip.models import TripPath, TripList
from .models import TemporaryTripCart, TemporaryUser

logger = logging.getLogger(__name__)


# ============================================================================
# SESSION & USER UTILITIES
# ============================================================================

def extract_session_id(data, parameters):
    """Extract session ID from webhook data"""
    session_full = data.get('session', '')
    session_id = session_full.split('/')[-1] if session_full else None

    if not session_id and 'originalDetectIntentRequest' in data:
        session_id = data['originalDetectIntentRequest'].get('payload', {}).get('sessionId')

    if not session_id:
        session_id = parameters.get('session_id')

    return session_id


def normalize_locations(loc):
    """Normalize location input to list format"""
    if loc is None:
        return []
    if isinstance(loc, str):
        return [loc]
    if isinstance(loc, list):
        return loc
    return [str(loc)]


def get_or_create_temp_user(session_id):
    """Get or create temporary user for session"""
    temp_user, _ = TemporaryUser.objects.get_or_create(session_id=session_id)
    return temp_user


def get_or_create_temp_cart(session_id, user=None):
    """Get or create temporary trip cart for session"""
    cart, created = TemporaryTripCart.objects.get_or_create(
        session_id=session_id,
        defaults={"user": user}
    )
    if not created and cart.user is None and user is not None:
        cart.user = user
        cart.save()
    return cart


# ============================================================================
# INTENT HANDLERS - FAVOURITES
# ============================================================================

def handle_add_favourite(user, locations_list):
    """Handle adding locations to favourites"""
    if not user or isinstance(user, TemporaryUser):
        return "Please log in to add favourites."
    if not locations_list:
        return "Please specify a location to add."

    for loc in locations_list:
        try:
            location_obj = Location.objects.get(location__iexact=loc)
            location_obj.favourited_by.add(user)
        except Location.DoesNotExist:
            return f"Location '{loc}' does not exist."
    return True


def handle_remove_favourite(user, locations_list):
    """Handle removing locations from favourites"""
    if not user or isinstance(user, TemporaryUser):
        return "Please log in to remove favourites."
    if not locations_list:
        return "Please specify a location to remove."

    for loc in locations_list:
        try:
            location_obj = Location.objects.get(location__iexact=loc)
            location_obj.favourited_by.remove(user)
        except Location.DoesNotExist:
            return f"Location '{loc}' does not exist."
    return True


# ============================================================================
# INTENT HANDLERS - SEARCH
# ============================================================================

def handle_find_location(locations_list):
    """Handle finding a specific location"""
    if not locations_list:
        return "Please specify the location you're looking for."

    for loc in locations_list:
        try:
            location_obj = Location.objects.get(location__iexact=loc)
            location_url = reverse('display_location', args=[location_obj.code])
            return f"Here is the path: {location_url}"
        except Location.DoesNotExist:
            return f"Sorry, I couldn't find a location named '{loc}'."
    return None


def handle_find_by_tags(parameters):
    """Handle finding locations by tags"""
    tag = parameters.get("tags")
    if not tag:
        return "Please provide a tag to search for."
    if isinstance(tag, list):
        tag = tag[0]

    base_url = reverse('locations')
    query_string = urlencode({'search': tag})
    return f"Here's a location search result for tag '{tag}': {base_url}?{query_string}"


# ============================================================================
# INTENT HANDLERS - TRIP MANAGEMENT
# ============================================================================

def handle_start_trip(trip_cart_obj):
    """Handle starting a new trip"""
    trip_cart_obj.locations = []
    trip_cart_obj.start_location = None
    trip_cart_obj.end_location = None
    trip_cart_obj.save()
    return True


def handle_set_start_location(trip_cart_obj, locations_list):
    """Handle setting start location"""
    if locations_list:
        trip_cart_obj.start_location = locations_list[0]
        trip_cart_obj.save()
        return True
    return "Please tell me which location to set as the starting point."


def handle_set_end_location(trip_cart_obj, locations_list):
    """Handle setting end location"""
    if locations_list:
        trip_cart_obj.end_location = locations_list[0]
        trip_cart_obj.save()
        return True
    return "Please tell me which location to set as the ending point."


def handle_add_locations(trip_cart_obj, locations_list):
    """Handle adding locations to trip"""
    updated = False
    for loc in locations_list:
        if loc not in trip_cart_obj.locations:
            trip_cart_obj.locations.append(loc)
            updated = True

    if updated:
        trip_cart_obj.save()
        return True
    return "Those locations are already in your trip."


def handle_remove_locations(trip_cart_obj, locations_list):
    """Handle removing locations from trip"""
    removed, not_found = [], []
    for loc in locations_list:
        if loc in trip_cart_obj.locations:
            trip_cart_obj.locations.remove(loc)
            removed.append(loc)
        else:
            not_found.append(loc)

    trip_cart_obj.save()

    if removed and not not_found:
        return f"Removed {', '.join(removed)} from your trip."
    if removed and not_found:
        return f"Removed {', '.join(removed)}. However, {', '.join(not_found)} were not in your trip list."
    return f"{', '.join(not_found)} is/are not in your trip list."


# ============================================================================
# TRIP COMPLETION - HELPER FUNCTIONS
# ============================================================================

def ensure_start_end_in_locations(trip_cart_obj):
    """Ensure start and end locations are included in trip"""
    locations = trip_cart_obj.locations or []
    start_name = trip_cart_obj.start_location
    end_name = trip_cart_obj.end_location
    existing = [loc.lower() for loc in locations]
    updated = False

    if start_name and start_name.lower() not in existing:
        locations.insert(0, start_name)
        updated = True
        existing = [loc.lower() for loc in locations]

    if end_name and end_name.lower() not in existing:
        locations.append(end_name)
        updated = True

    if updated:
        trip_cart_obj.locations = locations
        trip_cart_obj.save()

    return trip_cart_obj.locations


def get_ordered_locations(locations, start_name, end_name):
    """Get ordered list of location names"""
    middles = [loc for loc in locations if loc not in (start_name, end_name)]
    ordered_names = []

    if start_name:
        ordered_names.append(start_name)
    ordered_names.extend(middles)
    if end_name:
        ordered_names.append(end_name)

    return ordered_names


def validate_and_fetch_locations(ordered_names):
    """Validate that all locations exist in database and fetch them"""
    loc_objs = list(Location.objects.filter(location__in=ordered_names))
    found = {l.location for l in loc_objs}
    missing = [n for n in ordered_names if n not in found]

    if missing:
        logger.warning(f"Missing from DB: {missing}")
    else:
        logger.debug("All names found in DB.")

    name2obj = {l.location: l for l in loc_objs}
    location_list = [name2obj[n] for n in ordered_names if n in name2obj]
    return location_list


def build_distance_graph(location_list):
    """Build graph with distances and durations between locations"""
    coords = [loc.coordinate for loc in location_list]
    index_to_id = {i: location_list[i].id for i in range(len(location_list))}
    distances, durations = [], {}

    for i in range(len(coords)):
        for j in range(len(coords)):
            if i == j:
                continue
            d, t = distance(coords[i], coords[j])
            distances.append((i, j, d))
            durations[(i, j)] = t

    graph = Graph(len(location_list))
    for u, v, w in distances:
        graph.add_edge(u, v, w)

    return graph, durations, index_to_id


def calculate_optimal_path(graph, location_list, end_name):
    """Calculate optimal path using TSP algorithm"""
    start_idx = 0
    end_idx = (len(location_list) - 1) if end_name else start_idx
    logger.debug(f"start_idx={start_idx}, end_idx={end_idx}")

    best_path, total_dist = graph.find_hamiltonian_path(
        fixed_position=None,
        precedence_constraints=None,
        start=start_idx,
        end=end_idx
    )

    if not best_path:
        logger.error("No valid path found.")
        return None, None

    if best_path[0] == best_path[-1]:
        best_path = best_path[:-1]

    logger.debug(f"best_path indices: {best_path}")
    return best_path, total_dist


def create_trip_path_record(user, location_list, best_path, total_dist, durations, index_to_id):
    """Create and save TripPath database record"""
    total_time = sum(
        durations.get((best_path[i], best_path[i + 1]), 0)
        for i in range(len(best_path) - 1)
    )

    trip_list, _ = TripList.objects.get_or_create(user=user)
    middle_ids = [index_to_id[i] for i in best_path[1:-1]]

    trip_path = TripPath.objects.create(
        trip_list=trip_list,
        total_duration=total_time,
        total_distance=total_dist,
        locations_ordered=json.dumps(middle_ids),
        path_name=f"{user.username} chatbot TripPath",
        start_point=location_list[best_path[0]],
        end_point=location_list[best_path[-1]]
    )
    trip_path.locations.add(*[location_list[i] for i in best_path])
    trip_path.save()

    return trip_path, total_time


def format_trip_completion_response(location_list, best_path, total_dist, total_time, trip_list):
    """Format the trip completion response message"""
    itinerary_names = [location_list[i].location for i in best_path]
    logger.debug(f"itinerary: {itinerary_names}")

    header = [
        f"Total distance: {total_dist:.1f} km",
        f"Total duration: {total_time:.1f} minutes",
        ""
    ]
    reply = "\n".join(header + itinerary_names)
    url = reverse('my_trip') + "?" + urlencode({'id': trip_list.id})

    return f"{reply}\n\nView it here: {url}"


def handle_complete_trip(trip_cart_obj, user):
    """Handle trip completion and optimization"""
    locations = trip_cart_obj.locations or []
    start_name = trip_cart_obj.start_location
    end_name = trip_cart_obj.end_location

    if not locations and not (start_name and end_name):
        return "Your trip has no locations. Please add some before finishing."

    locations = ensure_start_end_in_locations(trip_cart_obj)
    ordered_names = get_ordered_locations(locations, start_name, end_name)
    location_list = validate_and_fetch_locations(ordered_names)

    if not location_list:
        return "Unable to find valid locations for your trip."

    graph, durations, index_to_id = build_distance_graph(location_list)
    best_path, total_dist = calculate_optimal_path(graph, location_list, end_name)

    if not best_path:
        return "Unable to generate a valid trip with the selected start/end points."

    trip_path, total_time = create_trip_path_record(
        user, location_list, best_path, total_dist, durations, index_to_id
    )

    return format_trip_completion_response(
        location_list, best_path, total_dist, total_time, trip_path.trip_list
    )


# ============================================================================
# MAIN INTENT ROUTER
# ============================================================================

def handle_intent(request, intent_name, parameters, user=None, session_id=None):
    """Main intent router with reduced complexity"""
    if not session_id:
        try:
            data = json.loads(request.body.decode("utf-8"))
            session_id = extract_session_id(data, parameters)
        except Exception:
            pass

    if not session_id:
        return "Missing session ID."

    if user is None:
        user = get_or_create_temp_user(session_id)

    try:
        trip_cart_obj = get_or_create_temp_cart(session_id, user)
    except Exception:
        return "Could not create or access your trip session."

    locations_list = normalize_locations(parameters.get("locations") or [])

    # Intent routing with handler functions
    intent_handlers = {
        "Default Welcome Intent": lambda: True,
        "Default Fallback Intent": lambda: True,
        "discovering.ability": lambda: True,
        "favourite.add.location": lambda: handle_add_favourite(user, locations_list),
        "favourite.remove.location": lambda: handle_remove_favourite(user, locations_list),
        "find.location.particular": lambda: handle_find_location(locations_list),
        "find.location.tags": lambda: handle_find_by_tags(parameters),
        "start.trip": lambda: handle_start_trip(trip_cart_obj),
        "set.start.location": lambda: handle_set_start_location(trip_cart_obj, locations_list),
        "set.end.location": lambda: handle_set_end_location(trip_cart_obj, locations_list),
        "trip.create.add.location": lambda: handle_add_locations(trip_cart_obj, locations_list),
        "trip.create.remove.location": lambda: handle_remove_locations(trip_cart_obj, locations_list),
        "trip.create.complete": lambda: handle_complete_trip(trip_cart_obj, user),
    }

    handler = intent_handlers.get(intent_name)
    if handler:
        return handler()

    return False


# ============================================================================
# WEBHOOK ENDPOINT
# ============================================================================

@csrf_exempt
def dialogflow_webhook(request):
    """Main webhook endpoint for Dialogflow"""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are allowed."}, status=405)

    try:
        logger.debug(f"Request body: {request.body}")
        data = json.loads(request.body.decode("utf-8"))
        intent_name = data['queryResult']['intent']['displayName']
        parameters = data['queryResult'].get('parameters', {})

        session_id = extract_session_id(data, parameters)
        user_id = data.get('originalDetectIntentRequest', {}) \
                      .get('payload', {}) \
                      .get('userId') or parameters.get('user_id')

        user = None
        if user_id:
            user_model = get_user_model()
            try:
                user = user_model.objects.get(id=user_id)
            except user_model.DoesNotExist:
                user = None

        result = handle_intent(request, intent_name, parameters, user, session_id)

        if result is True:
            return JsonResponse({
                "fulfillmentText": data['queryResult'].get('fulfillmentText', "Done.")
            })
        elif isinstance(result, str):
            return JsonResponse({"fulfillmentText": result})
        else:
            return JsonResponse({"fulfillmentText": "Unhandled case."})

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        traceback.print_exc()
        return JsonResponse({
            "fulfillmentText": "An error occurred while processing your request.",
            "debug": str(e)
        })
